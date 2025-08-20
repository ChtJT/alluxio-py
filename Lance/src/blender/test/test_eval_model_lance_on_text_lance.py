import os
import numpy as np
import pyarrow as pa
import lance
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from Lance.src.blender.main.integration.core.mapping import ColumnMapping
from Lance.src.blender.main.integration.pytorch.lance_torch_dataset import LanceTorchDataset


def text_collate_fn(batch):
    out = {}
    if not batch: return out
    keys = set().union(*[b.keys() for b in batch])
    for k in ["input_ids", "attention_mask"]:
        if k in keys:
            seqs = [torch.as_tensor(b[k], dtype=torch.long) for b in batch if k in b]
            out[k] = pad_sequence(seqs, batch_first=True, padding_value=0)
    if "labels" in keys:
        ys = [torch.as_tensor(b["labels"]) for b in batch if "labels" in b]
        out["labels"] = torch.stack(ys, 0)
    return out

def _pick_text_column(ds) -> str:
    schema = ds.schema
    names = schema.names
    preferred = ["text", "src_text", "sentence", "content", "input", "utterance"]
    for p in preferred:
        if p in names and pa.types.is_string(schema.field(p).type):
            return p
    for n in names:
        if pa.types.is_string(schema.field(n).type):
            return n
    raise ValueError("未找到字符串列作为文本。")

class CappedTokenizer:
    def __init__(self, texts_iter, max_vocab: int):
        vocab = {"[PAD]": 0, "[UNK]": 1}
        idx = 2
        for t in texts_iter:
            for tok in str(t).lower().split():
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1
                    if idx >= max_vocab:
                        break
            if idx >= max_vocab:
                break
        self.vocab = vocab
        self.unk = 1
    def __call__(self, text, truncation=True, padding=False, return_tensors=None):
        toks = str(text).lower().split()
        ids = np.array([self.vocab.get(tok, self.unk) for tok in toks], dtype=np.int32)
        attn = np.ones_like(ids, dtype=np.int32)
        return {"input_ids": ids, "attention_mask": attn}

class TinyHead(nn.Module):
    def __init__(self, embedding_weight: torch.Tensor, num_labels: int = 2):
        super().__init__()
        V, D = embedding_weight.shape
        self.emb = nn.Embedding(V, D, padding_idx=0)
        with torch.no_grad():
            self.emb.weight.copy_(embedding_weight)
        self.fc = nn.Linear(D, num_labels)
    def forward(self, input_ids, attention_mask=None):
        x = self.emb(input_ids)  # (B,L,D)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            x = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        else:
            x = x.mean(1)
        return self.fc(x)  # (B,num_labels)

@pytest.mark.parametrize("max_batches", [3])
def test_eval_model_lance_on_text_lance_no_training(tmp_path, max_batches):
    model_uri = os.getenv("MODEL_LANCE_URI")
    data_uri  = os.getenv("DATA_LANCE_URI")
    if not model_uri or not data_uri:
        pytest.skip("请设置环境变量 MODEL_LANCE_URI 和 DATA_LANCE_URI 指向 .lance 文件。")

    ds_model = lance.dataset(model_uri)
    tbl_m = ds_model.to_table(columns=["shape", "data"])
    assert tbl_m.num_rows == 1, "模型 .lance 应只有一行（单张量）"
    shape = tbl_m.column("shape").to_pylist()[0]
    buf   = tbl_m.column("data").to_pylist()[0]
    W = np.frombuffer(buf, dtype=np.float32).reshape(shape)
    assert W.ndim == 2, f"期望二维嵌入矩阵，得到 ndim={W.ndim}"
    V, D = W.shape

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_weight = torch.from_numpy(W).to(device)

    ds_data = lance.dataset(data_uri)
    text_col = _pick_text_column(ds_data)
    head_tbl = ds_data.to_table(columns=[text_col], limit=min(5000, ds_data.count_rows()))
    texts_for_vocab = [row[text_col].as_py() for row in head_tbl]
    tok = CappedTokenizer(texts_for_vocab, max_vocab=V)

    mapping = ColumnMapping(text=text_col)
    torch_ds = LanceTorchDataset(
        lance_uri=data_uri,
        mapping=mapping,
        tokenizer=tok,
        batch_rows=4096,
    )
    dl = DataLoader(
        torch_ds,
        batch_size=16,
        collate_fn=text_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # 4) 前向评测：跑若干 batch，检查形状和数值
    model = TinyHead(emb_weight, num_labels=2).to(device).eval()
    seen = 0
    with torch.no_grad():
        for batch in dl:
            if "input_ids" not in batch:
                pytest.skip("未生成 input_ids；请检查 tokenizer 与文本列。")
            x = batch["input_ids"].to(device)
            m = batch.get("attention_mask")
            m = m.to(device) if m is not None else None
            logits = model(x, m)  # (B,2)
            assert logits.ndim == 2 and logits.shape[1] == 2
            assert torch.isfinite(logits).all()
            probs = F.softmax(logits, dim=-1)
            assert (probs >= 0).all() and (probs <= 1).all()
            seen += 1
            if seen >= max_batches:
                break

    assert seen > 0, "未遍历到任何 batch，请检查数据 .lance 内容。"
