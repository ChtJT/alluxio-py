import os
import io
import math
import json
import random
import numpy as np
import pyarrow as pa
import lance
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torch.utils.data import DataLoader

from Lance.src.cai_lance.main.integration.core.mapping import ColumnMapping
from Lance.src.cai_lance.main.integration.pytorch.collate import text_collate_fn, image_collate_fn
from Lance.src.cai_lance.main.integration.pytorch.lance_torch_dataset import LanceTorchDataset
from Lance.src.cai_lance.main.integration.pytorch.transforms import make_image_transform


# -------------------------------
# utils
# -------------------------------
def _maybe_cuda(model):
    return model.cuda() if torch.cuda.is_available() else model

def _to_device(batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def test_text_training(tmp_path):
    # 1) 准备 Lance 数据
    ds_uri = str(tmp_path / "text_cls.lance")
    texts = _write_lance_text(ds_uri)

    # 2) tokenizer + mapping + dataset / dataloader
    tok = SimpleTokenizer.build_from_texts(texts)
    mapping = ColumnMapping(text="text", label="label")

    train_ds = LanceTorchDataset(
        lance_uri=ds_uri,
        mapping=mapping,
        tokenizer=tok,        # 直接在 reader 里生成 input_ids/attention_mask
        batch_rows=4096,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=4,
        collate_fn=text_collate_fn,
        num_workers=0,       # 单元测试：0 更稳
        pin_memory=torch.cuda.is_available(),
    )

    model = _maybe_cuda(TinyTextClassifier(vocab_size=len(tok.vocab), dim=32, num_labels=2))
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # 4) 训练若干 step，验证 loss 下降
    model.train()
    initial_loss = None
    last_loss = None

    for epoch in range(5):
        for batch in train_dl:
            batch = _to_device(batch)
            out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), labels=batch["labels"])
            loss = out["loss"]
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            last_loss = loss.item()

    assert initial_loss is not None and last_loss is not None
    assert last_loss < initial_loss, f"loss did not decrease: {initial_loss:.4f} -> {last_loss:.4f}"

def _make_square_image(color, size=32):
    img = Image.new("RGB", (size, size), color=color)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return Image.open(bio)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x, labels=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels.long())
        return out

def _write_lance_images(uri: str, img_dir: str):
    os.makedirs(img_dir, exist_ok=True)
    paths, labels = [], []

    # 生成 8 张 32x32 PNG：4 张红色(0)，4 张绿色(1)
    colors = [("red", (255, 0, 0), 0), ("green", (0, 255, 0), 1)]
    idx = 0
    for name, rgb, lab in colors:
        for _ in range(4):
            img = Image.new("RGB", (32, 32), color=rgb)
            p = os.path.join(img_dir, f"{name}_{idx}.png")
            img.save(p)
            paths.append(f"{name}_{idx}.png")  # 存相对路径
            labels.append(lab)
            idx += 1

    table = pa.table({
        "image": pa.array(paths, type=pa.string()),
        "label": pa.array(labels, type=pa.int64()),
    })
    lance.write_dataset(table, uri, mode="overwrite")


# def test_image_training(tmp_path):
#     # 1) 准备 Lance 图像数据（相对路径 + image_root）
#     ds_uri = str(tmp_path / "image_cls.lance")
#     img_root = str(tmp_path / "imgs")
#     _write_lance_images(ds_uri, img_root)
#
#     mapping = ColumnMapping(image="image", label="label", image_root=img_root)
#
#     img_ds = LanceTorchDataset(
#         lance_uri=ds_uri,
#         mapping=mapping,
#         tokenizer=None,
#         batch_rows=2048,
#         sample_transform=make_image_transform(size=32, augment=False),
#     )
#     img_dl = DataLoader(
#         img_ds,
#         batch_size=4,
#         collate_fn=image_collate_fn,
#         num_workers=0,
#         pin_memory=torch.cuda.is_available(),
#     )
#
#     # 3) 模型与优化器
#     model = _maybe_cuda(TinyCNN(num_classes=2))
#     optim = torch.optim.AdamW(model.parameters(), lr=5e-3)
#
#     # 4) 训练若干 step，验证 loss 下降（此数据极易分）
#     model.train()
#     initial_loss, last_loss = None, None
#
#     for epoch in range(8):
#         for batch in img_dl:
#             batch = _to_device(batch)
#             out = model(batch["pixel_values"], labels=batch["labels"])
#             loss = out["loss"]
#             if initial_loss is None:
#                 initial_loss = loss.item()
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#             last_loss = loss.item()
#
#     assert initial_loss is not None and last_loss is not None
#     assert last_loss < initial_loss, f"loss did not decrease: {initial_loss:.4f} -> {last_loss:.4f}"
