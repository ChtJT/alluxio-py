from typing import Optional, List, Dict, Any

from torch.utils.data import IterableDataset
import pyarrow as pa
import numpy as np
import torch
import os
import lance

from Lance.src.cai_lance.main.model.mapping.column_mapping import ColumnMapping


class LanceTorchDataset(IterableDataset):
    def __init__(
        self,
        lance_uri: str,
        mapping: ColumnMapping,
        tokenizer=None,
        batch_rows: int = 8192,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        self._lance = lance
        self.uri = lance_uri
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.batch_rows = batch_rows
        self.filter = filter
        self.columns = columns

    def __iter__(self):
        ds = self._lance.dataset(self.uri)
        for batch in ds.to_batches(batch_size=self.batch_rows, columns=self.columns, filter=self.filter):
            table = pa.Table.from_batches([batch])
            # transfer the first line: sample dict -> torch tensor
            for i in range(table.num_rows):
                row = table.slice(i, 1).to_pydict()
                yield self._row_to_example(row)

    def _row_to_example(self, row: Dict[str, List[Any]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        # label
        if self.mapping.label:
            val = row[self.mapping.label][0]
            if isinstance(val, str):
                # hash
                lab = (hash(val) % (2**31 - 1))
                out["labels"] = torch.tensor(lab, dtype=torch.long)
            elif isinstance(val, (int, np.integer)):
                out["labels"] = torch.tensor(int(val), dtype=torch.long)
            else:
                out["labels"] = torch.tensor(float(val), dtype=torch.float32)

        if self.mapping.text:
            txt = row[self.mapping.text][0]
            if self.tokenizer:
                enc = self.tokenizer(
                    txt, truncation=True, padding=False, return_tensors="pt"
                )
                for k, v in enc.items():  # input_ids / attention_mask / token_type_ids
                    out[k] = v.squeeze(0)
            else:
                # no tokenizer
                arr = np.frombuffer(str(txt).encode("utf-8"), dtype=np.uint8)
                out["text_ids"] = torch.tensor(arr)

        # features -> float32
        if self.mapping.features:
            chunks = []
            for c in self.mapping.features:
                v = row[c][0]
                chunks.append(np.asarray(v).reshape(-1))
            feats = np.concatenate(chunks, axis=0).astype("float32")
            out["features"] = torch.from_numpy(feats)

        # image -> CHW float32 [0,1]
        if self.mapping.image:
            v = row[self.mapping.image][0]
            img_tensor = None
            from PIL import Image
            import io

            if isinstance(v, dict) and "bytes" in v:
                im = Image.open(io.BytesIO(v["bytes"])).convert("RGB")
                img_tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
            elif isinstance(v, str):
                path = v
                if self.mapping.image_root and not os.path.isabs(path):
                    path = os.path.join(self.mapping.image_root, path)
                im = Image.open(path).convert("RGB")
                img_tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0

            if img_tensor is not None:
                out["pixel_values"] = img_tensor

        return out
