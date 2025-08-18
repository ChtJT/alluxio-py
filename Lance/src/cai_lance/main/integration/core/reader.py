from typing import Any, Dict, Generator, Iterable, Optional, List
import numpy as np
import pyarrow as pa
import lance
from .mapping import ColumnMapping
from .utils import stable_label_id
from . import transforms as T

class LanceExampleReader:
    """
    统一的数据读取与样本“标准化”层：输出 Python/NumPy 为主的 dict（跨框架无关）
    """
    def __init__(
        self,
        lance_uri: str,
        mapping: ColumnMapping,
        tokenizer=None,
        batch_rows: int = 8192,
        filter: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ):
        self.uri = lance_uri
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.batch_rows = batch_rows
        self.filter = filter
        self.columns = columns

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        ds = lance.dataset(self.uri)
        for batch in ds.to_batches(batch_size=self.batch_rows, columns=self.columns, filter=self.filter):
            table = pa.Table.from_batches([batch])
            for i in range(table.num_rows):
                # pyarrow Row -> python dict（列名 -> 单值）
                row = {name: table.column(name)[i].as_py() for name in table.schema.names}
                yield self._row_to_example(row)

    def _row_to_example(self, row: Dict[str, Any]) -> Dict[str, Any]:
        ex: Dict[str, Any] = {}

        # label
        if self.mapping.label:
            ex["labels"] = stable_label_id(row[self.mapping.label])

        # text
        if self.mapping.text:
            txt = row[self.mapping.text]
            if self.tokenizer:
                # HuggingFace tokenizer: 返回 numpy/py 值，框架适配层再转 tensor
                enc = self.tokenizer(str(txt), truncation=True, padding=False, return_tensors=None)
                # 统一每个字段为 list[int] / list[list[int]] -> np.array
                for k, v in enc.items():
                    ex[k] = np.asarray(v, dtype=np.int32)
            else:
                # 不用 tokenizer 时，给出原始 UTF-8 bytes（下游可自定义）
                ex["text_bytes"] = np.frombuffer(str(txt).encode("utf-8"), dtype=np.uint8)

        # features
        if self.mapping.features:
            ex["features"] = T.to_feature_vector(row, self.mapping.features)

        # image
        if self.mapping.image:
            img = T.decode_image(row[self.mapping.image], self.mapping.image_root)  # HWC float32 [0,1]
            ex["pixel_values"] = np.transpose(img, (2, 0, 1))  # CHW

        return ex
