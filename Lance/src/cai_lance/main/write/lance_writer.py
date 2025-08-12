from typing import Iterable, Optional, Dict, Any

import lance
import pyarrow as pa

class LanceWriter:
    def __init__(self, path: str, mode: str="overwrite", primary_key: Optional[str]=None):
        self.path = path
        self.mode = mode
        self.primary_key = primary_key

    def write_batch(self, records: Iterable[Dict[str,Any]]):
        # 一次写一个 batch 到 Lance
        table = pa.Table.from_pylist(list(records))
        lance.write_dataset(
            table,
            self.path,
            mode=self.mode,
        )
        # 下一次 batch 用 append
        self.mode = "append"