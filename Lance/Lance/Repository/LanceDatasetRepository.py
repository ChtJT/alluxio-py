from pathlib import Path
import lance, pyarrow as pa
from pathlib import Path
from datetime import datetime

from lance.Lance.Repository.DatasetRepository import DatasetRepository


class LanceDatasetRepository(DatasetRepository):
    def __init__(self, meta_uri: str):
        self.meta_path = Path(meta_uri)
        self.init_meta()

    def init_meta(self):
        if not self.meta_path.exists():
            schema = pa.schema([...])
            empty = pa.Table.from_pydict({...}, schema=schema, mapping=None)
            lance.write_dataset(empty, str(self.meta_path))

    def append_meta(self, record: dict):
        table = pa.Table.from_pydict(record, mapping=None)
        lance.write_dataset(table, str(self.meta_path), mode="append")

    def query(self, name: str):
        ds = lance.dataset(str(self.meta_path))
        table = ds.to_table(filter=(pa.compute.equal(ds["lance"], name)))
        return table.to_pandas()