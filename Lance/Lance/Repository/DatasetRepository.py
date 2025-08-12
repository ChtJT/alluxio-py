from abc import ABC, abstractmethod
import lance, pyarrow as pa, pandas as pd
from pathlib import Path
from datetime import datetime

class DatasetRepository(ABC):
    @abstractmethod
    def init_meta(self): ...
    @abstractmethod
    def append_meta(self, record: dict): ...
    @abstractmethod
    def query(self, name: str) -> pd.DataFrame: ...