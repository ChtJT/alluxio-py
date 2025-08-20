from abc import ABC, abstractmethod
from typing import Any, Dict
from venv import logger

from Lance.src.blender.main.utils.handle_errors import handle_errors


class BaseConverter(ABC):
    @handle_errors("convert")
    def convert(self, source: str) -> Dict[str, Any]:
        return self._convert_impl(source)

    @abstractmethod
    def _convert_impl(self, source: str) -> Dict[str, Any]:
        ...