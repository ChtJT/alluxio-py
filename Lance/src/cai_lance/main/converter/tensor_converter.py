from typing import Dict, Any
from Lance.src.cai_lance.main.base.base_converter import BaseConverter
import torch

class TensorConverter(BaseConverter):
    def _convert_impl(self, source: str) -> Dict[str, Any]:
        tensor = torch.load(source, map_location="cpu")
        if isinstance(tensor, dict):
            tensor = next(iter(tensor.values()))
        arr = tensor.detach().cpu().numpy()

        return {
            "uri": source,
            "tensor": arr,
        }