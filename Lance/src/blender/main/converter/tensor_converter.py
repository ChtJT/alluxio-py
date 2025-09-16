import json
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import fsspec
import numpy as np
import pyarrow as pa
import torch

import lance
from Lance.src.blender.main.base.base_converter import BaseConverter


class TensorConverter(BaseConverter):
    # ---------- 基础加载 ----------
    def _load_any(self, source: str):
        ext = os.path.splitext(source)[1].lower()
        if ext == ".safetensors":
            from safetensors.torch import load_file

            return load_file(source)
        try:
            return torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(source, map_location="cpu")

    def _extract_state_dict(self, obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(obj, dict):
            # 1) 直接是 state_dict
            if obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
                return obj
            # 2) 常见包装
            for key in ("state_dict", "model", "weights"):
                if key in obj and isinstance(obj[key], dict):
                    inner = obj[key]
                    if inner and all(
                        isinstance(v, torch.Tensor) for v in inner.values()
                    ):
                        return inner
        # 兜底：单 tensor 或其它结构 -> 失败
        raise ValueError("checkpoint 不包含完整的 state_dict（需要 name->tensor 映射）")

    # ---------- state_dict <-> Lance 表 ----------
    def _state_to_table(self, sd: Dict[str, torch.Tensor]) -> pa.Table:
        names: List[str] = []
        datas: List[bytes] = []
        dtypes: List[str] = []
        shapes: List[str] = []

        for name, t in sd.items():
            if not isinstance(t, torch.Tensor):
                continue
            arr = t.detach().cpu().numpy()
            names.append(name)
            datas.append(arr.tobytes(order="C"))
            dtypes.append(str(arr.dtype))
            shapes.append(json.dumps(list(arr.shape)))

        return pa.table(
            {
                "name": pa.array(names, type=pa.string()),
                "data": pa.array(datas, type=pa.binary()),
                "dtype": pa.array(dtypes, type=pa.string()),
                "shape": pa.array(shapes, type=pa.string()),
            }
        )

    def _table_to_state(self, tbl: pa.Table) -> Dict[str, torch.Tensor]:
        sd: Dict[str, torch.Tensor] = {}
        names = tbl["name"].to_pylist()
        datas = tbl["data"].to_pylist()
        dtypes = tbl["dtype"].to_pylist()
        shapes = [json.loads(s) for s in tbl["shape"].to_pylist()]
        for n, b, dt, shp in zip(names, datas, dtypes, shapes):
            np_arr = np.frombuffer(b, dtype=np.dtype(dt)).reshape(shp)
            sd[n] = torch.from_numpy(np_arr)
        return sd

    # ---------- BaseConverter 接口：保持向后兼容（返回 first tensor + 表） ----------
    def _convert_impl(self, source: str) -> Dict[str, Any]:
        obj = self._load_any(source)
        sd = self._extract_state_dict(obj)
        tbl = self._state_to_table(sd)

        # 为兼容你旧逻辑，额外返回“第一个 tensor 的 numpy”
        first_name = next(iter(sd))
        first_arr = (
            sd[first_name]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        return {"uri": source, "tensor": first_arr, "table": tbl}

    # ---------- 一步写入 Lance（整套权重） ----------
    def convert_and_write_full(
        self,
        source: str,
        lance_uri: str,
        *,
        storage_options: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ) -> str:
        obj = self._load_any(source)
        sd = self._extract_state_dict(obj)
        tbl = self._state_to_table(sd)
        if overwrite:
            fs = fsspec.filesystem(
                "s3",
                key=(storage_options or {}).get("aws_access_key_id"),
                secret=(storage_options or {}).get("aws_secret_access_key"),
                client_kwargs={
                    "endpoint_url": (storage_options or {}).get("endpoint")
                    or (storage_options or {}).get("endpoint_override"),
                    "region_name": (storage_options or {}).get(
                        "region", "us-east-1"
                    ),
                },
                use_ssl=str((storage_options or {}).get("endpoint", ""))
                .lower()
                .startswith("https://"),
                anon=False,
            )
            if fs.exists(lance_uri):
                fs.rm(lance_uri, recursive=True)
        lance.write_dataset(
            tbl, lance_uri, storage_options=storage_options or {}
        )
        return lance_uri

    # ---------- 从 Lance 读取并 load 到模型 ----------
    @staticmethod
    def load_from_lance_into_model(
        lance_uri: str,
        model: torch.nn.Module,
        *,
        storage_options: Optional[Dict[str, str]] = None,
        strict: bool = False,
        strip_module_prefix: bool = True,
    ):
        ds = lance.dataset(lance_uri, storage_options=storage_options or {})
        tbl = ds.to_table()
        conv = TensorConverter()
        sd = conv._table_to_state(tbl)

        if strip_module_prefix:
            from collections import OrderedDict

            fixed = OrderedDict()
            for k, v in sd.items():
                nk = k[7:] if k.startswith("module.") else k
                fixed[nk] = v
            sd = fixed

        missing = model.load_state_dict(sd, strict=strict)
        return missing
