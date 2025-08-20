import os

import numpy as np
import pytest

from Lance.src.cai_lance.main.converter.tensor_converter import TensorConverter
import torch
import lance
from huggingface_hub.utils import HfHubHTTPError
import pyarrow as pa

from Lance.src.cai_lance.main.downloader.model_downloader import ModelDownloader


def test_hf_repo_pth_to_lance_via_tensor_converter(tmp_path):

    repo_id = "OpenNMT/nllb-200-onmt"
    max_files = int(os.getenv("MAX_FILES", "1"))  # 为避免超大下载/内存，默认取 1

    # 1) 下载仓库（真实联网）
    try:
        dl = ModelDownloader(name=repo_id, cache_dir=str(tmp_path), mode="repo")
        repo_path = dl.download()["path"]
    except HfHubHTTPError as e:
        pytest.skip(f"Hugging Face 下载失败或离线：{e}")
    except Exception as e:
        pytest.fail(f"ModelDownloader 运行失败：{e}")

    # 2) 遍历 .pt/.pth，限制数量
    pth_files = []
    for root, _, files in os.walk(repo_path):
        for fn in files:
            if fn.lower().endswith((".pt", ".pth")):
                pth_files.append(os.path.join(root, fn))
    if not pth_files:
        pytest.skip(f"仓库 {repo_id} 中未找到 .pt/.pth 文件")

    pth_files.sort()
    pth_files = pth_files[:max_files]

    out_dir = tmp_path / "lance_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    conv = TensorConverter()

    produced = {}
    for p in pth_files:
        try:
            out = conv.convert(p)
        except AttributeError:
            out = conv._convert_impl(p)
        arr = out["tensor"]
        assert isinstance(arr, np.ndarray), f"TensorConverter 未返回 ndarray：{type(arr)}"

        stem = os.path.splitext(os.path.basename(p))[0]
        lance_uri = str(out_dir / f"{stem}.lance")
        table = pa.table({
            "name":  pa.array([os.path.basename(p)], type=pa.string()),
            "dtype": pa.array([str(arr.dtype)], type=pa.string()),
            "shape": pa.array([list(arr.shape)], type=pa.list_(pa.int32())),
            "numel": pa.array([int(arr.size)], type=pa.int64()),
            "data":  pa.array([arr.astype(np.float32, copy=False).tobytes()], type=pa.binary()),
        })
        lance.write_dataset(table, lance_uri, mode="overwrite")
        produced[p] = (lance_uri, arr.astype(np.float32, copy=False))  # 记录基准

    for pth_path, (lance_uri, ref_arr) in produced.items():
        ds = lance.dataset(lance_uri)
        tbl = ds.to_table(columns=["shape", "data"])
        assert tbl.num_rows == 1
        shape = tbl.column("shape").to_pylist()[0]
        data_bytes = tbl.column("data").to_pylist()[0]
        back = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)

        np.testing.assert_allclose(back, ref_arr, rtol=1e-6, atol=1e-6), f"不一致：{pth_path}"