import os
from pathlib import Path
import numpy as np
import pytest
import torch

from Lance.src.cai_lance.main.converter.tensor_converter import TensorConverter
from Lance.src.cai_lance.main.downloader.model_downloader import ModelDownloader


def _fake_snapshot_download(repo_id, repo_type, local_dir, local_dir_use_symlinks, resume_download, revision=None, token=None, **_):
    """
    伪造 huggingface_hub.snapshot_download：
    在 local_dir 下创建一个“模型仓库”并写入一个很小的 .pth 文件。
    """
    os.makedirs(local_dir, exist_ok=True)
    # 建一个子目录模拟常见结构
    weights_dir = Path(local_dir) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 伪造一份很小的权重字典
    small_tensor = torch.randn(2, 3)
    state = {"linear.weight": small_tensor}

    # 保存为 .pth
    pth_path = weights_dir / "tiny_model.pth"
    torch.save(state, pth_path)

    # 顺便放点其它文件，模拟真实仓库
    (Path(local_dir) / "config.json").write_text('{"dummy":"ok"}', encoding="utf-8")

    return str(local_dir)

@pytest.mark.usefixtures()
def test_model_repo_pth_to_tensor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "Surfrider/mobilenet_v3_pretrained.pth",
        _fake_snapshot_download
    )

    cache_dir = tmp_path / "hf_cache"
    dl = ModelDownloader(
        name="someone/tiny-model",   # 名字随便，反正被打桩了
        cache_dir=str(cache_dir),
        mode="repo",                 # 走“原样仓库文件”下载
    )
    result = dl.download()
    repo_root = Path(result["path"]) if isinstance(result, dict) else Path(result)

    assert repo_root.is_dir(), f"模型仓库路径不存在：{repo_root}"

    # 找到刚才“下载”的 .pth
    pth_files = list(repo_root.rglob("*.pth")) + list(repo_root.rglob("*.pt"))
    assert pth_files, "未找到任何 .pth/.pt 文件"

    conv = TensorConverter()

    converted = 0
    for pth in pth_files:
        out = conv.convert(str(pth))
        assert "tensor" in out, f"转换结果缺少 'tensor' 键：{pth}"
        arr = out["tensor"]
        assert isinstance(arr, np.ndarray), f"应返回 numpy.ndarray，实际是"
