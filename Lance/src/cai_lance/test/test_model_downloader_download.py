import os
from pathlib import Path

import pytest
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from Lance.src.cai_lance.main.downloader.dataset_downloader import DatasetDownloader
from ..main.downloader.model_downloader import ModelDownloader


# def test_model_downloader_download(tmp_path):
#     cache_dir = str(tmp_path / "hf_cache")
#
#     downloader = DatasetDownloader("fka/awesome-chatgpt-prompts", cache_dir=cache_dir)
#     result = downloader.download()
#
#     # assert isinstance(result, dict), "下载结果应为 dict"
#     # model = result.get("model")
#     # tokenizer = result.get("tokenizer")
#     # assert isinstance(model, PreTrainedModel), "返回的 model 类型错误"
#     # assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), \
#     #     "返回的 tokenizer 类型错误"
#     #
#     # assert os.path.isdir(cache_dir), "缓存目录未被创建"
#
#     expected_subdir = os.path.join(cache_dir, "datasets--fka-awesome-chatgpt-prompts")
#     assert os.path.isdir(expected_subdir), f"模型目录 {expected_subdir} 不存在"
#     print("数据下载到：", cache_dir)

def test_dataset_downloader_download(tmp_path):
    tmp_path = Path(tmp_path)
    cache_dir = str(tmp_path / "hf_cache")
    downloader = DatasetDownloader(
        name="jontooy/Flickr8k-Image-Features",
        cache_dir=cache_dir,
    )
    result = downloader.download()
    assert os.path.isdir(cache_dir), "缓存目录未被创建"
    print("数据下载到：", cache_dir)

if __name__ == "__main__":
    test_dataset_downloader_download('jontooy')