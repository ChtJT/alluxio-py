import os
from pathlib import Path
from Lance.src.cai_lance.main.downloader.dataset_downloader import DatasetDownloader


def test_model_downloader_download(tmp_path):
    cache_dir = str(tmp_path / "hf_cache")

    downloader = DatasetDownloader("fka/awesome-chatgpt-prompts", cache_dir=cache_dir)
    result = downloader.download()
    print(result)

    expected_subdir = os.path.join(cache_dir, "datasets--fka-awesome-chatgpt-prompts")
    assert os.path.isdir(expected_subdir), f"模型目录 {expected_subdir} 不存在"
    print("数据下载到：", cache_dir)

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