import argparse
import io
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import fsspec
import pyarrow as pa
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

import lance
from Lance.src.blender.main.converter.png_converter import PngConverter
from Lance.src.blender.main.converter.tensor_converter import TensorConverter
from Lance.src.blender.main.downloader.dataset_downloader import (
    DatasetDownloader,
)
from Lance.src.blender.main.downloader.model_downloader import ModelDownloader


def _write_lance_overwrite(tbl: pa.Table, uri: str, so_lance: Dict[str, str]):
    import fsspec

    fs = fsspec.filesystem(
        "s3",
        key=so_lance["aws_access_key_id"],
        secret=so_lance["aws_secret_access_key"],
        client_kwargs={
            "endpoint_url": so_lance["endpoint"],
            "region_name": so_lance.get("region", "us-east-1"),
        },
        use_ssl=so_lance["endpoint"].lower().startswith("https://"),
        anon=False,
    )
    if fs.exists(uri):
        fs.rm(uri, recursive=True)
    lance.write_dataset(tbl, uri, storage_options=so_lance)


class LanceImageDataset(Dataset):
    def __init__(
        self,
        uri: str,
        storage_options: Dict[str, Any],
        limit: Optional[int] = None,
    ):
        ds = lance.dataset(uri, storage_options=storage_options)
        t = ds.to_table(limit=limit) if limit else ds.to_table()
        rows = t.to_pylist()
        self.images = [r["image"] for r in rows]
        self.labels = [int(r["label"]) for r in rows]
        self.tfm = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        return self.tfm(img), torch.tensor(self.labels[i], dtype=torch.long)


def _mk_s3fs_kwargs(o: Dict[str, Any]) -> Dict[str, Any]:
    k = o.get("aws_access_key_id") or o.get("key")
    s = o.get("aws_secret_access_key") or o.get("secret")
    r = o.get("region") or o.get("client_kwargs", {}).get("region_name")
    e = o.get("endpoint_override") or o.get("client_kwargs", {}).get(
        "endpoint_url"
    )
    kw = {}
    if k:
        kw["key"] = k
    if s:
        kw["secret"] = s
    ck = {}
    if e:
        ck["endpoint_url"] = e
    if r:
        ck["region_name"] = r
    if ck:
        kw["client_kwargs"] = ck
    return kw


def write_bytes_to_s3(data: bytes, s3_uri: str, so: Dict[str, str]):
    ep = so["endpoint"]
    use_ssl = ep.lower().startswith("https://")
    fs = fsspec.filesystem(
        "s3",
        key=so["aws_access_key_id"],
        secret=so["aws_secret_access_key"],
        client_kwargs={
            "endpoint_url": ep,
            "region_name": so.get("region", "us-east-1"),
        },
        use_ssl=use_ssl,
        anon=False,
    )
    with fs.open(s3_uri, "wb") as f:
        f.write(data)


def _resolve_ckpt_path(
    mres: Dict[str, Any], cache_dir: str, model_name: str
) -> str:
    kind = mres.get("kind")
    if kind == "url":
        p = mres["path"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"url path missing: {p}")
        return p
    if kind == "transformers":
        tmp = os.path.join(
            cache_dir, f"{Path(model_name).name or 'model'}_state.pth"
        )
        torch.save(mres["model"].state_dict(), tmp)
        return tmp
    root = Path(mres["path"])
    exts = [".pth", ".pt", ".bin", ".safetensors"]
    cands: List[Path] = []
    for ext in exts:
        cands.extend(root.rglob(f"*{ext}"))
    if not cands:
        listed = [
            str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        ]
        raise FileNotFoundError(
            f"no checkpoint under {root}; "
            f"looked for {exts}; "
            f"found: {listed[:50]}"
        )
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(cands[0])


def _model_tag_from_path(p: str) -> str:
    b = os.path.basename(p.split("?")[0])
    n, _ = os.path.splitext(b)
    return n or "model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./_cache")
    ap.add_argument(
        "--model",
        default="https://download.pytorch.org/models/resnet18-f37072fd.pth",
    )
    ap.add_argument(
        "--model_mode", default="url", choices=["url", "repo", "transformers"]
    )
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument(
        "--dataset_mode", default="arrow", choices=["repo", "arrow"]
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--s3-endpoint", default="http://localhost:9000")
    ap.add_argument("--aws-access-key", default="minioadmin")
    ap.add_argument("--aws-secret-key", default="minioadmin123")
    ap.add_argument("--aws-region", default="us-east-1")
    ap.add_argument("--bucket", default="my-bucket")
    ap.add_argument("--prefix", default="resnet_demo")
    args = ap.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    so = {
        "aws_access_key_id": str(args.aws_access_key),
        "aws_secret_access_key": str(args.aws_secret_key),
        "region": str(args.aws_region or "us-east-1"),
        "endpoint": str(args.s3_endpoint),
    }
    so_lance = {
        "aws_access_key_id": so["aws_access_key_id"],
        "aws_secret_access_key": so["aws_secret_access_key"],
        "region": so["region"],
        "endpoint": so["endpoint"],  # endpoint
        "allow_http": "true",  # string
        "force_path_style": "true",
        "virtual_hosted_style": "false",
    }

    md = ModelDownloader(
        name=args.model, cache_dir=args.cache, mode=args.model_mode
    )
    mres = md.download()
    ckpt_path = _resolve_ckpt_path(
        mres, cache_dir=args.cache, model_name=args.model
    )

    weights_uri = f"s3://{args.bucket}/{args.prefix}/weights/{_model_tag_from_path(ckpt_path)}.lance"
    TensorConverter().convert_and_write_full(
        ckpt_path,
        weights_uri,
        storage_options=so_lance,
        overwrite=True,
    )

    dd = DatasetDownloader(
        name=args.dataset,
        cache_dir=args.cache,
        mode=args.dataset_mode,
        split=args.split,
    )
    d = dd.download()
    split_dir = d.get(args.split) if isinstance(d, dict) else d

    if args.dataset_mode == "arrow":
        img_tbl = PngConverter.images_to_lance_from_arrow(
            split_dir, limit=args.limit
        )
    else:
        img_tbl = PngConverter.images_to_lance_from_files(
            split_dir, limit=args.limit
        )

    images_uri = f"s3://{args.bucket}/{args.prefix}/datasets/{args.dataset}-{args.split}.lance"
    _write_lance_overwrite(img_tbl, images_uri, so_lance)

    train_ds = LanceImageDataset(images_uri, storage_options=so_lance)
    num_classes = len(set(train_ds.labels))
    loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    missing = TensorConverter.load_from_lance_into_model(
        weights_uri,
        model,
        storage_options=so_lance,
        strict=False,
    )

    def _all_fc(keys):
        return all(k.startswith("fc.") for k in keys)

    if not (
        _all_fc(missing.missing_keys) and _all_fc(missing.unexpected_keys)
    ):
        raise RuntimeError(
            f"Unexpected state_dict mismatch:\n"
            f"  missing_keys={missing.missing_keys}\n"
            f"  unexpected_keys={missing.unexpected_keys}\n"
            f"It is expected that only fc.* will not match (because num_classes != 1000)。"
        )

    print(
        f"[weights] loaded backbone; skipped classifier: "
        f"missing={missing.missing_keys}, unexpected={missing.unexpected_keys}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        n = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(
                device, non_blocking=True
            )
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            n += 1
            if n >= args.steps:
                break

    out_local = os.path.join(args.cache, "resnet18_trained.pth")
    torch.save(model.state_dict(), out_local)
    with open(out_local, "rb") as f:
        write_bytes_to_s3(
            f.read(),
            f"s3://{args.bucket}/{args.prefix}/models/resnet18_trained.pth",
            so,
        )


if __name__ == "__main__":
    main()
