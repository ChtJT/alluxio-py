import io
from typing import Any
from typing import Dict

import numpy as np
import pyarrow as pa
from PIL import Image

from Lance.src.blender.main.base.base_converter import BaseConverter
from Lance.src.blender.main.utils.find_files import find_files


class PngConverter(BaseConverter):
    def _convert_impl(self, source: str) -> Dict[str, Any]:
        with open(source, "rb") as f:
            img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        ch = len(img.getbands())
        fmt = (img.format or "PNG").lower()
        return {
            "uri": source,
            "data": img_bytes,
            "height": h,
            "width": w,
            "channels": ch,
            "format": fmt,
        }

    @staticmethod
    def images_to_lance_from_arrow(split_dir: str, limit: int) -> pa.Table:
        from datasets import load_from_disk

        ds = load_from_disk(split_dir)
        imgs, labels, hh, ww, cc = [], [], [], [], []
        sel = ds.select(range(min(limit, len(ds))))
        for ex in sel:
            img = ex.get("image", ex.get("img"))
            pil = (
                img
                if hasattr(img, "convert")
                else Image.fromarray(np.array(img))
            )
            bio = io.BytesIO()
            pil.save(bio, format="PNG")
            data = bio.getvalue()
            im = Image.open(io.BytesIO(data))
            imgs.append(data)
            labels.append(int(ex["label"]))
            hh.append(im.height)
            ww.append(im.width)
            cc.append(len(im.getbands()))
        return pa.table(
            {
                "image": pa.array(imgs, type=pa.binary()),
                "height": pa.array(hh, type=pa.int32()),
                "width": pa.array(ww, type=pa.int32()),
                "channels": pa.array(cc, type=pa.int32()),
                "label": pa.array(labels, type=pa.int32()),
            }
        )

    @staticmethod
    def images_to_lance_from_files(dataset_dir: str, limit: int) -> pa.Table:
        conv = PngConverter()
        imgs, labels, h, w, c = [], [], [], [], []
        label_map: Dict[str, int] = {}
        idx = 0
        files = find_files(
            dataset_dir,
            [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"],
            limit=None,
        )
        for p in files[:limit]:
            out = conv.convert(str(p))
            cls = p.parent.name
            if cls not in label_map:
                label_map[cls] = idx
                idx += 1
            imgs.append(out["data"])
            h.append(out["height"])
            w.append(out["width"])
            c.append(out["channels"])
            labels.append(label_map[cls])
        return pa.table(
            {
                "image": pa.array(imgs, type=pa.binary()),
                "height": pa.array(h, type=pa.int32()),
                "width": pa.array(w, type=pa.int32()),
                "channels": pa.array(c, type=pa.int32()),
                "label": pa.array(labels, type=pa.int32()),
            }
        )
