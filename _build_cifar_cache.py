"""Build a local CIFAR-10 .npz cache from the HuggingFace parquet mirror.

The torchvision default mirror (cs.toronto.edu) is throttled on some VMs
(~30 KB/s), while the HF CDN serves the same images at >100 MB/s.  This decodes
the `uoft-cs/cifar10` parquet into the uint8 [N,32,32,3] arrays that
`metasmooth.load_cifar_subset` reads from `{data_dir}/cifar10_train.npz`.

Pixels are identical to torchvision's (PNG is lossless); only the example
*order* may differ, which is irrelevant — the subset is drawn by a fixed seed.

Usage: python _build_cifar_cache.py <data_dir>
"""
from __future__ import annotations

import io
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HF = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
SPLITS = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def build(split: str, fname: str, data_dir: str) -> None:
    url = f"{HF}/{fname}"
    print(f"[{split}] downloading {url}", flush=True)
    raw = urllib.request.urlopen(url).read()
    table = pq.read_table(io.BytesIO(raw))
    cols = table.column_names
    print(f"[{split}] {table.num_rows} rows, columns={cols}", flush=True)

    img_col = table.column("img").to_pylist()  # list of {'bytes':..., 'path':...}
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)

    images = np.empty((len(img_col), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(img_col):
        arr = np.array(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
        images[i] = arr
    out = f"{data_dir}/cifar10_{split}.npz"
    np.savez(out, images=images, labels=labels)
    print(f"[{split}] wrote {out}  images={images.shape}{images.dtype} "
          f"labels={labels.shape} (classes {sorted(set(labels.tolist()))})",
          flush=True)


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    import os
    os.makedirs(data_dir, exist_ok=True)
    for split, fname in SPLITS.items():
        build(split, fname, data_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
