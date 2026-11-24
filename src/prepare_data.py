"""Download CIFAR-10 (official uoft-cs/cifar10 copy on the Hugging Face Hub) and
cache it as uint8 numpy arrays: data/cifar10_{train,test}.npz.

Identical content to https://www.cs.toronto.edu/~kriz/cifar.html
(50,000 train / 10,000 test, 32x32 RGB, 10 classes); the HF mirror is used
only because it downloads much faster.
"""
import io
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from common import DATA_DIR

URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ("train", "test"):
        out = os.path.join(DATA_DIR, f"cifar10_{split}.npz")
        if os.path.exists(out):
            continue
        pqf = os.path.join(DATA_DIR, f"{split}.parquet")
        if not os.path.exists(pqf):
            urllib.request.urlretrieve(URL.format(split), pqf)
        t = pq.read_table(pqf).to_pydict()
        imgs = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
        labels = np.array(t["label"], dtype=np.int64)
        assert imgs.shape[1:] == (32, 32, 3) and imgs.dtype == np.uint8
        np.savez(out, x=imgs, y=labels)
        print(split, imgs.shape, np.bincount(labels))


if __name__ == "__main__":
    main()
