"""Pixel helpers shared by the image tools: tools/kit/img.py, tools/kit/diff-set.py and
tools/kit/compare-captures.py import them, so a number means the same in each.

All images are uint8 H x W x 3 RGB arrays (decoded once, never converted to float for a
difference). Every mean of a difference is an exact integer sum divided once, so it does not depend
on the order of summation (the value numpy's own mean() returns).
"""
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# the difference levels counted by default: a pixel counts when its largest channel difference
# exceeds the level
LEVELS = (1, 2, 8, 32)
# the difference amplified x8 and clipped, as a lookup on uint8 differences
X8 = np.minimum(np.arange(256) * 8, 255).astype(np.uint8)
# Rec. 709 luma weights (on the 8-bit code values, not linearised)
LUMA = np.array([0.2126, 0.7152, 0.0722])


def decode(data):
    """A PNG's bytes (or a path) decoded to uint8 RGB."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


class Files:
    """A set of PNGs, each read and decoded once (uint8 H x W x 3)."""

    def __init__(self):
        self.raw, self.img = {}, {}

    def bytes(self, f):
        k = str(f)
        if k not in self.raw:
            self.raw[k] = Path(f).read_bytes()
        return self.raw[k]

    def load(self, f):
        k = str(f)
        if k not in self.img:
            self.img[k] = decode(self.bytes(f))
        return self.img[k]

    def same(self, p, q):
        """Identical pixels: equal bytes, else equal decoded arrays (the bytes depend on the encoder)."""
        if self.bytes(p) == self.bytes(q):
            return True
        A, B = self.load(p), self.load(q)
        return A.shape == B.shape and np.array_equal(A, B)


def stats(A, B, levels=LEVELS):
    """|A - B| (uint8, per channel), its per-pixel max over the channels, and the numbers."""
    D = np.maximum(A, B)
    D -= np.minimum(A, B)
    M = np.maximum(np.maximum(D[..., 0], D[..., 1]), D[..., 2])
    # the mean: an exact integer sum over one division (what D.mean() returns, whatever the order);
    # a row of a frame sums to under 2^32
    total = int(D.reshape(D.shape[0], -1).sum(1, dtype=np.uint32).sum(dtype=np.int64))
    return D, M, {
        "max": int(M.max()),
        "mean": round(float(total / D.size), 5),
        "over": {str(l): int(np.count_nonzero(M > l)) for l in levels},
        "pixels": int(M.size),
    }


def bbox(M, level=0):
    """[x, y, w, h] of the pixels of M over level, or None when there are none."""
    rows, cols = np.flatnonzero((M > level).any(1)), np.flatnonzero((M > level).any(0))
    if not rows.size:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)]


def bias(A, B):
    """The mean signed difference B - A per channel (exact integer sums, one division each)."""
    n = A.shape[0] * A.shape[1]
    return [round(float((int(B[..., k].sum(dtype=np.int64)) - int(A[..., k].sum(dtype=np.int64))) / n), 5) for k in range(3)]


def sheet(A, B, D, box, title):
    """ref | test | diff x8 of one box (x, y, w, h), 1:1, under a title."""
    x, y, w, h = box
    tiles = (A[y:y + h, x:x + w], B[y:y + h, x:x + w], X8[D[y:y + h, x:x + w]])
    im = Image.new("RGB", (w * 3 + 8, h + 18), "#202020")
    for j, t in enumerate(tiles):
        im.paste(Image.fromarray(t), (j * (w + 4), 18))
    ImageDraw.Draw(im).text((4, 3), title, fill="white")
    return im


def metrics(im):
    """compare-captures.py's region metrics: mean RGB, luma mean and spread, local contrast (the
    mean luma difference of adjacent pixels) and the share of bright pixels (luma > 245)."""
    a = np.asarray(im, dtype=np.float64)
    y = a @ LUMA
    dx = np.abs(np.diff(y, axis=1)).mean()
    dy = np.abs(np.diff(y, axis=0)).mean()
    return {
        "meanRGB": np.round(a.mean(axis=(0, 1)), 2).tolist(),
        "lumaMean": round(float(y.mean()), 2),
        "lumaStdDev": round(float(y.std()), 2),
        "meanAdjacentLumaDifference": round(float((dx + dy) / 2), 3),
        "brightPixelPercent": round(float((y > 245).mean() * 100), 3),
    }
