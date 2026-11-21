# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Measures and arranges captures: differences, crops, sheets, thumbnails, means, statistics, flicker, ring profiles.

It does the image work around captures (tools/kit/capture.mjs, capture-set, capture-clip): how two
captures or sets differ, a region's colour and light, a crop or a labelled sheet for the eye, a
clip's flicker. Use it instead of a Pillow or NumPy script of one's own. Its numbers are
diff-set.py's and compare-captures.py's (the shared helpers in tools/kit/pixels.py). How to check a
visual change with it: skills/verifying-rendering-changes/SKILL.md.

Usage:
  uv run tools/kit/img.py diff  A B            [region] [--thr 1,2,8,32] [--bbox-thr 0] [--heat P] [--json]
  uv run tools/kit/img.py crop  IMG|DIR...     region [--zoom N] --out DIR
  uv run tools/kit/img.py sheet IMG|DIR...     [region] [--labels a,b] [--cols N] [--diff] [--zoom N | --scale F | --width W] --out FILE
  uv run tools/kit/img.py thumb IMG|DIR...     [--short 360] [--quality 88] --out DIR
  uv run tools/kit/img.py mean  IMG|DIR...     [--codes] --out FILE
  uv run tools/kit/img.py stats IMG|DIR...     [region] [--pct 1,5,50,95,99] [--oklab] [--linear] [--split] [--cover] [--mask M [--against B]] [--json]
  uv run tools/kit/img.py stats IMG|DIR... --deck [--deck-boxes auto|h|sun|none] [region] [--deck-rows [--horizon x0,x1,y]] [--trend 15] [--px-per-deg 24] [--json]
  uv run tools/kit/img.py flicker CLIP|DIR...  [region] [--frames a-b] [--spike 20,40] [--odd 2.5] [--drop-odd] [--map P] [--json] [--save]
  uv run tools/kit/img.py radial IMG|DIR...    --centre x,y | --centre-key KEY  [--rings 0,2,4,8,15,30,50] [--fov F] [--half up] [--json]

Options:
  region        --box [name=]x,y,w,h (repeatable) and/or --anchor name[,name]|all [--anchors FILE.json]
  --match glob  the files a directory stands for (default *.png, sorted by name)
  --gain g      the scale of diff's heat map, sheet's --diff column and flicker's map (default 8)
  --jobs n      diff and flicker: worker threads (default half the CPUs, 1 to 4)
  --json        prints JSON instead of a table (diff, stats, flicker, radial)
  diff          --thr levels counted (1,2,8,32); --bbox-thr the box's level (0); --heat a heat map
  crop          --zoom nearest-neighbour factor (1); --out the directory (required)
  sheet         --labels; --cols; --diff; --zoom (1), --scale or --width; --out the file (required)
  thumb         --short the short side in pixels (360); --quality JPEG quality (88); --out (required)
  mean          --codes averages the code values as stored; --out the file (required)
  stats         --pct percentiles (1,5,50,95,99); --oklab; --linear; --split; --cover; --mask; --against
  stats --deck  --deck-boxes (auto); --deck-rows; --horizon; --trend degrees (15); --px-per-deg (24)
  flicker       --frames a-b; --spike levels (20,40); --odd factor (2.5, 0 none); --drop-odd; --map; --save
  radial        --centre or --centre-key; --rings edges (0,2,4,8,15,30,50); --fov degrees; --half side

`uv run` reads the dependencies from the script's own block (PEP 723), so no system Pillow or
NumPy is needed. A directory stands for its PNGs; `diff` and `sheet` pair directories by file name.

Regions: --box takes pixels of the image as stored (a capture at dpr 2 has twice the CSS size).
--anchor reads the named crop anchors of the capture's JSON (<name>.json next to <name>.png,
`crops`: tools/island/parity/views.mjs; for diff, B's JSON, else A's), `all` every anchor in it;
--anchors FILE reads them from a capture JSON or a { name: {x, y, w, h} } file instead. An anchor
the capture reports missing (off screen) is skipped with a note on stderr.

diff: per pair and region, `identical` (equal decoded pixels), the max and mean absolute channel
difference, the count of pixels over each --thr level (a pixel counts when its largest channel
difference exceeds the level), the mean signed difference B - A per channel (`bias`), the luma
mean's change (`dL`, Rec. 709 on code values) and the bounding box [x, y, w, h] of the pixels over
--bbox-thr. Max, mean and the counts over 1, 2, 8 and 32 equal diff-set.py's report. --heat writes
the per-pixel max channel difference times --gain (clipped; as diff-set's <view>.diff.png).

crop: every region of every image, zoomed by --zoom. sheet: a labelled grid. Files (or one
directory's PNGs) take a column each (--cols wraps them into rows when no region is given); several
directories take a column each and a row per shared file name. Each region adds rows, cut with
the same box from every column (an anchor's box comes from the first column's capture JSON). --diff
adds a column after two inputs: their difference times --gain. Tiles are 1:1 unless --zoom
(nearest, integer), --scale or --width (box filter) is given; a full frame wider than 1280 px is
scaled down to 1280. thumb: a box-filtered JPEG, a frame kept to be recognised, not measured.

mean: the per-pixel mean of images of one size, written as a PNG: a many-sample reference made of
captures that each take one sample (a direction, a jitter, a moment), or the mean of a clip. The
mean is taken in linear light (sRGB decoded) and encoded back, as light adds; --codes averages the
code values instead, which keeps a share exact when each capture marks a pixel either fully in one
colour or in another (a debug view's hit and miss).

stats: per image and region, compare-captures.py's metrics (mean RGB, luma mean and standard
deviation, the mean luma difference of adjacent pixels, the share of pixels with luma over 245) and
the luma percentiles of --pct. --oklab adds the median Oklab L, C and hue (degrees) of the
region's pixels (sRGB decoded), the screen colour a look is judged by. --linear adds the mean
linear luminance Y (sRGB decoded, Rec. 709), the --pct percentiles of Y and the Oklab L, a and b of
the mean linear colour: light compared in stops (log2 of a ratio of Y), colour as an Oklab distance.
--split gives the means under and over the region's median L and of its darkest 5 % and brightest
10 % (oklab_split). --cover adds the share of pixels (%) whose red exceeds twice their green: the
blades in a capture with `flags=grassMask` (packages/island/world/Grass.ts). --mask M reads a mask
capture of the same view (a file, or a directory of the same names) and splits the region's pixels
into those it marks and the rest: the marked share, both parts' median Oklab L, C and hue with their
10th and 90th percentiles, and each marked pixel against the rest round it (mask_stats); --against B
compares them with the same pixels of the view without what the mask marks. `flags=streakMask`
(packages/render/sky/Sky.ts) marks the thin cloud layer's streaks; `flags=streaks:0` takes it away.

stats --deck: the cloud deck's structure on a sky instead of the metrics above: per region the
spread of its blocks of a degree, its structure about a broad trend, the std per band of size, the
edges' gradient, the masses' size and the dark and light blocks (deck_stats). The regions are named
presets on a 2560 x 1440 frame, scaled with the image's width (DECK_BOXES: `h`, the horizon views'
sky; `sun`, round the sun above the frame; `auto` picks by a file name ending in -h or -sun, or none
when --box or --anchor is given), plus every --box or --anchor. --deck-rows adds the horizon's rows
(--horizon x0,x1,horizonY, default 0,800,528 at 2560 wide): the low sky against the sky over it and
the far sea's fall a degree (deck_rows).

flicker: frame-to-frame flicker of consecutive frames per region: a capture-clip clip directory
(its whole frames, <clip>/full/<i>.png) or a directory of PNGs named by frame number. A gap is
refused, and at least 5 frames are needed. It gives F, the temporal high-pass Fhp (clip-metrics.py's),
one-frame flashes over the --spike levels, the per-pixel std and the mean, and lists odd frames
(flicker_clip). A clip of 120 frames at 60 fps is the usual run; the sea's own motion is a useful
control region (Fhp about 0.35, no spikes, on island-fair-noon). Only the regions are kept in memory,
so give regions rather than the whole frame for long 4K runs.

radial: the profile around a point (the sun), ring by ring: the median R, G, B (code values), luma
(Rec. 709 on code values), linear luma (sRGB decoded, 0-1) and the count of the ring's pixels in the
frame. --rings are edges in percent of the image's height, or degrees with --fov. --centre is in
pixels, or fractions when both are at most 1; --centre-key reads it, in fractions, from the capture
JSON's --eval-ready values (the first that holds KEY, as [x, y]). --half keeps the half of each ring
on that side of the point (up: the sky over a low sun, without the sea under it).

Output: tables on stdout (JSON with --json); notes on stderr. Files: diff --heat and flicker --map
write a .png for one pair, clip or region, else a directory of <name>[.<region>].diff.png or
<clip>[.<region>].flash.png; crop writes <out>/<name>.<region>.png; sheet writes --out; thumb writes
<out>/<name>.jpg; mean writes --out; flicker --save writes <clip>/flicker.json.

GPU lock and browser: neither; it reads image files. Cost: under a second to seconds per image; a
long 4K clip takes longer. It runs with the application policy on macOS (tools/kit/qos.mjs).

Exit codes: 0 done; 1 a usage error the tool reports (no region for crop, mismatched inputs, a gap
in a clip, no centre) or a file it cannot read; 2 a bad option (argparse).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help import answer_help  # noqa: E402

answer_help(__file__)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from project import project  # noqa: E402

# The application scheduling policy on macOS (tools/kit/qos.mjs says why), as in diff-set.py: the
# process re-executes itself in place under taskpolicy -a, and its pool threads ask for the
# user-initiated QoS.
QOS_ENV = project()["machine"]["envPrefix"] + "QOS"
QOS = sys.platform == "darwin" and os.environ.get(QOS_ENV) != "off"
if QOS and os.environ.get(QOS_ENV) != str(os.getpid()):
    os.environ[QOS_ENV] = str(os.getpid())
    os.execv("/usr/sbin/taskpolicy", ["/usr/sbin/taskpolicy", "-a", sys.executable, *sys.orig_argv[1:]])


def raise_thread():
    if QOS:
        import ctypes
        ctypes.CDLL(None).pthread_set_qos_class_self_np(0x19, 0)  # QOS_CLASS_USER_INITIATED


import numpy as np
from PIL import Image, ImageDraw

from pixels import LEVELS, LUMA, Files, bbox, bias, decode, metrics, stats  # noqa: E402

FULL_MAX = 1280


def ints(s):
    return [int(v) for v in s.split(",") if v != ""]


def expand(paths, match):
    """Files as given; a directory stands for its PNGs matching the pattern, sorted by name."""
    out = []
    for p in map(Path, paths):
        out += sorted(p.glob(match)) if p.is_dir() else [p]
    return out


def anchors_of(img, a):
    """The capture's crop anchors: --anchors FILE, else <img>.json's `crops`."""
    f = Path(a.anchors) if a.anchors else Path(img).with_suffix(".json")
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    return d.get("crops", {} if "url" in d else d)


def box_arg(s):
    """(label, (x, y, w, h)) of a --box value, `x,y,w,h` or `name=x,y,w,h`."""
    name, _, v = s.rpartition("=")
    x, y, w, h = ints(v)
    return (name or f"{x}_{y}_{w}_{h}", (x, y, w, h))


def regions(imgs, a, shape):
    """[(label, (x, y, w, h))] for the first image of imgs that has what is asked; the whole frame
    when no region is asked."""
    out = [box_arg(b) for b in a.box or []]
    if a.anchor:
        crops = {}
        for img in imgs:
            crops = anchors_of(img, a)
            if crops:
                break
        names = list(crops) if a.anchor == "all" else a.anchor.split(",")
        for n in names:
            c = crops.get(n)
            if c is None or "x" not in c:
                why = "not in the capture's anchors" if c is None else f"missing: {c.get('missing', '?')}"
                print(f"note: anchor {n} skipped ({why})", file=sys.stderr)
                continue
            out.append((n, (c["x"], c["y"], c["w"], c["h"])))
    if not a.box and not a.anchor:
        out.append((None, (0, 0, shape[1], shape[0])))
    return out


def cut(X, box):
    x, y, w, h = box
    return X[y:y + h, x:x + w]


def gain_lut(g):
    return np.minimum(np.arange(256) * g, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------- diff

def diff_pair(name, fa, fb, a, levels):
    fs = Files()
    A, B = fs.load(fa), fs.load(fb)
    if A.shape != B.shape:
        return [{"name": name, "a": str(fa), "b": str(fb), "sizeMismatch": [list(A.shape), list(B.shape)]}]
    ident = fs.same(fa, fb)
    rows = []
    for label, box in regions([fb, fa], a, A.shape):
        RA, RB = cut(A, box), cut(B, box)
        D, M, r = stats(RA, RB, levels)
        n = M.size
        sa, sb = RA.reshape(-1, 3).sum(0, dtype=np.int64), RB.reshape(-1, 3).sum(0, dtype=np.int64)
        r = {"name": name, "region": label, "box": list(box), "identical": ident if label is None else r["max"] == 0} | r
        r["bias"] = bias(RA, RB)
        r["dL"] = round(float(LUMA @ (sb - sa) / n), 5)
        b = bbox(M, a.bbox_thr)
        r["bbox"] = None if b is None else [b[0] + box[0], b[1] + box[1], b[2], b[3]]
        if a.heat and r["max"] > 0:
            heat = Path(a.heat)
            if heat.suffix.lower() != ".png":
                heat.mkdir(parents=True, exist_ok=True)
                heat = heat / (Path(name).stem + (f".{label}" if label else "") + ".diff.png")
            else:
                heat.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(gain_lut(a.gain)[M]).save(heat)
            r["heat"] = str(heat)
        rows.append(r)
    return rows


def cmd_diff(a):
    pa, pb = Path(a.a), Path(a.b)
    if pa.is_dir() != pb.is_dir():
        sys.exit("diff: give two images or two directories")
    if pa.is_dir():
        names = [p.name for p in sorted(pa.glob(a.match))]
        pairs = [(n, pa / n, pb / n) for n in names if (pb / n).exists()]
        missing = [n for n in names if not (pb / n).exists()] + [p.name for p in sorted(pb.glob(a.match)) if not (pa / p.name).exists()]
    else:
        pairs, missing = [(pb.name if pa.name == pb.name else f"{pa.name} vs {pb.name}", pa, pb)], []
    if a.heat and len(pairs) > 1 and Path(a.heat).suffix.lower() == ".png":
        sys.exit("diff: --heat takes a directory when there are several pairs")
    levels = tuple(ints(a.thr))
    with ThreadPoolExecutor(max_workers=max(1, a.jobs), initializer=raise_thread) as pool:
        results = [r for rows in pool.map(lambda p: diff_pair(*p, a, levels), pairs) for r in rows]
    if a.json:
        print(json.dumps({"pairs": results, "missing": missing}, indent=2))
        return
    w = max([len(r["name"] + (f" / {r['region']}" if r.get("region") else "")) for r in results] + [4])
    print(f"{'name':{w}} {'ident':5} {'max':>4} {'mean':>8} " + " ".join(f"{'>' + str(l):>9}" for l in levels)
          + f" {'bias R,G,B':>22} {'dL':>8}  bbox x,y,w,h")
    for r in results:
        n = r["name"] + (f" / {r['region']}" if r.get("region") else "")
        if "sizeMismatch" in r:
            print(f"{n:{w}} size {r['sizeMismatch'][0]} vs {r['sizeMismatch'][1]}")
            continue
        bb = ",".join(map(str, r["bbox"])) if r["bbox"] else "-"
        print(f"{n:{w}} {('yes' if r['identical'] else 'no'):5} {r['max']:4} {r['mean']:8.4f} "
              + " ".join(f"{r['over'][str(l)]:9}" for l in levels)
              + f" {','.join(f'{v:+.3f}' for v in r['bias']):>22} {r['dL']:+8.3f}  {bb}" + (f"  {r['heat']}" if r.get("heat") else ""))
    for n in missing:
        print(f"{n:{w}} only on one side")


# ---------------------------------------------------------------------------------------------- crop

def zoomed(X, z):
    return X if z == 1 else np.repeat(np.repeat(X, z, axis=0), z, axis=1)


def cmd_crop(a):
    if not a.box and not a.anchor:
        sys.exit("crop: give --box and/or --anchor")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in expand(a.images, a.match):
        X = decode(f)
        for label, box in regions([f], a, X.shape):
            p = out / f"{f.stem}.{label}.png"
            Image.fromarray(zoomed(cut(X, box), a.zoom)).save(p)
            print(p)


# --------------------------------------------------------------------------------------------- thumb

def cmd_thumb(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in expand(a.images, a.match):
        im = Image.fromarray(decode(f))
        k = a.short / min(im.size)
        if k < 1:
            im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.BOX)
        p = out / f"{f.stem}.jpg"
        im.save(p, quality=a.quality)
        print(p)


# ---------------------------------------------------------------------------------------------- mean

def cmd_mean(a):
    files = expand(a.images, a.match)
    if not files:
        sys.exit("mean: no images")
    total = None
    for f in files:
        X = decode(f).astype(np.float64) / 255
        if not a.codes:
            X = np.where(X <= 0.04045, X / 12.92, ((X + 0.055) / 1.055) ** 2.4)
        if total is not None and X.shape != total.shape:
            sys.exit(f"mean: {f} is {X.shape[1]}x{X.shape[0]}, the first {total.shape[1]}x{total.shape[0]}")
        total = X if total is None else total + X
    M = total / len(files)
    if not a.codes:
        M = np.where(M <= 0.0031308, M * 12.92, 1.055 * M ** (1 / 2.4) - 0.055)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.round(M * 255), 0, 255).astype(np.uint8)).save(a.out)
    print(f"{a.out} (the mean of {len(files)})")


# --------------------------------------------------------------------------------------------- sheet

def tile_size(X, a, region):
    h, w = X.shape[:2]
    if a.width:
        return a.width, max(1, round(h * a.width / w))
    if a.scale:
        return max(1, round(w * a.scale)), max(1, round(h * a.scale))
    if region is None and a.zoom == 1 and w > FULL_MAX:
        return FULL_MAX, max(1, round(h * FULL_MAX / w))
    return w * a.zoom, h * a.zoom


def to_tile(X, a, region, nearest=False):
    w, h = tile_size(X, a, region)
    if (w, h) == (X.shape[1], X.shape[0]):
        return Image.fromarray(X)
    if a.zoom > 1 and not a.width and not a.scale:
        return Image.fromarray(zoomed(X, a.zoom))
    return Image.fromarray(X).resize((w, h), Image.NEAREST if nearest else Image.BOX)


def cmd_sheet(a):
    inputs = [Path(p) for p in a.images]
    dirs = [p.is_dir() for p in inputs]
    if any(dirs) and not all(dirs):
        sys.exit("sheet: give images or directories, not both")
    if all(dirs) and len(inputs) > 1:
        names = [p.name for p in sorted(inputs[0].glob(a.match)) if all((d / p.name).exists() for d in inputs[1:])]
        groups = [(Path(n).stem, [d / n for d in inputs]) for n in names]
        labels = a.labels.split(",") if a.labels else [d.name for d in inputs]
    else:
        files = expand(a.images, a.match)
        labels = a.labels.split(",") if a.labels else [f.stem for f in files]
        groups = [(None, files)]
        if a.cols and not a.box and not a.anchor:
            groups = [(None, files[i:i + a.cols]) for i in range(0, len(files), a.cols)]
            labels = [labels[i:i + a.cols] for i in range(0, len(files), a.cols)]
    if a.diff and len(groups[0][1]) != 2:
        sys.exit("sheet: --diff needs exactly two inputs")
    lut = gain_lut(a.gain)
    rows = []  # [(title, [(label, PIL image)])]
    for gi, (gname, files) in enumerate(groups):
        imgs = [decode(f) for f in files]
        glabels = labels[gi] if a.cols and isinstance(labels[0], list) else labels
        for label, box in regions(files, a, imgs[0].shape):
            tiles = [cut(X, box) for X in imgs]
            row = [(glabels[j] if j < len(glabels) else files[j].stem, to_tile(t, a, label)) for j, t in enumerate(tiles)]
            if a.diff:
                if tiles[0].shape != tiles[1].shape:
                    sys.exit(f"sheet: --diff on images of different sizes ({files[0]}, {files[1]})")
                _, M, st = stats(tiles[0], tiles[1])
                row.append((f"diff x{a.gain:g} (max {st['max']})", to_tile(lut[M], a, label, nearest=True)))
            title = " / ".join(s for s in (gname, label) if s)
            rows.append((title, row))
    gap, bar = 4, 18
    cols = max(len(r) for _, r in rows)
    cw = [max((r[j][1].width for _, r in rows if j < len(r)), default=0) for j in range(cols)]
    rh = [max(t.height for _, t in r) + bar for _, r in rows]
    im = Image.new("RGB", (sum(cw) + gap * (cols - 1), sum(rh) + gap * (len(rows) - 1)), "#202020")
    dr = ImageDraw.Draw(im)
    y = 0
    for (title, row), h in zip(rows, rh):
        x = 0
        for j, (label, t) in enumerate(row):
            im.paste(t, (x, y + bar))
            dr.text((x + 4, y + 3), (f"{title}: {label}" if title and j == 0 else label), fill="white")
            x += cw[j] + gap
        y += h + gap
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(f"{out} ({im.width}x{im.height}, {len(rows)} rows x {cols} columns)")


# --------------------------------------------------------------------------------------------- stats

def cmd_stats(a):
    if a.deck:
        res = []
        for f in expand(a.images, a.match):
            d = deck_stats(f, a, decode(f))
            if d is not None:
                res.append({"name": f.name} | d)
        if a.json:
            print(json.dumps(res, indent=2))
            return
        for r in res:
            print(f"{r['name']}  ({r['pxPerDeg']:g} px a degree)")
            print_deck(r["name"], r)
        return
    pct = [float(p) for p in a.pct.split(",")] if a.pct else []
    res = []
    for f in expand(a.images, a.match):
        X = decode(f)
        for label, box in regions([f], a, X.shape):
            R = cut(X, box)
            m = metrics(R)
            y = R.astype(np.float64) @ LUMA
            m["lumaPercentiles"] = {f"{p:g}": round(float(v), 2) for p, v in zip(pct, np.percentile(y, pct))}
            if a.oklab:
                m["oklab"] = oklab_median(R)
            if a.split:
                m["split"] = oklab_split(R)
            if a.linear:
                m["linear"] = linear_stats(R, pct)
            if a.cover:
                rgb = R[..., :3].astype(np.int32)
                m["cover"] = round(float(np.mean(rgb[..., 0] > 2 * rgb[..., 1])) * 100, 2)
            if a.mask:
                bare = cut(decode(mask_of(f, a.against)), box) if a.against else None
                m["mask"] = mask_stats(R, cut(decode(mask_of(f, a.mask)), box), bare)
            res.append({"name": f.name, "region": label, "box": list(box)} | m)
    if a.json:
        print(json.dumps(res, indent=2))
        return
    w = max([len(r["name"] + (f" / {r['region']}" if r["region"] else "")) for r in res] + [4])
    print(f"{'name':{w}} {'mean R,G,B':>20} {'luma':>7} {'std':>6} {'adj':>6} {'>245%':>7}  " + " ".join(f"{'p' + k:>7}" for k in (f"{p:g}" for p in pct)))
    for r in res:
        n = r["name"] + (f" / {r['region']}" if r["region"] else "")
        print(f"{n:{w}} {','.join(f'{v:.1f}' for v in r['meanRGB']):>20} {r['lumaMean']:7.2f} {r['lumaStdDev']:6.2f} "
              f"{r['meanAdjacentLumaDifference']:6.3f} {r['brightPixelPercent']:7.3f}  "
              + " ".join(f"{v:7.2f}" for v in r["lumaPercentiles"].values())
              + (f"  oklab L {r['oklab']['L']:.3f} C {r['oklab']['C']:.3f} h {r['oklab']['h']:.1f}" if "oklab" in r else "")
              + (f"  cover {r['cover']:.1f}%" if "cover" in r else ""))
        if "linear" in r:
            k = r["linear"]
            print(f"  linear Y {k['Y']:.5f}  " + " ".join(f"p{p} {v:.5f}" for p, v in k["percentiles"].items())
                  + f"  oklab L {k['L']:.4f} a {k['a']:+.4f} b {k['b']:+.4f}")
        if "split" in r:
            print(f"  median L {r['split']['medianL']:.3f}")
            for part, p in r["split"]["parts"].items():
                print(f"  {part:5} L {p['L']:.3f} C {p['C']:.3f} h {p['h']:5.1f}  rgb {','.join(f'{v:.4f}' for v in p['rgb'])}  ({p['px']} px)")
        if "mask" in r:
            k = r["mask"]
            print(f"  marked {k['share']:.2f} %  edge {k['edgeShare']:.2f} %"
                  + (f"  dL {k['dL']:+.4f} dC {k['dC']:+.4f}" if "dL" in k else ""))
            for key, what in (("near", "against the rest round them"), ("against", "against the same pixels")):
                if key in k:
                    g = k[key]
                    print(f"  {what}: dL {g['dL']:+.4f} [{g['dL10']:+.4f}, {g['dL90']:+.4f}]  dC {g['dC']:+.4f} [{g['dC10']:+.4f}, {g['dC90']:+.4f}]")
            for part in ("marked", "rest"):
                p = k.get(part)
                if p:
                    print(f"  {part:6} L {p['L']:.3f} [{p['L10']:.3f}, {p['L90']:.3f}]  C {p['C']:.3f} [{p['C10']:.3f}, {p['C90']:.3f}]"
                          f"  h {p['h']:5.1f} [{p['h10']:5.1f}, {p['h90']:5.1f}]  ({p['px']} px)")


def mask_of(f, mask):
    """The mask capture of image f: --mask a file, or a directory holding one of f's name."""
    m = Path(mask)
    return m / Path(f).name if m.is_dir() else m


def mask_stats(R, M, bare=None):
    """The Oklab medians and 10-90 percentiles of a region's pixels split by a mask capture of the same
    view: `marked` where the mask differs from the image by more than 16 levels in a channel and its
    red is its largest, `edge` where it differs so and its green is, `rest` the others (the edge left
    out of both). With `bare`, the same view without what the mask marks: the marked pixels' own
    differences in L and C from it.
    Why the largest channel and not a ratio: a bright paint leaves the display chain desaturated;
    about the sun a marked red came out salmon, its red under twice its green, and a test by that
    ratio found a sixth of the marked sky there. The hue's percentiles are taken about its circular
    mean. Each marked pixel's dL and dC are against the rest round it (their Gaussian mean over the
    rest's pixels only, a sigma of 1 % of the region's width: median and 10-90 percentiles), since
    the rest of a sky is lighter or darker by its elevation. Against `bare`, the differences leave out
    a change the marked pixels made to the whole frame (an exposure that meters the frame)."""
    if M.shape != R.shape or (bare is not None and bare.shape != R.shape):
        return {"error": f"mask shape {list(M.shape)} against {list(R.shape)}"}
    c = M[..., :3].reshape(-1, 3).astype(np.int32)
    painted = np.abs(c - R[..., :3].reshape(-1, 3).astype(np.int32)).max(axis=1) > 16
    marked = painted & (c[:, 0] > c[:, 1]) & (c[:, 0] > c[:, 2])
    edge = painted & (c[:, 1] > c[:, 0]) & (c[:, 1] > c[:, 2])
    rest = ~marked & ~edge
    _, L, A, B = oklab_pixels(R)
    C, H = np.hypot(A, B), np.degrees(np.arctan2(B, A)) % 360

    def part(keep):
        if not keep.any():
            return None
        h = H[keep]
        # the hue's percentiles about its circular mean, so a spread across 0 / 360 reads as one
        centre = np.degrees(np.arctan2(np.sin(np.radians(h)).mean(), np.cos(np.radians(h)).mean())) % 360
        h = centre + (h - centre + 180) % 360 - 180
        q = lambda v: [round(float(x), 4) for x in np.percentile(v, [10, 50, 90])]
        (l10, l50, l90), (c10, c50, c90), (h10, h50, h90) = q(L[keep]), q(C[keep]), q(h)
        return {"L": l50, "L10": l10, "L90": l90, "C": c50, "C10": c10, "C90": c90,
                "h": round(h50 % 360, 1), "h10": round(h10 % 360, 1), "h90": round(h90 % 360, 1), "px": int(keep.sum())}

    out = {"share": round(float(marked.mean()) * 100, 2), "edgeShare": round(float(edge.mean()) * 100, 2),
           "marked": part(marked), "rest": part(rest)}
    if out["marked"] and out["rest"]:
        out["dL"] = round(out["marked"]["L"] - out["rest"]["L"], 4)
        out["dC"] = round(out["marked"]["C"] - out["rest"]["C"], 4)
    if marked.any() and rest.any():
        # each marked pixel against the rest round it: their Gaussian mean (a sigma of 1 % of the
        # region's width) over the rest's pixels only, so neither the marked nor the edge pull in
        h, w = R.shape[:2]
        sigma = max(4.0, 0.01 * w)
        keep = rest.reshape(h, w).astype(np.float64)
        dL = (L.reshape(h, w) - masked_blur(L.reshape(h, w), keep, sigma)).reshape(-1)[marked]
        dC = (C.reshape(h, w) - masked_blur(C.reshape(h, w), keep, sigma)).reshape(-1)[marked]
        (l10, l50, l90), (c10, c50, c90) = ([round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] for v in (dL, dC))
        out["near"] = {"dL": l50, "dL10": l10, "dL90": l90, "dC": c50, "dC10": c10, "dC90": c90}
    if bare is not None and marked.any():
        _, L0, A0, B0 = oklab_pixels(bare)
        dL, dC = (L - L0)[marked], (C - np.hypot(A0, B0))[marked]
        (l10, l50, l90), (c10, c50, c90) = ([round(float(x), 4) for x in np.percentile(v, [10, 50, 90])] for v in (dL, dC))
        out["against"] = {"dL": l50, "dL10": l10, "dL90": l90, "dC": c50, "dC10": c10, "dC90": c90}
    return out


def oklab_of_linear(lin):
    """The Oklab L, a and b of linear sRGB colours (..., 3) (Björn Ottosson's Oklab)."""
    lms = lin @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                          [0.5363325363, 0.6806995451, 0.2817188376],
                          [0.0514459929, 0.1073969566, 0.6299787005]])
    l_, m_, s_ = np.moveaxis(np.cbrt(lms), -1, 0)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return L, A, B


def oklab_pixels(R):
    """The linear RGB (n, 3) and the Oklab L, a and b (each n) of 8-bit sRGB pixels."""
    c = R[..., :3].reshape(-1, 3).astype(np.float64) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    L, A, B = oklab_of_linear(lin)
    return lin, L, A, B


def linear_stats(R, pct):
    """The mean linear luminance Y (sRGB decoded, Rec. 709 weights), its percentiles, and the Oklab L, a, b of the mean linear colour."""
    lin, _, _, _ = oklab_pixels(R)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    L, A, B = oklab_of_linear(lin.mean(axis=0))
    return {"Y": round(float(y.mean()), 6),
            "percentiles": {f"{p:g}": round(float(v), 6) for p, v in zip(pct, np.percentile(y, pct))},
            "L": round(float(L), 5), "a": round(float(A), 5), "b": round(float(B), 5)}


def oklab_median(R):
    """The median Oklab L, C and hue (degrees) of 8-bit sRGB pixels."""
    _, L, A, B = oklab_pixels(R)
    h = np.degrees(np.arctan2(B, A)) % 360
    return {"L": round(float(np.median(L)), 4), "C": round(float(np.median(np.hypot(A, B))), 4), "h": round(float(np.median(h)), 1)}


def oklab_split(R):
    """The pixels' means under and over their median Oklab L, and over the darkest 5 % and the brightest 10 %.
    Four parts, each the mean Oklab L, C and hue (the mean of the pixels' L, a and b, C and h taken
    from the mean a and b) and the mean linear RGB of its pixels: `under` and `over` the region's own
    median L (a crown's dark and lit halves), `p5` the darkest 5 % and `p90` the brightest 10 %
    (`medianL` and `parts` in --json)."""
    lin, L, A, B = oklab_pixels(R)
    median, low, high = np.percentile(L, [50, 5, 90])

    def mean_of(keep):
        a, b = float(A[keep].mean()), float(B[keep].mean())
        return {"L": round(float(L[keep].mean()), 4), "C": round(float(np.hypot(a, b)), 4), "h": round(float(np.degrees(np.arctan2(b, a)) % 360), 1),
                "rgb": [round(float(v), 5) for v in lin[keep].mean(axis=0)], "px": int(keep.sum())}

    return {"medianL": round(float(median), 4),
            "parts": {"under": mean_of(L <= median), "over": mean_of(L >= median), "p5": mean_of(L <= low), "p90": mean_of(L >= high)}}


# ---------------------------------------------------------------------------------------------- deck

# The sky's regions on a 2560 x 1440 frame, [x, y, w, h] each, scaled with the image's width. A region
# of several boxes is read as one. The horizon views (`h`) keep the low band to the columns of skyL
# and skyR, clear of the island; the view toward the sun (`sun`) has the sun above the frame.
DECK_WIDTH = 2560
DECK_BOXES = {
    "h": {"skyL": [(0, 0, 820, 500)], "skyR": [(1780, 0, 780, 500)], "sky": [(0, 0, 820, 500), (1780, 0, 780, 500)],
          "skyHigh": [(0, 0, 2560, 140)], "skyLowBand": [(0, 400, 820, 110), (1780, 400, 780, 110)]},
    "sun": {"sunSide": [(1500, 0, 1060, 560)], "offSide": [(0, 0, 640, 560)]},
}
DECK_PRESET_OF = (("-h", "h"), ("-sun", "sun"))  # a file name's ending -> its preset under --deck auto
DECK_ROWS = (0, 800, 528)  # x0, x1, the horizon's row on 2560 x 1440: --deck-rows' default
DECK_BANDS = (1, 3, 10, 30)  # the scale bands' edges, degrees


def deck_regions(f, a, shape):
    """[(label, [boxes])] the deck's measures read on an image: the preset's regions (--deck-boxes h|sun, or
    auto by the file name) and every --box / --anchor, each its own region."""
    w = shape[1]
    preset = a.deck_boxes
    if preset == "auto":
        preset = "none" if a.box or a.anchor else next((p for end, p in DECK_PRESET_OF if Path(f).stem.endswith(end)), None)
        if preset is None:
            print(f"note: {Path(f).name} skipped (no preset for its name: give --deck-boxes h|sun or --box)", file=sys.stderr)
            return []
    k = w / DECK_WIDTH
    out = [(n, [tuple(round(v * k) for v in b) for b in boxes]) for n, boxes in DECK_BOXES.get(preset, {}).items()]
    if a.box or a.anchor:
        out += [(label, [box]) for label, box in regions([f], a, shape)]
    return out


def gauss_blur(X, sigma):
    """X blurred by a Gaussian of sigma pixels, zero outside the frame (separable, by FFT)."""
    if sigma <= 0:
        return X
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (1, 0):
        n = X.shape[axis]
        m = 1 << int(np.ceil(np.log2(n + 2 * r + len(k))))
        F = np.fft.rfft(X, n=m, axis=axis) * np.fft.rfft(k, n=m).reshape([-1, 1] if axis == 0 else [1, -1])
        X = np.fft.irfft(F, n=m, axis=axis)
        X = X[r:r + n] if axis == 0 else X[:, r:r + n]
    return X


def masked_blur(X, M, sigma):
    """The Gaussian of X over the mask's pixels only, normalised by the mask's own blur, so the land
    and the frame's edge pull nothing in."""
    return gauss_blur(X * M, sigma) / np.maximum(gauss_blur(M, sigma), 1e-6)


def deck_stats(f, a, X):
    """The deck's measures on each of the image's sky regions; --deck-rows adds the horizon's rows.
    Oklab L and C and linear luminance Y (Rec. 709) come from the sRGB code values. The sky mask is
    the union of every box read; each blur is masked_blur, so neither the land nor the frame's edge
    pulls anything in. Degrees: --px-per-deg, by default 24 at 2560 wide (the horizon view's vfov of
    55 on 1440 rows), scaled with the width. Per region:
      spread     the region's blocks of a degree (24 px square, each the median of its sky pixels, a
                 block kept when nine tenths of it is sky): p90 - p10 of their L; `stops` log2 of
                 their Y's p90 over p10
      structure  L minus its broad trend (a sigma of --trend / 2 degrees, 15 by default): p10 and p90
      scale std  the std of L - G1, G1 - G3, G3 - G10 and G10 - G30, G_d a sigma of d / 2 degrees:
                 the bands under 1, 1-3, 3-10 and 10-30 degrees
      gradient   |grad L| a degree on a sigma of .125 degree (normalised over the whole frame): p95
                 and p99, how hard the edges are
      mass       the half length (to .5) of the structure's autocorrelation along x and y, in degrees,
                 averaged over the region's boxes: about a third to a half of a mass's width
      dark/light the median L and C of the blocks in the lowest and the highest quarter of the
                 p10..p90 range
    Read every view at the same scale: 8-bit quantisation floors the std under 1 degree at about
    .0014 on a flat sky, and a frame read at half size gives the same numbers within a few per cent
    above 1 degree and lower ones under it."""
    H, W = X.shape[:2]
    regs = deck_regions(f, a, X.shape)
    if not regs:
        return None
    ppd = a.px_per_deg or 24.0 * W / DECK_WIDTH
    lin, L, A, B = oklab_pixels(X)
    L, C, Y = L.reshape(H, W), np.hypot(A, B).reshape(H, W), (lin @ np.array([0.2126, 0.7152, 0.0722])).reshape(H, W)
    mask = np.zeros((H, W))
    for _, boxes in regs:
        for x, y, w, h in boxes:
            mask[y:y + h, x:x + w] = 1
    # the structure: L minus its broad trend (a sigma of half --trend degrees, over the sky alone)
    S = L - masked_blur(L, mask, a.trend * ppd / 2)
    G = [L] + [masked_blur(L, mask, d * ppd / 2) for d in DECK_BANDS]
    bands = {"<1": G[0] - G[1], "1-3": G[1] - G[2], "3-10": G[2] - G[3], "10-30": G[3] - G[4]}
    # the edges' hardness: |grad L| a degree on a sigma of .125 degree, over the whole frame
    gy, gx = np.gradient(masked_blur(L, np.ones_like(L), 0.125 * ppd))
    grad = np.hypot(gx, gy) * ppd
    bs = max(1, round(ppd))
    out = {"pxPerDeg": round(ppd, 3), "regions": {}}
    for label, boxes in regs:
        bl, by, bc, hx, hy, sel = [], [], [], [], [], []
        for x, y, w, h in boxes:
            m = mask[y:y + h, x:x + w] > 0.5
            sel.append((np.s_[y:y + h, x:x + w], m))
            # blocks of a degree, each its median
            for j in range(y, y + h - bs + 1, bs):
                for i in range(x, x + w - bs + 1, bs):
                    mm = mask[j:j + bs, i:i + bs] > 0.5
                    if mm.mean() < 0.9:
                        continue
                    bl.append(np.median(L[j:j + bs, i:i + bs][mm]))
                    by.append(np.median(Y[j:j + bs, i:i + bs][mm]))
                    bc.append(np.median(C[j:j + bs, i:i + bs][mm]))
            # the masses: the half length of the structure's autocorrelation along x and y
            Z = S[y:y + h, x:x + w] - S[y:y + h, x:x + w].mean()
            R = np.fft.irfft2(np.abs(np.fft.rfft2(Z, s=(2 * h, 2 * w))) ** 2, s=(2 * h, 2 * w))
            R = R / max(R[0, 0], 1e-20)
            rx, ry = R[0, :w], R[:h, 0]
            hx.append((np.argmax(rx < 0.5) if (rx < 0.5).any() else w) / ppd)
            hy.append((np.argmax(ry < 0.5) if (ry < 0.5).any() else h) / ppd)

        def cat(V):
            return np.concatenate([V[s][m] for s, m in sel])

        if not bl:
            out["regions"][label] = {"boxes": [list(b) for b in boxes], "note": "no block of a degree inside it"}
            continue
        bl, by, bc = np.array(bl), np.array(by), np.array(bc)
        p10, p90 = np.percentile(bl, [10, 90])
        y10, y90 = np.percentile(by, [10, 90])
        dark, light = bl <= p10 + (p90 - p10) / 4, bl >= p90 - (p90 - p10) / 4
        sv, g = cat(S), cat(grad)
        out["regions"][label] = {
            "boxes": [list(b) for b in boxes],
            "median": round(float(np.median(cat(L))), 4),
            "spread": round(float(p90 - p10), 4),
            "stops": round(float(np.log2(max(y90, 1e-9) / max(y10, 1e-9))), 3),
            "structure": [round(float(v), 4) for v in np.percentile(sv, [10, 90])],
            "scaleStd": {k: round(float(np.std(cat(v))), 4) for k, v in bands.items()},
            "gradient": [round(float(v), 4) for v in np.percentile(g, [95, 99])],
            "mass": [round(float(np.mean(hx)), 2), round(float(np.mean(hy)), 2)],
            "dark": [round(float(np.median(bl[dark])), 4), round(float(np.median(bc[dark])), 4)],
            "light": [round(float(np.median(bl[light])), 4), round(float(np.median(bc[light])), 4)],
            "blocks": int(len(bl)),
        }
    if a.deck_rows:
        out["rows"] = deck_rows(L, a, ppd, W)
    return out


def deck_rows(L, a, ppd, W):
    """The horizon's profile: row medians of L over x0..x1 every 6 px from 15 degrees up to 6 down,
    and what section 4 reads from it: the sky 0-6 degrees up against 6-15 up (their medians, their
    difference and their blocks' spreads), the far sea's fall a degree over 0-5 down (a line's
    slope, positive when the sea darkens downward) and its largest step between rows within one
    degree, and `line`, the pixel row within a degree of the horizon that stands furthest from the
    rows 3 px above and below it (a thin dark or bright line)."""
    k = W / DECK_WIDTH
    x0, x1, hz = ints(a.horizon) if a.horizon else (round(v * k) for v in DECK_ROWS)
    H = L.shape[0]
    prof = []
    for yy in range(max(0, round(hz - 15 * ppd)), min(H - 6, round(hz + 6 * ppd)) + 1, 6):
        prof.append(((hz - (yy + 3)) / ppd, float(np.median(L[yy:yy + 6, x0:x1]))))
    deg = np.array([d for d, _ in prof])
    val = np.array([v for _, v in prof])

    def strip(lo, hi):
        # the strip lo..hi degrees up: its median L and the spread of its blocks of a degree
        y0, y1 = round(hz - hi * ppd), round(hz - lo * ppd)
        bs = max(1, round(ppd))
        blocks = [np.median(L[j:j + bs, i:i + bs]) for j in range(y0, y1 - bs + 1, bs) for i in range(x0, x1 - bs + 1, bs)]
        p10, p90 = np.percentile(blocks, [10, 90]) if blocks else (np.nan, np.nan)
        return float(np.median(L[y0:y1, x0:x1])), float(p90 - p10)

    low, lowSpread = strip(0, 6)
    above, aboveSpread = strip(6, 15)
    sea = (deg <= 0) & (deg >= -5)
    fall = float(np.polyfit(deg[sea], val[sea], 1)[0]) if sea.sum() > 2 else None  # L lost a degree going down
    ds, vs = deg[sea], val[sea]
    step = max((abs(vs[i] - vs[j]) for i in range(len(ds)) for j in range(i + 1, len(ds)) if abs(ds[i] - ds[j]) <= 1), default=None)
    # a thin line at the horizon: the row (each pixel row's median) furthest from its neighbours 3 px off
    r = np.array([np.median(L[y, x0:x1]) for y in range(round(hz - ppd), round(hz + ppd) + 1)])
    dev = r[3:-3] - (r[:-6] + r[6:]) / 2
    i = int(np.argmax(np.abs(dev)))
    return {"x0": x0, "x1": x1, "horizon": hz,
            "skyLow": round(low, 4), "skyAbove": round(above, 4), "skyLowOver": round(low - above, 4),
            "skyLowSpread": round(lowSpread, 4), "skyAboveSpread": round(aboveSpread, 4),
            "seaFall": None if fall is None else round(fall, 4), "seaStep": None if step is None else round(float(step), 4),
            "line": round(float(dev[i]), 4), "lineY": round(hz - ppd) + 3 + i,
            "profile": [[round(d, 2), round(v, 4)] for d, v in prof]}


def print_deck(name, d):
    for label, r in d["regions"].items():
        if "note" in r:
            print(f"  {label:10} {r['note']}")
            continue
        s = r["scaleStd"]
        print(f"  {label:10} median {r['median']:.3f}  spread {r['spread']:.3f} ({r['stops']:.2f} stops)"
              f"  structure {r['structure'][0]:+.3f}/{r['structure'][1]:+.3f}"
              f"  std <1 {s['<1']:.4f} 1-3 {s['1-3']:.4f} 3-10 {s['3-10']:.4f} 10-30 {s['10-30']:.4f}"
              f"  grad {r['gradient'][0]:.3f}/{r['gradient'][1]:.3f}  mass {r['mass'][0]:.1f}/{r['mass'][1]:.1f} deg"
              f"  dark {r['dark'][0]:.3f} C {r['dark'][1]:.3f}  light {r['light'][0]:.3f} C {r['light'][1]:.3f}")
    if "rows" in d:
        w = d["rows"]
        # (the sea's fall and step are None where too few rows lie 0-5 degrees down)
        na = lambda v, form: "n/a" if v is None else format(v, form)
        print(f"  rows x {w['x0']}-{w['x1']}, horizon y {w['horizon']}: sky 0-6 up {w['skyLow']:.3f} over 6-15 up {w['skyAbove']:.3f}"
              f" ({w['skyLowOver']:+.3f}; spreads {w['skyLowSpread']:.3f} / {w['skyAboveSpread']:.3f})"
              f"  sea 0-5 down falls {na(w['seaFall'], '.3f')}/deg, largest step within 1 deg {na(w['seaStep'], '.3f')}"
              f"  line {w['line']:+.3f} at y {w['lineY']}")
        print("  profile (deg, L): " + " ".join(f"{p[0]:g}:{p[1]:.3f}" for p in w["profile"]))


# --------------------------------------------------------------------------------------------- radial

def centre_of(f, a, shape):
    """The radial profile's centre in pixels: --centre, else the capture JSON's --eval-ready value."""
    h, w = shape[:2]
    if a.centre:
        x, y = (float(v) for v in a.centre.split(","))
    else:
        j = Path(f).with_suffix(".json")
        vals = [e.get("value") for e in json.loads(j.read_text()).get("readyEvals", [])] if j.exists() else []
        hit = next((v[a.centre_key] for v in vals if isinstance(v, dict) and a.centre_key in v), None)
        if hit is None:
            sys.exit(f"radial: {f}: no --eval-ready value holds {a.centre_key}")
        x, y = hit
    return (x * w, y * h) if x <= 1 and y <= 1 else (x, y)


def cmd_radial(a):
    """With --fov (the capture's vertical field of view) a ring's radius r in degrees maps to
    tan(r) / tan(fov / 2) half heights, exact at the frame's centre."""
    if not a.centre and not a.centre_key:
        sys.exit("radial: give --centre x,y or --centre-key KEY")
    edges = [float(v) for v in a.rings.split(",")]
    res = []
    for f in expand(a.images, a.match):
        X = decode(f)
        h, w = X.shape[:2]
        cx, cy = centre_of(f, a, X.shape)
        if a.fov:
            half = np.tan(np.radians(a.fov) / 2)
            px = [np.tan(np.radians(e)) / half * h / 2 for e in edges]
        else:
            px = [e / 100 * h for e in edges]
        yy, xx = np.mgrid[0:h, 0:w]
        dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
        r = np.hypot(dx, dy)
        side = {"up": dy < 0, "down": dy > 0, "left": dx < 0, "right": dx > 0}.get(a.half, np.ones_like(r, bool))
        rgb = X[..., :3].astype(np.float64)
        lin = rgb / 255
        lin = np.where(lin <= 0.04045, lin / 12.92, ((lin + 0.055) / 1.055) ** 2.4) @ LUMA
        rings = []
        for k in range(len(edges) - 1):
            m = (r >= px[k]) & (r < px[k + 1]) & side
            n = int(m.sum())
            ring = {"from": edges[k], "to": edges[k + 1], "pixels": n}
            if n:
                med = np.median(rgb[m], axis=0)
                ring |= {"rgb": [round(float(v), 1) for v in med], "luma": round(float(np.median(rgb[m] @ LUMA)), 1),
                         "linear": round(float(np.median(lin[m])), 5)}
            rings.append(ring)
        res.append({"name": f.name, "centre": [round(cx, 1), round(cy, 1)], "unit": "deg" if a.fov else "%h", "half": a.half, "rings": rings})
    if a.json:
        print(json.dumps(res, indent=2))
        return
    for r in res:
        half = f", {r['half']} half" if r["half"] else ""
        print(f"{r['name']}  centre {r['centre'][0]:.0f},{r['centre'][1]:.0f}  ({r['unit']}{half})")
        for g in r["rings"]:
            span = f"{g['from']:g}-{g['to']:g}"
            if "rgb" not in g:
                print(f"  {span:>9}  (outside the frame)")
                continue
            print(f"  {span:>9}  rgb {','.join(f'{v:.0f}' for v in g['rgb']):>11}  luma {g['luma']:6.1f}  linear {g['linear']:.4f}  n {g['pixels']}")


# -------------------------------------------------------------------------------------------- flicker

# an odd frame also flashes this share of pixels (0.5 per mille) above the median, so a clip that
# hardly flashes (a ground truth) lists none
ODD_FLOOR = 0.5e-3
LUMA32 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def clip_frames(d, span):
    """[(frame, path)] of a clip directory's whole frames (or a directory of numbered PNGs), in
    frame order, within span (a, b) when given."""
    d = Path(d)
    src = d / "full" if (d / "full").is_dir() else d
    files = sorted((int(p.stem), p) for p in src.glob("*.png") if p.stem.isdigit())
    if span:
        files = [(i, p) for i, p in files if span[0] <= i <= span[1]]
    gaps = [files[k][0] for k in range(1, len(files)) if files[k][0] != files[k - 1][0] + 1]
    if gaps:
        sys.exit(f"flicker: {src}: frames not consecutive (a gap before frame {gaps[0]}); pick a run with --frames")
    if len(files) < 5:
        sys.exit(f"flicker: {src}: {len(files)} frames, the measures need at least 5")
    return files


def flicker_clip(d, a):
    """The flicker measures of one clip, per region. Luma is Rec. 709 of the 8-bit values (float32),
    in levels:
      F       mean |L_t - L_t-1|
      Fhp     mean |L_t - ( L_t-2 + L_t+2 ) / 2|, the temporal high-pass (clip-metrics.py's)
      spike   s = L_t - ( L_t-1 + L_t+1 ) / 2, a one-frame flash: the share of pixel-frames with |s|
              over each --spike level and with s over the first (`bright`), in per mille, its 99.9th
              percentile (`p999`), and the share of pixels that flash over the first level at least
              once (`once`, %)
      std     the mean per-pixel standard deviation over the frames; `mean` the luma mean
    The map is the share of frames whose |s| exceeds the first level, times 255 x --gain.
    Odd frames: a paused page's stepped frames now and then come out as a discrete alternative image
    (scattered vegetation pixels off by up to 190 levels: capture-clip.mjs, beside RECHECK_SHOTS),
    which in a 120-frame meadow clip raised Fhp by about 5 % and the flashes by about 6 % from one
    frame. A frame whose flash share over all the regions exceeds --odd times the clip's median and
    the median by ODD_FLOOR is listed (`odd`); --drop-odd leaves out every term that reads one, so
    A/Bs of small effects compare the wind's flicker rather than which clip caught an odd frame."""
    files = clip_frames(d, tuple(ints(a.frames.replace("-", ","))) if a.frames else None)
    first = decode(files[0][1])
    regs = regions([files[0][1]], a, first.shape)
    levels = [float(v) for v in a.spike.split(",")]

    def lumas(f):
        Y = decode(f).astype(np.float32) @ LUMA32
        return [cut(Y, box).copy() for _, box in regs]

    with ThreadPoolExecutor(max_workers=max(1, a.jobs), initializer=raise_thread) as pool:
        per = list(pool.map(lumas, [p for _, p in files]))
    n = len(files)
    stacks = [np.stack([f[k] for f in per]) for k in range(len(regs))]
    # odd frames: a frame's flash share over all the regions' pixels against the clip's median
    share = sum((np.abs(s[1:-1] - (s[:-2] + s[2:]) / 2) > levels[0]).sum(axis=(1, 2)) for s in stacks) / sum(s[0].size for s in stacks)
    med = float(np.median(share))
    odd = [int(i) + 1 for i in np.nonzero((share > a.odd * med) & (share > med + ODD_FLOOR))[0]] if a.odd > 0 else []
    out = {"clip": str(d), "frames": [files[0][0], files[-1][0]], "odd": [files[i][0] for i in odd], "dropped": bool(a.drop_odd and odd), "regions": []}
    bad = np.zeros(n, bool)
    if a.drop_odd:
        bad[odd] = True
    # the terms kept: none that reads an odd frame
    k1 = ~(bad[:-1] | bad[1:])
    k2 = ~(bad[:-4] | bad[2:-2] | bad[4:])
    ks = ~(bad[:-2] | bad[1:-1] | bad[2:])
    for k, (label, box) in enumerate(regs):
        s = stacks[k]
        d1 = np.abs(np.diff(s, axis=0))[k1]
        hp = np.abs(s[2:-2] - (s[:-4] + s[4:]) / 2)[k2]
        sp = (s[1:-1] - (s[:-2] + s[2:]) / 2)[ks]
        ab = np.abs(sp)
        r = {"region": label, "box": list(box), "F": round(float(d1.mean()), 3), "Fhp": round(float(hp.mean()), 3)}
        for v in levels:
            r[f"spike>{v:g}"] = round(float((ab > v).mean()) * 1e3, 3)
        r[f"bright>{levels[0]:g}"] = round(float((sp > levels[0]).mean()) * 1e3, 3)
        r["p999"] = round(float(np.percentile(ab, 99.9)), 2)
        flashes = ab > levels[0]
        r["once"] = round(float(flashes.any(axis=0).mean()) * 100, 2)
        r["std"] = round(float(s[~bad].std(axis=0).mean()), 3)
        r["mean"] = round(float(s[~bad].mean()), 2)
        if a.map:
            m = Path(a.map)
            if m.suffix.lower() != ".png":
                m.mkdir(parents=True, exist_ok=True)
                m = m / (Path(d).name + (f".{label}" if label else "") + ".flash.png")
            else:
                m.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.clip(flashes.mean(axis=0) * 255 * a.gain, 0, 255).astype(np.uint8)).save(m)
            r["map"] = str(m)
        out["regions"].append(r)
    if a.save:
        (Path(d) / "flicker.json").write_text(json.dumps(out, indent=1))
    return out


def cmd_flicker(a):
    if a.map and Path(a.map).suffix.lower() == ".png" and (len(a.clips) > 1 or len(a.box or []) + (1 if a.anchor else 0) > 1):
        sys.exit("flicker: --map takes a directory when there are several clips or regions")
    res = [flicker_clip(d, a) for d in a.clips]
    if a.json:
        print(json.dumps(res, indent=2))
        return
    lv = [f"{float(v):g}" for v in a.spike.split(",")]
    w = max([len(Path(c["clip"]).name + (f" / {r['region']}" if r["region"] else "")) for c in res for r in c["regions"]] + [4])
    print(f"{'clip':{w}} {'F':>6} {'Fhp':>6} " + " ".join(f"{'>' + v + '‰':>8}" for v in lv)
          + f" {'bright‰':>8} {'p999':>6} {'once%':>6} {'std':>6} {'mean':>7}")
    for c in res:
        for r in c["regions"]:
            n = Path(c["clip"]).name + (f" / {r['region']}" if r["region"] else "")
            print(f"{n:{w}} {r['F']:6.3f} {r['Fhp']:6.3f} " + " ".join(f"{r['spike>' + v]:8.3f}" for v in lv)
                  + f" {r['bright>' + lv[0]]:8.3f} {r['p999']:6.2f} {r['once']:6.2f} {r['std']:6.3f} {r['mean']:7.2f}"
                  + (f"  {r['map']}" if r.get("map") else ""))
        if c["odd"]:
            print(f"  odd frames {c['odd']}" + (" (dropped)" if c["dropped"] else " (kept: --drop-odd leaves them out)"))


# ---------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Image differences, crops, sheets and statistics for captures.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--box", action="append", help="x,y,w,h in image pixels (repeatable)")
        p.add_argument("--anchor", help="named crop anchors from the capture JSON: a,b or all")
        p.add_argument("--anchors", help="anchors from this JSON file instead of <image>.json")
        p.add_argument("--match", default="*.png", help="the files a directory stands for")

    p = sub.add_parser("diff", help="differences of two images or two directories")
    p.add_argument("a")
    p.add_argument("b")
    common(p)
    p.add_argument("--thr", default=",".join(map(str, LEVELS)), help="levels to count pixels over")
    p.add_argument("--bbox-thr", type=int, default=0, help="the level the bounding box encloses")
    p.add_argument("--heat", help="heat map: a .png file for one pair, else a directory")
    p.add_argument("--gain", type=float, default=8)
    p.add_argument("--json", action="store_true")
    p.add_argument("--jobs", type=int, default=min(4, max(1, (os.cpu_count() or 2) // 2)))
    p.set_defaults(fn=cmd_diff)

    p = sub.add_parser("crop", help="cut boxes or named anchors")
    p.add_argument("images", nargs="+")
    common(p)
    p.add_argument("--zoom", type=int, default=1)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_crop)

    p = sub.add_parser("sheet", help="a labelled grid of images or crops")
    p.add_argument("images", nargs="+")
    common(p)
    p.add_argument("--labels")
    p.add_argument("--cols", type=int)
    p.add_argument("--diff", action="store_true")
    p.add_argument("--gain", type=float, default=8)
    p.add_argument("--zoom", type=int, default=1)
    p.add_argument("--scale", type=float)
    p.add_argument("--width", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_sheet)

    p = sub.add_parser("thumb", help="small JPEGs of images")
    p.add_argument("images", nargs="+")
    p.add_argument("--match", default="*.png", help="the files a directory stands for")
    p.add_argument("--short", type=int, default=360)
    p.add_argument("--quality", type=int, default=88)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_thumb)

    p = sub.add_parser("mean", help="the per-pixel mean of images, in linear light")
    p.add_argument("images", nargs="+")
    p.add_argument("--match", default="*.png", help="the files a directory stands for")
    p.add_argument("--codes", action="store_true", help="average the code values as stored")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_mean)

    p = sub.add_parser("stats", help="mean, luma, percentiles per image or region")
    p.add_argument("images", nargs="+")
    common(p)
    p.add_argument("--pct", default="1,5,50,95,99")
    p.add_argument("--oklab", action="store_true", help="the median Oklab L, C and hue")
    p.add_argument("--linear", action="store_true", help="the mean linear luminance Y, its --pct percentiles and the Oklab L, a, b of the mean linear colour")
    p.add_argument("--split", action="store_true", help="the means under and over the median L, of the darkest 5 %% and the brightest 10 %%")
    p.add_argument("--cover", action="store_true", help="the share of red pixels (a grassMask capture's blades)")
    p.add_argument("--mask", help="a mask capture of the same view (a file, or a directory of the same names): Oklab medians of its marked pixels and the rest")
    p.add_argument("--against", help="with --mask, the same view without what the mask marks (a file or a directory): the marked pixels' own dL and dC")
    p.add_argument("--deck",action="store_true", help="the cloud deck's measures on named sky regions and any --box")
    p.add_argument("--deck-boxes", default="auto", choices=["auto", "h", "sun", "none"], help="with --deck, the named regions (auto: by the file name, -h or -sun)")
    p.add_argument("--deck-rows", action="store_true", help="with --deck, the horizon's rows (section 4 of the deck's rows)")
    p.add_argument("--horizon", help="with --deck-rows, x0,x1,horizonY in image pixels (default 0,800,528 at 2560 wide, scaled)")
    p.add_argument("--trend", type=float, default=15, help="with --deck, the broad trend's width in degrees")
    p.add_argument("--px-per-deg", type=float, help="with --deck, pixels a degree (default 24 at 2560 wide, scaled with the width)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("flicker", help="frame-to-frame flicker of a clip's consecutive frames per region")
    p.add_argument("clips", nargs="+")
    common(p)
    p.add_argument("--frames", help="a-b: the frames to keep")
    p.add_argument("--spike", default="20,40", help="the flash levels counted (the first also for `bright`, `once` and --map)")
    p.add_argument("--map", help="flash map: a .png file for one clip and region, else a directory")
    p.add_argument("--gain", type=float, default=8)
    p.add_argument("--json", action="store_true")
    p.add_argument("--save", action="store_true", help="write <clip>/flicker.json")
    p.add_argument("--odd", type=float, default=2.5, help="an odd frame flashes this many times the clip's median (0: none)")
    p.add_argument("--drop-odd", action="store_true", help="leave out the terms that read an odd frame")
    p.add_argument("--jobs", type=int, default=min(4, max(1, (os.cpu_count() or 2) // 2)))
    p.set_defaults(fn=cmd_flicker)

    p = sub.add_parser("radial", help="the median ring by ring around a point (the sun)")
    p.add_argument("images", nargs="+")
    p.add_argument("--match", default="*.png", help="the files a directory stands for")
    p.add_argument("--centre", help="x,y in pixels, or fractions when both are at most 1")
    p.add_argument("--centre-key", help="the centre, in fractions, from this key of the capture JSON's --eval-ready values")
    p.add_argument("--rings", default="0,2,4,8,15,30,50", help="ring edges: percent of the height, or degrees with --fov")
    p.add_argument("--fov", type=float, help="the capture's vertical fov (degrees): --rings in degrees")
    p.add_argument("--half", choices=["up", "down", "left", "right"], help="keep the half of each ring on this side of the point")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_radial)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
