"""The parity tolerances: when two paused frames of a temporal AA mode (taa, taau) count as the same
image, and when two MSAA paused frames do inside the region the sea's held reflection reaches.
Imported by diff-set.py (--temporal, --held); every threshold lives in TEMPORAL and HELD below.

Why a tolerance: MSAA paused frames are bit-identical between runs, temporal ones are not. The GPU
ends a render pass on its own (a tile's binned geometry overflows, preemption) at points that vary from draw to
draw, and a split pass shades a few samples differently (mostly through fragment derivatives). MSAA
shows one frame, so a glitch in it is gone the next frame; the temporal rest averages 40 frames and
freezes whatever glitches those frames held. Making that bit-exact would cost product changes we
chose not to make, so temporal frames are compared the way engine screenshot tests compare frames:
per-pixel differences under a level are ignored, the share of pixels over it is bounded on the
frame and in every local window, and a coherent shift that stays under the level everywhere is
caught by the window's mean signed difference. A failure counts only after one recapture
(diff-set.py: the best of the pairings of both sides' captures, <view>.png and <view>.repeat.png).

The model (all levels are 8-bit code values of the PNG, per channel):
  - Unity's Graphics Test Framework (ImageComparisonSettings): a per-pixel threshold, the share of
    pixels over it (IncorrectPixelsThreshold) and the mean error (AverageCorrectnessThreshold).
  - Unreal's screenshot comparison (FAutomationScreenshotTolerance): a per-channel tolerance with a
    global error (share over tolerance in the frame) and a local error (the same share in any
    block), so that a small, concentrated change cannot hide in a large frame.
  - NVIDIA FLIP (perceptual difference), reported only and only when flip-evaluator is installed:
    the levels above are the gate, since FLIP's mean over a frame is too forgiving of small regions.

Calibration: every A/A pair of one build recorded (taa, taau and msaa; quiet and contended GPU;
forced strike, storm, rain; vegPrepass=0), and synthetic regressions cut into those frames (a 1-level change over a 48 px
square, a recoloured flower patch, a 0.25 px shift of a 384 px crop, a missing 6 px flower or the
islander's head) that must fail. The worst A/A pair (dense vegetation on a busy GPU) uses 0.43 of
the tolerance (`use`: the largest measurement / threshold); the weakest regression 3.9 times it.
Each threshold sits at two to four times the worst A/A value of its measurement. Windows are
W x W at a stride of W / 2, so any square of 1.5 W lies whole inside one.

The held rule (HELD): MSAA paused frames are bit-identical between runs except where the sea's
reflection reaches. The reflection averages a fixed sequence of frames of the opaque image and
holds the result (packages/render/ocean/SeaReflection.ts, Rest); a few of those frames can differ in their
last bits from load to load, and the held result keeps what they held. The region is the cells of
the render the reflection last traced from, which each capture records (`held`:
packages/render/debug/HeldRegion.ts), grown by `dilate` cells for the upscale and the sharpening
(held_region()). Outside it nothing may differ (`outside`: 0); inside it the measurements and
names are TEMPORAL's, over the region's pixels, at the level such frames show.

Calibration: 12 fresh loads of each of the sea's parity views (island-fair-noon,
island-clear-golden, horizon-fair-noon, island-night-clear, water-skim-cliffs, island-from-sea,
sunrise-long-shadows, night-moon, low-36, rim-sea, cliff-sea-golden, meadow-close; the GPU shared
with other captures), every pair compared, and the regressions above cut at three points of each
view's region, plus a change of every region pixel by 1 % and of one pixel outside it. The A/A
pairs differ only in the region (but for a lone pair whose grass outside it differed, which the
recapture absorbs), by 1 level and on one pixel in one pair by 2: up to 206 pixels a frame, 28 in a
32 px window. So `level` is 1 and `big` 4 with no big pixel allowed; the window's share of off
pixels and the frame's mean sit at four and five times their worst A/A values; the frame's and the
crops' shares, whose worst A/A value is one pixel or none, allow ten pixels in a region of a million
and six in a crop; the biases stay above the largest cluster taken as all of one sign (28 px in a
window, 0.027), not only the mixed signs seen (0.006).
The worst A/A pair uses 0.24 of the tolerance (a window's 2-level pixel); the weakest regression
3.4 times it (a missing 6 px object on the sea at night, low in contrast), the next 6.4 times (3
levels of green over an 8 px square).
"""
import numpy as np

# Every threshold of the temporal rule. A pixel is "off" when its largest channel difference
# exceeds `level`; "big" when it exceeds `big`.
TEMPORAL = {
	"level": 2,
	"big": 16,
	# the frame: share of off pixels, mean |difference| over all channels
	"frame_off": 0.0005,
	"frame_mean": 0.02,
	# every window: share of off pixels, big pixels, |mean signed difference| of any channel
	"window": 32,
	"window_off": 0.06,
	"window_big": 6,
	"window_bias": 0.25,
	# every named crop (views.mjs anchors): share of off pixels, |mean signed difference| of any channel
	"crop_off": 0.004,
	"crop_bias": 0.02,
}

# Every threshold of the held rule (MSAA paused frames, the held region only; see the header). The
# names are TEMPORAL's; `outside`: pixels that may differ outside the region; `dilate`: cells the
# region is grown by.
HELD = {
	"level": 1,
	"big": 4,
	"frame_off": 0.00001,
	"frame_mean": 0.0001,
	"window": 32,
	"window_off": 0.004,
	"window_big": 0,
	"window_bias": 0.03,
	"crop_off": 0.0001,
	"crop_bias": 0.001,
	"outside": 0,
	"dilate": 1,
}


def held_region(helds, shape, dilate=None):
	"""The pixels of a frame of `shape` (H x W [x C]) where a held effect may differ: the union of
	the captures' `held` records (the page's cells, packages/render/debug/HeldRegion.ts), each cell
	grown by `dilate` cells (HELD's); None when no capture has one."""
	helds = [h for h in helds if h]
	if not helds:
		return None
	H, W = shape[0], shape[1]
	dilate = HELD["dilate"] if dilate is None else dilate
	out = np.zeros((H, W), bool)
	for h in helds:
		nx, ny = h["grid"]
		digits = np.frombuffer(h["cells"].encode(), np.uint8)
		nib = np.where(digits >= 97, digits - 87, digits - 48).astype(np.uint8)
		cells = ((nib[:, None] >> np.arange(4, dtype=np.uint8)) & 1).astype(bool).ravel()[:nx * ny].reshape(ny, nx)
		for _ in range(dilate):
			c = cells.copy()
			c[1:] |= cells[:-1]
			c[:-1] |= cells[1:]
			c[:, 1:] |= cells[:, :-1]
			c[:, :-1] |= cells[:, 1:]
			cells = c
		# a pixel's texel, then its cell (the render may be smaller than the image)
		rw, rh = h["size"]
		cx = (np.arange(W) * rw // W) // h["cell"]
		cy = (np.arange(H) * rh // H) // h["cell"]
		out |= cells[cy[:, None], cx[None, :]]
	return out


def window_sums(X, w, s):
	"""Sums of X (H x W [x C]) over w x w windows whose corners lie on a grid of stride s: returns the
	sums (rows x cols [x C], int64) and the windows' top and left edges. A window that would cross the
	frame's far edge is clipped to it (only when the frame is smaller than a window).

	Exact integer sums at the cost of one pass over X: every window edge along an axis is a multiple
	of g, the greatest common divisor of those edges (s when w = 2 s), so X is summed in blocks of
	gy x gx and the integral image is built on the block grid only, not on every pixel."""
	ys = np.arange(0, max(1, X.shape[0] - w + 1), s)
	xs = np.arange(0, max(1, X.shape[1] - w + 1), s)
	y1, x1 = np.minimum(ys + w, X.shape[0]), np.minimum(xs + w, X.shape[1])
	gy, gx = int(np.gcd.reduce(np.concatenate((ys, y1)))), int(np.gcd.reduce(np.concatenate((xs, x1))))
	ny, nx = int(y1[-1]) // gy, int(x1[-1]) // gx
	C = X.shape[2:]
	B = X[:ny * gy].reshape((ny, gy, X.shape[1]) + C).sum(1, dtype=np.int64)
	B = B[:, :nx * gx].reshape((ny, nx, gx) + C).sum(2)
	I = np.zeros((ny + 1, nx + 1) + C, dtype=np.int64)
	I[1:, 1:] = B.cumsum(0).cumsum(1)
	j0, j1, i0, i1 = (ys // gy)[:, None], (y1 // gy)[:, None], (xs // gx)[None, :], (x1 // gx)[None, :]
	return I[j1, i1] - I[j0, i1] - I[j1, i0] + I[j0, i0], ys, xs


def _windows(X, w):
	"""Sums of X over w x w windows at a stride of w / 2."""
	return window_sums(X, w, w // 2)


def _region(S, M, T, absS=None):
	# every float here is an exact integer sum divided once, so it does not depend on the order of
	# summation: the same value as the arrays' own mean()
	n = M.size
	absS = np.abs(S) if absS is None else absS
	return {
		"off": float(np.count_nonzero(M > T["level"]) / n),
		"big": int(np.count_nonzero(M > T["big"])),
		"mean": float(int(absS.sum(dtype=np.int64)) / absS.size),
		"bias": float(np.abs(np.array([S[..., k].sum(dtype=np.int64) for k in range(3)]) / n).max()),
		"max": int(M.max()),
	}


def measure(A, B, crops=None, T=TEMPORAL, region=None):
	"""The rule's measurements for frames A (reference) and B (test), H x W x 3 (uint8 or int16).
	With `region` (H x W bool, held_region()): the pixels that differ outside it are counted
	(`outside`), the difference outside it is left out, and the frame's measurements are over the
	region's pixels."""
	S = np.subtract(B, A, dtype=np.int16)
	absS = np.abs(S)
	M = np.maximum(np.maximum(absS[..., 0], absS[..., 1]), absS[..., 2])
	if region is None:
		r = {"frame": _region(S, M, T, absS)}
	else:
		r = {"outside": int(np.count_nonzero(M[~region]))}
		S[~region], absS[~region], M[~region] = 0, 0, 0
		r["frame"] = _region(S[region], M[region], T, absS[region]) if region.any() else _region(S[:1, :1], M[:1, :1], T)
	w = T["window"]
	off, ys, xs = _windows((M > T["level"]).astype(np.uint8), w)
	big, _, _ = _windows((M > T["big"]).astype(np.uint8), w)
	bias, _, _ = _windows(S, w)
	bias = np.abs(bias).max(axis=2) / (w * w)
	worst = lambda X: tuple(int(i) for i in np.unravel_index(np.argmax(X), X.shape))
	r["window"] = {}
	for k, X, scale in (("off", off, 1 / (w * w)), ("big", big, 1), ("bias", bias, 1)):
		j, i = worst(X)
		r["window"][k] = {"value": float(X[j, i] * scale) if k != "big" else int(X[j, i]), "at": [int(xs[i]), int(ys[j])]}
	r["crops"] = {}
	for name, c in (crops or {}).items():
		if "x" not in c:
			continue
		x, y, cw, ch = c["x"], c["y"], c["w"], c["h"]
		r["crops"][name] = _region(S[y:y + ch, x:x + cw], M[y:y + ch, x:x + cw], T)
	return r


def checks(r, T=TEMPORAL):
	"""Every measurement against its threshold: (criterion, value, limit, where) per check."""
	f, wn = r["frame"], r["window"]
	if "outside" in r:
		yield "px outside the region", r["outside"], T["outside"], ""
	yield f"frame px>{T['level']}", f["off"], T["frame_off"], ""
	yield "frame mean", f["mean"], T["frame_mean"], ""
	yield f"window px>{T['level']}", wn["off"]["value"], T["window_off"], wn["off"]["at"]
	yield f"window px>{T['big']}", wn["big"]["value"], T["window_big"], wn["big"]["at"]
	yield "window bias", wn["bias"]["value"], T["window_bias"], wn["bias"]["at"]
	for name, c in r["crops"].items():
		yield f"crop {name} px>{T['level']}", c["off"], T["crop_off"], ""
		yield f"crop {name} bias", c["bias"], T["crop_bias"], ""


def judge(r, T=TEMPORAL):
	"""(pass, use, reasons): use is the largest value / limit over the checks (pass: use <= 1)."""
	why, use = [], 0.0
	for name, v, lim, at in checks(r, T):
		use = max(use, v / lim if lim else (float("inf") if v else 0.0))
		if v > lim:
			fv = f"{v:.3%}" if "px>" + str(T["level"]) in name else (f"{v:.4f}" if isinstance(v, float) else str(v))
			fl = f"{lim:.3%}" if "px>" + str(T["level"]) in name else str(lim)
			why.append(f"{name} {fv} > {fl}" + (f" @{at[0]},{at[1]}" if at else ""))
	return not why, round(use, 3), why


def flip_mean(A, B):
	"""Mean LDR FLIP of B against A, or None without flip-evaluator (NVIDIA, BSD-3-Clause)."""
	try:
		import flip_evaluator as flip
	except ImportError:
		return None
	_, mean, _ = flip.evaluate((A / 255.0).astype(np.float32), (B / 255.0).astype(np.float32), "LDR")
	return float(mean)
