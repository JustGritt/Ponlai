# /// script
# dependencies = [ "numpy", "pillow", "scipy" ]
# ///
"""Measures a stepped clip's image and temporal metrics against MSAA and a ground truth, with gates.

It answers how a mode under test compares with MSAA and with the ground truth on the crops of a
clip captured by tools/kit/parity/capture-clip.mjs: brightness, flicker, sharpness, ghosting, lag,
vegetation coverage, flowers, stars, glints, edges and more, each as a ratio to MSAA and judged
against a thresholds file when one is given. tools/kit/parity/baseline.mjs runs it for every entry
of a plan; tools/kit/table.py and tools/kit/gates.py read its output. For paused sets, use
tools/kit/diff-set.py.

Usage:
    uv run [--with flip-evaluator] tools/kit/clip-metrics.py --test DIR --msaa DIR [--gt DIR] [options]
    uv run tools/kit/clip-metrics.py --calibrate masks.json [--size 288] DIR [DIR ...] [--overlays DIR]

Options:
    --test, --msaa, --gt DIR  capture-clip folders of one clip: the mode under test, MSAA, the ground truth
    --crops a,b | boxes.json  the crops measured (default every crop folder of the GT, else the test's)
    --split-redraws           Fhp also near and between the shadow map redraws the clip's notes mark
    --thresholds t.json       the masks and the gates
    --baseline metrics.json   the MSAA baseline from an earlier run instead of --msaa's values
    --key tier/kind/name      the entry of a baseline.mjs metrics.json given as --baseline
    --out DIR                 default <test>/metrics
    --no-flip                 MAE instead of NVIDIA FLIP even when flip-evaluator is installed
    --no-gt-gates             leave out the thresholds' `gt` rules (a GT that is not the side's own)
    --workers N               worker processes; default CLIP_METRICS_WORKERS, else a quarter of the CPUs, 1 to 4
    --calibrate masks.json    derive the mask parameters from MSAA folders (the positional DIRs)
    --size WxH|N              with --calibrate: the GT crop size the metrics run at (288 for gt-hi)
    --overlays DIR            with --calibrate: write the masks over the images

The ground truth is capture-clip's `--aa gt`, whose crops are the test's size; the older `--aa
gt-hi` crops are smaller (288 px against 384), so a larger test or MSAA crop is area-resampled
(PIL's box filter) to the GT's size first and every metric is taken at the GT's resolution. Luma L
is Rec. 709 luma of the 8-bit sRGB-encoded values (0-255 levels): differences in levels are what
the eye and the PNGs see.

Frames: metrics against the GT use the frames present on all three sides (the GT may be sampled:
every 4th frame of a long clip plus dense runs, tools/island/parity/clips.mjs gtSampling). A side's
own metrics use every frame it captured. The temporal metrics against the GT pair consecutive frames
only, so they read the GT's dense runs.

Metrics, per crop, for the test and for MSAA ("ratio" is test / MSAA); the definitions are beside
own_metrics() and measure() below:
    B brightness, F flicker, Fhp high-pass flicker (FhpNear, FhpBetween with --split-redraws),
    TI temporal instability, S sharpness, V and Vown sharpness steps, FLIP (MAE without
    flip-evaluator), dMSAA difference to MSAA, G ghosting, L lag, R ringing, Rec recovery after a
    flick, W vegetation coverage, fringe, holes; by crop name: flowers (N_f, C_f), meadow (luma,
    saturation, green share, texture, coverage, lumaLin, chroma), twigs, stars (count, excess,
    peak, twinkle), glints (share, energy, sparkle), streaks (rain), peak (a flash), edgeW and
    edgeWgt (edge width), shift, shiftX and shiftY (an offset left in the output).
Without --gt only the sides' own metrics are taken (F, Fhp, Vown, Rec, holes, twinkle, sparkle,
dMSAA): a clip that has no ground truth, such as a shadow-shimmer study of MSAA and a temporal mode.

Crops: the "full" folder (whole frames, capture-clip.mjs --full) is measured only when --crops
names it. --crops boxes.json cuts crops from the whole frames instead, for clips captured whole
(every frame in full/): { view: { crop: [ x0, y0, x1, y1 ] } } in the whole frame's pixels, the
entry of clip.json's view; every side's full/ frames must be the same size.

Masks: the thresholds file's `masks` holds the calibrated mask parameters: `masks.default`, then per
view `masks[view]['*']` and per crop `masks[view][crop]`, each a partial copy of DEFAULT_MASKS
(merged in that order). The view is clip.json's.

Gates: the thresholds file's `metrics` holds rules against the MSAA baseline, `gt` rules against the
GT. A rule is looked up as "<clip>@<view>/<crop>/<metric>", "<clip>@<view>/*/<metric>",
"<clip>/<crop>/<metric>", "<clip>/*/<metric>", "*/<crop>/<metric>", "*/*/<metric>",
"<crop>/<metric>", then "<metric>" (nested metrics named as in the table, such as "meadow.luma";
the clip of a rest image is "rest"). The baseline is the MSAA side measured here, or with
--baseline the "msaa" values of an earlier metrics.json: a clip's own, or the entry --key of a
baseline.mjs metrics.json. Magnitudes and ratios to the GT take ratio forms ("max", "min", "band",
with an absolute "floor"); the signed, integer and absolute-difference metrics (fringe, dMSAA,
edgeW, edgeWgt, shiftX, shiftY, holes, L, Rec) take "diff"; conditions ("tiers", "requires",
"except") limit where a rule is judged. `gt` rules bound the value against the GT directly, named
"gt:<metric>". The forms are written out beside DIFF_METRICS below. Without thresholds the table
is informational.

Output: <out>/metrics.json (per crop: test, msaa, ratio, per-frame series and verdicts, a box
crop's box, with --split-redraws each side's redraw frames; `gates`: the failed count by kind) and
<out>/metrics.md, the table, also on stdout. --calibrate writes the masks file and prints it.

GPU lock and browser: neither; it reads PNG files. The frames are measured in worker processes;
on macOS the main process and the workers run with the application scheduling policy
(tools/kit/qos.mjs).

Cost: seconds to a minute per clip, with the frame count and the crops. The output does not depend
on the worker count (byte-identical metrics.json).

Exit codes: 0 done, no gate failed; 1 a gate failed, or a bad thresholds file, a missing crop box
or baseline entry; 2 a usage error (--test or --msaa missing).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help import answer_help  # noqa: E402

answer_help(__file__)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from threshold_rules import find_rule  # noqa: E402
from project import project  # noqa: E402

# The application scheduling policy on macOS (tools/kit/qos.mjs says why): the main process
# re-executes itself in place under taskpolicy -a; the workers are new processes, which do not
# inherit it, so they start through a shim that does the same (pool_context).
QOS_ENV = project()[ 'machine' ][ 'envPrefix' ] + 'QOS'
QOS = sys.platform == 'darwin' and os.environ.get( QOS_ENV ) != 'off'
if QOS and __name__ == '__main__' and os.environ.get( QOS_ENV ) != str( os.getpid() ):
	os.environ[ QOS_ENV ] = str( os.getpid() )
	os.execv( '/usr/sbin/taskpolicy', [ '/usr/sbin/taskpolicy', '-a', sys.executable, *sys.orig_argv[ 1: ] ] )


def pool_context():
	ctx = multiprocessing.get_context( 'spawn' )
	if QOS:
		# written afresh into a private directory of this run (never a file found at a known path)
		import atexit
		import shlex
		import shutil
		import tempfile
		folder = tempfile.mkdtemp( prefix=project()[ 'machine' ][ 'tempPrefix' ] + 'qos-' )
		atexit.register( shutil.rmtree, folder, True )
		shim = os.path.join( folder, 'python.sh' )
		with open( shim, 'w' ) as f:
			f.write( f'#!/bin/sh\n{ QOS_ENV }=$$ exec /usr/sbin/taskpolicy -a { shlex.quote( sys.executable ) } "$@"\n' )
		os.chmod( shim, 0o700 )
		ctx.set_executable( shim )
	return ctx


import numpy as np
from PIL import Image
from scipy import ndimage as ndi

try:
	import flip_evaluator as flip  # NVIDIA FLIP (BSD-3-Clause), optional
except Exception:  # noqa: BLE001
	flip = None


# The mask parameters before calibration (calibrate() replaces them per view from MSAA images).
DEFAULT_MASKS = {
	# flower heads: outside the green hue band [ lo, hi ] (or below greenSat), at least `chroma`
	# (Y'CbCr units) from the crop's median green chroma, and bright (L above brightLuma) or
	# saturated (s above minSat and value above minValue)
	'flower': { 'greenHue': [ 60, 170 ], 'greenSat': 0.2, 'chroma': 0.1, 'brightLuma': 150, 'minSat': 0.35, 'minValue': 0.4, 'minArea': 2, 'maxArea': 400 },
	# blade coverage: green's lead over red and blue, as a share of green, above `rel` (the olive
	# soil and the yellowed grass lead by less), and luma above `floor` (black shade has no hue)
	'meadow': { 'rel': 0.2, 'floor': 20 },
	# twigs: black top-hat of L over `window` px deeper than `tophat` levels, pieces of minArea px
	'twig': { 'window': 9, 'tophat': 12, 'minArea': 6 },
	# stars: local maxima at least `excess` levels over the local sky (the twinkle's star finder);
	# the count, excess and peak read the visible stars, at least `visible` levels over it: under
	# MSAA's sharpening the cloud texture's maxima reach 8-11 levels and counted as stars
	'stars': { 'excess': 12, 'visible': 12 },
	# glints: L at or above `level` (calibrated as the MSAA crop's `percentile`)
	'glints': { 'percentile': 99.9, 'level': 235 },
	# holes: no channel above `level`
	'holes': { 'level': 2 },
}


def frames_of( folder, crop ):

	d = os.path.join( folder, crop )
	if not os.path.isdir( d ):
		return {}
	return { int( f[ :-4 ] ): os.path.join( d, f ) for f in os.listdir( d ) if f.endswith( '.png' ) and f[ :-4 ].isdigit() }


def load_u8( path, size=None, box=None ):

	im = Image.open( path ).convert( 'RGB' )
	if box is not None:
		# a crop box of --crops <json>, in the whole frame's pixels
		im = im.crop( tuple( box ) )
	if size is not None and im.size != size:
		im = im.resize( size, Image.BOX )
	return np.asarray( im )


def load( path, size=None ):

	return np.asarray( load_u8( path, size ), dtype=np.float64 )


def luma( rgb ):

	return rgb[ ..., 0 ] * 0.2126 + rgb[ ..., 1 ] * 0.7152 + rgb[ ..., 2 ] * 0.0722


def chroma( rgb ):

	"""Y'CbCr (BT.709) chroma of 8-bit values, in units of the full range (about -0.5 .. 0.5)."""
	y = luma( rgb ) / 255.0
	return ( rgb[ ..., 2 ] / 255.0 - y ) / 1.8556, ( rgb[ ..., 0 ] / 255.0 - y ) / 1.5748


def hsv( rgb ):

	# per channel, and the max and min of three elementwise: the same values as ( rgb / 255 ).max( -1 ),
	# which reduces over a 3-long axis several times slower
	r, g, b = rgb[ ..., 0 ] / 255.0, rgb[ ..., 1 ] / 255.0, rgb[ ..., 2 ] / 255.0
	mx, mn = np.maximum( np.maximum( r, g ), b ), np.minimum( np.minimum( r, g ), b )
	d = mx - mn
	h = np.zeros_like( mx )
	safe = np.where( d > 1e-9, d, 1.0 )
	h = np.where( mx == r, ( ( g - b ) / safe ) % 6, h )
	h = np.where( mx == g, ( b - r ) / safe + 2, h )
	h = np.where( mx == b, ( r - g ) / safe + 4, h )
	h = np.where( d > 1e-9, h * 60.0, 0.0 )
	s = np.where( mx > 1e-9, d / np.where( mx > 1e-9, mx, 1.0 ), 0.0 )
	return h, s, mx


def sobel( lum ):

	p = np.pad( lum, 1, mode='edge' )
	gx = ( p[ :-2, 2: ] + 2 * p[ 1:-1, 2: ] + p[ 2:, 2: ] ) - ( p[ :-2, :-2 ] + 2 * p[ 1:-1, :-2 ] + p[ 2:, :-2 ] )
	gy = ( p[ 2:, :-2 ] + 2 * p[ 2:, 1:-1 ] + p[ 2:, 2: ] ) - ( p[ :-2, :-2 ] + 2 * p[ :-2, 1:-1 ] + p[ :-2, 2: ] )
	return np.hypot( gx, gy ) / 4.0


def green_mask( rgb, hsv3=None ):

	h, s, v = hsv3 or hsv( rgb )
	return ( h >= 60 ) & ( h <= 170 ) & ( s > 0.2 ) & ( v > 0.08 )


def blue_mask( rgb, hsv3=None ):

	h, s, v = hsv3 or hsv( rgb )
	return ( h >= 180 ) & ( h <= 250 ) & ( s > 0.15 )


def dilate( mask, r ):

	# binary dilation by a ( 2r + 1 )^2 square, as a separable maximum: the edge padding repeats a
	# pixel already in the window, so it is the dilation with nothing outside the image, exactly
	return ndi.maximum_filter( mask, size=2 * r + 1, mode='nearest' )


def merge( a, b ):

	out = copy.deepcopy( a )
	for k, v in ( b or {} ).items():
		if isinstance( v, dict ) and isinstance( out.get( k ), dict ):
			out[ k ] = merge( out[ k ], v )
		else:
			out[ k ] = copy.deepcopy( v )
	return out


def masks_for( cfg, view, crop ):

	m = merge( DEFAULT_MASKS, ( cfg or {} ).get( 'default' ) )
	per = ( cfg or {} ).get( view ) or {}
	return merge( merge( m, per.get( '*' ) ), per.get( crop ) )


# ------------------------------------------------------------------ masks

def flower_mask( rgb, m, hsv3=None ):

	f = m[ 'flower' ]
	h, s, v = hsv3 or hsv( rgb )
	lum = luma( rgb )
	lo, hi = f[ 'greenHue' ]
	band = ( h >= lo ) & ( h <= hi )
	green = band & ( s > f[ 'greenSat' ] ) & ( v > 0.08 )
	cb, cr = chroma( rgb )
	if green.sum() >= 50:
		ref = ( float( np.median( cb[ green ] ) ), float( np.median( cr[ green ] ) ) )
	else:
		ref = tuple( f.get( 'greenRef', ( -0.08, -0.05 ) ) )
	d = np.hypot( cb - ref[ 0 ], cr - ref[ 1 ] )
	return ( ~band | ( s < f[ 'greenSat' ] ) ) & ( d > f[ 'chroma' ] ) & ( ( lum > f[ 'brightLuma' ] ) | ( ( s > f[ 'minSat' ] ) & ( v > f[ 'minValue' ] ) ) )


def flower_heads( rgb, m, away=None, hsv3=None ):

	"""Flower heads: (count, contrast). `away` masks out what is not meadow (the GT frame's sea and
	sky, dilated): the same pixels on every side. A component of up to maxArea is one head; a larger
	one, up to MERGED x maxArea, is a cluster of touching heads (umbels in a bed) and counts one head
	per peak of its distance transform (the watershed's seeds: local maxima over 5 x 5 at least
	1.5 px from its border, a plateau one peak), at least one; anything larger is a region, not
	flowers. A head that breaks into fragments counts as more: the gates read a loss only."""
	f = m[ 'flower' ]
	mask = flower_mask( rgb, m, hsv3 )
	if away is not None:
		mask &= ~away
	scale = ( rgb.shape[ 0 ] * rgb.shape[ 1 ] ) / ( 384 * 384 )
	lo, hi = max( 2, int( round( f[ 'minArea' ] * scale ) ) ), int( round( f[ 'maxArea' ] * scale ) )
	labels, n = ndi.label( mask )
	if n == 0:
		return 0, 0.0
	sizes = np.bincount( labels.ravel() )
	single = np.zeros( n + 1, dtype=bool )
	single[ 1: ] = ( sizes[ 1: ] >= lo ) & ( sizes[ 1: ] <= hi )
	merged = np.zeros( n + 1, dtype=bool )
	merged[ 1: ] = ( sizes[ 1: ] > hi ) & ( sizes[ 1: ] <= MERGED * hi )
	heads = ( single | merged )[ labels ]
	count = int( single.sum() )
	inside = merged[ labels ]
	if inside.any():
		dt = ndi.distance_transform_edt( inside )
		peaks, _ = ndi.label( ( dt == ndi.maximum_filter( dt, size=5 ) ) & ( dt >= 1.5 ) & inside, structure=np.ones( ( 3, 3 ) ) )
		# the distinct peaks of each cluster (a peak touching two counts in both), from the pairs
		# ( component, peak ) present: one pass over the pixels instead of one per cluster
		sel = peaks > 0
		span = int( peaks.max() ) + 1
		pairs = np.unique( labels[ sel ].astype( np.int64 ) * span + peaks[ sel ] )
		per = np.bincount( pairs // span, minlength=n + 1 )
		count += int( np.maximum( per[ merged ], 1 ).sum() )
	if not count:
		return 0, 0.0
	lum = luma( rgb )
	ring = dilate( heads, 3 ) & ~mask
	contrast = float( lum[ heads ].mean() - lum[ ring ].mean() ) if ring.any() else 0.0
	return count, contrast


# a flower component up to this many times maxArea is a cluster of heads (flower_heads)
MERGED = 8
# the fewest meadow pixels (meadow_region) in every GT frame for the meadow statistics
MIN_MEADOW_PX = 400


def meadow_region( gt_rgb, m, away=None ):

	"""The meadow of a GT frame: its blades dilated over the soil between them (4 px at the GT's
	288 px crop), less its sea and sky (dilated 2 px). Every side's meadow statistics are taken over
	this one mask, so a crop's sea, sky and bare stone do not move them and a test / MSAA ratio
	compares the same pixels."""
	r = max( 2, int( round( 4 * gt_rgb.shape[ 1 ] / 288 ) ) )
	return dilate( blade_mask( gt_rgb, m ), r ) & ~( dilate( blue_mask( gt_rgb ), 2 ) if away is None else away )


# the display S-curve's contrast of the default look (Pipeline.ts, neutral-hue by day), undone with
# the sRGB transfer for the meadow's linear statistics (linear luminance and chroma)
LOOK_CONTRAST = 0.04


def to_linear( rgb ):

	"""Linear light of 8-bit display values: the sRGB transfer and the look's S-curve
	( s = x + c ( x^2 ( 3 - 2x ) - x ) ) undone (Newton from x = s)."""
	s = rgb / 255.0
	x = s.copy()
	for _ in range( 6 ):
		f = x + LOOK_CONTRAST * ( x * x * ( 3 - 2 * x ) - x ) - s
		d = 1 + LOOK_CONTRAST * ( 6 * x - 6 * x * x - 1 )
		x = np.clip( x - f / d, 0, 1 )
	return np.where( x <= 0.04045, x / 12.92, ( ( x + 0.055 ) / 1.055 ) ** 2.4 )


# to_linear() of the 256 levels: the frames hold 8-bit values, so a lookup gives the same numbers
# (checked element for element, at every offset of numpy's vector loops)
LINEAR_LUT = to_linear( np.arange( 256, dtype=np.float64 ) )


def blade_mask( rgb, m ):

	g = m[ 'meadow' ]
	r, gg, b = rgb[ ..., 0 ], rgb[ ..., 1 ], rgb[ ..., 2 ]
	return ( ( gg - np.maximum( r, b ) ) / np.maximum( gg, 1.0 ) > g[ 'rel' ] ) & ( luma( rgb ) > g[ 'floor' ] )


def twig_mask( lum, m ):

	t = m[ 'twig' ]
	depth = ndi.grey_closing( lum, size=( t[ 'window' ], t[ 'window' ] ) ) - lum
	return depth > t[ 'tophat' ], depth


def twigs( lum, m ):

	mask, depth = twig_mask( lum, m )
	labels, n = ndi.label( mask, structure=np.ones( ( 3, 3 ) ) )
	sizes = np.bincount( labels.ravel() )[ 1: ] if n else np.zeros( 0 )
	return float( depth[ mask ].mean() ) if mask.any() else 0.0, int( ( sizes >= m[ 'twig' ][ 'minArea' ] ).sum() )


def star_points( lum, m, bg=None ):

	# excess over the local sky (a 7 x 7 median, `bg` when given), so a gradient or thin cloud behind
	# the stars does not hide the faint ones or count as stars; visible stars only (the mask's
	# `visible` level)
	bg = ndi.median_filter( lum, size=7 ) if bg is None else bg
	peak = ndi.maximum_filter( lum, size=3 )
	points = ( lum == peak ) & ( lum >= bg + m[ 'stars' ][ 'visible' ] )
	return np.sort( ( lum - bg )[ points ] )[ ::-1 ]


SHIFT_UPSAMPLE = 100


def phase_ref( a ):

	"""The window and the reference's windowed spectrum (phase_shift()), taken once per GT frame."""
	h, w = a.shape
	wy = np.hanning( h )[ :, None ] * np.hanning( w )[ None, : ]
	return wy, np.fft.fft2( ( a - a.mean() ) * wy )


def phase_shift( a, b, ref=None ):

	"""Sub-pixel shift ( dx, dy ) of b against a by phase correlation, refined on the correlation
	surface upsampled by SHIFT_UPSAMPLE around its integer peak (a matrix-multiply DFT, as in
	Guizar-Sicairos, Thurman and Fienup, "Efficient subpixel image registration algorithms", 2008).
	A parabola through the peak's neighbours read about half the true shift on the whitened
	correlation's narrow peak (0.0077 px measured as 0.004), so bounds in px meant twice that."""
	h, w = a.shape
	wy, fa = ref or phase_ref( a )
	fb = np.fft.fft2( ( b - b.mean() ) * wy )
	r = fb * np.conj( fa )
	r /= np.abs( r ) + 1e-12
	c = np.real( np.fft.ifft2( r ) )
	y, x = np.unravel_index( np.argmax( c ), c.shape )
	y0 = y - h if y > h / 2 else y
	x0 = x - w if x > w / 2 else x
	# the correlation at y0 + t, x0 + t for t in [ -1.5, 1.5 ] px in steps of 1 / SHIFT_UPSAMPLE
	t = np.arange( - 1.5, 1.5 + 1e-9, 1.0 / SHIFT_UPSAMPLE )
	ky, kx = np.fft.fftfreq( h ) * h, np.fft.fftfreq( w ) * w
	ey = np.exp( 2j * np.pi * np.outer( y0 + t, ky ) / h )
	ex = np.exp( 2j * np.pi * np.outer( kx, x0 + t ) / w )
	up = np.real( ey @ r @ ex )
	j, i = np.unravel_index( np.argmax( up ), up.shape )
	# a parabola through the upsampled peak's neighbours: the surface is smooth at this step

	def refine( cm, c0, cp ):
		den = cm - 2 * c0 + cp
		return 0.0 if abs( den ) < 1e-12 else 0.5 * ( cm - cp ) / den

	n = len( t ) - 1
	fj = refine( up[ j - 1, i ], up[ j, i ], up[ j + 1, i ] ) if 0 < j < n else 0.0
	fi = refine( up[ j, i - 1 ], up[ j, i ], up[ j, i + 1 ] ) if 0 < i < n else 0.0
	return float( x0 + t[ i ] + fi / SHIFT_UPSAMPLE ), float( y0 + t[ j ] + fj / SHIFT_UPSAMPLE )


def edge_width( lum, ref, rng=4.0, step=0.25, max_edges=1500 ):

	"""Median 10-90 % rise distance (px) of `lum` across the strongest edges of `ref`, sampled along
	ref's gradient normals, so both sides of a comparison are measured across the same edges."""
	p = np.pad( ref, 1, mode='edge' )
	gx = ( ( p[ :-2, 2: ] + 2 * p[ 1:-1, 2: ] + p[ 2:, 2: ] ) - ( p[ :-2, :-2 ] + 2 * p[ 1:-1, :-2 ] + p[ 2:, :-2 ] ) ) / 4.0
	gy = ( ( p[ 2:, :-2 ] + 2 * p[ 2:, 1:-1 ] + p[ 2:, 2: ] ) - ( p[ :-2, :-2 ] + 2 * p[ :-2, 1:-1 ] + p[ :-2, 2: ] ) ) / 4.0
	mag = np.hypot( gx, gy )
	m = int( np.ceil( rng ) ) + 1
	# local maxima along the (rounded) gradient direction, strong, away from the border
	ang = np.round( np.arctan2( gy, gx ) / ( np.pi / 4 ) ).astype( int ) % 4
	keep = np.zeros_like( mag, dtype=bool )
	for k, ( dy, dx ) in { 0: ( 0, 1 ), 1: ( 1, 1 ), 2: ( 1, 0 ), 3: ( 1, -1 ) }.items():
		a = np.roll( mag, ( dy, dx ), axis=( 0, 1 ) )
		b = np.roll( mag, ( -dy, -dx ), axis=( 0, 1 ) )
		keep |= ( ang == k ) & ( mag >= a ) & ( mag >= b )
	keep &= mag > 24
	keep[ :m, : ] = False
	keep[ -m:, : ] = False
	keep[ :, :m ] = False
	keep[ :, -m: ] = False
	ys, xs = np.nonzero( keep )
	if len( ys ) < 8:
		return None
	order = np.argsort( mag[ ys, xs ] )[ ::-1 ][ :max_edges ]
	ys, xs = ys[ order ], xs[ order ]
	nx, ny = gx[ ys, xs ] / mag[ ys, xs ], gy[ ys, xs ] / mag[ ys, xs ]
	t = np.arange( -rng, rng + 1e-9, step )
	cy = ys[ :, None ] + ny[ :, None ] * t[ None, : ]
	cx = xs[ :, None ] + nx[ :, None ] * t[ None, : ]
	prof = ndi.map_coordinates( lum, [ cy.ravel(), cx.ravel() ], order=1, mode='nearest' ).reshape( cy.shape )
	lo, hi = prof[ :, :4 ].mean( 1 ), prof[ :, -4: ].mean( 1 )
	d = hi - lo
	ok = np.abs( d ) >= 12
	u = ( prof[ ok ] - lo[ ok, None ] ) / d[ ok, None ]  # rises from about 0 to 1 along the normal

	def cross( level ):
		above = u >= level
		j = np.argmax( above, axis=1 )
		valid = above.any( 1 ) & ( j > 0 )
		j = np.maximum( j, 1 )
		r = np.arange( len( u ) )
		f = ( level - u[ r, j - 1 ] ) / np.maximum( u[ r, j ] - u[ r, j - 1 ], 1e-9 )
		return t[ j - 1 ] + f * step, valid

	a, va = cross( 0.1 )
	b, vb = cross( 0.9 )
	sel = va & vb & ( b >= a )
	return float( np.median( ( b - a )[ sel ] ) ) if sel.sum() >= 8 else None


def is_edge_crop( crop ):

	return any( k in crop for k in ( 'cliff-edge', 'horizon', 'islander', 'moon' ) )


def safe_ratio( a, b ):

	if a is None or b is None:
		return None
	if abs( b ) < 1e-12:
		return None if abs( a ) >= 1e-12 else 1.0
	return a / b


def mean_or_none( xs ):

	xs = [ x for x in xs if x is not None ]
	return float( np.mean( xs ) ) if xs else None


# ------------------------------------------------------------------ measuring

# The work is split in two (the header's "Speed"): frame_job() reads one frame index of a crop (the
# test's, MSAA's and, where it has one, the GT's frame) and takes every per-frame value, in a worker
# process; measure() and own_metrics() then combine them, and the metrics that pair frames (F, Fhp,
# TI, G, Rec, the twinkle) are taken from the frames' luma, in the main process. Every value is the
# same expression on the same pixels as when one loop took them all, and the per-frame values are
# combined in frame order, so the results do not depend on the worker count.

def is_vegetation( crop ):

	return any( k in crop for k in ( 'meadow', 'flower', 'rim', 'tree', 'twig' ) )


def is_flower_crop( crop ):

	return 'flower' in crop or 'meadow' in crop


def meadow_stats( rgb, region, m, u8, hsv3 ):

	"""The meadow statistics of one frame over the GT frame's meadow region (measure()); u8 the
	frame's 8-bit values (the linear values by lookup) and hsv3 its hsv()."""
	lum = luma( rgb )
	_, s, _ = hsv3
	local = np.sqrt( np.maximum( ndi.uniform_filter( lum * lum, 5 ) - ndi.uniform_filter( lum, 5 ) ** 2, 0 ) )
	lin = LINEAR_LUT[ u8 ]
	co = ( lin[ ..., 0 ] - lin[ ..., 2 ] ) / 2
	cg = ( 2 * lin[ ..., 1 ] - lin[ ..., 0 ] - lin[ ..., 2 ] ) / 4
	return np.array( [ lum[ region ].mean(), s[ region ].mean(), green_mask( rgb, hsv3 )[ region ].mean(), local[ region ].mean(), blade_mask( rgb, m )[ region ].mean(),
		luma( lin )[ region ].mean(), np.hypot( co, cg )[ region ].mean() ] )


def gt_frame( G, u8, crop, m ):

	"""What the metrics read of one GT frame: the masks the sides are measured over (kept here, in
	the worker) and the GT's own values (`own`, returned). Taken once for both sides."""
	Lg = luma( G )
	sob = sobel( Lg )
	hsv3 = hsv( G )
	green = green_mask( G, hsv3 )
	own = { 'mean': Lg.mean(), 'sharp': float( sob.mean() ), 'green': float( green.mean() ) }
	ctx = { 'G': G, 'Lg': Lg, 'own': own, 'phase': phase_ref( Lg ) }
	# ringing: the GT's edges and its own 3 x 3 range
	edges = sob > 32
	if edges.any():
		ctx[ 'edges' ] = edges
		ctx[ 'hi' ] = ndi.maximum_filter( Lg, size=3 )
		ctx[ 'lo' ] = ndi.minimum_filter( Lg, size=3 )
	# the sea and sky dilated 2 px: the fringe's band (where it meets the dilated green) and what the
	# flower heads and the meadow statistics leave out
	away = dilate( blue_mask( G, hsv3 ), 2 )
	band = dilate( green, 2 ) & away
	if band.sum() >= 20:
		ctx[ 'band' ] = band
		own[ 'bandMean' ] = Lg[ band ].mean()
	if is_flower_crop( crop ):
		ctx[ 'away' ] = away
		own[ 'heads' ] = flower_heads( G, m, away, hsv3 )
		region = meadow_region( G, m, away )
		ctx[ 'region' ] = region
		own[ 'regionMean' ] = region.mean()
		own[ 'regionSum' ] = int( region.sum() )
		# a frame below the floor drops the meadow statistics of the whole crop (measure())
		if own[ 'regionSum' ] >= MIN_MEADOW_PX:
			own[ 'stats' ] = meadow_stats( G, region, m, u8, hsv3 )
	if 'twig' in crop:
		own[ 'twigs' ] = twigs( Lg, m )
	if 'star' in crop:
		own[ 'stars' ] = star_points( Lg, m )
	if 'glint' in crop:
		level = m[ 'glints' ][ 'level' ]
		own[ 'glintShare' ] = float( ( Lg >= level ).mean() )
		own[ 'glintEnergy' ] = float( np.maximum( Lg - level, 0 ).sum() )
	if 'rain' in crop:
		own[ 'streaks' ] = float( np.maximum( Lg - ndi.uniform_filter( Lg, 5 ), 0 ).mean() )
	if is_edge_crop( crop ):
		own[ 'edgeW' ] = edge_width( Lg, Lg )
	return ctx


def side_vs_gt( X, u8, Lx, sharp, gt, crop, m, use_flip, bg=None ):

	"""One side's frame against the GT frame (measure()'s per-frame values); bg the 7 x 7 median of
	Lx in star crops, when side_own() took it."""
	G, Lg, g = gt[ 'G' ], gt[ 'Lg' ], gt[ 'own' ]
	hsv3 = hsv( X )
	v = { 'mean': Lx.mean(), 'S': safe_ratio( sharp, g[ 'sharp' ] ) }
	if use_flip:
		if 'G32' not in gt:
			gt[ 'G32' ] = ( G / 255.0 ).astype( np.float32 )
		_, mean, _ = flip.evaluate( gt[ 'G32' ], ( X / 255.0 ).astype( np.float32 ), 'LDR' )
		v[ 'FLIP' ] = float( mean )
	else:
		v[ 'MAE' ] = float( np.abs( X - G ).mean() )
	if 'edges' in gt:
		over = np.maximum( Lx - gt[ 'hi' ], 0 ) + np.maximum( gt[ 'lo' ] - Lx, 0 )
		v[ 'R' ] = float( over[ gt[ 'edges' ] ].mean() )
	v[ 'W' ] = safe_ratio( float( green_mask( X, hsv3 ).mean() ), g[ 'green' ] )
	if 'band' in gt:
		v[ 'fringe' ] = float( Lx[ gt[ 'band' ] ].mean() - g[ 'bandMean' ] )
	if is_flower_crop( crop ):
		v[ 'heads' ] = flower_heads( X, m, gt[ 'away' ], hsv3 )
		if 'stats' in g:
			v[ 'stats' ] = meadow_stats( X, gt[ 'region' ], m, u8, hsv3 )
	if 'twig' in crop:
		v[ 'twigs' ] = twigs( Lx, m )
	if 'star' in crop:
		v[ 'stars' ] = star_points( Lx, m, bg )
	if 'glint' in crop:
		level = m[ 'glints' ][ 'level' ]
		v[ 'glintShare' ] = float( ( Lx >= level ).mean() )
		v[ 'glintEnergy' ] = float( np.maximum( Lx - level, 0 ).sum() )
	if 'rain' in crop:
		v[ 'streaks' ] = float( np.maximum( Lx - ndi.uniform_filter( Lx, 5 ), 0 ).mean() )
	if is_edge_crop( crop ):
		v[ 'edgeW' ] = edge_width( Lx, Lg )
	v[ 'shift' ] = phase_shift( Lg, Lx, gt[ 'phase' ] )
	return v


def side_own( X, Lx, crop, m, twinkle ):

	"""One side's own per-frame values (own_metrics()); with the twinkle, the frame's 7 x 7 median
	under `bg` (star_points() reads it too; not returned)."""
	v = { 'sharp': float( sobel( Lx ).mean() ) }
	if is_vegetation( crop ):
		# the max of three elementwise: the same as X.max( -1 ), faster
		v[ 'holes' ] = int( ( np.maximum( np.maximum( X[ ..., 0 ], X[ ..., 1 ] ), X[ ..., 2 ] ) <= m[ 'holes' ][ 'level' ] ).sum() )
	if 'glint' in crop:
		v[ 'glintCount' ] = float( ( Lx >= m[ 'glints' ][ 'level' ] ).sum() )
	if twinkle:
		# each pixel's excess over the local sky, its 3 x 3 maximum: read at the stars of the mean frame
		v[ 'bg' ] = ndi.median_filter( Lx, size=7 )
		v[ 'excess' ] = ndi.maximum_filter( Lx - v[ 'bg' ], size=3 )
	return v


def frame_job( job ):

	"""Every per-frame value of one frame index of a crop: the sides' own, and with a GT frame, the
	GT's and each side's against it. Returns them with the frames' 8-bit pixels (the main process
	takes the metrics that pair frames from their luma)."""
	crop, m, size, use_flip, twinkle = job[ 'crop' ], job[ 'm' ], job[ 'size' ], job[ 'flip' ], job[ 'twinkle' ]
	out = { 'rgb': {}, 'own': {}, 'vs': {} }
	frames = {}
	for side in ( 'test', 'msaa' ):
		if job[ side ] is not None:
			out[ 'rgb' ][ side ] = load_u8( job[ side ], size, job[ 'box' ] )
			frames[ side ] = np.asarray( out[ 'rgb' ][ side ], dtype=np.float64 )
	lumas = { side: luma( X ) for side, X in frames.items() }
	for side, X in frames.items():
		out[ 'own' ][ side ] = side_own( X, lumas[ side ], crop, m, twinkle )
	bgs = { side: out[ 'own' ][ side ].pop( 'bg', None ) for side in frames }
	if 'msaa' in frames:
		out[ 'dMSAA' ] = float( np.abs( lumas[ 'test' ] - lumas[ 'msaa' ] ).mean() )
	if job[ 'gt' ] is not None:
		u8 = load_u8( job[ 'gt' ], None, job[ 'box' ] )
		out[ 'rgb' ][ 'gt' ] = u8
		gt = gt_frame( np.asarray( u8, dtype=np.float64 ), u8, crop, m )
		out[ 'gt' ] = gt[ 'own' ]
		for side, X in frames.items():
			out[ 'vs' ][ side ] = side_vs_gt( X, out[ 'rgb' ][ side ], lumas[ side ], out[ 'own' ][ side ][ 'sharp' ], gt, crop, m, use_flip, bgs[ side ] )
	return out


def redraws_of( clip ):

	"""The frames drawn first on a new shadow map, from clip.json's notes: a hook in the clip writes
	`sun <i> <draws> <weight>` at step i, the map's draw count as the step begins, so it shows the
	state after frame i - 1; where the count changes from one note to the next, frame i - 1 is the
	first one drawn on the new map. None when the notes hold no such lines (nothing to split by)."""
	rows = []
	for note in ( clip.get( 'stepped' ) or {} ).get( 'notes', [] ):
		w = note.split()
		if len( w ) == 4 and w[ 0 ] == 'sun':
			rows.append( ( int( w[ 1 ] ), int( w[ 2 ] ) ) )
	if not rows:
		return None
	return [ i - 1 for ( i, d ), ( _, pd ) in zip( rows[ 1: ], rows ) if d != pd ]


# Metric definitions (own_metrics() and measure()).
#
# The temporal metrics against the GT (TI, V, G, L) pair consecutive frames only, so they read the
# GT's dense runs: a change over four frames would hide a shimmer of a shorter period and see a
# jitter sequence whose length divides four at one phase. A side's own metrics (F, Fhp, Vown, Rec,
# the glints' sparkle, the stars' twinkle, dMSAA) use every frame that side captured (the test and
# MSAA capture every frame alike).
#
# Per crop, for the test and for MSAA; "ratio" is test / MSAA:
#
#   B       brightness: mean L of the clip / the GT's (1 = as bright).
#   F       flicker: mean over consecutive frame pairs of the mean |L_t - L_t-1| (levels), own frames.
#           Raw frame-to-frame change, true motion included: compare like with like.
#   Fhp     temporal high-pass flicker: mean |L_t - (L_t-2 + L_t+2) / 2| (levels), own frames: steps
#           and shimmer without the steady change of a slow motion (the shadow-shimmer check).
#           With --split-redraws, where the side's clip.json notes mark shadow map redraws
#           (`sun <i> <draws> <weight>` lines, redraws_of()), also FhpNear over the frames t within 2
#           of a redraw (the first frame drawn on the new map) and FhpBetween over the others: a
#           redraw's step shows as FhpNear well above FhpBetween, the shadow filter's own noise as
#           FhpBetween high against a lit control crop. The series' Fhp holds the per-frame values.
#   TI      temporal instability: mean over consecutive GT frame pairs of the mean |dL_t - dL_gt,t| with
#           dL_t = L_t - L_t-1: the change that the GT does not have (shimmer, crawl, jitter left in
#           the output), 0 = as steady as the GT.
#   S       sharpness: mean Sobel gradient magnitude of L / the GT's (below 1 softer, above 1
#           sharper or noisier).
#   V       sharpness step against the GT: the largest change of the per-frame S between
#           consecutive GT frames.
#   Vown    sharpness step, own frames: the largest change of the mean Sobel magnitude from one frame
#           to the next / its mean (a visible softening or sharpening step: dynamic resolution).
#   FLIP    mean NVIDIA FLIP (LDR) per frame against the GT, averaged (needs flip-evaluator);
#   MAE     else the mean |RGB - GT| in levels.
#   dMSAA   mean |L - L_msaa| over the frames both captured (levels; 0 for MSAA itself).
#   G       ghosting: among pixels where the GT changed by more than 16 levels over two frames (GT
#           frames t and t - 2 both captured)
#           (moving content), the share where the clip at t is off the GT at t by more than 8 levels
#           and closer to the GT at t-2 by more than 4: a trail of where things were. Fine detail
#           that a render resolves differently from the GT scores a little too: judge it against
#           MSAA's.
#   L       lag: the shift in frames (0 to 4) that best lines up the frame-to-frame changes of the
#           clip's mean L with the GT's (a flash or a cut that the clip shows late), over the longest
#           run of consecutive GT frames (at least 7); 0 = none.
#   R       ringing: over the GT's edges (Sobel above 32), the mean overshoot of L beyond the GT's
#           own 3 x 3 range (levels): sharpening halos.
#   Rec     recovery: in a paused clip whose camera stops moving (a flick's end from its path, else
#           the last change of the per-frame view hash in clip.json), the frames after it stops until every later frame is within 1 level (mean
#           |L - L_last|) of the last frame: the history rebuilding after a flick.
#   W       vegetation coverage: the share of green pixels (hue 60-170 degrees, saturation above
#           0.2, value above 0.08) / the GT's: blades and leaves thinner (below 1) or fatter.
#   fringe  mean L in a 2 px band along the border of green and blue (sea or sky: hue 180-250,
#           saturation above 0.15) pixels, minus the GT's (levels; above 0 a pale fringe).
#   holes   pixels with no channel above the holes level (the clear colour showing through the
#           vegetation's depth test), mean count per frame, in vegetation crops.
#   N_f     flower heads (crops named flower or meadow): connected heads of the calibrated flower
#           mask (outside the view's green hue band or unsaturated, at least `chroma` away from the
#           crop's median green chroma, bright or saturated), off the GT frame's sea and sky, 2 to
#           400 px scaled to the crop's area; a component of up to 8 times that is touching heads
#           and counts one per peak of its distance transform / the GT's count, mean over frames.
#           flowers.count is the side's own count (the gates' validity condition).
#   C_f     flower head contrast: mean L of the heads minus that of a 3 px ring around them / the
#           GT's.
#   meadow  for meadow and flower crops, over the meadow region of each GT frame (its blades dilated
#           4 px over the soil between them, less its sea and sky dilated 2 px; the same pixels on
#           every side, so the crop's sea, sky and stone do not move them): mean L, mean saturation,
#           green share, local standard deviation of L (5 x 5, texture) and blade coverage (the
#           calibrated classifier: green's lead over red and blue as a share of green,
#           (G - max(R, B)) / G, above the view's `rel`, and L above `floor`), each / the GT's, and
#           the mean linear luminance (lumaLin) and linear chroma (chroma: the length of YCoCg's
#           Co, Cg) with the sRGB transfer and the default look's display S-curve undone (the
#           measures in linear light); none with fewer than 400 region pixels in a frame. meadowShare is the
#           region's share of the crop. The classifier reads chromaticity, not brightness, so a darker meadow of the same
#           blades keeps its coverage: luma and coverage are separate gates.
#   twigs   for twig crops: the twig mask (a dark-detail black top-hat of L over `window` px above
#           `tophat` levels): its mean depth (contrast) and its count of connected pieces of at least
#           `minArea` px (components), each / the GT's.
#   stars   for star crops: visible stars (local maxima at least the mask's `visible` level, 12,
#           over a 7 x 7 median: below it MSAA's sharpened cloud texture counts) / the GT's count,
#           their mean excess / the GT's, the mean excess of the ten brightest (peak) / the GT's,
#           and twinkle (stars at the calibrated `excess`): on a still camera (the clip's path), the
#           standard deviation over the clip of each star's excess (its 3 x 3 maximum), the stars found in the
#           mean frame, mean over the stars (levels; own frames): the horizon's scintillation. None
#           under a moving camera, where a sub-pixel star's brightness follows its pixel phase.
#   glints  for sea-glint crops: the share of pixels at or above the calibrated level (the MSAA
#           crop's 99.9th percentile) / the GT's, their energy above it / the GT's, and sparkle: the
#           mean change of the count from frame to frame / its mean (own frames).
#   streaks for rain crops: the energy of the bright detail (mean of L above its 5 x 5 mean) / the
#           GT's.
#   peak    the largest rise of the crop's mean L over its median across the GT frames / the GT's
#           (a lightning flash's brightness); only where the GT's rise exceeds 2 levels.
#   edgeW   edge width (crops named cliff-edge, horizon, islander or moon): the median 10-90 % rise
#           distance in GT pixels of the clip's L across the GT's strongest edges, sampled along the
#           GT's gradient normals (non-maximum suppressed, Sobel above 24, rises of at least 12
#           levels), mean over frames. edgeWgt: edgeW minus the GT's own width across the same
#           edges (0 = as soft as the GT). Gated as differences to MSAA (px).
#   shift   the global image shift against the GT (phase correlation refined on a 100x upsampled
#           correlation surface, gain 0.99 on known shifts), its length in GT pixels, mean over
#           frames: a jitter or reprojection offset left in the output; shiftX and shiftY its signed
#           components (x right, y down).

def own_metrics( L, own, crop, m, hashes, clock, path=None, redraws=None ):

	"""Metrics of a side's own frames (every frame it captured), no GT: L its luma and own its
	per-frame values (side_own()), by frame index; with redraws (redraws_of()), Fhp near them and
	between them too. Returns the metrics and the per-frame Fhp series."""
	idx = sorted( L )
	out = {}
	pairs = [ ( a, b ) for a, b in zip( idx, idx[ 1: ] ) if b == a + 1 ]
	out[ 'F' ] = float( np.mean( [ np.abs( L[ b ] - L[ a ] ).mean() for a, b in pairs ] ) ) if pairs else None
	trip = [ i for i in idx if i - 2 in L and i + 2 in L ]
	hp = { i: float( np.abs( L[ i ] - ( L[ i - 2 ] + L[ i + 2 ] ) / 2 ).mean() ) for i in trip }
	out[ 'Fhp' ] = float( np.mean( list( hp.values() ) ) ) if trip else None
	if redraws is not None:
		# a map redraw shows as a step: the frames within 2 of one (those whose high-pass reaches it)
		# are near, the rest between; near well above between is the redraw's step, between itself
		# high (against a lit control crop) is the filter's own noise
		near = { k for r in redraws for k in range( r - 2, r + 3 ) }
		hn, hb = [ hp[ i ] for i in trip if i in near ], [ hp[ i ] for i in trip if i not in near ]
		out[ 'FhpNear' ] = float( np.mean( hn ) ) if hn else None
		out[ 'FhpBetween' ] = float( np.mean( hb ) ) if hb else None
	sharp = { i: own[ i ][ 'sharp' ] for i in idx }
	steps = [ abs( sharp[ b ] - sharp[ a ] ) for a, b in pairs ]
	out[ 'Vown' ] = float( max( steps ) / np.mean( list( sharp.values() ) ) ) if steps and np.mean( list( sharp.values() ) ) > 1e-9 else None
	# recovery after the camera stops (paused clips)
	out[ 'Rec' ] = None
	if clock == 'paused' and hashes and len( idx ) > 2:
		# the stop: from the path where it says (a jittered projection changes the view hash every
		# frame), else the last frame whose view differs from the one before
		if path and path[ 0 ] == 'flick':
			changes = [ int( path[ 2 ] + path[ 3 ] ) ]
		else:
			changes = [ i for i in idx[ 1: ] if i - 1 in hashes and i in hashes and hashes[ i ] != hashes[ i - 1 ] ]
		if changes and idx[ 0 ] < changes[ -1 ] < idx[ -1 ]:
			stop, last = changes[ -1 ], L[ idx[ -1 ] ]
			after = [ i for i in idx if i >= stop ]
			bad = [ i for i in after if np.abs( L[ i ] - last ).mean() > 1.0 ]
			out[ 'Rec' ] = ( bad[ -1 ] + 1 - stop ) if bad else 0
	if is_vegetation( crop ):
		out[ 'holes' ] = float( np.mean( [ own[ i ][ 'holes' ] for i in idx ] ) )
	if 'star' in crop:
		# scintillation: on a still camera (the clip's path), the stars found in the mean frame, each
		# one's excess over the local sky (its 3 x 3 maximum, so a sub-pixel offset does not count)
		# frame by frame, its standard deviation over the clip, the mean over the stars; under a pan
		# a star's brightness changes with its pixel phase, so no value
		out[ 'stars.twinkle' ] = None
		if path and path[ 0 ] == 'still' and len( idx ) > 2:
			mean = np.mean( [ L[ i ] for i in idx ], axis=0 )
			pts = ( mean == ndi.maximum_filter( mean, size=3 ) ) & ( mean >= ndi.median_filter( mean, size=7 ) + m[ 'stars' ][ 'excess' ] )
			if pts.any():
				ex = np.array( [ own[ i ][ 'excess' ][ pts ] for i in idx ] )
				out[ 'stars.twinkle' ] = float( ex.std( axis=0 ).mean() )
	if 'glint' in crop:
		counts = { i: own[ i ][ 'glintCount' ] for i in idx }
		mean = np.mean( list( counts.values() ) )
		out[ 'glints.sparkle' ] = float( np.mean( [ abs( counts[ b ] - counts[ a ] ) for a, b in pairs ] ) / mean ) if pairs and mean > 0 else None
	return out, { 'Fhp': [ hp[ i ] for i in trip ], 'FhpFrames': trip }


def measure( L, Lg, vs, gt, crop, use_flip ):

	"""Metrics of one side's frames against the GT's, at the GT's frames: L and Lg the luma, vs the
	side's per-frame values against the GT (side_vs_gt()) and gt the GT's own (gt_frame()), by GT
	frame index."""
	idx = sorted( Lg )
	out, series = {}, {}
	out[ 'B' ] = safe_ratio( float( np.mean( [ vs[ i ][ 'mean' ] for i in idx ] ) ), float( np.mean( [ gt[ i ][ 'mean' ] for i in idx ] ) ) )
	# temporal metrics against the GT use consecutive GT frames only (its dense runs): a change over
	# four frames hides a shimmer of a shorter period and aliases a jitter sequence to one phase
	pairs = [ ( a, b ) for a, b in zip( idx, idx[ 1: ] ) if b == a + 1 ]
	out[ 'TI' ] = float( np.mean( [ np.abs( ( L[ b ] - L[ a ] ) - ( Lg[ b ] - Lg[ a ] ) ).mean() for a, b in pairs ] ) ) if pairs else None
	sharp = { i: vs[ i ][ 'S' ] for i in idx }
	series[ 'S' ] = [ sharp[ i ] for i in idx ]
	out[ 'S' ] = mean_or_none( sharp.values() )
	steps = [ abs( sharp[ b ] - sharp[ a ] ) for a, b in pairs if sharp[ a ] is not None and sharp[ b ] is not None ]
	out[ 'V' ] = float( max( steps ) ) if steps else None
	name = 'FLIP' if use_flip else 'MAE'
	errs = [ vs[ i ][ name ] for i in idx ]
	series[ name ] = errs
	out[ name ] = float( np.mean( errs ) )
	# ghosting over moving content (two GT frames back)
	ghost, moving = 0, 0
	for t in idx:
		p = t - 2
		if p not in Lg:
			continue
		mv = np.abs( Lg[ t ] - Lg[ p ] ) > 16
		moving += int( mv.sum() )
		err = np.abs( L[ t ] - Lg[ t ] )
		ghost += int( ( mv & ( err > 8 ) & ( np.abs( L[ t ] - Lg[ p ] ) + 4 < err ) ).sum() )
	out[ 'G' ] = ghost / moving if moving else None
	# lag of the mean luma's changes, in frames: over the longest run of consecutive GT frames
	run, cur = [], []
	for i in idx:
		cur = cur + [ i ] if cur and i == cur[ -1 ] + 1 else [ i ]
		if len( cur ) > len( run ):
			run = cur
	if len( run ) > 6:
		mt = np.diff( [ vs[ i ][ 'mean' ] for i in run ] )
		mg = np.diff( [ gt[ i ][ 'mean' ] for i in run ] )
		if np.abs( mg ).max() > 0.05:
			best, lag = - np.inf, 0
			for s in range( 0, 5 ):
				c = float( np.dot( mt[ s: ], mg[ :len( mg ) - s ] ) )
				if c > best:
					best, lag = c, s
			out[ 'L' ] = lag
		else:
			out[ 'L' ] = 0
	else:
		out[ 'L' ] = None
	# a flash: the largest rise of the mean over its median
	if len( idx ) > 6:
		mt = np.array( [ vs[ i ][ 'mean' ] for i in idx ] )
		mg = np.array( [ gt[ i ][ 'mean' ] for i in idx ] )
		rise_g = float( mg.max() - np.median( mg ) )
		out[ 'peak' ] = float( mt.max() - np.median( mt ) ) / rise_g if rise_g > 2 else None
	# ringing over the GT's edges
	ring = [ vs[ i ][ 'R' ] for i in idx if 'R' in vs[ i ] ]
	out[ 'R' ] = float( np.mean( ring ) ) if ring else None
	# vegetation coverage and the fringe against sea or sky
	out[ 'W' ] = mean_or_none( vs[ i ][ 'W' ] for i in idx )
	fr = [ vs[ i ][ 'fringe' ] for i in idx if 'fringe' in vs[ i ] ]
	out[ 'fringe' ] = float( np.mean( fr ) ) if fr else None
	# flowers
	if is_flower_crop( crop ):
		nf, cf, counts = [], [], []
		for i in idx:
			n, c = vs[ i ][ 'heads' ]
			ng, cg = gt[ i ][ 'heads' ]
			nf.append( safe_ratio( float( n ), float( ng ) ) )
			cf.append( safe_ratio( c, cg ) )
			counts.append( n )
		out[ 'N_f' ] = mean_or_none( nf )
		out[ 'C_f' ] = mean_or_none( cf )
		out[ 'flowers.count' ] = float( np.mean( counts ) )
		share = float( np.mean( [ gt[ i ][ 'regionMean' ] for i in idx ] ) )
		keys = [ 'luma', 'saturation', 'green', 'texture', 'coverage', 'lumaLin', 'chroma' ]
		if min( gt[ i ][ 'regionSum' ] for i in idx ) >= MIN_MEADOW_PX:
			a = np.mean( [ vs[ i ][ 'stats' ] for i in idx ], axis=0 )
			g = np.mean( [ gt[ i ][ 'stats' ] for i in idx ], axis=0 )
			out[ 'meadow' ] = { k: safe_ratio( float( x ), float( y ) ) for k, x, y in zip( keys, a, g ) }
		else:
			out[ 'meadow' ] = { k: None for k in keys }
		# the share of the crop the statistics cover (the same on every side: not a ratio to gate)
		out[ 'meadowShare' ] = share
	if 'twig' in crop:
		tw = [ vs[ i ][ 'twigs' ] for i in idx ]
		tg = [ gt[ i ][ 'twigs' ] for i in idx ]
		out[ 'twigs' ] = { 'contrast': safe_ratio( float( np.mean( [ t[ 0 ] for t in tw ] ) ), float( np.mean( [ t[ 0 ] for t in tg ] ) ) ),
			'components': safe_ratio( float( np.mean( [ t[ 1 ] for t in tw ] ) ), float( np.mean( [ t[ 1 ] for t in tg ] ) ) ) }
	if 'star' in crop:
		pts = [ vs[ i ][ 'stars' ] for i in idx ]
		gpts = [ gt[ i ][ 'stars' ] for i in idx ]
		mean_ex = lambda ps: float( np.mean( [ p.mean() if len( p ) else 0.0 for p in ps ] ) )
		peak_ex = lambda ps: float( np.mean( [ p[ :10 ].mean() if len( p ) else 0.0 for p in ps ] ) )
		out[ 'stars' ] = { 'count': safe_ratio( float( np.mean( [ len( p ) for p in pts ] ) ), float( np.mean( [ len( p ) for p in gpts ] ) ) ),
			'excess': safe_ratio( mean_ex( pts ), mean_ex( gpts ) ), 'peak': safe_ratio( peak_ex( pts ), peak_ex( gpts ) ) }
	if 'glint' in crop:
		out[ 'glints' ] = { 'share': safe_ratio( float( np.mean( [ vs[ i ][ 'glintShare' ] for i in idx ] ) ), float( np.mean( [ gt[ i ][ 'glintShare' ] for i in idx ] ) ) ),
			'energy': safe_ratio( float( np.mean( [ vs[ i ][ 'glintEnergy' ] for i in idx ] ) ), float( np.mean( [ gt[ i ][ 'glintEnergy' ] for i in idx ] ) ) ) }
	if 'rain' in crop:
		out[ 'streaks' ] = { 'energy': safe_ratio( float( np.mean( [ vs[ i ][ 'streaks' ] for i in idx ] ) ), float( np.mean( [ gt[ i ][ 'streaks' ] for i in idx ] ) ) ) }
	if is_edge_crop( crop ):
		out[ 'edgeW' ] = mean_or_none( vs[ i ][ 'edgeW' ] for i in idx )
		own = mean_or_none( gt[ i ][ 'edgeW' ] for i in idx )
		out[ 'edgeWgt' ] = out[ 'edgeW' ] - own if out[ 'edgeW' ] is not None and own is not None else None
	vec = [ vs[ i ][ 'shift' ] for i in idx ]
	shifts = [ float( np.hypot( dx, dy ) ) for dx, dy in vec ]
	series[ 'shift' ] = shifts
	out[ 'shift' ] = float( np.mean( shifts ) )
	# the signed offset (x right, y down, GT px): a jitter bias moves every crop the same way, an
	# upscaler's content-dependent offset does not
	out[ 'shiftX' ] = float( np.mean( [ v[ 0 ] for v in vec ] ) )
	out[ 'shiftY' ] = float( np.mean( [ v[ 1 ] for v in vec ] ) )
	return out, series


def flatten( d, prefix='' ):

	out = {}
	for k, v in d.items():
		if isinstance( v, dict ):
			out.update( flatten( v, f'{ prefix }{ k }.' ) )
		else:
			out[ f'{ prefix }{ k }' ] = v
	return out


def unflatten( d ):

	out = {}
	for k, v in d.items():
		parts = k.split( '.' )
		node = out
		for p in parts[ :-1 ]:
			node = node.setdefault( p, {} )
		node[ parts[ -1 ] ] = v
	return out


# ------------------------------------------------------------------ gates

# The gates' forms (a thresholds file's `metrics` and `gt` rules; the lookup order is in the
# docstring). Keys "kind", "gate" and "note" carry the gate's description. Forms against MSAA:
#
#   ratio       for magnitudes and the ratios to the GT:
#                 "max": r      PASS if test <= max( r x b, floor ), or with "gtPlus": d, if
#                               test <= the GT's own value + d (F only: the GT's frame-to-frame change
#                               over its consecutive frames; a gate the truth itself cannot meet, such
#                               as rain that moves every frame, then passes within d of it)
#                 "min": r      PASS if test >= r x b
#                 "band": [lo, hi]  PASS if lo <= test / b <= hi; with |b| <= floor (b about 0, where
#                               a ratio means nothing) PASS if |test| <= floor
#               "floor" (default 0) is an absolute allowance in the metric's own units, not an MSAA
#               value: under MSAA a paused clip's frames are identical, so its F and TI are exactly 0.
#   conditions  "tiers": [..] judges the rule only for those tiers (the first part of --key);
#               "requires": { metric: least } only where the baseline's metric reaches it (a count
#               large enough to measure); "except": [ key, .. ] not for those --key entries (a view
#               whose image is not reproducible between two loads, so the gate's precision is out
#               of reach). Elsewhere the rule gives no verdict.
#   difference  for the signed, integer and absolute-difference metrics: fringe and dMSAA (levels),
#               edgeW, edgeWgt, shiftX and shiftY (px), holes (pixels), L and Rec (frames):
#                 "diff": [lo, hi]  PASS if lo <= test - b <= hi; with "floor": f the baseline is
#                               max( b, f ) (edgeWgt with floor 0: as wide as the wider of MSAA's
#                               and the GT's edge)
#
# Rules in `gt` (left out with --no-gt-gates: a GT that is not the side's own, such as high's GT
# for medium and low) read the test's value against the GT directly (its metrics are already ratios to the
# GT or errors against it): "min", "max" or "band" (absolute bounds on the value). Their verdicts are
# named "gt:<metric>".
#
# metrics gated by their difference to the baseline, never by a ratio (see the docstring)
DIFF_METRICS = { 'fringe', 'L', 'Rec', 'edgeW', 'edgeWgt', 'holes', 'dMSAA', 'shiftX', 'shiftY' }
RATIO_KEYS = { 'max', 'min', 'band' }
# options of a ratio gate (see the docstring): an allowance over the GT's own value
OPTION_KEYS = { 'gtPlus' }
# descriptive keys a rule may carry (thresholds.json): the GT rules' provenance (`gtSource`,
# `provisional`), the gate it implements (`id`, `section`), the side it is judged against
# (`judge`) and a decision it waits for (`pending`)
# conditions under which a rule is judged at all (applies()): `tiers` (the tiers of --key) and
# `requires` ({ metric: least baseline value }: a count too small to measure, such as a crop with a
# handful of flower heads whose count moves by a third between two loads) and `except` (--key
# entries the rule skips: a view that is not reproducible between two loads)
CONDITION_KEYS = { 'tiers', 'requires', 'except' }
META_KEYS = { 'floor', 'kind', 'gate', 'note', 'id', 'section', 'gtSource', 'provisional', 'pending', 'judge' }


def check_thresholds( rules, gt=False ):

	for key, rule in rules.items():
		name = key.split( '/' )[ -1 ]
		keys = set( rule ) - META_KEYS
		if not gt and name in DIFF_METRICS and keys & RATIO_KEYS:
			raise SystemExit( f'thresholds: { key } is a signed or integer metric: gate it with "diff": [lo, hi], not { sorted( keys & RATIO_KEYS ) }' )
		if 'diff' in rule and ( keys & RATIO_KEYS or gt ):
			raise SystemExit( f'thresholds: { key } mixes "diff" with ratio gates, or is a GT rule with "diff"' )
		unknown = keys - RATIO_KEYS - CONDITION_KEYS - OPTION_KEYS - { 'diff' }
		if unknown:
			raise SystemExit( f'thresholds: { key } has unknown keys { sorted( unknown ) }' )


def applies( rule, tier, base, key=None ):

	if rule is None:
		return False
	if key is not None and key in ( rule.get( 'except' ) or [] ):
		return False
	if 'tiers' in rule and tier is not None and tier not in rule[ 'tiers' ]:
		return False
	for k, least in ( rule.get( 'requires' ) or {} ).items():
		if base.get( k ) is None or base[ k ] < least:
			return False
	return True


def verdict( rule, test, base, gt_value=None ):

	if rule is None or test is None or base is None:
		return None
	eps = 1e-12
	if 'diff' in rule:
		lo, hi = rule[ 'diff' ]
		# a floor raises the baseline: edgeWgt's 0 is the GT's own width, so an edge may be as wide
		# as the wider of MSAA's and the GT's
		if 'floor' in rule:
			base = max( base, float( rule[ 'floor' ] ) )
		return 'PASS' if lo - eps <= test - base <= hi + eps else 'FAIL'
	floor = float( rule.get( 'floor', 0.0 ) )
	plus = rule[ 'gtPlus' ] + gt_value if 'gtPlus' in rule and gt_value is not None else float( '-inf' )
	if 'max' in rule and test > max( rule[ 'max' ] * base, floor, plus ) + eps:
		return 'FAIL'
	if 'min' in rule and test < rule[ 'min' ] * base - eps:
		return 'FAIL'
	if 'band' in rule:
		if abs( base ) <= max( floor, eps ):
			if abs( test ) > floor + eps:
				return 'FAIL'
		elif not ( rule[ 'band' ][ 0 ] - eps <= test / base <= rule[ 'band' ][ 1 ] + eps ):
			return 'FAIL'
	return 'PASS'


def verdict_gt( rule, test ):

	if rule is None or test is None:
		return None
	eps = 1e-12
	if 'max' in rule and test > rule[ 'max' ] + eps:
		return 'FAIL'
	if 'min' in rule and test < rule[ 'min' ] - eps:
		return 'FAIL'
	if 'band' in rule and not ( rule[ 'band' ][ 0 ] - eps <= test <= rule[ 'band' ][ 1 ] + eps ):
		return 'FAIL'
	return 'PASS'


# ------------------------------------------------------------------ calibration

def otsu( values, bins=256, lo=0.0, hi=255.0 ):

	h, edges = np.histogram( values, bins=bins, range=( lo, hi ) )
	h = h.astype( np.float64 )
	centres = ( edges[ :-1 ] + edges[ 1: ] ) / 2
	w0 = np.cumsum( h )
	w1 = w0[ -1 ] - w0
	m0 = np.cumsum( h * centres )
	mu0 = m0 / np.maximum( w0, 1e-9 )
	mu1 = ( m0[ -1 ] - m0 ) / np.maximum( w1, 1e-9 )
	between = w0 * w1 * ( mu0 - mu1 ) ** 2
	return float( centres[ int( np.argmax( between ) ) ] )


def calibrate( sources, overlays=None, size=None ):

	"""Mask parameters per view from MSAA capture-clip folders (the first frame of each), each crop
	box-resampled to `size` first (the GT's crop size, as main() resamples it before measuring), so
	the levels are those of the pixels the metrics read:

	  flower.greenHue  the view's green band: the mode of the hue of saturated pixels (55-175 degrees)
	                   in its meadow and flower crops, +- 4 robust deviations (1.4826 MAD), within
	                   55-175 (yellow heads at 40-55 degrees stay outside it). Rim crops are left out:
	                   their sea would pull the mode to blue.
	  flower.chroma    0.85 x the 99th percentile of the chroma distance of the band's green pixels
	                   from their median, within 0.04-0.08: the heads (pink, lilac, orange, cream) lie
	                   further from the green than that, and the olive soil, which is outside the band
	                   too, nearer (checked on the overlays: below 0.06 the lit soil joins the heads,
	                   above 0.1 the cream daisies leave them)
	  flower.brightLuma 1.1 x the 99th percentile of the band's green luma, within 100-150: an
	                   unsaturated head brighter than any lit blade (white petals on a dim day)
	  meadow.rel       Otsu's split of green's relative lead, (G - max(R, B)) / G, over the
	                   green-dominant pixels above the floor in the meadow and flower crops (the
	                   saturated blades and leaves against the olive soil and yellowed grass), within
	                   0.1-0.35
	  twig.tophat      per twig crop, Otsu's split of the black top-hat over its pixels above 2 levels,
	                   at least 6
	  stars.excess     per star crop, max( 8, 6 x the MAD of L over its 7 x 7 median ): above the
	                   sky's own grain and the thin cloud layer's texture
	  glints.level     per glint crop, the MSAA crop's 99.9th percentile of L (at least 160); where that
	                   is 250 or more, the crop lies in the sun's burned-out glitter path (the glints
	                   clip into white, post/ToneCurve.ts), whose top thousandth is the clipped plateau itself: then 250,
	                   the near-white glitter (`clipped`)
	"""
	out = {}
	for src in sources:
		clip = json.load( open( os.path.join( src, 'clip.json' ) ) )
		view = clip.get( 'view' )
		entry = out.setdefault( view, {} )
		crops = sorted( c for c in os.listdir( src ) if os.path.isdir( os.path.join( src, c ) ) and c not in ( 'metrics', 'full' ) )
		imgs = {}
		for c in crops:
			fr = frames_of( src, c )
			if fr:
				imgs[ c ] = load( fr[ sorted( fr )[ 0 ] ], size )
		veg = [ c for c in imgs if any( k in c for k in ( 'meadow', 'flower' ) ) ]
		if veg:
			hs = []
			for c in veg:
				h, s, v = hsv( imgs[ c ] )
				sel = ( s > 0.2 ) & ( v > 0.08 ) & ( h >= 55 ) & ( h <= 175 )
				hs.append( h[ sel ] )
			hs = np.concatenate( hs )
			if len( hs ) > 500:
				hist, edges = np.histogram( hs, bins=120, range=( 55, 175 ) )
				mode = float( ( edges[ np.argmax( hist ) ] + edges[ np.argmax( hist ) + 1 ] ) / 2 )
				mad = float( np.median( np.abs( hs - mode ) ) ) * 1.4826
				band = [ round( max( 55.0, mode - 4 * mad ), 1 ), round( min( 175.0, mode + 4 * mad ), 1 ) ]
				ds, refs, gl = [], [], []
				for c in veg:
					rgb = imgs[ c ]
					h, s, v = hsv( rgb )
					green = ( h >= band[ 0 ] ) & ( h <= band[ 1 ] ) & ( s > DEFAULT_MASKS[ 'flower' ][ 'greenSat' ] ) & ( v > 0.08 )
					if green.sum() < 50:
						continue
					cb, cr = chroma( rgb )
					ref = ( float( np.median( cb[ green ] ) ), float( np.median( cr[ green ] ) ) )
					refs.append( ref )
					ds.append( np.hypot( cb[ green ] - ref[ 0 ], cr[ green ] - ref[ 1 ] ) )
					gl.append( luma( rgb )[ green ] )
				flower = { 'greenHue': band }
				if ds:
					flower[ 'chroma' ] = round( float( np.clip( 0.85 * np.percentile( np.concatenate( ds ), 99 ), 0.04, 0.08 ) ), 4 )
					flower[ 'brightLuma' ] = round( float( np.clip( 1.1 * np.percentile( np.concatenate( gl ), 99 ), 100, 150 ) ), 1 )
					flower[ 'greenRef' ] = [ round( float( np.median( [ r[ 0 ] for r in refs ] ) ), 4 ), round( float( np.median( [ r[ 1 ] for r in refs ] ) ), 4 ) ]
				entry.setdefault( '*', {} )[ 'flower' ] = flower
			mead = [ c for c in imgs if 'meadow' in c or 'flower' in c ]
			if mead:
				vals = []
				for c in mead:
					rgb = imgs[ c ]
					g, mx = rgb[ ..., 1 ], np.maximum( rgb[ ..., 0 ], rgb[ ..., 2 ] )
					sel = ( g > mx ) & ( luma( rgb ) > DEFAULT_MASKS[ 'meadow' ][ 'floor' ] )
					vals.append( ( ( g - mx ) / np.maximum( g, 1.0 ) )[ sel ] )
				vals = np.concatenate( vals )
				if len( vals ) > 500:
					entry.setdefault( '*', {} )[ 'meadow' ] = { 'rel': round( float( np.clip( otsu( vals, lo=0.0, hi=1.0 ), 0.1, 0.35 ) ), 3 ) }
		for c, rgb in imgs.items():
			lum = luma( rgb )
			if 'twig' in c:
				depth = ndi.grey_closing( lum, size=( DEFAULT_MASKS[ 'twig' ][ 'window' ], ) * 2 ) - lum
				sel = depth[ depth > 2 ]
				if len( sel ) > 100:
					entry.setdefault( c, {} )[ 'twig' ] = { 'tophat': round( max( 6.0, otsu( sel, lo=0, hi=float( sel.max() ) ) ), 1 ) }
			if 'star' in c:
				res = lum - ndi.median_filter( lum, size=7 )
				mad = float( np.median( np.abs( res - np.median( res ) ) ) ) * 1.4826
				entry.setdefault( c, {} )[ 'stars' ] = { 'excess': round( max( 8.0, 6 * mad ), 1 ) }
			if 'glint' in c:
				top = float( np.percentile( lum, 99.9 ) )
				entry.setdefault( c, {} )[ 'glints' ] = { 'percentile': 99.9, 'level': round( max( 160.0, top ), 1 ) } if top < 250 else { 'percentile': 99.9, 'level': 250.0, 'clipped': True }
		if overlays:
			os.makedirs( overlays, exist_ok=True )
			for c, rgb in imgs.items():
				m = masks_for( out, view, c )
				panels = [ rgb ]
				tint = lambda mask, col: np.where( mask[ ..., None ], np.array( col, dtype=np.float64 ), rgb * 0.35 )
				if any( k in c for k in ( 'meadow', 'flower', 'rim' ) ):
					panels += [ tint( flower_mask( rgb, m ), ( 255, 0, 255 ) ), tint( blade_mask( rgb, m ), ( 255, 255, 0 ) ) ]
				if 'twig' in c:
					panels.append( tint( twig_mask( luma( rgb ), m )[ 0 ], ( 255, 0, 0 ) ) )
				if 'star' in c:
					lum = luma( rgb )
					p = ( lum == ndi.maximum_filter( lum, size=3 ) ) & ( lum >= ndi.median_filter( lum, size=7 ) + m[ 'stars' ][ 'excess' ] )
					panels.append( tint( dilate( p, 2 ), ( 255, 80, 80 ) ) )
				if 'glint' in c:
					panels.append( tint( luma( rgb ) >= m[ 'glints' ][ 'level' ], ( 255, 80, 80 ) ) )
				if len( panels ) > 1:
					Image.fromarray( np.concatenate( panels, axis=1 ).clip( 0, 255 ).astype( np.uint8 ) ).save( os.path.join( overlays, f'{ view }.{ c }.png' ) )
	return out


# ------------------------------------------------------------------ main

# Speed: the frames are measured in worker processes, one frame index of a crop per job (its test,
# MSAA and GT frames: the GT's masks, filters and spectrum are taken once for both sides), and the
# per-frame values are combined in frame order in the main process, which also takes the metrics
# that pair frames from the frames' luma. Every number is the same expression on the same pixels as
# in one loop, so the output does not depend on the worker count (byte-identical metrics.json).
# Workers: --workers N, else the environment's CLIP_METRICS_WORKERS, else a quarter of the CPUs
# within 1 to 4 (the CPU is shared with captures, and baseline.mjs runs several metric runs at once);
# 1 measures in the main process. On macOS the main process and the workers run with the application
# scheduling policy (tools/kit/qos.mjs says why; see QOS above).


def default_workers():

	"""Worker processes by default: CLIP_METRICS_WORKERS, else a quarter of the CPUs, 1 to 4 (the
	CPU is shared with captures, world generation and other metric runs: baseline.mjs runs several)."""
	env = os.environ.get( 'CLIP_METRICS_WORKERS' )
	if env:
		return max( 1, int( env ) )
	return max( 1, min( 4, ( os.cpu_count() or 1 ) // 4 ) )


def main():

	ap = argparse.ArgumentParser( description=__doc__.split( '\n' )[ 0 ] )
	ap.add_argument( '--test' )
	ap.add_argument( '--msaa' )
	ap.add_argument( '--gt' )
	ap.add_argument( '--crops', help='crop folders a,b, or a JSON file of crop boxes cut from the whole frames (full/)' )
	ap.add_argument( '--split-redraws', action='store_true', help='Fhp also near and between the shadow map redraws the clip\'s notes mark' )
	ap.add_argument( '--thresholds' )
	ap.add_argument( '--baseline' )
	ap.add_argument( '--key' )
	ap.add_argument( '--out' )
	ap.add_argument( '--no-flip', action='store_true' )
	ap.add_argument( '--no-gt-gates', action='store_true' )
	ap.add_argument( '--calibrate' )
	ap.add_argument( '--overlays' )
	ap.add_argument( '--size', help='--calibrate: the GT crop size the metrics run at, WxH or one number (none for aa=gt, whose crops are the test\'s; 288 for gt-hi)' )
	ap.add_argument( '--workers', type=int, help='worker processes measuring frames (default: CLIP_METRICS_WORKERS, else a quarter of the CPUs, 1 to 4)' )
	ap.add_argument( 'sources', nargs='*' )
	a = ap.parse_args()
	if a.calibrate:
		size = None
		if a.size:
			w, _, h = a.size.partition( 'x' )
			size = ( int( w ), int( h or w ) )
		result = calibrate( a.sources, a.overlays, size )
		json.dump( result, open( a.calibrate, 'w' ), indent=1 )
		print( json.dumps( result, indent=1 ) )
		return
	if not ( a.test and a.msaa ):
		ap.error( '--test and --msaa are required' )
	use_flip = flip is not None and not a.no_flip
	clip_json = { 'gt': {} }
	for side, d in ( ( 'test', a.test ), ( 'msaa', a.msaa ), ( 'gt', a.gt ) ):
		if d is None:
			continue
		p = os.path.join( d, 'clip.json' )
		clip_json[ side ] = json.load( open( p ) ) if os.path.exists( p ) else {}
	names = { c.get( 'clip' ) for c in clip_json.values() if c.get( 'clip' ) }
	if len( names ) > 1:
		print( f'warning: the folders hold different clips: { sorted( names ) }', file=sys.stderr )
	clip_name = clip_json[ 'test' ].get( 'clip' ) or clip_json[ 'msaa' ].get( 'clip' )
	view = clip_json[ 'test' ].get( 'view' ) or clip_json[ 'msaa' ].get( 'view' )
	clock = ( clip_json[ 'test' ].get( 'clipDef' ) or {} ).get( 'clock' )
	cam_path = ( clip_json[ 'test' ].get( 'clipDef' ) or clip_json[ 'msaa' ].get( 'clipDef' ) or {} ).get( 'path' )
	tier = a.key.split( '/' )[ 0 ] if a.key else None
	# whole frames only when named (see the docstring); crop boxes are the view's in the JSON file
	boxes = {}
	if a.crops and a.crops.endswith( '.json' ):
		boxes = json.load( open( a.crops ) ).get( view )
		if not boxes:
			raise SystemExit( f'{ a.crops }: no crop boxes for the view { view!r }' )
		crops = list( boxes )
	elif a.crops:
		crops = a.crops.split( ',' )
	else:
		listed = a.gt or a.test
		crops = sorted( c for c in os.listdir( listed ) if os.path.isdir( os.path.join( listed, c ) ) and c not in ( 'metrics', 'full' ) )
	redraws = { side: redraws_of( clip_json[ side ] ) if a.split_redraws else None for side in ( 'test', 'msaa' ) }
	cfg = json.load( open( a.thresholds ) ) if a.thresholds else {}
	rules, gt_rules, mask_cfg = cfg.get( 'metrics', {} ), cfg.get( 'gt', {} ), cfg.get( 'masks', {} )
	if a.no_gt_gates:
		gt_rules = {}
	check_thresholds( rules )
	check_thresholds( gt_rules, gt=True )
	baseline = json.load( open( a.baseline ) ) if a.baseline else None
	if baseline and 'entries' in baseline:
		if not a.key or a.key not in baseline[ 'entries' ]:
			raise SystemExit( f'baseline: no entry { a.key!r } in { a.baseline } (give --key tier/kind/name)' )
		baseline = baseline[ 'entries' ][ a.key ]
	hashes = lambda side: { b[ 'i' ]: b.get( 'view' ) for b in clip_json[ side ].get( 'boxes', [] ) }
	result = { 'clip': sorted( names ), 'view': view, 'key': a.key, 'metric': 'FLIP' if use_flip else 'MAE', 'test': a.test, 'msaa': a.msaa, 'gt': a.gt,
		'baseline': a.baseline, 'thresholds': a.thresholds, 'crops': {} }
	failed = { }
	# every crop's frames are measured by frame_job() in a pool of worker processes (the header's
	# "Speed"), then combined crop by crop in the order the frames were given
	same = os.path.realpath( a.test ) == os.path.realpath( a.msaa )
	plans = []
	for crop in crops:
		box = boxes.get( crop )
		folder = 'full' if box else crop
		ft, fm = frames_of( a.test, folder ), frames_of( a.msaa, folder )
		fg = frames_of( a.gt, folder ) if a.gt else {}
		own_idx = sorted( set( ft ) & set( fm ) )
		idx = sorted( set( own_idx ) & set( fg ) )
		if not ( idx if a.gt else own_idx ):
			plans.append( { 'crop': crop } )
			continue
		m = masks_for( mask_cfg, view, crop )
		# the GT frame's size, which every metric runs at (without a GT, the test's)
		if box:
			size = ( box[ 2 ] - box[ 0 ], box[ 3 ] - box[ 1 ] )
		else:
			size = Image.open( fg[ idx[ 0 ] ] if a.gt else ft[ own_idx[ 0 ] ] ).size
		twinkle = 'star' in crop and bool( cam_path ) and cam_path[ 0 ] == 'still' and len( own_idx ) > 2
		gidx = set( idx )
		jobs = [ { 'crop': crop, 'm': m, 'size': size, 'box': box, 'flip': use_flip, 'twinkle': twinkle, 'test': ft[ i ], 'msaa': None if same else fm[ i ],
			'gt': fg[ i ] if i in gidx else None } for i in own_idx ]
		plans.append( { 'crop': crop, 'idx': idx, 'own_idx': own_idx, 'm': m, 'size': size, 'jobs': jobs } )
	jobs = [ j for p in plans for j in p.get( 'jobs', [] ) ]
	workers = max( 1, min( a.workers if a.workers is not None else default_workers(), len( jobs ) ) )
	pool = ProcessPoolExecutor( max_workers=workers, mp_context=pool_context() ) if workers > 1 else None
	results = pool.map( frame_job, jobs ) if pool else map( frame_job, jobs )
	for plan in plans:
		crop = plan[ 'crop' ]
		if 'jobs' not in plan:
			result[ 'crops' ][ crop ] = { 'missing': 'no frame in all three folders' if a.gt else 'no frame in both folders' }
			continue
		idx, own_idx, m, size = plan[ 'idx' ], plan[ 'own_idx' ], plan[ 'm' ], plan[ 'size' ]
		recs = { i: next( results ) for i in own_idx }
		lumas = lambda side, frames: { i: luma( np.asarray( recs[ i ][ 'rgb' ][ side ], dtype=np.float64 ) ) for i in frames }
		Lt, Lg = lumas( 'test', own_idx ), lumas( 'gt', idx )
		gt = { i: recs[ i ][ 'gt' ] for i in idx }
		side_of = lambda side, key, frames: { i: recs[ i ][ key ][ side ] for i in frames }
		def side_metrics( side, L ):
			# without a GT only the side's own metrics; merged, not updated: own_metrics adds to nested
			# groups (stars, glints) that measure() fills
			mx, sx = measure( { i: L[ i ] for i in idx }, Lg, side_of( side, 'vs', idx ), gt, crop, use_flip ) if a.gt else ( {}, {} )
			mo, so = own_metrics( L, side_of( side, 'own', own_idx ), crop, m, hashes( side ), clock, cam_path, redraws[ side ] )
			return merge( mx, unflatten( mo ) ), { **sx, **so }
		mt, st = side_metrics( 'test', Lt )
		if same:
			mm, sm = copy.deepcopy( mt ), copy.deepcopy( st )
		else:
			mm, sm = side_metrics( 'msaa', lumas( 'msaa', own_idx ) )
		mt[ 'dMSAA' ] = 0.0 if same else float( np.mean( [ recs[ i ][ 'dMSAA' ] for i in own_idx ] ) )
		mm[ 'dMSAA' ] = 0.0
		ft_, fm_ = flatten( mt ), flatten( mm )
		if baseline and crop in baseline.get( 'crops', {} ) and 'msaa' in baseline[ 'crops' ][ crop ]:
			base = flatten( baseline[ 'crops' ][ crop ][ 'msaa' ] )
		else:
			base = fm_
		# the GT's own frame-to-frame change, over its consecutive frames (its dense runs)
		gpairs = [ ( p, q ) for p, q in zip( idx, idx[ 1: ] ) if q == p + 1 ]
		gown = { 'F': float( np.mean( [ np.abs( Lg[ q ] - Lg[ p ] ).mean() for p, q in gpairs ] ) ) if gpairs else None }
		entry = { 'frames': len( own_idx ), 'gtFrames': len( idx ), 'size': list( size ), **( { 'box': boxes[ crop ] } if crop in boxes else {} ),
			**( { 'redraws': redraws } if a.split_redraws else {} ), 'test': mt, 'msaa': mm, 'gtOwn': gown,
			'ratio': { k: safe_ratio( v, fm_.get( k ) ) for k, v in ft_.items() if isinstance( v, ( int, float ) ) },
			'series': { 'test': st, 'msaa': sm } }
		if rules or gt_rules:
			v = {}
			for k, x in ft_.items():
				r = find_rule( rules, clip_name, view, crop, k )
				res = verdict( r, x, base.get( k ), gown.get( k ) ) if applies( r, tier, base, a.key ) else None
				if res:
					v[ k ] = res
					if res == 'FAIL':
						failed.setdefault( r.get( 'kind', 'gate' ), [] ).append( f'{ crop }/{ k }' )
				rg = find_rule( gt_rules, clip_name, view, crop, k )
				res = verdict_gt( rg, x ) if applies( rg, tier, base, a.key ) else None
				if res:
					v[ f'gt:{ k }' ] = res
					if res == 'FAIL':
						failed.setdefault( 'gt:' + rg.get( 'kind', 'gate' ), [] ).append( f'{ crop }/{ k }' )
			entry[ 'verdict' ] = v
		result[ 'crops' ][ crop ] = entry
	if pool:
		pool.shutdown()
	if rules or gt_rules:
		result[ 'gates' ] = { 'failed': sum( len( x ) for x in failed.values() ), 'byKind': failed }
	out = a.out or os.path.join( a.test, 'metrics' )
	os.makedirs( out, exist_ok=True )
	json.dump( result, open( os.path.join( out, 'metrics.json' ), 'w' ), indent=1 )
	# the table
	fmt = lambda v: '–' if v is None else ( f'{ v:.4g}' if isinstance( v, float ) else str( v ) )
	lines = [ f'# clip metrics: { ", ".join( result[ "clip" ] ) } ({ view })', '', f'test `{ a.test }`, MSAA `{ a.msaa }`, ' + ( f'GT `{ a.gt }`; { result[ "metric" ] }' if a.gt else 'no GT: own metrics only' ) + ( f'; baseline `{ a.baseline }` { a.key or "" }' if a.baseline else '' ), '' ]
	for crop, e in result[ 'crops' ].items():
		lines.append( f'## { crop }' )
		if 'missing' in e:
			lines += [ '', e[ 'missing' ], '' ]
			continue
		gf = e.get( 'gtOwn', {} ).get( 'F' )
		lines += [ '', f'{ e[ "frames" ] } frames ({ e[ "gtFrames" ] } with the GT) at { e[ "size" ][ 0 ] }x{ e[ "size" ][ 1 ] }' + ( f'; the GT\'s own F { gf:.4g}' if gf is not None else '' ), '', '| metric | test | MSAA | test / MSAA | gate | GT gate |', '|---|---|---|---|---|---|' ]
		ft_, fm_ = flatten( e[ 'test' ] ), flatten( e[ 'msaa' ] )
		for k in ft_:
			v = e.get( 'verdict', {} )
			lines.append( f'| { k } | { fmt( ft_[ k ] ) } | { fmt( fm_.get( k ) ) } | { fmt( e[ "ratio" ].get( k ) ) } | { v.get( k, "" ) } | { v.get( "gt:" + k, "" ) } |' )
		lines.append( '' )
	if 'gates' in result:
		g = result[ 'gates' ]
		lines.append( f'**{ "FAIL" if g[ "failed" ] else "PASS" }** ({ g[ "failed" ] } gates failed' + ( ': ' + '; '.join( f'{ k } { len( x ) }' for k, x in g[ 'byKind' ].items() ) if g[ 'failed' ] else '' ) + ')' )
	open( os.path.join( out, 'metrics.md' ), 'w' ).write( '\n'.join( lines ) + '\n' )
	print( '\n'.join( lines ) )
	sys.exit( 1 if result.get( 'gates', {} ).get( 'failed' ) else 0 )


if __name__ == '__main__':
	main()
