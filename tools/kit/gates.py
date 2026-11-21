# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Regroups one baseline.mjs label's gate verdicts per gate id, with a task's decisions applied.

It answers which gates a label passes. <metrics.json> is a label's `<root>/<sha>/<label>/metrics.json`
measured with --thresholds (tools/kit/parity/baseline.mjs): every crop of every entry carries
tools/kit/clip-metrics.py's verdicts, "<metric>" against MSAA and "gt:<metric>" against the ground
truth. clip-metrics judges each crop by the rule that the thresholds file gives it; a task's later
decisions (judge this look against the ground truth instead of MSAA, leave that gate
informational) are not in those verdicts. This tool applies them and says, per gate id of the
thresholds file, how many verdicts pass and fail and which items fail. To compare labels' values
side by side, use tools/kit/table.py.

What it guarantees: every verdict comes from the metrics file, either as clip-metrics gave it or
rejudged by a policy rule from the test's own value in the file; nothing is measured again. The
tool holds no decision itself: every one is a rule of the policy file, so the decisions stay with
the task that made them and can be read, diffed and copied.

Usage:
    uv run tools/kit/gates.py <metrics.json> --thresholds <thresholds.json> [options]

Options:
    --thresholds <file>   the thresholds file the metrics were judged by (required)
    --policy <file>       the task's decisions (below); without one, clip-metrics' verdicts stand
    --normal              count special-weather entries apart: they fail no gate
    --md <file>           the report as Markdown too
    --json <file>         the counts per gate with the failing items
    --title <text>        the report's title; default the label

Gate ids: a verdict's gate is the id of the thresholds rule that judged it, found as clip-metrics
finds it (tools/kit/threshold_rules.py gives the order; `gt` rules for "gt:" verdicts); "?" when
none matches. Gates print in the thresholds' `gates` order.

The policy (JSON):

    { "about": "...",
      "informational": [ { "match": { ... }, "note": "why" }, ... ],
      "judge": [ { "match": { ... }, "gate": "IQ-x", "band": [ lo, hi ] | "min": x | "max": x,
                   "skipBelow": { "a/b": least }, "note": "why" }, ... ] }

  match       any of "gate", "tier", "kind" (rest|clip), "entry" (the entry's name, `clip@view` for a
              clip on another view), "clip" ("rest" for rest images), "view", "crop", "metric" (the
              name without "gt:"), "against" (msaa|gt): each a glob (fnmatch) or a list of globs,
              all given fields must match; a field left out matches anything.
  informational  every matching verdict is counted apart and never fails a gate (its note says
              why). A verdict matched by several rules takes the last rule's note.
  judge       an MSAA verdict ("against" is msaa) that matches is judged again from the test's value
              of that metric in the file (a ratio to the GT, or a difference from it: clip-metrics'
              test values are already against the GT) by the absolute bounds given (no tolerance:
              lo <= value <= hi), as a new item "GT:<metric>" under "gate" (default: the matched
              verdict's gate). The first matching rule judges a crop's metric; later rules do not
              judge it again, so put the specific rule (a clip's own gate) before the general one.
              "skipBelow" { "a/b": least } leaves the crop unjudged where the test's a / b is below
              least (a quotient of two test values: count / N_f is the GT's flower count); a
              quotient that cannot be taken (a value missing or zero) does not skip. A missing
              test value is not judged. The original verdict is not touched: mark it informational
              with its own rule when the GT judgement replaces it.

--normal: verdicts of special-weather entries (tools/island/parity/views.mjs's group `special`, and
the clips whose view is in it, tools/island/parity/clips.mjs) are counted in their own column and
fail no gate: normal weather is accepted first. Both lists are read from those modules through node.

Output: stdout and --md, a table per gate (PASS, FAIL, informational, special weather) and per
gate the failing items (test and MSAA values, their ratio, the rule) and the informational ones
that fail; --json the counts per gate with the failing items ({ label, aa, verdict, gates: { id:
{ PASS, FAIL, info, special, fails: [ { tier, entry, crop, metric, test, msaa } ] } }, aggregates,
thresholds, policy }). baseline.mjs's aggregates (IQ-fidelity, IQ-alignment) are copied from the
metrics file and count as judged gates.

GPU lock and browser: neither; it reads JSON files, and with --normal runs node to read the two
tables. Cost: under a second.

Exit codes: 0 when every judged gate passes (no FAIL outside the informational and, with --normal,
the special-weather columns, and no aggregate FAIL); 1 otherwise; 2 for bad input (a file that
cannot be read or parsed, an unknown policy key, a missing --thresholds).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help import answer_help  # noqa: E402

answer_help(__file__)

import argparse  # noqa: E402
import fnmatch  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from threshold_rules import find_rule  # noqa: E402

HERE = os.path.dirname( os.path.abspath( __file__ ) )
ROOT = os.path.dirname( os.path.dirname( HERE ) )
# where the project's configuration says the views' and clips' tables are (its `tables`)
TABLE_OF = "const p = ( await import( process.argv[ 1 ] ) ).default; console.log( p.tables[ process.argv[ 2 ] ] )"


def node_list( table, group ):

	# the groups live in the views' and clips' tables alone, so ask them rather than keep a copy here
	config = os.path.join( os.path.dirname( HERE ), 'project.mjs' )
	module = subprocess.run( [ 'node', '--input-type=module', '-e', TABLE_OF, config, table ], capture_output=True, text=True, check=True ).stdout.strip()
	out = subprocess.run( [ 'node', os.path.join( ROOT, module ), 'list', group ], capture_output=True, text=True, check=True ).stdout
	return { line.strip() for line in out.splitlines() if line.strip() }


def gate_of( rules, clip, view, crop, metric ):

	# the rule clip-metrics judged by, found the same way
	rule = find_rule( rules, clip, view, crop, metric )
	return ( rule.get( 'id' ), rule ) if rule is not None else ( None, None )


def get( d, k ):

	# a metric name as clip-metrics flattens it: "meadow.lumaLin" is test[ "meadow" ][ "lumaLin" ],
	# but a flat key with a dot ("flowers.count") is read as it is
	if d is None:
		return None
	if k in d:
		return d[ k ]
	if '.' in k:
		x, y = k.split( '.', 1 )
		v = d.get( x )
		return v.get( y ) if isinstance( v, dict ) else None
	return None


def matches( match, fields ):

	for k, pat in ( match or {} ).items():
		if k not in fields:
			raise ValueError( f'policy: unknown match field { k } (known: { ", ".join( fields ) })' )
		pats = pat if isinstance( pat, list ) else [ pat ]
		v = fields[ k ]
		if v is None or not any( fnmatch.fnmatchcase( str( v ), p ) for p in pats ):
			return False
	return True


def within( rule, value ):

	if 'band' in rule:
		lo, hi = rule[ 'band' ]
		if ( lo is not None and value < lo ) or ( hi is not None and value > hi ):
			return False
	if 'min' in rule and rule[ 'min' ] is not None and value < rule[ 'min' ]:
		return False
	if 'max' in rule and rule[ 'max' ] is not None and value > rule[ 'max' ]:
		return False
	return True


def bounds( rule ):

	if 'band' in rule:
		return rule[ 'band' ]
	return [ rule.get( 'min' ), rule.get( 'max' ) ]


def skipped( rule, test ):

	for q, least in ( rule.get( 'skipBelow' ) or {} ).items():
		a, b = q.split( '/', 1 )
		x, y = get( test, a ), get( test, b )
		if isinstance( x, ( int, float ) ) and isinstance( y, ( int, float ) ) and x and y and x / y < least:
			return True
	return False


def collect( M, T, policy, special_views, special_clips ):

	rules, gtrules = T[ 'metrics' ], T[ 'gt' ]
	infos, judges = policy.get( 'informational', [] ), policy.get( 'judge', [] )
	items = []
	for key, e in M[ 'entries' ].items():
		tier, kind, name = key.split( '/', 2 )
		clip = e.get( 'clip' ) or ( 'rest' if kind == 'rest' else name.split( '@' )[ 0 ] )
		view = e.get( 'view' )
		special = view in special_views or clip in special_clips
		for crop, c in ( e.get( 'crops' ) or {} ).items():
			if not isinstance( c, dict ) or c.get( 'missing' ):
				continue
			ver = c.get( 'verdict' ) or {}
			test, msaa = c.get( 'test' ) or {}, c.get( 'msaa' ) or {}
			msaa_gates = {}
			for vk, v in ver.items():
				gt = vk.startswith( 'gt:' )
				metric = vk[ 3: ] if gt else vk
				gid, rule = gate_of( gtrules if gt else rules, clip, view, crop, metric )
				if not gt:
					msaa_gates[ metric ] = gid
				it = dict( tier=tier, entry=name, kind=kind, clip=clip, view=view, crop=crop, metric=vk, verdict=v, gate=gid or '?', rule=rule,
					test=get( test, metric ), msaa=get( msaa, metric ), informational=False, note='', special=special )
				fields = dict( gate=gid, tier=tier, kind=kind, entry=name, clip=clip, view=view, crop=crop, metric=metric, against='gt' if gt else 'msaa' )
				for r in infos:
					if matches( r.get( 'match' ), fields ):
						it[ 'informational' ] = True
						it[ 'note' ] = r.get( 'note', 'informational (policy)' )
				items.append( it )
			# the policy's judgements against the GT, after the crop's own verdicts
			judged = set()
			for r in judges:
				for metric, gid in msaa_gates.items():
					if metric in judged:
						continue
					fields = dict( gate=gid, tier=tier, kind=kind, entry=name, clip=clip, view=view, crop=crop, metric=metric, against='msaa' )
					if not matches( r.get( 'match' ), fields ):
						continue
					judged.add( metric )
					val = get( test, metric )
					if val is None or skipped( r, test ):
						continue
					items.append( dict( tier=tier, entry=name, kind=kind, clip=clip, view=view, crop=crop, metric='GT:' + metric,
						verdict='PASS' if within( r, val ) else 'FAIL', gate=r.get( 'gate' ) or gid or '?', rule={ 'gtJudge': bounds( r ) },
						test=val, msaa=None, informational=False, note=r.get( 'note', 'judged vs the GT (policy)' ), special=special ) )
	return items


def fmt( x ):

	if x is None:
		return '-'
	if isinstance( x, float ):
		return f'{ x:.4g}'
	return str( x )


def main():

	ap = argparse.ArgumentParser( description='gate verdicts of a baseline.mjs metrics.json (see the header)' )
	ap.add_argument( 'metrics' )
	ap.add_argument( '--thresholds', required=True )
	ap.add_argument( '--policy' )
	ap.add_argument( '--normal', action='store_true', help='special-weather entries fail no gate' )
	ap.add_argument( '--md' )
	ap.add_argument( '--json' )
	ap.add_argument( '--title', default='' )
	a = ap.parse_args()

	try:
		M = json.load( open( a.metrics ) )
		T = json.load( open( a.thresholds ) )
		policy = json.load( open( a.policy ) ) if a.policy else {}
		unknown = set( policy ) - { 'about', 'informational', 'judge' }
		if unknown:
			raise ValueError( f'policy: unknown key { ", ".join( sorted( unknown ) ) }' )
		special_views = node_list( 'views', 'special' ) if a.normal else set()
		special_clips = node_list( 'clips', 'special' ) if a.normal else set()
		items = collect( M, T, policy, special_views, special_clips )
	except ( OSError, ValueError, KeyError, subprocess.CalledProcessError ) as error:
		print( f'gates.py: { error }', file=sys.stderr )
		sys.exit( 2 )

	gates = {}
	for it in items:
		g = gates.setdefault( it[ 'gate' ], dict( PASS=0, FAIL=0, info=0, special=0, fails=[], infos=[] ) )
		if it[ 'informational' ]:
			g[ 'info' ] += 1
			if it[ 'verdict' ] == 'FAIL':
				g[ 'infos' ].append( it )
			continue
		if it[ 'special' ] and a.normal:
			g[ 'special' ] += 1
			if it[ 'verdict' ] == 'FAIL':
				g[ 'infos' ].append( it )
			continue
		g[ it[ 'verdict' ] ] = g.get( it[ 'verdict' ], 0 ) + 1
		if it[ 'verdict' ] == 'FAIL':
			g[ 'fails' ].append( it )

	aggregates = M.get( 'aggregates' ) or []
	failed = any( g[ 'FAIL' ] for g in gates.values() ) or any( ag.get( 'verdict' ) == 'FAIL' for ag in aggregates )
	order = [ x[ 'id' ] for x in T.get( 'gates', [] ) ]
	ranked = sorted( gates, key=lambda g: order.index( g ) if g in order else 999 )

	lines = [ f'## { a.title or M.get( "label" ) } ({ M.get( "aa" ) }, sha { str( M.get( "sha" ) )[ :8 ] }): { "FAIL" if failed else "PASS" }', '',
		f'policy { os.path.basename( a.policy ) if a.policy else "none" }{ ", special weather apart (--normal)" if a.normal else "" }', '' ]
	for ag in aggregates:
		lines.append( f"- aggregate { ag.get( 'gate' ) }: { ag.get( 'verdict' ) } ({ { k: v for k, v in ag.items() if k not in ( 'gate', 'verdict' ) } })" )
	lines += [ '', '| gate | PASS | FAIL | informational | special weather |', '|---|---|---|---|---|' ]
	for gid in ranked:
		g = gates[ gid ]
		lines.append( f"| { gid } | { g[ 'PASS' ] } | { g[ 'FAIL' ] } | { g[ 'info' ] } | { g[ 'special' ] } |" )
	lines.append( '' )
	for gid in ranked:
		g = gates[ gid ]
		if not g[ 'fails' ] and not g[ 'infos' ]:
			continue
		lines.append( f'### { gid }' )
		for it in g[ 'fails' ]:
			ratio = f" (ratio { it[ 'test' ] / it[ 'msaa' ]:.3f})" if isinstance( it[ 'test' ], ( int, float ) ) and isinstance( it[ 'msaa' ], ( int, float ) ) and it[ 'msaa' ] else ''
			lines.append( f"- FAIL { it[ 'tier' ] } { it[ 'entry' ] } / { it[ 'crop' ] } / { it[ 'metric' ] }: test { fmt( it[ 'test' ] ) }, msaa { fmt( it[ 'msaa' ] ) }{ ratio } rule { json.dumps( it[ 'rule' ] )[ :120 ] }" )
		for it in g[ 'infos' ]:
			lines.append( f"- (info{ ' special' if it[ 'special' ] and not it[ 'informational' ] else '' }) { it[ 'verdict' ] } { it[ 'tier' ] } { it[ 'entry' ] } / { it[ 'crop' ] } / { it[ 'metric' ] }: test { fmt( it[ 'test' ] ) }, msaa { fmt( it[ 'msaa' ] ) } { it[ 'note' ] }" )
		lines.append( '' )
	out = '\n'.join( lines )
	print( out )
	if a.md:
		open( a.md, 'w' ).write( out + '\n' )
	if a.json:
		keep = ( 'tier', 'entry', 'crop', 'metric', 'test', 'msaa' )
		json.dump( dict( label=M.get( 'label' ), aa=M.get( 'aa' ), verdict='FAIL' if failed else 'PASS',
			gates={ gid: { k: v for k, v in g.items() if k not in ( 'fails', 'infos' ) } | { 'fails': [ { k: it[ k ] for k in keep } for it in g[ 'fails' ] ] } for gid, g in gates.items() },
			aggregates=M.get( 'aggregates' ), thresholds=os.path.abspath( a.thresholds ), policy=os.path.abspath( a.policy ) if a.policy else None ),
			open( a.json, 'w' ), indent=1 )
	sys.exit( 1 if failed else 0 )


if __name__ == '__main__':
	main()
