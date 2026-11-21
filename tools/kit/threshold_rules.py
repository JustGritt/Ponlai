"""Finds the thresholds rule that judges a crop's metric. Imported by clip-metrics.py, which judges
by it, and gates.py, which names the gate that judged, so the two always agree on the rule.

A thresholds file's `metrics` (and `gt`) map keys to rules. The first key found wins, in this order:
"<clip>@<view>/<crop>/<metric>", "<clip>@<view>/*/<metric>", "<clip>/<crop>/<metric>",
"<clip>/*/<metric>", "*/<crop>/<metric>", "*/*/<metric>", then the older forms "<crop>/<metric>"
and "<metric>". So a clip's own rule for a view beats its rule for every view, and within one scope
a crop's own rule beats the scope's rule for every crop.
"""


def find_rule( rules, clip, view, crop, metric ):

	for scope in ( f'{ clip }@{ view }', clip, '*' ):
		for key in ( f'{ scope }/{ crop }/{ metric }', f'{ scope }/*/{ metric }' ):
			if key in rules:
				return rules[ key ]
	return rules.get( f'{ crop }/{ metric }', rules.get( metric ) )
