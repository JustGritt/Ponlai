"""Measures two matched overview captures region by region and saves unretouched comparison crops.

It answers how a change moved the island's main regions (the cliff, the meadow, the open water and
the water's glare) in one before-and-after pair of overview captures: the same metrics per region,
and the crops side by side for the eye.

Usage:
    uv run --with numpy --with pillow tools/kit/compare-captures.py before.png after.png

Options: none. Both captures are the overview camera at 1672 by 941 pixels; any other size stops
the tool with an assertion.

Output: per region its box and the metrics of each image (pixels.metrics in tools/kit/pixels.py,
which `tools/kit/img.py stats` reports too: mean RGB, luma mean and spread, local contrast) as JSON
on stdout and in artifacts/comparison/metrics.json, and in artifacts/comparison/ each region's
before and after crops and a pair with both side by side (the folder under the working directory
is the configuration's output.comparison, tools/project.mjs).

GPU lock and browser: neither; it reads two PNG files. Cost: under a second.

Exit codes: 0 done; 1 a missing file or captures of another size.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help import answer_help  # noqa: E402

answer_help(__file__)

import json  # noqa: E402

from PIL import Image, ImageDraw  # noqa: E402

from pixels import metrics  # noqa: E402
from project import project  # noqa: E402


before, after = [Image.open(path).convert("RGB") for path in sys.argv[1:3]]
assert before.size == after.size == (1672, 941), "Use matching overview cameras at 1672 x 941"
out = Path(project()["output"]["root"]) / project()["output"]["comparison"]
out.mkdir(parents=True, exist_ok=True)
regions = {
    "cliff": (320, 600, 1100, 790),
    "meadow": (480, 240, 1110, 560),
    "open-water": (25, 30, 375, 155),
    "water-glare": (1480, 310, 1650, 720),
}


report = {}
for name, box in regions.items():
    a, b = before.crop(box), after.crop(box)
    a.save(out / f"{name}-before.png")
    b.save(out / f"{name}-after.png")
    paired = Image.new("RGB", (a.width * 2, a.height + 24), "#172c32")
    paired.paste(a, (0, 24))
    paired.paste(b, (a.width, 24))
    draw = ImageDraw.Draw(paired)
    draw.text((8, 5), f"BEFORE / {name}", fill="white")
    draw.text((a.width + 8, 5), f"AFTER / {name}", fill="white")
    paired.save(out / f"{name}-comparison.png")
    report[name] = {"bounds": box, "before": metrics(a), "after": metrics(b)}

(out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
