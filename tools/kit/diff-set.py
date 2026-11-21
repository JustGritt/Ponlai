"""Compares a capture set with a reference set view by view, exactly or within a tolerance.

It answers whether a set of paused captures (tools/kit/parity/capture-set.mjs) holds the
reference's pixels: identical, within the reference's own run-to-run spread (--aa-floor), or within
the temporal tolerance (--temporal). capture-set.mjs --check runs it. Which rule fits which set is
method: tools/kit/parity/README.md. For two single images, use `tools/kit/img.py diff`.

Usage:
    uv run --with numpy --with pillow tools/kit/diff-set.py <refDir> <testDir> [options]
    uv run --with numpy --with pillow tools/kit/diff-set.py ref/ test/ --aa-floor --crops-from test

Options:
    --views a,b          the views compared; default every <view>.png in <refDir>
    --out <dir>          default <testDir>/diff
    --max-px1 <n>        pass: fewer than n pixels over 1 level and none over 2; default 200, 0 for identity
    --crops <n>          worst regions saved per differing view; default 3
    --crop <px>          their size; default 256
    --aa-floor           judge by each view's own run-to-run spread (both sets captured with --repeat)
    --temporal           judge temporal AA captures by the tolerance (not with --aa-floor)
    --held               MSAA sets: two captures also count as equal within the held tolerance (below)
    --flip               with --temporal: add the mean NVIDIA FLIP, reported only (--with flip-evaluator)
    --crops-from <src>   cut the named crop anchors: ref, test, a directory or a JSON file
    --md <file.md>       the table as Markdown too
    --jobs <n>           views at once, in threads; default 4, or half the cores below 8

Per view (each <view>.png present in both): the max absolute channel difference, the mean absolute
difference (over all channels), and how many pixels differ by more than 1, 2, 8 and 32 levels (a
pixel counts when its largest channel difference exceeds the level). `identical` is equality of
the decoded pixels: a PNG's bytes depend on the encoder (the capture tools' fast PNG against
Playwright's default one), its pixels do not, so sets written by either encoder compare alike.

--aa-floor: a view passes when the test frame is pixel-identical to the reference frame (either
capture of each side), or when the reference is nondeterministic (its A/A pair differs) and the
cross difference stays within the larger of the two sides' A/A differences (max level and the count
of pixels over 1 and 2 levels). A test that is deterministic while the reference is not passes when
its frame lies within the reference's spread (noted as such). A view that is deterministic on the
reference must be deterministic on the test side, and must match it pixel for pixel. Comparing a set with itself (refDir =
testDir) lists its A/A floor.

--temporal judges temporal AA captures (taa, taau) by the tolerance in tools/kit/tolerance.py, whose
header holds the thresholds, the model and their calibration: per-pixel differences up to a level
are ignored, the share of pixels over it is bounded on the frame, in every 32 px window and on
every named crop, and the windows' and crops' mean signed difference bounds coherent shifts under
the level. A failure counts only after one recapture: the view passes when any pairing of the two
sides' captures (<view>.png, <view>.repeat.png) passes, and a failing view without a recapture on
the test side is marked unconfirmed, so capture temporal sets with --repeat. `use` is the largest
measurement / threshold ratio of the best pairing (pass: at most 1). The crops are those of
--crops-from, else of the test side's JSON. FLIP is reported only: a frame's mean hides a missing
flower entirely. MSAA sets stay pixel-exact (the default and --aa-floor).

--held: MSAA paused frames are bit-identical between runs, but for the region a held history-dependent
effect reaches (the sea's reflection: packages/render/debug/HeldRegion.ts), which each capture of a
set records in its JSON (`held`). With --held, two captures count as equal (in the exact rule and in
every comparison --aa-floor makes) when they are identical, or identical outside the union of the
two sides' regions and within tools/kit/tolerance.py's HELD inside it; a view with no region on
either side stays exact. Rows say `(held)` where a pair passed so, and the report keeps each such
pair's `use` and measurements under `held`. capture-set.mjs and verify-scene-jump.mjs pass it for
every MSAA comparison.

--crops-from: `ref` or `test` reads <view>.json on that side, a directory reads <view>.json there,
and a JSON file holds either one capture's `crops` (used for every view) or { view: crops }.

Both sets must come from the same browser build (`browser` in each capture's JSON): a mismatch is
reported as a warning, since no frame then matches pixel for pixel. Compare sets of one build, or a
stored set verified for the other build (tools/kit/reference.mjs).

Output: a table on stdout, one row per view; <out>/report.json with everything. For views that
differ, <out>/<view>.diff.png (the per-pixel max channel difference, x8 and clipped) and the N
worst regions as side-by-side crops (ref | test | diff x8) at 1:1, the windows holding the most
difference, non-overlapping. With --crops-from, per anchor its difference counts and a sheet
<out>/<view>.<anchor>.png, and per view <out>/<view>.anchors.png (the test frame at a quarter size
with its anchor boxes).

GPU lock and browser: neither; it reads PNG files. On macOS the process and its threads run with
the application scheduling policy (tools/kit/qos.mjs).

Cost: seconds for a set; views run in parallel threads. The output does not depend on --jobs: the
table prints in view order and every mean is an exact integer sum divided once.

Exit codes: 0 every view passes; 1 a view fails or is missing; 2 a usage error (--temporal with
--aa-floor or --held).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from help import answer_help  # noqa: E402

answer_help(__file__)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from project import project  # noqa: E402

# The application scheduling policy on macOS (tools/kit/qos.mjs says why): this process re-executes
# itself in place under taskpolicy -a, and its pool threads ask for the user-initiated QoS, which
# such a process is granted (threads do not inherit the main thread's).
QOS_ENV = project()["machine"]["envPrefix"] + "QOS"
QOS = sys.platform == "darwin" and os.environ.get(QOS_ENV) != "off"
if QOS and os.environ.get(QOS_ENV) != str(os.getpid()):
    os.environ[QOS_ENV] = str(os.getpid())
    os.execv("/usr/sbin/taskpolicy", ["/usr/sbin/taskpolicy", "-a", sys.executable, *sys.orig_argv[1:]])


def raise_thread():
    if QOS:
        import ctypes
        ctypes.CDLL(None).pthread_set_qos_class_self_np(0x19, 0)  # QOS_CLASS_USER_INITIATED

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import tolerance
from pixels import LEVELS, X8, Files, sheet, stats

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("test")
ap.add_argument("--out", default=None)
ap.add_argument("--crops", type=int, default=3)
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--max-px1", type=int, default=200)
ap.add_argument("--views", default="")
ap.add_argument("--aa-floor", action="store_true")
ap.add_argument("--md", default=None)
ap.add_argument("--crops-from", default=None)
ap.add_argument("--temporal", action="store_true")
ap.add_argument("--flip", action="store_true")
ap.add_argument("--held", action="store_true")
# views compared at once: 4, or half the cores when the CPU has fewer than 8, since captures and
# world generation share the CPU
ap.add_argument("--jobs", type=int, default=min(4, max(1, (os.cpu_count() or 2) // 2)))
a = ap.parse_args()
if a.temporal and a.aa_floor:
    ap.error("--temporal and --aa-floor are two different rules: pick one")
if a.held and a.temporal:
    ap.error("--held is for MSAA sets (exact or --aa-floor), not --temporal")

ref, test = Path(a.ref), Path(a.test)
out = Path(a.out) if a.out else test / "diff"
out.mkdir(parents=True, exist_ok=True)
views = a.views.split(",") if a.views else sorted(p.stem for p in ref.glob("*.png") if "." not in p.stem)
def worst_windows(v, A, B, D, M, r):
    # worst windows: sum of the difference over crop-sized windows on a half-crop grid, the largest
    # first (ties: the larger x, then the larger y), picked greedily without overlap
    c = a.crop
    S, ys, xs = tolerance.window_sums(M, c, c // 2)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    S, xx, yy = S.ravel(), xx.ravel(), yy.ravel()
    order = np.lexsort((yy, xx, S))[::-1]
    picked = []
    for k in order:
        s, x, y = int(S[k]), int(xx[k]), int(yy[k])
        if s == 0 or len(picked) >= a.crops:
            break
        if all(abs(x - px) >= c or abs(y - py) >= c for _, px, py in picked):
            picked.append((s, x, y))
    r["crops"] = []
    for i, (s, x, y) in enumerate(picked):
        box = (x, y, x + c, y + c)
        name = f"{v}.crop{i}.png"
        sheet(A, B, D, (x, y, c, c), f"{v} @ {x},{y}: ref | test | diff x8 (max {int(M[y:y+c, x:x+c].max())})").save(out / name)
        r["crops"].append({"file": name, "box": box, "sum": s})


def crops_for(v):
    src = a.crops_from
    if src in ("ref", "test"):
        f = (ref if src == "ref" else test) / f"{v}.json"
    else:
        p = Path(src)
        if p.is_dir():
            f = p / f"{v}.json"
        else:
            d = json.loads(p.read_text())
            return d["crops"] if "crops" in d else d.get(v, {})
    return json.loads(f.read_text()).get("crops", {}) if f.exists() else {}


def named_crops(v, A, B, D, r):
    anchors = crops_for(v)
    r["anchors"] = {}
    boxes = []
    for name, c in anchors.items():
        if "x" not in c:
            r["anchors"][name] = {"missing": c.get("missing", "?")}
            continue
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        _, M, st = stats(A[y:y + h, x:x + w], B[y:y + h, x:x + w])
        st["box"] = [x, y, w, h]
        st["file"] = f"{v}.{name}.png"
        r["anchors"][name] = st
        sheet(A, B, D, (x, y, w, h), f"{v} / {name} @ {x},{y}: ref | test | diff x8 (max {st['max']})").save(out / st["file"])
        boxes.append((name, x, y, w, h))
    # where the anchors fall: the test frame at a quarter size
    k = 4
    im = Image.fromarray(B).resize((B.shape[1] // k, B.shape[0] // k), Image.BOX)
    dr = ImageDraw.Draw(im)
    for name, x, y, w, h in boxes:
        dr.rectangle((x // k, y // k, (x + w) // k, (y + h) // k), outline="#ff3080", width=2)
        dr.text((x // k + 3, y // k + 2), name, fill="#ff3080")
    missing = [n for n, c in r["anchors"].items() if "missing" in c]
    if missing:
        dr.text((6, im.height - 14), "missing: " + ", ".join(f"{n} ({r['anchors'][n]['missing']})" for n in missing), fill="#ffd040")
    im.save(out / f"{v}.anchors.png")


def view_crops(v):
    """The named crops the tolerances judge: --crops-from's, else the test side's JSON's."""
    if a.crops_from:
        return crops_for(v)
    f = test / f"{v}.json"
    return json.loads(f.read_text()).get("crops", {}) if f.exists() else {}


def helds(v):
    """The `held` records of the view's captures on both sides (tools/kit/tolerance.py held_region)."""
    out = []
    for d in (ref, test):
        for f in (d / f"{v}.json", d / f"{v}.repeat.json"):
            try:
                out.append(json.loads(f.read_text()).get("held") if f.exists() else None)
            except ValueError:
                pass
    return out


class Held:
    """--held: two MSAA captures count as equal when identical, or identical outside the view's held
    region and within the held tolerance inside it (tools/kit/tolerance.py HELD)."""

    def __init__(self, fs, v):
        self.fs, self.v, self.region, self.crops, self.seen = fs, v, False, None, []

    def equal(self, p, q):
        fs = self.fs
        if fs.same(p, q):
            return True
        if not a.held:
            return False
        A, B = fs.load(p), fs.load(q)
        if A.shape != B.shape:
            return False
        if self.region is False:
            self.region, self.crops = tolerance.held_region(helds(self.v), A.shape), view_crops(self.v)
        if self.region is None:
            return False
        m = tolerance.measure(A, B, self.crops, tolerance.HELD, self.region)
        ok, use, why = tolerance.judge(m, tolerance.HELD)
        self.seen.append({"files": [str(p), str(q)], "pass": ok, "use": use, "why": why, "outside": m["outside"], "max": m["frame"]["max"], "off": m["frame"]["off"]})
        return ok


def aa_pair(held, d, v):
    fa, fb = d / f"{v}.png", d / f"{v}.repeat.png"
    if not fa.exists() or not fb.exists():
        return None
    if held.fs.same(fa, fb):
        return {"identical": True, "equal": True, "max": 0, "over": {str(l): 0 for l in LEVELS}}
    _, _, st = stats(held.fs.load(fa), held.fs.load(fb))
    st["identical"] = False
    st["equal"] = held.equal(fa, fb)
    return st


def fmt(st):
    if st is None:
        return "no repeat"
    if st.get("identical"):
        return "identical"
    return f"max {st['max']}, {st['over']['1']} px>1, {st['over']['2']} px>2" + (" (held)" if st.get("equal") else "")


def temporal(fs, v):
    """The tolerance on the best pairing of the two sides' captures (one recapture before a failure counts)."""
    same = ref.resolve() == test.resolve()
    sides = [("", ".repeat")] if same else [(s, t) for s in ("", ".repeat") for t in ("", ".repeat")]
    crops = view_crops(v)
    best = None
    for s, t in sides:
        p, q = ref / f"{v}{s}.png", test / f"{v}{t}.png"
        if not (p.exists() and q.exists()):
            continue
        if fs.same(p, q):
            ok, use, why, m = True, 0.0, [], None
        else:
            m = tolerance.measure(fs.load(p), fs.load(q), crops)
            ok, use, why = tolerance.judge(m)
        e = {"pair": ("repeat" if s else "png") + " / " + ("repeat" if t else "png"), "files": [str(p), str(q)], "pass": ok, "use": use, "why": why, "measure": m}
        if best is None or (ok, -use) > (best["pass"], -best["use"]):
            best = e
        if ok and use == 0:
            break
    if best is None:
        return {"pass": False, "use": None, "why": ["no repeat capture" if same else "missing"], "pair": ""}
    if not best["pass"] and not (test / f"{v}.repeat.png").exists():
        best["why"].append("unconfirmed: recapture the view (or capture with --repeat) before counting it")
    return best


def browser(d, v):
    f = d / f"{v}.json"
    try:
        return json.loads(f.read_text()).get("browser") if f.exists() else None
    except ValueError:
        return None


def compare(v):
    """One view: (printed lines, report entry, table row, passed)."""
    fa, fb = ref / f"{v}.png", test / f"{v}.png"
    if not fb.exists() or not fa.exists():
        return [f"{v:24} missing"], {"missing": True}, (v, "missing", "", "", "FAIL"), False
    fs = Files()
    ident = fs.same(fa, fb)
    A, B = fs.load(fa), fs.load(fb)
    if A.shape != B.shape:
        return [f"{v:24} size {A.shape} vs {B.shape}"], {"sizeMismatch": [A.shape, B.shape]}, (v, "size mismatch", "", "", "FAIL"), False
    D, M, r = stats(A, B)
    r["identical"] = ident
    held = Held(fs, v)
    if a.aa_floor:
        ra, ta = aa_pair(held, ref, v), aa_pair(held, test, v)
        r["refAA"], r["testAA"] = ra, ta
        # the cross difference: the closest of the four pairings of the two sides' captures
        cross = r
        pairs = [(ref / f"{v}{s}.png", test / f"{v}{t}.png") for s in ("", ".repeat") for t in ("", ".repeat")]
        if any(p.exists() and q.exists() and fs.same(p, q) for p, q in pairs):
            cross, note = {"identical": True, "max": 0, "over": {str(l): 0 for l in LEVELS}}, "identical"
        elif any(p.exists() and q.exists() and held.equal(p, q) for p, q in pairs):
            cross, note = {"identical": True, "held": True, "max": 0, "over": {str(l): 0 for l in LEVELS}}, "within the held tolerance"
        elif ra is None or ta is None:
            note, ok = "a side has no repeat capture", False
        else:
            for p, q in pairs[1:]:
                if p.exists() and q.exists():
                    _, _, st = stats(fs.load(p), fs.load(q))
                    if (st["max"], st["over"]["1"]) < (cross["max"], cross["over"]["1"]):
                        cross = st
        if cross.get("identical"):
            ok = not (ra and ra["equal"] and ta and not ta["equal"])
            note = ("within the held tolerance" if cross.get("held") else "identical") if ok else "test is nondeterministic where the reference is deterministic"
        elif ra is not None and ta is not None:
            if ra["equal"]:
                ok, note = False, "differs where the reference is deterministic"
            else:
                floor = {k: max(ra[k] if k == "max" else ra["over"][k], ta[k] if k == "max" else ta["over"][k]) for k in ("max", "1", "2")}
                ok = cross["max"] <= floor["max"] and cross["over"]["1"] <= floor["1"] and cross["over"]["2"] <= floor["2"]
                note = "within the A/A floor" if ok else f"over the A/A floor (max {floor['max']}, {floor['1']} px>1, {floor['2']} px>2)"
                if ta["equal"]:
                    note += "; test is deterministic, the reference is not"
        r["cross"] = {k: cross[k] for k in ("max", "over") if k in cross} | {"identical": bool(cross.get("identical"))}
        r["note"] = note
    elif a.temporal:
        t = temporal(fs, v)
        ok = t["pass"]
        if t.get("files") and t["files"] != [str(fa), str(fb)]:
            # the diff images and crops show the pairing that was judged
            A, B = fs.load(t["files"][0]), fs.load(t["files"][1])
            D, M, r = stats(A, B)
            r["identical"] = t["measure"] is None
        r["temporal"] = t
        if a.flip:
            r["flip"] = tolerance.flip_mean(A, B)
    else:
        ok = r["over"]["1"] < a.max_px1 and r["over"]["2"] == 0 if a.max_px1 > 0 else (ident or held.equal(fa, fb))
    if held.seen:
        r["held"] = held.seen
    r["pass"] = bool(ok)
    if r["max"] > 0:
        Image.fromarray(X8[M]).save(out / f"{v}.diff.png")
        worst_windows(v, A, B, D, M, r)
    if a.crops_from:
        named_crops(v, A, B, D, r)
    if a.aa_floor:
        row = (v, fmt(r["refAA"]), fmt(r["testAA"]), fmt(r["cross"]), ("ok" if ok else "FAIL") + f" ({r['note']})")
        lines = [f"{v:24} {row[1]:28} {row[2]:28} {row[3]:28} {row[4]}"]
    elif a.temporal:
        t, m = r["temporal"], r["temporal"].get("measure")
        mx, off = (m["frame"]["max"], f"{m['frame']['off']:.4%}") if m else (0, f"{0:.4%}")
        verdict = ("ok" if ok else "FAIL") + (f" ({'; '.join(t['why'])})" if t["why"] else "")
        if a.flip and r.get("flip") is not None:
            verdict += f" FLIP {r['flip']:.6f}"
        row = (v, t["pair"], str(mx), off, f"{t['use']:.3f}" if t["use"] is not None else "", verdict)
        lines = [f"{v:24} {t['pair']:15} {mx:4} {off:>8} {row[4]:>6}  {verdict}"]
    else:
        verdict = ("ok" + (" (held)" if ok and not ident and a.max_px1 == 0 else "")) if ok else "FAIL"
        row = (v, "yes" if ident else "no", str(r["max"]), f"{r['mean']:.4f}", " ".join(str(r["over"][str(l)]) for l in LEVELS), verdict)
        lines = [f"{v:24} {('yes' if ident else 'no'):5} {r['max']:4} {r['mean']:8.4f} " + " ".join(f"{r['over'][str(l)]:9}" for l in LEVELS) + "  " + verdict]
    if a.crops_from:
        for name, c in r["anchors"].items():
            lines.append(f"    {name:20} " + (f"missing ({c['missing']})" if "missing" in c else f"max {c['max']:3}  >1 {c['over']['1']:6}  >2 {c['over']['2']:6}  {c['file']}"))
    return lines, r, row, bool(ok)


# frames from two browser builds never match pixel for pixel: say so before the table
seen = {(browser(ref, v), browser(test, v)) for v in views}
mixed = sorted({f"{r} vs {t}" for r, t in seen if r and t and r != t})
if mixed:
    print("WARNING: the sets come from different browser builds (" + "; ".join(mixed) + ")")

report, failed, rows = {}, False, []
if a.aa_floor:
    print(f"{'view':24} {'ref A/A':28} {'test A/A':28} {'cross':28} pass")
elif a.temporal:
    print(f"{'view':24} {'pairing':15} {'max':>4} {'px>' + str(tolerance.TEMPORAL['level']):>8} {'use':>6}  verdict")
else:
    print(f"{'view':24} {'ident':5} {'max':>4} {'mean':>8} " + " ".join(f"{'>' + str(l):>9}" for l in LEVELS) + "  pass")
sys.stdout.flush()
# Views run in threads: numpy and Pillow's PNG codec release the GIL. Each view decodes each of its
# PNGs once and keeps them as uint8; the output is the same as one view at a time, printed in view
# order. Window sums are exact integer sums on a coarse integral image (tolerance.window_sums), and
# every mean is an exact integer sum divided once, so no number depends on the order of summation.
with ThreadPoolExecutor(max_workers=max(1, a.jobs), initializer=raise_thread) as pool:
    # map() yields in view order: the table prints as the views finish, in order
    for v, (lines, r, row, ok) in zip(views, pool.map(compare, views)):
        print("\n".join(lines), flush=True)
        report[v], failed = r, failed or not ok
        rows.append(row)
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
if a.md:
    head = ("view", "ref A/A", "test A/A", "cross", "verdict") if a.aa_floor else \
        ("view", "pairing", "max", f"px >{tolerance.TEMPORAL['level']}", "use", "verdict") if a.temporal else \
        ("view", "identical", "max", "mean", "px >1 >2 >8 >32", "pass")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    Path(a.md).write_text("\n".join(lines) + "\n")
print(f"report: {out / 'report.json'}")
sys.exit(1 if failed else 0)
