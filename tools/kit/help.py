"""Answers `--help` for the Python tool being run from that tool's own module docstring.

The Python tools' twin of tools/kit/help.mjs, by the same rules: the docstring is the tool's
manual, a section opens with an unindented line of one to four words and a colon and ends before
the next blank line, and only the arguments before the first `--` count. An entry tool calls it
right after its docstring, before it imports a third-party package or re-executes itself:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the kit's folder, if elsewhere
    from help import answer_help
    answer_help(__file__)

What it prints:
    uv run <tool> --help            the docstring's first line, its Usage and Options sections
    uv run <tool> --help --full     the whole docstring
Both exit 0. Without `--help` the call does nothing. Standard library only, so the answer needs
none of the tool's dependencies.
"""
import ast
import re
import sys
from pathlib import Path

HEADING = re.compile(r"^[A-Z][A-Za-z0-9]*(?: [A-Za-z0-9]+){0,3}:(?: |$)")
ROOT = Path(__file__).resolve().parents[2]


def header_of(text):
    """The module docstring of a Python file's text, as lines."""
    doc = ast.get_docstring(ast.parse(text), clean=True)
    return doc.split("\n") if doc else []


def section_of(header, name):
    """The section that opens with `name:`: from its heading to the next blank line or heading."""
    for start, line in enumerate(header):
        if line.startswith(name + ":") and HEADING.match(line):
            end = start + 1
            while end < len(header) and header[end].strip() and not HEADING.match(header[end]):
                end += 1
            return header[start:end]
    return []


def manual_of(header, file, full=False):
    """What `--help` prints for a header (`--help --full`: the whole header)."""
    if full:
        return "\n".join(header) + "\n"
    rel = Path(file).resolve().relative_to(ROOT)
    parts = [header[:1], section_of(header, "Usage"), section_of(header, "Options"),
             [f"The whole manual: uv run {rel} --help --full"]]
    return "\n\n".join("\n".join(p) for p in parts if p) + "\n"


def answer_help(file):
    """Print the manual of `file` and exit 0 when `--help` is among the arguments before `--`."""
    args = sys.argv[1:]
    own = args[:args.index("--")] if "--" in args else args
    if "--help" in own:
        sys.stdout.write(manual_of(header_of(Path(file).read_text(encoding="utf-8")), file, "--full" in own))
        sys.stdout.flush()
        sys.exit(0)
