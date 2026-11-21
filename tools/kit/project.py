"""The project's configuration (tools/project.mjs) for the kit's Python tools, so that both
languages read one file: node prints it as JSON, once per process, on the first call.

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))   # the kit's folder, if elsewhere
    from project import project
    out = Path(project()["output"]["root"]) / project()["output"]["comparison"]

Standard library only; it needs `node` on the PATH, as every kit tool does.
"""
import functools
import json
import subprocess
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "project.mjs"


@functools.cache
def project():
    """The configuration's default export, as plain JSON values."""
    script = f"import p from {json.dumps(CONFIG.as_uri())}; process.stdout.write(JSON.stringify(p));"
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)
