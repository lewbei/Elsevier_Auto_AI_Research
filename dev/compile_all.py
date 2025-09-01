import sys
import compileall
from pathlib import Path

# Compile only first-party code to avoid third-party example trees
ROOTS = [
    Path("agents"),
    Path("lab"),
    Path("utils"),
    Path("run_pipeline.py"),
]

ok = True
for p in ROOTS:
    if p.is_dir():
        ok = compileall.compile_dir(str(p), force=True, quiet=1) and ok
    elif p.is_file():
        ok = compileall.compile_file(str(p), force=True, quiet=1) and ok

sys.exit(0 if ok else 1)
