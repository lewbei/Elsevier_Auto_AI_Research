import sys
import compileall

ok = compileall.compile_dir('.', force=True, quiet=1)
sys.exit(0 if ok else 1)

