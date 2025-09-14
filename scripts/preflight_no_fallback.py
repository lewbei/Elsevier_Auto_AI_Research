import os, re, sys, pathlib

BAD = [
    r"except\s*:\s*pass",
    r"except\s+Exception\s*:\s*pass?",
    r"os\.getenv\([^\)]*\)\s*or\s*['\"]",
    r"if\s+not\s+torch\.cuda\.is_available\(\).*:.*cpu",
]

violations = []
root = pathlib.Path('.')
for p in root.rglob('*.py'):
    try:
        s = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    for pat in BAD:
        for m in re.finditer(pat, s, flags=re.IGNORECASE | re.DOTALL):
            violations.append((str(p), m.start(), pat))

out = pathlib.Path('artifacts/fallbacks_found.md')
out.parent.mkdir(parents=True, exist_ok=True)

if violations:
    out.write_text('\n'.join(f"{f}:{pos} pattern={pat}" for f, pos, pat in violations))
    print("no fallback: patterns found. see artifacts/fallbacks_found.md")
    sys.exit(1)
else:
    out.write_text("")
    print("preflight ok")
    sys.exit(0)

