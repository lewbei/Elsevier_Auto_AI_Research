#!/usr/bin/env bash
set -Eeuo pipefail

OUTDIR="artifacts/code_snapshot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

: "${MAX_BYTES:=0}"
: "${CHUNK_BYTES:=600000}"

git ls-files -co --exclude-standard > "$OUTDIR/all_files.txt"

awk 'BEGIN{IGNORECASE=1}
  !/^(.git\/|node_modules\/|vendor\/|dist\/|build\/|\.next\/|\.cache\/|__pycache__\/|\.mypy_cache\/|\.pytest_cache\/|coverage\/|\.idea\/|\.vscode\/)/ {
    print
  }' "$OUTDIR/all_files.txt" \
| grep -E '\\.(py|ipynb|ts|tsx|js|jsx|json|md|yml|yaml|toml|ini|cfg|conf|sh|bash|bat|ps1|html|css|scss|c|cc|cpp|h|hpp|cu|rs|go)$' \
> "$OUTDIR/code_files.txt"

if [ "$MAX_BYTES" -gt 0 ]; then
  tmp="$OUTDIR/code_files.sz.txt"
  : > "$tmp"
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    size=$(wc -c <"$f" | tr -dc '0-9')
    if [ "${size:-0}" -le "$MAX_BYTES" ]; then
      printf '%s\n' "$f" >> "$tmp"
    fi
  done < "$OUTDIR/code_files.txt"
  mv "$tmp" "$OUTDIR/code_files.txt"
fi

lang() {
  f="$1"; ext="${f##*.}"
  case "$ext" in
    py) echo python;; ipynb) echo json;; ts) echo ts;; tsx) echo tsx;; js) echo javascript;; jsx) echo jsx;;
    sh|bash) echo bash;; ps1) echo powershell;; bat) echo bat;; yml|yaml) echo yaml;; toml) echo toml;; ini|cfg|conf) echo '';;
    html) echo html;; css|scss) echo css;; c) echo c;; cc|cpp) echo cpp;; h|hpp) echo c;; cu) echo cuda;; rs) echo rust;; go) echo go;;
    json) echo json;; md) echo markdown;; *) echo '';;
  esac
}

chunk=1; cur=0
outfile="$OUTDIR/chunk_$(printf '%04d' "$chunk").md"
echo "# code snapshot chunk $chunk" > "$outfile"

while IFS= read -r f; do
  [ -f "$f" ] || continue
  size=$(wc -c <"$f" | tr -dc '0-9')
  if [ $((cur + size)) -ge "$CHUNK_BYTES" ] && [ "$cur" -gt 0 ]; then
    chunk=$((chunk+1)); cur=0
    outfile="$OUTDIR/chunk_$(printf '%04d' "$chunk").md"
    echo "# code snapshot chunk $chunk" > "$outfile"
  fi
  l=$(lang "$f")
  {
    echo
    echo "<file path=\"$f\">"
    echo '```'"$l"
    cat "$f"
    echo '```'
    echo "</file>"
  } >> "$outfile"
  cur=$((cur + size))
done < "$OUTDIR/code_files.txt"

ls -1 "$OUTDIR"/chunk_*.md > "$OUTDIR/index.txt"
chunks=$(wc -l < "$OUTDIR/index.txt" | tr -dc '0-9')
echo "wrote $chunks chunks to $OUTDIR"

