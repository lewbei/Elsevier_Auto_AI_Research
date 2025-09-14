#!/usr/bin/env bash
set -Eeuo pipefail

# hard-pin interpreter (Windows path)
export PY_WIN='C:\Users\lewka\miniconda3\envs\deep_learning\python.exe'

# wrappers: always go through CMD with the Windows path
py()      { cmd.exe /C "\"$PY_WIN\" $*"; }
pipw()    { cmd.exe /C "\"$PY_WIN\" -m pip $*"; }
pytestw() { cmd.exe /C "\"$PY_WIN\" -m pytest $*"; }

# compile all tracked .py files
pycompile() {
  files=$(git ls-files "*.py"); if [ -n "$files" ]; then py -m py_compile $files; fi
}

# progress helpers
progress_update() {
  pct="$1"; shift; msg="$*"
  mkdir -p artifacts
  ts=$(date -Ins)
  esc_msg=$(printf '%s' "$msg" | sed 's/"/\\"/g')
  printf '{"percent":%s,"message":"%s","ts":"%s"}\n' "$pct" "$esc_msg" "$ts" > artifacts/progress.json
  printf '%s  %3s%%  %s\n' "$ts" "$pct" "$msg" >> artifacts/progress.md
}

progress_phase() { # convenience wrapper
  phase="$1"; pct="$2"; shift 2; note="$*"
  progress_update "$pct" "$phase: $note"
}

# no-fallback run harness
export NO_FALLBACK=1
run_no_fallback() {
  [ "${NO_FALLBACK:-1}" = "1" ] || { echo "[error] NO_FALLBACK must be 1"; exit 1; }
  mkdir -p artifacts
  git ls-files > artifacts/filelist.txt

  progress_phase map 5 "repo files indexed"

  # gpu policy
  if [ -z "${ALLOW_CPU_FOR_TESTS:-}" ]; then
    if ! py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
      progress_phase compile 0 "no fallback: CUDA/GPU required"; echo "no fallback: CUDA/GPU required"; exit 1
    fi
  fi

  progress_phase plan 15 "plan.md reviewed"
  pycompile && progress_phase compile 60 "compiled"

  if pytestw -q --maxfail=1 --disable-warnings -W error --strict-markers --strict-config \
       --cov=. --cov-report=xml:artifacts/coverage.xml --cov-report=term-missing --cov-branch \
       --junitxml=artifacts\\test-results.xml; then
    progress_phase unit 90 "tests green"
    progress_phase done 100 "done"
  else
    progress_phase unit 70 "tests failed"
    exit 1
  fi
}

