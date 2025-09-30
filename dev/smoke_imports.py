import importlib, sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mods = [
    'agents.orchestrator',
    'agents.summarize',
    'agents.novelty',
    'agents.planner',
    'agents.iterate',
    'lab.experiment_runner',
    'utils.llm_utils',
    'utils.embeddings',
]

ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        has_entry = any(hasattr(mod, k) for k in ('main','iterate','process_pdfs','run_experiment'))
        print(f"{m} OK entry={has_entry}")
    except Exception as e:
        print(f"{m} ERR {type(e).__name__} {e}")
        ok = False

sys.exit(0 if ok else 1)
