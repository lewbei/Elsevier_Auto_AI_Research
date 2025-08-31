from pathlib import Path
from typing import List, Dict, Any


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Experiment Dashboard</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 6px 8px; }
    th { background: #f5f5f5; }
    .ok { color: green; }
    .stub { color: #999; }
    .bar { background: #eee; height: 10px; position: relative; }
    .fill { background: #4caf50; height: 10px; }
  </style>
  </head>
  <body>
    <h1>Experiment Dashboard</h1>
    <p>Total runs: {{TOTAL}}</p>
    <table>
      <thead>
        <tr>
          <th>Iter</th><th>Name</th><th>Acc</th><th>Bar</th><th>Mode</th><th>Run ID</th>
        </tr>
      </thead>
      <tbody>
      {{ROWS}}
      </tbody>
    </table>
  </body>
</html>
"""


def render_dashboard(runs: List[Dict[str, Any]]) -> str:
    def row(r: Dict[str, Any]) -> str:
        acc = r.get("result", {}).get("metrics", {}).get("val_accuracy", 0.0)
        mode = r.get("result", {}).get("mode", "stub")
        cls = "ok" if mode == "real" else "stub"
        width = max(0, min(100, int(acc * 100)))
        bar = f"<div class='bar'><div class='fill' style='width:{width}%'></div></div>"
        return (
            f"<tr><td>{r.get('iter')}</td><td>{r.get('name')}</td>"
            f"<td>{acc:.4f}</td><td>{bar}</td><td class='{cls}'>{mode}</td><td>{r.get('run_id')}</td></tr>"
        )

    rows = "\n".join(row(r) for r in runs)
    return TEMPLATE.replace("{{TOTAL}}", str(len(runs))).replace("{{ROWS}}", rows)


def write_dashboard(path: Path, runs: List[Dict[str, Any]]) -> None:
    html = render_dashboard(runs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
