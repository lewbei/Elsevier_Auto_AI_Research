"""Generate a paper draft from experiment artifacts, with optional LLM drafting.

Default path composes Markdown+LaTeX from artifacts. When enabled via
config/env, an LLM drafts a full Markdown paper using section scaffolds
and a brief reflection pass for clarity/consistency.
"""

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from lab.config import get, get_bool
from utils.llm_utils import chat_text_cached, LLMError


load_dotenv()

DATA_DIR = pathlib.Path("data")
RUNS_DIR = pathlib.Path("runs")
PAPER_DIR = pathlib.Path("paper")


def _ensure_dirs() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: pathlib.Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _best_acc(runs: List[Dict[str, Any]], name_contains: str, prefer_metric: str = "test_accuracy") -> float:
    best = 0.0
    for r in runs:
        if name_contains in str(r.get("name")):
            metrics = r.get("result", {}).get("metrics", {})
            acc = metrics.get(prefer_metric)
            if acc is None:
                acc = metrics.get("val_accuracy", 0.0)
            try:
                acc = float(acc or 0.0)
            except Exception:
                acc = 0.0
            if acc > best:
                best = acc
    return best


def _render_md(title: str, novelty: Dict[str, Any], plan: Dict[str, Any], summary: Dict[str, Any]) -> str:
    # Guard against non-dict inputs; callers may pass loose JSON content
    if not isinstance(novelty, dict):
        novelty = {}
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(summary, dict):
        summary = {}
    runs_val = summary.get("runs") if isinstance(summary, dict) else None
    runs = runs_val if isinstance(runs_val, list) else []
    baseline_acc = _best_acc(runs, "baseline")
    novelty_acc = _best_acc(runs, "novelty")
    ablation_acc = _best_acc(runs, "ablation")
    delta = novelty_acc - baseline_acc
    best = None
    try:
        best = _read_json(RUNS_DIR / "best.json")
    except Exception:
        best = None

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "We investigate a compact novelty for skin-cancer classification using a minimal, reproducible pipeline. "
        "Our experiments compare a baseline, a novelty variant, and an ablation."
    )
    lines.append("")

    # Structured Introduction
    lines.append("## Introduction")
    lines.append("### Executive Introduction (E-Intro)")
    goal_txt = str(get("project.goal", "") or "This work explores a compact, reproducible pipeline for skin-cancer classification under tight compute constraints.")
    lines.append(goal_txt)
    lines.append("")
    lines.append("### Problem Statement")
    lines.append("Automated skin-lesion analysis requires accurate, data-efficient models that are easy to reproduce and extend. We target a minimal baseline plus a small novelty under constrained training budgets.")
    lines.append("")
    lines.append("### Objectives (2)")
    obj = str(plan.get("objective", ""))
    if obj:
        lines.append(f"- O1: {obj}")
    else:
        lines.append("- O1: Establish a minimal, reproducible baseline for skin-cancer classification.")
    lines.append("- O2: Evaluate a lightweight novelty against the baseline under a small budget.")
    lines.append("")
    lines.append("### Research Question (1)")
    rq = "Can a small, well-scoped novelty outperform a minimal baseline on skin-cancer classification under limited compute?"
    lines.append(f"- RQ1: {rq}")
    lines.append("")
    lines.append("### Contributions (2)")
    lines.append("- C1: A lean, end-to-end research pipeline (planning → execution → reporting) with guardrails.")
    lines.append("- C2: A lightweight novelty and evaluation protocol suitable for tight budgets.")
    lines.append("")

    # Literature Review
    lines.append("## Literature Review")
    ths = novelty.get("themes") or []
    if ths:
        lines.append("We summarize recent themes and position our approach accordingly:")
        for t in ths[:5]:
            name = t.get("name") or "Theme"
            summary = t.get("summary") or ""
            lines.append(f"- {name}: {summary}")
    else:
        lines.append("Prior work spans strong baselines (e.g., ResNet variants) and modest augmentations; our focus is a minimal, reproducible path with a compact novelty.")
    lines.append("")

    # Methodology with equations
    lines.append("## Methodology")
    lines.append("We train a compact CNN backbone with a standard classification head. Below we outline the objective functions used.")
    # Try to infer task/loss from best run spec
    task_kind = None
    loss_name = None
    if best and isinstance(best, dict):
        try:
            spec_b = best.get("spec", {}) or {}
            task_kind = str(spec_b.get("task") or "").strip().lower() or None
            loss_name = str(spec_b.get("loss") or "").strip().lower() or None
        except Exception:
            task_kind = None
            loss_name = None
    # Display equations (Markdown + LaTeX)
    if task_kind == "regression":
        lines.append("### Regression Loss")
        if loss_name in {"l1", "mae"}:
            lines.append("We use the mean absolute error (MAE):")
            lines.append("$$\\mathcal{L}_{\\text{MAE}} = \\frac{1}{N} \\sum_{i=1}^{N} \\lVert y_i - \\hat{y}_i \\rVert_1.$$")
        elif loss_name in {"huber", "smooth_l1"}:
            lines.append("We use the Huber (Smooth L1) loss with threshold \\delta:")
            lines.append("$$\\mathcal{L}_{\\delta}(y, \\hat{y}) = \\begin{cases} \\tfrac{1}{2}(y-\\hat{y})^2 & \\text{if } |y-\\hat{y}| \\le \\delta \\cr \\delta |y-\\hat{y}| - \\tfrac{1}{2} \\delta^2 & \\text{otherwise.} \\end{cases}$$")
        else:
            lines.append("We use mean squared error (MSE):")
            lines.append("$$\\mathcal{L}_{\\text{MSE}} = \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2.$$")
    else:
        lines.append("### Classification Loss")
        if loss_name in {"bce", "bcewithlogits"}:
            lines.append("For multi-label settings, we may use binary cross-entropy with logits:")
            lines.append("$$\\mathcal{L}_{\\text{BCE}} = - \\frac{1}{N} \\sum_{i=1}^{N} \\Big[y_i \\log(\\sigma(z_i)) + (1-y_i) \\log(1-\\sigma(z_i))\\Big].$$")
        else:
            lines.append("We use the standard (multi-class) cross-entropy:")
            lines.append("$$\\mathcal{L}_{\\text{CE}} = - \\frac{1}{N} \\sum_{i=1}^{N} \\log \\frac{e^{z_{i,y_i}}}{\\sum_{c} e^{z_{i,c}}}.$$")
    lines.append("")

    lines.append("## Experiments")
    lines.append("We run small, CPU-friendly experiments with <=1 epoch and <=100 steps per run.")
    lines.append("We evaluate baseline, novelty, and ablation, plus minor variants.")
    lines.append("")
    lines.append("### Results")
    lines.append("| Setting | Acc (test) |\n|---|---:|")
    lines.append(f"| Baseline | {baseline_acc:.4f} |")
    lines.append(f"| Novelty | {novelty_acc:.4f} |")
    lines.append(f"| Ablation | {ablation_acc:.4f} |")
    lines.append("")
    lines.append(f"Delta vs baseline: {delta:+.4f}")
    lines.append("")
    if best:
        lines.append("Best run (across matrix):")
        lines.append(f"- Name: {best.get('name')} | Acc: {best.get('result',{}).get('metrics',{}).get('val_accuracy',0.0):.4f}")
        spec = best.get("spec", {})
        keep = ["model", "input_size", "batch_size", "epochs", "lr", "max_train_steps", "seed"]
        parts = [f"{k}={spec.get(k)}" for k in keep if k in spec]
        lines.append("- Spec: " + ", ".join(parts))
        lines.append("")

    lines.append("## Discussion")
    lines.append("The novelty shows modest differences under a small compute budget. Further work includes broader datasets and more rigorous sweeps.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("Our runs are short and primarily CPU-bound; results are indicative rather than definitive.")
    lines.append("")

    # Pipeline decision and environment info
    if isinstance(summary.get("goal_reached") if isinstance(summary, dict) else None, bool):
        lines.append(f"Decision: goal_reached = {summary.get('goal_reached')}")
        lines.append("")
    try:
        env = _read_json(RUNS_DIR / "env.json")
        if env:
            lines.append("Environment:")
            py = env.get("python", "")
            exe = env.get("executable", "")
            plat = env.get("platform", "")
            lines.append(f"- Python: {py}")
            lines.append(f"- Executable: {exe}")
            lines.append(f"- Platform: {plat}")
            lines.append("")
    except Exception:
        pass

    # Embed accuracy chart if present
    img_path = RUNS_DIR / "accuracy.png"
    if img_path.exists():
        lines.append("## Accuracy Chart")
        lines.append("![Validation Accuracy](../runs/accuracy.png)")
        lines.append("")

    lines.append("## Conclusion")
    lines.append("We present a lean research pipeline with planning, execution, and reporting. It supports quick iteration and extensions.")
    lines.append("")

    # Future Work (explicit section)
    lines.append("## Future Work")
    lines.append("We outline next steps consistent with the current scope and constraints:")
    lines.append("- Scale experiments with longer training budgets and larger datasets when resources permit.")
    lines.append("- Evaluate additional lightweight novelties (e.g., alternative heads, mild augmentations) under the same budget.")
    lines.append("- Extend tasks (e.g., lesion segmentation or detection) while maintaining reproducibility and guardrails.")
    lines.append("")

    return "\n".join(lines)


def _cap(s: str, n: int) -> str:
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 3] + "..."


SECTION_TIPS = {
    "Title": [
        "Concise and informative; hint at method/setting/outcome",
        "Avoid hype; under 2 lines"
    ],
    "Abstract": [
        "Single paragraph TL;DR: problem, approach, results, takeaway",
        "No hyperbole; numbers only if provided in inputs"
    ],
    "Introduction": [
        "Context and motivation; why it matters",
        "Contributions as bullets; scope under tight compute"
    ],
    "Related Work": [
        "Compare/contrast closest work; cite when relevant",
        "Explain why baselines are appropriate"
    ],
    "Methods": [
        "Describe model/augmentations/training succinctly",
        "Keep to primitives; avoid missing details"
    ],
    "Experimental Setup": [
        "Datasets/paths/splits; metrics and evaluation",
        "Training budget: <=1 epoch; steps and seeds"
    ],
    "Results": [
        "Report numbers exactly as given; no invention",
        "Compare baseline/novelty/ablation; call out deltas"
    ],
    "Discussion": [
        "Interpret results under constraints; note trade-offs",
        "State limitations and future work"
    ],
}


def _llm_paper_md(novelty: Dict[str, Any], plan: Dict[str, Any], summary: Dict[str, Any], lit_review: Optional[str]) -> str:
    title = str(get("project.title", "") or "Draft: Compact Research Report")
    goal = str(get("project.goal", "your goal") or "your goal")
    novelty_small = {
        "themes": (novelty.get("themes") or [])[:6],
        "new_ideas": (novelty.get("new_ideas") or [])[:8],
        "unique_ideas": (novelty.get("unique_ideas") or [])[:8],
    }
    plan_small = {k: plan.get(k) for k in [
        "objective", "hypotheses", "success_criteria", "datasets", "baselines",
        "novelty_focus", "stopping_rules"
    ] if k in plan}
    runs = summary.get("runs", []) if isinstance(summary.get("runs"), list) else []
    compact_runs: List[Dict[str, Any]] = []
    for r in runs[:50]:
        compact_runs.append({
            "name": r.get("name"),
            "metrics": r.get("result", {}).get("metrics", {}),
        })
    system = (
        "You are an expert research writer drafting a Markdown paper under tight compute constraints.\n"
        f"Project goal: {goal}.\n"
        "Return ONLY Markdown (no code fences); use LaTeX math ($$...$$) for equations.\n"
        "Include sections in this order with required structure: \n"
        "1) Title\n"
        "2) Abstract\n"
        "3) Introduction with subsections: Executive Introduction, Problem Statement, Objectives (exactly 2), Research Question (exactly 1), Contributions (exactly 2)\n"
        "4) Literature Review\n"
        "5) Methodology (show the training objective equations explicitly; choose CE/BCE/MSE/Huber as appropriate)\n"
        "6) Experimental Setup\n"
        "7) Results\n"
        "8) Discussion\n"
        "9) Conclusion\n"
        "10) Future Work\n"
        "11) Limitations\n"
        "12) References (optional)\n"
        "Grounding: use ONLY provided data; do NOT invent numbers, citations, or URLs. If a value is missing, write 'TBD'.\n"
        "Constraints: experiments are <=1 epoch with small steps; keep claims modest and reproducible.\n"
        "Style: objective, concise paragraphs + short bullet lists where helpful."
    )
    tips_lines = []
    for sec, tips in SECTION_TIPS.items():
        tips_lines.append(f"- {sec} tips: " + "; ".join(tips))
    user = {
        "title": title,
        "tips": tips_lines,
        "novelty": novelty_small,
        "plan": plan_small,
        "runs": compact_runs,
        "lit_review": _cap(lit_review or "", 18000),
        "require": [
            "Report numbers exactly as given (val/test metrics only if present)",
            "Cite only if citation strings are provided (else omit)",
            "Keep each section substantial but not verbose"
        ],
        "self_check": "Before responding, scan for invented numbers/claims and remove them; ensure headings present; return Markdown only.",
    }
    text = chat_text_cached([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ], temperature=0.2)
    draft = text.strip()

    reflect_system = (
        "You are a careful reviewer. Revise the draft Markdown to improve clarity, fix inconsistencies, and remove any invented content.\n"
        "Maintain the same sections. Return FULL Markdown (no code fences)."
    )
    reflect_user = {
        "draft": _cap(draft, 40000),
        "checks": [
            "Numbers appear only where provided in runs/summary",
            "Methods/setup match constraints and plan",
            "No missing sections; remove hype; keep objective tone"
        ],
        "self_check": "Before responding, ensure the output is complete Markdown, not partial, and contains no fenced code.",
    }
    try:
        revised = chat_text_cached([
            {"role": "system", "content": reflect_system},
            {"role": "user", "content": json.dumps(reflect_user, ensure_ascii=False)},
        ], temperature=0.0)
        md = revised.strip()
        if md:
            return md
    except LLMError:
        pass
    return draft


def _render_latex(title: str, md_path: pathlib.Path, latex_table: str | None = None, include_fig: bool = False, nocite_all: bool = False) -> str:
    nocite_cmd = "\\nocite{*}" if nocite_all else ""
    return f"""\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{lmodern}}
\\usepackage{{graphicx}}
\\usepackage[numbers]{{natbib}}
\\title{{{title}}}
\\author{{Auto Writer}}
\\date{{}}
\\begin{{document}}
\\maketitle
\\section*{{Draft}}
This is an auto-generated draft. Refer to {md_path.name} for Markdown version.
{('\\section*{Results}\n' + latex_table) if latex_table else ''}
    {('\\begin{figure}[h!]\\centering\\includegraphics[width=\\linewidth]{../runs/accuracy.png}\\caption{Validation Accuracy}\\end{figure}' if include_fig else '')}
    {nocite_cmd}
    \\bibliographystyle{{plainnat}}
    \\bibliography{{refs}}
    \\end{{document}}
    """


def _sanitize_bib_key(s: str) -> str:
    keep = "".join(c for c in s if c.isalnum())
    return keep[:40] or "ref"


def build_bibtex_from_csv(csv_path: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path | None:
    if not csv_path.exists():
        return None
    import csv
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
    except Exception:
        return None
    if not rows:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    bib_path = out_dir / "refs.bib"
    lines: List[str] = []
    for i, row in enumerate(rows, start=1):
        title = str(row.get("title") or f"Untitled-{i}")
        year = str(row.get("year") or "")
        doi = str(row.get("doi") or "")
        key = _sanitize_bib_key((title or "") + year)
        lines.append(f"@article{{{key},")
        lines.append(f"  title={{ {title} }},")
        if year:
            lines.append(f"  year={{ {year} }},")
        if doi:
            lines.append(f"  doi={{ {doi} }},")
        lines.append("}")
        lines.append("")
    bib_path.write_text("\n".join(lines), encoding="utf-8")
    return bib_path


def _aggregate_from_summary(summary: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    import math
    runs = summary.get("runs", []) if isinstance(summary.get("runs"), list) else []
    groups = {"baseline": [], "novelty": []}
    for r in runs:
        name = str(r.get("name") or "")
        base = name.split("_rep", 1)[0]
        if base in groups:
            try:
                acc = float(r.get("result", {}).get("metrics", {}).get("val_accuracy", 0.0) or 0.0)
                groups[base].append(acc)
            except Exception:
                pass
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in groups.items():
        if not vals:
            continue
        n = float(len(vals))
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = math.sqrt(var)
        out[k] = {"mean": mean, "std": std, "n": n}
    return out


def render_mean_std_table_md(agg: Dict[str, Dict[str, float]]) -> List[str]:
    lines = ["| Setting | N | Mean | Std |", "|---|---:|---:|---:|"]
    for k in ("baseline", "novelty"):
        if k in agg:
            a = agg[k]
            lines.append(f"| {k} | {int(a['n'])} | {a['mean']:.4f} | {a['std']:.4f} |")
    return lines


def render_mean_std_table_tex(agg: Dict[str, Dict[str, float]]) -> str:
    rows = []
    for k in ("baseline", "novelty"):
        if k in agg:
            a = agg[k]
            rows.append(f"{k} & {int(a['n'])} & {a['mean']:.4f} & {a['std']:.4f} \\ ")
    if not rows:
        return ""
    return (
        "\\begin{table}[h!]\\centering\\begin{tabular}{lrrr}\\hline Setting & N & Mean & Std \\ \\hline "
        + " ".join(rows)
        + " \\hline\\end{tabular}\\caption{Mean and standard deviation of validation accuracy.}\\end{table}"
    )


def try_pdflatex(tex_path: pathlib.Path) -> None:
    exe = shutil.which("pdflatex")
    if not exe:
        return
    try:
        subprocess.run([exe, "-interaction=nonstopmode", tex_path.name], cwd=str(tex_path.parent), check=True, timeout=120)
    except Exception:
        pass


def main() -> None:
    _ensure_dirs()
    novelty = _read_json(DATA_DIR / "novelty_report.json")
    plan = _read_json(DATA_DIR / "plan.json")
    summary = _read_json(RUNS_DIR / "summary.json")
    # Robustness: guard against files that contain valid JSON but not an object (e.g., a string)
    if not isinstance(novelty, dict):
        novelty = {}
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(summary, dict):
        summary = {}

    project_title = str(get("project.title", "") or "").strip()
    project_topic = str(get("project.topic", "Skin-Cancer Classification") or "Skin-Cancer Classification")
    title = project_title or f"A Minimal Novelty for {project_topic}"
    if novelty.get("new_ideas"):
        idea = str(novelty["new_ideas"][0])[:60]
        if idea:
            title = f"Toward: {idea}"

    # Optional LLM drafting mode
    use_llm = (
        get_bool("pipeline.write_paper.llm.enable", False)
        or get_bool("pipeline.write_paper_llm.enable", False)
        or (
        str(os.getenv("WRITE_PAPER_LLM", "")).lower() in {"1", "true", "yes"}
        )
    )
    lit_review: Optional[str] = None
    try:
        lp = DATA_DIR / "lit_review.md"
        if lp.exists():
            lit_review = lp.read_text(encoding="utf-8")
    except Exception:
        lit_review = None

    if use_llm:
        try:
            md = _llm_paper_md(novelty, plan, summary, lit_review)
        except LLMError as exc:
            print(f"[WARN] LLM writer failed: {exc}; falling back to artifact composer.")
            md = _render_md(title, novelty, plan, summary)
    else:
        md = _render_md(title, novelty, plan, summary)
    md_path = PAPER_DIR / "paper.md"
    md_path.write_text(md, encoding="utf-8")

    agg = _aggregate_from_summary(summary)
    latex_table = render_mean_std_table_tex(agg) if agg else None
    tex_path = PAPER_DIR / "main.tex"
    img_path = RUNS_DIR / "accuracy.png"
    # Try to build refs.bib; then include all entries via \nocite{*} if present
    csv_default = pathlib.Path("abstract_screen_deepseek.csv")
    try:
        bib_path = build_bibtex_from_csv(csv_default, PAPER_DIR)
    except Exception:
        bib_path = None
    nocite_all = bool(bib_path and pathlib.Path(bib_path).exists())
    tex_path.write_text(
        _render_latex(title, md_path, latex_table=latex_table, include_fig=img_path.exists(), nocite_all=nocite_all),
        encoding="utf-8",
    )

    try_pdflatex(tex_path)
    print(f"[DONE] Wrote paper draft to {md_path} and {tex_path}")


if __name__ == "__main__":
    main()
