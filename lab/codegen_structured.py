"""Structured code generation orchestrator with safety and git hygiene.

High-level flow (programmatic, no CLI):
1. gather_context(): repo tree listing + selected file excerpts
2. generate_plan(objective): ask LLM for strict JSON matching CodegenPlan
3. apply_plan_safely(plan):
   - create git branch (optional)
   - write artifacts to a staging directory run_dir/
   - write patch file instead of mutating repo if auto_commit disabled
   - run tests in sandbox
   - if tests pass and auto_commit enabled: write files to repo, create commit
4. return CodegenResult with structured metadata & logs.

Config keys (all under pipeline.codegen.*):
  structured.enable: bool
  model: str (llm model name)
  temperature: float (determinism knob)
  auto_commit: bool (if false, only patch file is produced)
  open_pr: bool (placeholder, not implemented - would require network)
  branch_prefix: str (default 'codegen')
  sandbox.timeout: int seconds for pytest
  deterministic.seed: int for RNG seeding

Idempotency: a registry JSONL file .codegen_registry.jsonl stores checksum
manifests per objective to avoid duplicating work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import os
import random
import hashlib
import datetime as _dt

from lab.config import get
from lab.logging_utils import vprint, is_verbose
from utils.llm_utils import chat_text_cached, LLMError
from lab.codegen_schema import CodegenPlan, CodegenResult, AppliedFile, TestResult, FileArtifact
from lab.sandbox import run_pytest

REGISTRY_PATH = Path('.codegen_registry.jsonl')
RUNS_DIR = Path('runs') / 'codegen'


def _seed_all(seed: int) -> None:
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed)
    except Exception:
        pass
    random.seed(seed)


def gather_context(max_files: int = 200, max_chars: int = 8000) -> str:
    """Return a compact textual context: repo tree + selected small python files heads.
    Avoids reading huge files; truncates sensibly.
    """
    root = Path('.')
    entries: List[str] = ["# REPO TREE"]
    for p in sorted(root.rglob('*')):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if any(part.startswith('.') for part in rel.parts):
            continue
        if rel.parts[0] in {'.venv', 'mlruns', 'downloads', '__pycache__'}:
            continue
        entries.append(str(rel))
        if len(entries) > max_files:
            break
    heads: List[str] = ["\n# FILE HEADS"]
    for rel in entries[1:50]:  # first 50 files
        p = Path(rel)
        if p.suffix != '.py':
            continue
        try:
            text = p.read_text(encoding='utf-8')[:500]
            heads.append(f"--- {rel} ---\n{text}")
        except Exception:
            pass
    payload = "\n".join(entries + heads)
    return payload[-max_chars:]


def _parse_plan(json_text: str) -> Optional[CodegenPlan]:
    try:
        data = json.loads(json_text)
        return CodegenPlan(**data)
    except Exception:
        return None


def generate_plan(objective: str, extra_instructions: Optional[str] = None) -> Optional[CodegenPlan]:
    ctx = gather_context()
    sys_prompt = (
        "You are a senior software engineer generating a structured code plan.\n"
        "Return ONLY valid JSON matching the CodegenPlan schema: {objective, summary, files:[{path,language,intent,action,content,tests}]}.\n"
        "Rules: always create accompanying pytest tests for new python modules. Use relative paths. Keep changes minimal and focused."
    )
    user_prompt = f"Objective: {objective}\nContext:\n{ctx}\n" + (f"Extra: {extra_instructions}\n" if extra_instructions else "") + "Return JSON only."
    temperature = float(get('pipeline.codegen.temperature', 0.2) or 0.2)
    model = get('pipeline.codegen.model', None)
    profile = get('pipeline.codegen.llm', None)
    try:
        raw = chat_text_cached([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=temperature, model=model, profile=profile)
    except LLMError as exc:
        if is_verbose():
            vprint(f"plan generation error: {exc}")
        return None
    # Extract first JSON object heuristically
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1:
        return None
    plan = _parse_plan(raw[start:end+1])
    return plan


def _load_registry() -> List[dict]:
    if not REGISTRY_PATH.exists():
        return []
    out = []
    for line in REGISTRY_PATH.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _append_registry(entry: dict) -> None:
    with REGISTRY_PATH.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _already_done(plan: CodegenPlan) -> bool:
    manifest = plan.checksum_manifest()
    for row in _load_registry():
        if row.get('manifest') == manifest:
            return True
    return False


def apply_plan_safely(plan: CodegenPlan) -> CodegenResult:
    seed = int(get('pipeline.codegen.deterministic.seed', 1234) or 1234)
    _seed_all(seed)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    run_dir = RUNS_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'plan.json').write_text(plan.model_dump_json(indent=2), encoding='utf-8')
    auto_commit = bool(get('pipeline.codegen.auto_commit', False))
    timeout = int(get('pipeline.codegen.sandbox.timeout', 60) or 60)
    applied: List[AppliedFile] = []

    if _already_done(plan):
        return CodegenResult(plan=plan, applied=[], success=True, message='Plan previously applied (idempotent skip)', run_dir=str(run_dir))

    # Stage files into run_dir/staging only
    staging = run_dir / 'staging'
    staging.mkdir(exist_ok=True)
    for fa in plan.files:
        target_repo_path = Path(fa.path)
        status = 'applied'
        reason = None
        try:
            # Read existing if any for patch generation info
            existing = ''
            if target_repo_path.exists() and fa.action in {'update', 'patch'}:
                existing = target_repo_path.read_text(encoding='utf-8')
            # Always write staged file full content (patch diff is informational)
            staged_file = staging / fa.path
            staged_file.parent.mkdir(parents=True, exist_ok=True)
            if fa.action == 'patch' and existing:
                # Convert patch diff to final content by applying naive override (future: real patch apply)
                # For safety keep content as-is; expectation: LLM provides final desired file body, not diff
                pass
            staged_file.write_text(fa.content, encoding='utf-8')
        except Exception as exc:
            status = 'error'
            reason = str(exc)
        applied.append(AppliedFile(path=fa.path, action=fa.action, status=status, reason=reason, checksum=fa.checksum))

    # Write a patch bundle file for review
    patch_lines: List[str] = []
    for fa in plan.files:
        repo_path = Path(fa.path)
        old = ''
        if repo_path.exists():
            try:
                old = repo_path.read_text(encoding='utf-8')
            except Exception:
                old = ''
        if old != fa.content:
            import difflib
            diff = difflib.unified_diff(old.splitlines(), fa.content.splitlines(), fromfile=f"a/{fa.path}", tofile=f"b/{fa.path}")
            patch_lines.extend(list(diff))
    (run_dir / 'changes.patch').write_text('\n'.join(patch_lines), encoding='utf-8')

    # Copy staged files into repo only if auto_commit enabled & tests pass
    # Run sandbox pytest first (tests referencing staged content by adjusting PYTHONPATH)
    # Collect test file paths from planned artifacts
    test_paths = sorted({f.path for f in plan.files if f.path.startswith('tests/')})
    extra_env = {'PYTHONPATH': str(staging) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    test_out = run_pytest(test_paths or None, timeout=timeout, extra_env=extra_env)
    tr = TestResult(
        passed=test_out['returncode'] == 0,
        total=0,
        failed=0 if test_out['returncode'] == 0 else 1,
        errors=0,
        duration_sec=float(test_out['duration_sec']),
        stdout=test_out['stdout'][:4000],
    )

    success = tr.passed
    if success and auto_commit:
        for fa in plan.files:
            repo_file = Path(fa.path)
            repo_file.parent.mkdir(parents=True, exist_ok=True)
            staged_file = staging / fa.path
            try:
                repo_file.write_text(staged_file.read_text(encoding='utf-8'), encoding='utf-8')
            except Exception as exc:
                success = False
                if is_verbose():
                    vprint(f"write failure {fa.path}: {exc}")
                break
        if success:
            # best-effort git commit if git repo
            if (Path('.git').exists()):
                import subprocess
                try:
                    branch_prefix = get('pipeline.codegen.branch_prefix', 'codegen') or 'codegen'
                    branch_name = f"{branch_prefix}/{ts}"
                    subprocess.run(['git', 'checkout', '-b', branch_name], check=False, capture_output=True)
                    subprocess.run(['git', 'add'] + [fa.path for fa in plan.files], check=False)
                    subprocess.run(['git', 'commit', '-m', f"codegen: {plan.objective}"], check=False)
                except Exception:
                    pass

    if success:
        _append_registry({'objective': plan.objective, 'manifest': plan.checksum_manifest(), 'ts': ts})

    return CodegenResult(
        plan=plan,
        applied=applied,
        test_result=tr,
        success=success,
        message='ok' if success else 'tests failed',
        run_dir=str(run_dir),
    )


def run_structured_codegen(objective: str, extra_instructions: Optional[str] = None) -> Optional[CodegenResult]:
    if not bool(get('pipeline.codegen.structured.enable', False)):
        if is_verbose():
            vprint('structured codegen disabled by config')
        return None
    # Optionally derive objective/extra from plan.json when enabled
    use_plan = bool(get('pipeline.codegen.structured.use_plan_requirements', True))
    if use_plan:
        plan_path = Path('data/plan.json')
        if plan_path.exists():
            try:
                pj = json.loads(plan_path.read_text(encoding='utf-8'))
                # Construct an enriched objective using plan objective + first success criterion
                plan_obj = str(pj.get('objective') or '').strip()
                succ = pj.get('success_criteria') or []
                metric_hint = ''
                if isinstance(succ, list) and succ:
                    first = succ[0]
                    if isinstance(first, dict):
                        m = first.get('metric')
                        dv = first.get('delta_vs_baseline')
                        if m:
                            metric_hint = f" target metric {m} (delta {dv})" if dv is not None else f" target metric {m}"
                if plan_obj and not objective:
                    objective = plan_obj[:200]
                # Build extra instructions summarizing datasets, tasks, hypotheses for guidance
                if extra_instructions is None:
                    ds = pj.get('datasets') or []
                    tasks = pj.get('tasks') or []
                    hyp = pj.get('hypotheses') or []
                    ds_txt = ', '.join(sorted({(d.get('name') if isinstance(d, dict) else str(d)) for d in ds if d}))
                    task_names = ', '.join([t.get('name') for t in tasks if isinstance(t, dict) and t.get('name')])
                    hyp_txt = '; '.join([str(h) for h in hyp if isinstance(h, str)])
                    extra_instructions = (
                        f"Plan-derived requirements:{metric_hint}. Datasets: {ds_txt}. Tasks: {task_names}. Hypotheses: {hyp_txt}."
                    )[:1500]
            except Exception:
                pass
    plan = generate_plan(objective, extra_instructions)
    if not plan:
        return None
    return apply_plan_safely(plan)
