import json
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from lab.config import get_bool, get
from agents.stage_manifest import (
    ROOT,
    DATA,
    RUNS,
    PDF_DIR,
    SUM_DIR,
    NOVELTY_REPORT,
    stage_ready,
)
from lab.plan_store import (
    PLAN_PATH as ACTIVE_PLAN_PATH,
    list_plan_files,
    set_active_plan_from_path,
    active_plan_exists,
)
from agents.persona_support import gather_notes

try:
    from agents.personas import DialogueManager  # type: ignore
except Exception:
    DialogueManager = None  # type: ignore


HERE = ROOT


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _now() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _persona_phase_notes(phase: str, context: Dict[str, Any], steps: int = 2) -> List[str]:
    enable = get_bool("pipeline.orchestrator.personas.enable", False) or (
        str(os.getenv("ORCH_PERSONAS", "")).lower() in {"1", "true", "yes"}
    )
    if not enable or DialogueManager is None:
        return []
    try:
        dm = DialogueManager()
        dm.post(
            "User",
            (
                f"Phase: {phase}. Provide structured, actionable guidance tailored to limited compute (<=1 epoch, small steps).\n"
                "Include: top 3 priorities, key risks with mitigations, and the minimal concrete actions to advance this phase. Be specific."
            ),
        )
        dm.post("User", f"Context: {json.dumps(context, ensure_ascii=False)}")
        notes: List[str] = []
        for i in range(max(1, steps)):
            # Auto step cycles personas in DialogueManager (JSON-aware)
            r = dm.step_auto()
            txt = r.get('text','')
            if not txt and r.get('json') is not None:
                try:
                    txt = json.dumps(r.get('json'), ensure_ascii=False)
                except Exception:
                    txt = str(r.get('json'))
            notes.append(f"[{r.get('role','')}] {txt}")
        return notes
    except Exception:
        return []


def _run_mod(mod: str, env_overrides: Dict[str, str] | None = None) -> None:
    cmd = [sys.executable, "-m", mod]
    print(f"[ORCH] RUN {' '.join(cmd)}")
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    subprocess.run(cmd, cwd=str(HERE), check=True, env=env)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    session = DATA / "orchestrator_session.jsonl"
    notes_dir = DATA / "phase_notes"
    steps = int(get("pipeline.orchestrator.phase_steps", 2) or 2)
    stage_time: dict[str, float] = {}

    # Embeddings config snapshot (used for prefetch/force summaries)
    try:
        emb_enable = bool(get("embeddings.enable", False))
    except Exception:
        emb_enable = False
    try:
        ret_enable = bool(get("embeddings.retrieval.enable", False))
    except Exception:
        ret_enable = False
    try:
        emb_provider = str(get("embeddings.provider", "huggingface") or "huggingface").strip().lower()
    except Exception:
        emb_provider = "huggingface"
    try:
        emb_model = str(get("embeddings.model", "") or "").strip()
    except Exception:
        emb_model = ""

    def _safe_model_dir_name(model_name: str) -> str:
        s = str(model_name).strip().replace("/", "-").replace(":", "-")
        s = "".join(ch for ch in s if ch.isalnum() or ch in {"-", "_", "."})
        return s or "model"

    def _vectors_exist() -> bool:
        if not emb_model:
            return False
        root = HERE / "data" / "embeddings" / _safe_model_dir_name(emb_model)
        if not root.exists():
            return False
        try:
            # Any .npy vector file indicates prior embedding run
            return any(root.rglob("*.npy"))
        except Exception:
            return False

    def _prefetch_embeddings_model() -> None:
        if not (emb_enable and emb_provider == "huggingface" and emb_model):
            return
        try:
            print(f"[ORCH] Prefetch embeddings model: {emb_model}")
        except Exception:
            pass
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            _ = AutoTokenizer.from_pretrained(emb_model)
            _ = AutoModel.from_pretrained(emb_model)
            try:
                print(f"[ORCH] Prefetch OK: {emb_model}")
            except Exception:
                pass
        except Exception as exc:
            # Do not fail the run here; the summarize stage will surface detailed errors
            try:
                print(f"[ORCH] Prefetch failed for embeddings model '{emb_model}': {exc}")
            except Exception:
                pass

    def _backfill_vectors_from_summaries() -> None:
        """Generate embedding vectors from existing summary JSONs when no vectors exist.

        Uses utils.embeddings.compute_summary_text + embed_and_store with
        config-driven provider/model settings. Requires GPU as per embedding policy.
        """
        if not (emb_enable and ret_enable and emb_model):
            return
        from pathlib import Path as _P
        sum_dir = HERE / "data" / "summaries"
        if not sum_dir.exists():
            return
        try:
            from lab.config import get as _get
            provider = str(_get("embeddings.provider", "huggingface") or "huggingface").strip().lower()
            model_name = str(_get("embeddings.model", emb_model) or emb_model)
            dtype = str(_get("embeddings.dtype", "float16") or "float16")
            bs = int(_get("embeddings.batch_size", 2) or 2)
            max_len = int(_get("embeddings.max_length", 1024) or 1024)
        except Exception:
            provider, model_name, dtype, bs, max_len = "huggingface", emb_model, "float16", 2, 1024
        try:
            print("[ORCH] Backfill embeddings from summaries …")
        except Exception:
            pass
        try:
            import json as _json
            from utils.embeddings import embed_and_store, compute_summary_text
            n = 0
            for p in sorted(sum_dir.glob("*.json")):
                try:
                    rec = _json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                summ = rec.get("summary") or {}
                if not isinstance(summ, dict):
                    continue
                st = compute_summary_text(summ)
                key = p.stem + "_summary"
                try:
                    _ = embed_and_store(key, st, provider=provider, model=model_name, batch_size=bs, max_length=max_len, dtype=dtype)
                    n += 1
                except Exception as exc:
                    # keep going; failures will surface later if no vectors exist
                    try:
                        print(f"[ORCH] Backfill fail for {p.name}: {exc}")
                    except Exception:
                        pass
            try:
                print(f"[ORCH] Backfill complete. Created {n} vectors (summary_text).")
            except Exception:
                pass
        except Exception as exc:
            try:
                print(f"[ORCH] Backfill error: {exc}")
            except Exception:
                pass

    def log(role: str, content: Any) -> None:
        line = json.dumps({"ts": _now(), "role": role, "content": content}, ensure_ascii=False)
        with session.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # Phase: literature review / paper finding
    if not (get_bool("pipeline.skip.find_papers", False) or (str(os.getenv("SKIP_FIND_PAPERS", "")).lower() in {"1", "true", "yes"})):
        notes = gather_notes(
            "find_papers",
            {},
            steps=steps,
            config_key="pipeline.orchestrator.personas.enable",
            env_name="ORCH_PERSONAS",
        )
        if notes:
            log("find_papers_notes", notes)
            _write_text(notes_dir / "find_papers.txt", "\n\n".join(notes))
        t0 = time.time()
        _run_mod("agents.paper_finder")
        stage_time["find_papers"] = time.time() - t0
    else:
        print("[ORCH] Skipping paper_finder per config/env")

    # Phase: summaries (per-paper)
    _skip_cfg = get_bool("pipeline.skip.summaries", False)
    _skip_env = ((str(os.getenv("SKIP_SUMMARIES", "")).strip().lower()) in {"1", "true", "yes"})
    _force_summaries = False
    # If embeddings+retrieval enabled and no vectors exist, force summaries to generate vectors
    if emb_enable and ret_enable and (not _vectors_exist()):
        _force_summaries = True
        try:
            print("[ORCH] Forcing summaries to generate embeddings (no vectors found).")
        except Exception:
            pass
    if not (_skip_cfg or _skip_env) or _force_summaries:
        ready, reason = stage_ready("summaries")
        if not ready:
            print(f"[ORCH] {reason}")
        else:
            notes = gather_notes(
                "summaries",
                {},
                steps=steps,
                config_key="pipeline.orchestrator.personas.enable",
                env_name="ORCH_PERSONAS",
            )
            if notes:
                log("summaries_notes", notes)
                _write_text(notes_dir / "summaries.txt", "\n\n".join(notes))
            try:
                from agents.summarize import process_pdfs
                _prefetch_embeddings_model()
                try:
                    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf") if p.is_file()]) if PDF_DIR.exists() else []
                    missing = [p for p in pdfs if not (SUM_DIR / f"{p.stem}.json").exists()]
                    print(f"[ORCH] Summaries: found {len(pdfs)} PDFs; missing={len(missing)}")
                    summaries_up_to_date = (len(pdfs) > 0 and len(missing) == 0)
                except Exception:
                    summaries_up_to_date = False
                profile = (get("pipeline.summarize.llm", None) or None)
                model = (get("pipeline.summarize.model", None) or None)
                default_model = (get("llm.default", None) or None)
                prof = str(profile or "").strip().lower()
                md = str(model or "").strip().lower()
                dm = str(default_model or "").strip().lower()
                chunk_size = int(get("pipeline.summarize.pass1_chunk", 20000) or 20000)
                if (md.startswith("gpt-5-")) or (not md and prof == "default" and dm.startswith("gpt-5-")):
                    chunk_size = -1
                # Always skip existing summaries; reprocessing handled only for missing PDFs.
                skip_existing = True
                try:
                    from lab.config import get_bool as _get_bool
                    summ_stream = bool(_get_bool("pipeline.summarize.stream", get_bool("llm.stream", False)))
                except Exception:
                    summ_stream = False
                prev_stream = os.getenv("LLM_STREAM")
                if summ_stream:
                    os.environ["LLM_STREAM"] = "1"
                else:
                    os.environ.pop("LLM_STREAM", None)
                try:
                    if not summaries_up_to_date:
                        t0 = time.time()
                        process_pdfs(
                            pdf_dir=str(PDF_DIR),
                            out_dir=str(SUM_DIR),
                            max_pages=int(get("pipeline.summarize.max_pages", 0) or 0),
                            max_chars=int(get("pipeline.summarize.max_chars", 0) or 0),
                            chunk_size=chunk_size,
                            timeout=int(get("pipeline.summarize.timeout", 60) or 60),
                            max_tries=int(get("pipeline.summarize.max_tries", 4) or 4),
                            model=model,
                            profile=profile,
                            verbose=bool(get("pipeline.summarize.progress", True) or get("pipeline.summarize.detail", False)),
                            skip_existing=skip_existing,
                            selected=missing,
                        )
                        stage_time["summaries"] = time.time() - t0
                    else:
                        print("[ORCH] Summaries step skipped: all PDFs already summarized.")
                finally:
                    if prev_stream is None:
                        os.environ.pop("LLM_STREAM", None)
                    else:
                        os.environ["LLM_STREAM"] = prev_stream
            except Exception as exc:
                print(f"[ORCH] Summaries failed: {exc}")
    # If vectors still do not exist but we have summaries, try summary-based backfill
    if emb_enable and ret_enable and (not _vectors_exist()):
        _prefetch_embeddings_model()
        _backfill_vectors_from_summaries()
    else:
        print("[ORCH] Skipping summaries per config/env")

    # Phase: novelty
    if not (get_bool("pipeline.skip.novelty", False) or (str(os.getenv("SKIP_NOVELTY", "")).lower() in {"1", "true", "yes"})):
        ready, reason = stage_ready("novelty")
        if not ready:
            print(f"[ORCH] {reason}")
        else:
            notes = gather_notes(
                "novelty",
                {},
                steps=steps,
                config_key="pipeline.orchestrator.personas.enable",
                env_name="ORCH_PERSONAS",
            )
            if notes:
                log("novelty_notes", notes)
                _write_text(notes_dir / "novelty.txt", "\n\n".join(notes))
            try:
                nov_stream = bool(get_bool("pipeline.novelty.stream", get_bool("llm.stream", False)))
            except Exception:
                nov_stream = False
            env = {"LLM_STREAM": "1"} if nov_stream else {}
            t0 = time.time()
            _run_mod("agents.novelty", env_overrides=env)
            stage_time["novelty"] = time.time() - t0
    else:
        print("[ORCH] Skipping novelty per config/env")

    # Phase: idea_blueprints (expand top ideas into runnable blueprints)
    if not (get_bool("pipeline.skip.idea_blueprints", False) or (str(os.getenv("SKIP_IDEA_BLUEPRINTS", "")).lower() in {"1", "true", "yes"})):
        ready, reason = stage_ready("idea_blueprints")
        if not ready:
            print(f"[ORCH] {reason}")
        else:
            notes = gather_notes(
                "idea_blueprints",
                {},
                steps=steps,
                config_key="pipeline.orchestrator.personas.enable",
                env_name="ORCH_PERSONAS",
            )
            if notes:
                log("idea_blueprints_notes", notes)
                _write_text(notes_dir / "idea_blueprints.txt", "\n\n".join(notes))
            t0 = time.time()
            _run_mod("agents.idea_blueprints")
            stage_time["idea_blueprints"] = time.time() - t0
    else:
        print("[ORCH] Skipping idea_blueprints per config/env")

    # Phase: plan
    if not (get_bool("pipeline.skip.planner", False) or (str(os.getenv("SKIP_PLANNER", "")).lower() in {"1", "true", "yes"})):
        ready, reason = stage_ready("planner")
        if not ready:
            print(f"[ORCH] {reason}")
        else:
            notes = gather_notes(
                "plan",
                {},
                steps=steps,
                config_key="pipeline.orchestrator.personas.enable",
                env_name="ORCH_PERSONAS",
            )
            if notes:
                log("plan_notes", notes)
                _write_text(notes_dir / "plan.txt", "\n\n".join(notes))
            t0 = time.time()
            _run_mod("agents.planner")
            stage_time["planner"] = time.time() - t0
            try:
                from lab.config import get_bool as _gb  # local import for safety
                if _gb("feedback.plan_eval.enable", _gb("feedback.enable", False)):
                    ready_eval, eval_reason = stage_ready("plan_eval")
                    if not ready_eval:
                        print(f"[ORCH] {eval_reason}")
                    else:
                        print("[ORCH] Evaluating plan(s)…")
                        try:
                            from agents.plan_evaluator import evaluate_plans  # type: ignore
                        except Exception:
                            evaluate_plans = None  # type: ignore
                        if evaluate_plans is not None:
                            plan_files = list_plan_files(include_active=True)
                            if plan_files:
                                try:
                                    res = evaluate_plans(plan_files)
                                    (DATA / "plan_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                                    print("[ORCH] Plan evaluation complete: plan_eval.json written")
                                except Exception as _pexc:
                                    print(f"[ORCH] Plan evaluation failed: {_pexc}")
            except Exception:
                pass
    else:
        print("[ORCH] Skipping planner per config/env")

    # Phase: structured codegen (optional) – runs after plan so it can react to plan.json
    try:
        use_structured_codegen = bool(get_bool("pipeline.codegen.structured.enable", False))
    except Exception:
        use_structured_codegen = False
    if use_structured_codegen:
        try:
            from lab.codegen_structured import run_structured_codegen  # type: ignore
            objective = str(get("pipeline.codegen.structured.objective", "") or "")
            extra = None
            # Provide current plan.json if present
            plan_path = ACTIVE_PLAN_PATH
            if plan_path.exists():
                try:
                    plan_json = plan_path.read_text(encoding="utf-8")
                    extra = f"Current plan artifact: {plan_json[:4000]}"
                except Exception:
                    pass
            cg_res = run_structured_codegen(objective, extra_instructions=extra)
            if cg_res is not None:
                try:
                    _write_text(DATA / "structured_codegen_result.json", cg_res.model_dump_json(indent=2))
                except Exception:
                    pass
                try:
                    print(f"[ORCH] Structured codegen success={cg_res.success} run_dir={cg_res.run_dir}")
                except Exception:
                    pass
        except Exception as exc:
            try:
                print(f"[ORCH] Structured codegen failed: {exc}")
            except Exception:
                pass

    # Phase: data prep (advice only; iterate consumes persona notes when enabled)
    notes = gather_notes(
        "data_prep",
        {},
        steps=steps,
        config_key="pipeline.orchestrator.personas.enable",
        env_name="ORCH_PERSONAS",
    )
    if notes:
        log("data_prep_notes", notes)
        _write_text(notes_dir / "data_prep.txt", "\n\n".join(notes))

    # Phase: interactive code loop (optional)
    enable_interactive = get_bool("pipeline.interactive.enable", False) or (str(os.getenv("INTERACTIVE_ENABLE", "")).lower() in {"1", "true", "yes"})
    if enable_interactive and not (get_bool("pipeline.skip.interactive", False) or (str(os.getenv("SKIP_INTERACTIVE", "")).lower() in {"1", "true", "yes"})):
        notes = gather_notes(
            "interactive",
            {},
            steps=steps,
            config_key="pipeline.orchestrator.personas.enable",
            env_name="ORCH_PERSONAS",
        )
        if notes:
            log("interactive_notes", notes)
            _write_text(notes_dir / "interactive.txt", "\n\n".join(notes))
        t0 = time.time()
        _run_mod("agents.interactive")
        stage_time["interactive"] = time.time() - t0

    # Phase: run (iterate) — supports per-blueprint multi-plan mode
    if not (get_bool("pipeline.skip.iterate", False) or (str(os.getenv("SKIP_ITERATE", "")).lower() in {"1", "true", "yes"})):
        ready, reason = stage_ready("iterate")
        if not ready:
            print(f"[ORCH] {reason}")
        else:
            multi = get_bool("pipeline.planner.plan_each_blueprint", False) or (str(os.getenv("PLAN_EACH_BLUEPRINT", "")).lower() in {"1", "true", "yes"})
            plan_files = list_plan_files(include_active=False)
            if multi:
                if not plan_files and active_plan_exists():
                    plan_files = [ACTIVE_PLAN_PATH]
                print(f"[ORCH] Multi-plan iterate enabled: {len(plan_files)} plans found")
                for idx, pfile in enumerate(plan_files, start=1):
                    try:
                        set_active_plan_from_path(pfile)
                    except Exception as exc:
                        print(f"[ORCH] Failed to set active plan from {pfile}: {exc}")
                        continue
                    notes = gather_notes(
                        "run_iterate",
                        {"plan_file": str(pfile.name)},
                        steps=steps,
                        config_key="pipeline.orchestrator.personas.enable",
                        env_name="ORCH_PERSONAS",
                    )
                if notes:
                    log("iterate_notes", notes)
                    _write_text(notes_dir / f"iterate_{idx}.txt", "\n\n".join(notes))
                    t0 = time.time()
                    env = {"LLM_STAGE": f"iterate_{idx}"}
                    _run_mod("agents.iterate", env_overrides=env)
                    stage_time[f"iterate_{idx}"] = time.time() - t0
            else:
                notes = gather_notes(
                    "run_iterate",
                    {},
                    steps=steps,
                    config_key="pipeline.orchestrator.personas.enable",
                    env_name="ORCH_PERSONAS",
                )
                if notes:
                    log("iterate_notes", notes)
                    _write_text(notes_dir / "iterate.txt", "\n\n".join(notes))
                t0 = time.time()
                _run_mod("agents.iterate")
                stage_time["iterate"] = time.time() - t0
    else:
        print("[ORCH] Skipping iterate per config/env")

    # Phase: interpretation
    try:
        summary = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    # Optional result analysis before persona interpretation notes
    try:
        from lab.config import get_bool as _gb2
        if _gb2("feedback.analysis.enable", _gb2("feedback.enable", False)):
            print("[ORCH] Running result analyzer…")
            try:
                from agents.result_analyzer import analyze, write_outputs  # type: ignore
                ins = analyze(summary)
                write_outputs(ins)
                # include insights in persona context if available
                summary["_insights"] = ins
            except Exception as _aexc:
                print(f"[ORCH] Analyzer failed: {_aexc}")
    except Exception:
        pass
    notes = gather_notes(
        "interpretation",
        {"summary": summary},
        steps=steps,
        config_key="pipeline.orchestrator.personas.enable",
        env_name="ORCH_PERSONAS",
    )
    if notes:
        log("interpret_notes", notes)
        _write_text(RUNS / "interpretation.txt", "\n\n".join(notes))

    # Phase: report (optional)
    write_paper = get_bool("pipeline.write_paper", False) or (str(os.getenv("WRITE_PAPER", "")).lower() in {"1", "true", "yes"})
    if write_paper:
        notes = gather_notes(
            "report",
            {},
            steps=steps,
            config_key="pipeline.orchestrator.personas.enable",
            env_name="ORCH_PERSONAS",
        )
        if notes:
            log("report_notes", notes)
            _write_text(notes_dir / "report.txt", "\n\n".join(notes))
        t0 = time.time()
        _run_mod("agents.write_paper")
        stage_time["write_paper"] = time.time() - t0

    print("[ORCH] Done")

    # Persist simple stage latency metrics and a minimal HTML view
    try:
        import json as _json
        lat_path = RUNS / "stage_latency.json"
        lat_path.write_text(_json.dumps(stage_time, indent=2), encoding="utf-8")
        html = [
            "<html><head><meta charset='utf-8'><title>Stage Latency</title>",
            "<style>body{font-family:system-ui,Segoe UI,Arial} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:8px} th{text-align:left;background:#f5f5f5}</style>",
            "</head><body>",
            "<h1>Stage Latency</h1>",
            "<table><thead><tr><th>Stage</th><th>Seconds</th></tr></thead><tbody>",
        ]
        for k, v in stage_time.items():
            html.append(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>")
        html += ["</tbody></table>", "</body></html>"]
        (RUNS / "stage_latency.html").write_text("\n".join(html), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
