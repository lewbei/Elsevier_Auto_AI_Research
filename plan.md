# Update summarize.py for Must-Capture Fields

Plan

- Tighten LLM prompts to require normalized training knobs, metrics with explicit AUC mode, and complete dataset_details keys.
- Add deterministic post-processing in `agents/summarize.py` to ensure placeholders (empty values) exist when unknown and to normalize metrics list.
- Keep schema unchanged; avoid fabricating values; only add normalized empty-key lines where required.
- Validate by compiling sources and running pytest.

Steps

1. Update finalize normalization rules and main summarization instructions.
2. Add enforcement in post-processing: training_procedure, hyperparameters, metrics, dataset_details.
3. Make behavior togglable via YAML (`pipeline.summarize.enforce_required`) or `SUM_ENFORCE`.
4. Patch novelty to emit numbered detailed ideas (`novelty_ideas`) via `agents/novelty.py` with toggles: `pipeline.novelty.detailed_ideas` and `NOVELTY_DETAILED`, and support ideas-only output via `pipeline.novelty.only_ideas` or `NOVELTY_ONLY_IDEAS=1`.
5. Compile and run tests.

Risks

- Might introduce duplicate metric entries if user-provided config also injects metrics.
- Some summaries may gain placeholder lines (e.g., `batch_size:`) even when unknown.
- Overly generic AUC tag `(unspecified)` when mode truly absent.
- Novelty ideas quality depends on LLM responses and input summaries; ensure secrets are configured.

Rollback

- Set `SUM_ENFORCE=0` or `pipeline.summarize.enforce_required: false` to disable.
- Set `NOVELTY_DETAILED=0` or `pipeline.novelty.detailed_ideas: false` to disable detailed novelty ideas.
- Set `NOVELTY_ONLY_IDEAS=1` for ideas-only reports or remove the flag to restore full report.
- Revert this patch if undesired.
