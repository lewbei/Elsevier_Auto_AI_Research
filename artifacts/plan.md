# plan.md

## Plan
- Set a single instruct model for all agents by configuring the global LLM profile in `config.yaml` to DeepSeek `deepseek-chat` and pointing all stage profiles at this default.

- Adjust paper writer to avoid repeated content and remove non-preferred subheads; ensure the draft stops after Future Work.

## Steps
1) Update `llm` block in `config.yaml` to `provider: deepseek`, `model: deepseek-chat`, `api_key_env: DEEPSEEK_API_KEY`, and set `default: deepseek`, `use: default`.
2) Verify per-stage `pipeline.*.llm` remain `default` so they resolve to the global instruct model without per-stage overrides.
3) Record a short change summary in artifacts for traceability.
4) Modify `agents/write_paper.py` LLM prompt: remove "What/Why/How" subsections, include "Future Work" as the last section, and instruct to stop after it.
5) Update fallback composer to avoid "What/Why/How" headings; keep coherent prose intro.
6) Add sanitizer rule to truncate any content after the "## Future Work" section.

## Risks
- Missing `DEEPSEEK_API_KEY` will cause strict LLM guard to fail at call-sites.
- If you previously relied on a custom OpenAI-compatible endpoint (e.g., Z.AI GLM), this change switches traffic to DeepSeek.
- Streaming and retry behavior remain as configured; any provider-specific rate limits or formats will surface at runtime.
- LLMs might still attempt to add References; sanitizer now cuts after Future Work in Markdown, but LaTeX draft still includes bibliography scaffolding (non-blocking for Markdown).

## Rollback
- Create a git checkpoint, then revert via `git reset --hard <checkpoint>` to restore the prior `config.yaml`.
- To undo writer behavior, revert `agents/write_paper.py` to the previous revision.

## Done Criteria
- Global LLM profile points to an instruct model (`deepseek-chat`).
- All stages that reference `llm: default` resolve to this instruct model.
- No silent fallbacks; strict mode remains enabled.
- Artifacts updated with this plan and a change summary.
- Paper Markdown contains no "What/Why/How" subsections and ends at "## Future Work" with no trailing sections.
