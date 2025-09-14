- Updated `config.yaml` → `llm` block to use ZLM (Z.AI GLM‑4.5) as the global instruct model for all agents via OpenAI‑compatible endpoint:
  - `provider: openai`
  - `model: glm-4.5`
  - `chat_url: https://api.z.ai/api/coding/paas/v4/chat/completions`
  - `api_key_env: ZAI_API_KEY`
  - `default: ""`, `use: default` so stages with `llm: default` resolve to this global model
- Kept `stream: true`, `strict: true`, and retry settings intact.
- No code changes; only configuration. Ensure `ZAI_API_KEY` is set in environment before running agents.

- Moved `llm:` config to top‑level in `config.yaml` (was nested under `research:`), so all agents that call `get('llm.*')` actually pick up the intended provider/model/url.

- Fixed LLM client payload to be OpenAI‑spec compliant for strict endpoints:
  - utils/llm_utils.py: removed non‑spec keys (`provider`, `chat_url`) from HTTP request bodies while preserving them in local logs/cache keys.
  - Resolves 400 errors like "Unknown parameter: 'provider'" on Z.AI.
- Improved paper drafting (journal-style, longer output):
  - `agents/write_paper.py`: enriched non-LLM composer with Related Work, Methods (Data/Architecture/Objective), Experimental Setup with plan-linked success criteria, Ablations, and Broader Impact & Ethics.
  - `agents/write_paper.py`: LLM drafting now targets peer‑review style, ≥ configurable word count, per‑section minimal paragraphs, expanded section list (Related Work, Methods, Experimental Setup, Ablations, Ethics).
  - `config.yaml`: added `pipeline.write_paper_llm.min_words` (default 2200) and `section_min_paragraphs` to tune length/structure.
  - Style control: added `pipeline.write_paper.style.referent` (default: "This study") to enforce third-person referent across non‑LLM composer and LLM prompts.
  - Post-processing: added markdown sanitizer to drop duplicate top-level titles and repeated `##` sections to prevent accidental repeated papers or sections.
  - Page reminder: added optional one-paragraph 'Page reminder' after Introduction describing what each section explains (enabled via `pipeline.write_paper_style.page_reminder_after_intro`).
  - LLM-only mode: removed fallback composer from write_paper execution path. If LLM is disabled or errors, the writer hard-fails (no placeholder output). Enable with `pipeline.write_paper_llm.enable: true` and provide a valid `ZAI_API_KEY`.
- Abstract handling:
  - LLM-only: the non‑LLM fallback no longer writes an abstract; it inserts a clear placeholder. The LLM prompt enforces the requested abstract format (Problem → Gap → Contribution → Setup → Results with numbers/95% CI → Implication/Scope; 150–250 words, active voice, no citations/acronyms/forward refs).

- Paper structure fixes per preference:
  - `agents/write_paper.py` (LLM path): Removed the "What/Why/How" Introduction subsections; Introduction is now cohesive prose. Section order now ends with "Future Work" and explicitly instructs the model to stop there (no References). Page reminder updated accordingly (no "Broader Impact & Ethics").
  - `agents/write_paper.py` (fallback composer): Replaced the "### What/Why/How" blocks with a single introductory paragraph; otherwise unchanged.
  - Markdown sanitizer now truncates any content after the "## Future Work" section to prevent trailing sections or accidental duplication.
  - Upgraded LaTeX output: full sanitized Markdown is rendered to `paper/main.tex` (headings, paragraphs, math) via a minimal converter; still stops at "Future Work".
  - Literature review gap: Related Work ends with a `### Research Gap` subsection; prompt and reminder updated to require this.
  - Formal tone + explicit intro sentences: enforced formal tone (no hype) and explicit sentences in Introduction: “The first objective is…”, “The second objective is…”, “The research question is…”, “The first contribution is…”, “The second contribution is…”. Post-pass ensures these exist.

- Planner enhancements:
  - Added blueprint-first multi-plan mode. When `pipeline.planner.plan_each_blueprint: true` and `data/ideas/blueprints.json` exists, the planner generates one plan per blueprint to `data/plans/plan_1.json`, `plan_2.json`, … and updates `data/plan.json` with the first plan for downstream stages. Falls back to single-plan behavior otherwise.
  - No change to orchestrator flow yet; iterate still consumes `data/plan.json`. Optionally, we can add an iterate-per-plan loop next.
