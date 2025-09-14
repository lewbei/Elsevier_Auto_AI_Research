# plan.md

## Plan
- Analyze the entire repository to understand current LLM usage and whether an embedding model is needed. Assess architecture, config, and pipeline stages for advanced capabilities.

## Steps
1) Add helper scripts (env wrappers, code snapshot, preflight check) and write this plan.
2) Generate a complete code snapshot into artifacts for offline review.
3) Search for embedding/LLM/vector usage across the codebase.
4) Read key modules (config.yaml, pipeline entrypoints, agents/*, utils/*) in full.
5) Decide if an embedding model should be integrated (or not), and propose exact integration points.
6) Produce a concise recommendation and change list.

## Risks
- Large codebase may produce heavy snapshots; ensure exclusions handle node_modules/build caches.
- Windows shell differences; rely on git-bash as specified.
- Changes to scripts only; no modifications to core runtime to avoid accidental breakage.

## Rollback
- Create a git checkpoint before multi-file edits; revert via `git reset --hard <checkpoint>`.

## Done Criteria
- Code snapshot created under artifacts/ with index.
- Repository scanned; key files reviewed.
- Written recommendation on embedding model usage with precise rationale.
- No secrets printed; artifacts written.
- Preliminary preflight scan produced `artifacts/fallbacks_found.md` for review.
