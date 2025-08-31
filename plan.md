 Data Leakage Review Plan

Checklist

- Map repo structure and key entry points
- Inspect dataset loading and splits
- Review training/validation loops and metrics
- Check augmentation application by split
- Review caching, logging, and secrets handling
- Summarize risks and concrete mitigations

Plan

- Analyze ML experiment code for train/val/test separation and any cross‑contamination.
- Examine selection/reporting logic for validation overfitting risk.
- Review fallback/synthetic data usage for inadvertent coupling.
- Audit LLM and HTTP utilities for secret or content leakage.

Steps

1) Inventory files
2) Read lab/experiment_runner.py and agents_iterate.py
3) Read lab/codegen_utils.py and generated_aug usage
4) Scan tests for assumptions about datasets and selection
5) Inspect llm_utils.py, paper_finder, logging/reporting
6) Produce findings and recommended fixes

Risks

- CIFAR10 path may be using the test split as validation (selection on test).
- Repeated runs may reuse identical seeds (lack of independence, not leakage).
- FakeData fallback could create overly similar train/val distributions.
- Potential accidental logging of sensitive env values (unlikely but checked).

Rollback

- This is a read‑only audit; no code changes required. If fixes are desired, implement behind env flags (e.g., EVAL_ON_TEST, VAL_FRACTION) to keep defaults stable.

