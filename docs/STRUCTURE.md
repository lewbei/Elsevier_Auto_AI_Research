# Project Structure

- agents/: Packaged steps (novelty, planner, iterate, write_paper).
- agents/paper_finder.py: Downloader + relevance filter (Elsevier + DeepSeek).
- lab/: Shared library utilities used across agents.
  - config.py: YAML-first configuration loader.
  - experiment_runner.py: Minimal train/val/test runner with PyTorch.
  - codegen_utils.py: Safe, tiny codegen for aug/head modules.
  - logging_utils.py, report_html.py, plot_utils.py, mutations.py, prompt_overrides.py.
- data/: Pipeline artifacts and derived JSON.
- paper/: Draft outputs (Markdown + LaTeX).
- tests/: Unit tests.
- config.yaml: Domain-agnostic settings (goal, dataset, research outline sizes).

Conventions

- Prompts are domain-agnostic; only `project.goal` shapes intent.
- Dataset supports imagefolder, CIFAR10, or custom loaders; train/val/test are explicit.
- Evaluation reports highlight test metrics; selection remains on validation.
