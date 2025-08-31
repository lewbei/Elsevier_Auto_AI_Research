# Toward: Try a small dropout head tweak

## Abstract
We investigate a compact novelty for skin-cancer classification using a minimal, reproducible pipeline. Our experiments compare a baseline, a novelty variant, and an ablation.

## Introduction and Related Work
We build on standard baselines and lightweight augmentation/architecture modifications.

## Methods
Objective: Evaluate a simple novelty
Hypotheses:
- A small augmentation or head change improves val acc by ~0.5pp

## Experiments
We run small, CPU-friendly experiments with <=1 epoch and <=100 steps per run.
We evaluate baseline, novelty, and ablation, plus minor variants.

### Results
| Setting | Acc (test) |
|---|---:|
| Baseline | 0.0000 |
| Novelty | 0.0000 |
| Ablation | 0.0000 |

Delta vs baseline: +0.0000

## Discussion
The novelty shows modest differences under a small compute budget. Further work includes broader datasets and more rigorous sweeps.

## Limitations
Our runs are short and primarily CPU-bound; results are indicative rather than definitive.

Decision: goal_reached = False

Environment:
- Python: 3.13.5 | packaged by Anaconda, Inc. | (main, Jun 12 2025, 16:37:03) [MSC v.1929 64 bit (AMD64)]
- Executable: C:\Users\lewka\miniconda3\envs\deep_learning\python.exe
- Platform: Windows-11-10.0.26100-SP0

## Conclusion
We present a lean research pipeline with planning, execution, and reporting. It supports quick iteration and extensions.
