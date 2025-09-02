# Draft: Compact Research Report

## Abstract
This compact probe evaluates Scale-Selective Spectral Attention (SSSA) integrated into an EfficientNet-B0 backbone with a lightweight FPN for skin lesion classification under strict compute limits (≤1 epoch or ≤100 gradient steps). The approach uses 8×8 DCT blocks per FPN scale, aggregates coefficients into three spectral bands, projects each band to channel space and learns per-band×channel scalar gates (init = 1.0, L1 λ = 1e-4). Experiments target ISIC2018 Task3 (stratified small probe subset at /data/ISIC2018/) with batch_size = 8, img_size = 384, lr = 3e-4 and seeds {0,1,2}. Provided run records contain placeholder val_accuracy = 0.0 and test_accuracy = 0.0 for listed runs. Key validation metrics specified in the plan (ROC AUC, small-lesion recall) are TBD pending execution of the constrained probe runs.

## 1. Introduction

### Executive Introduction
Automated skin lesion classification can assist clinicians by prioritizing suspicious images for review. Under tight compute budgets, compact architectural interventions that selectively emphasize informative spectral content at multiple scales are attractive because they aim to improve small-lesion sensitivity without large parameter increases.

### Problem Statement
Design and evaluate a low-overhead spectral attention module (SSSA) that can be inserted into an EfficientNet-B0 + lightweight FPN pipeline to improve validation ROC AUC and recall on small lesions within a strict probe budget (≤1 epoch or ≤100 gradient steps per run), ensuring the added capacity is accounted for by a parameter-matched control.

### Objectives
- Implement and probe SSSA integrated into EfficientNet-B0+FPN under the stated compute constraints and seeds {0,1,2}.
- Compare SSSA against (A) the baseline EfficientNet-B0+FPN and (B) a parameter-matched channel-attention control using the same budget and logging.

### Research Question
Does inserting Scale-Selective Spectral Attention (SSSA) into each FPN scale produce measurable improvements in validation ROC AUC and small-lesion recall over a parameter-matched baseline within the constrained probe budget?

### Contributions
- Specification and probe plan for SSSA: a spectral-band gating module (DCT 8×8 → 3 bands → 1×1 projections → per-band×channel scalars) with initialization and L1 regularization to avoid collapse.
- A constrained, reproducible experimental protocol (dataset path, seeds, optimizer and stopping rules) designed for cheap falsification runs (≤100 steps / ≤1 epoch).

## 2. Literature Review
Relevant themes and prior approaches that motivate this probe include:

- Use of pretrained CNN backbones and transfer learning for skin lesion classification; common concerns include dataset merging and class imbalance handling.
- Metaheuristic or hybrid optimization approaches that add search cost and often lack parameter-matched baselines.
- Detection and localization methods that address small-lesion handling via higher resolution or tailored anchors.
- Explainability and heatmap evaluation methods (e.g., Grad-CAM variants) and the lack of reproducible sufficiency-focused benchmarks.

This probe is positioned to (A) add a compact spectral module (SSSA) targeting small-lesion sensitivity and (B) follow a constrained experimental protocol with a parameter-matched control, addressing gaps in compute-aware probes and careful controls.

## 3. Methodology
Overview: integrate SSSA into each FPN scale of an EfficientNet-B0 (ImageNet pretrained) encoder. Keep lateral projections and fusion identical across baseline and ablations; the control is a parameter-matched channel-attention module.

SSSA module (specification from plan):
- For each FPN scale feature map F with shape C × H × W:
  - Compute non-overlapping 8×8 2D DCT blocks aligned to the scale's receptive regions (dct_block = 8).
  - Aggregate DCT coefficients into three bands: low (first 6 coefficients), mid (next 12 coefficients), high (remaining coefficients up to the block total).
  - For each band b in {low, mid, high}: apply a 1×1 convolution that maps the band summary to C channels, producing a per-channel scalar gate vector g_b of length C.
  - Combine bands and modulate F by a learned gate: F' = F ⊙ (1 + G), where G = sum_b g_b expanded/broadcast to H × W. Gates are initialized to 1.0.
- Per-band×channel scalar L1 regularization applied with λ = 1e-4 to discourage collapse to zero.

Training objective
- Binary classification loss: use binary cross-entropy (BCE). For a batch of N examples with labels y_i ∈ {0,1} and logits s_i:
  - L_BCE = -(1/N) * sum_i [ y_i * log(sigmoid(s_i)) + (1 - y_i) * log(1 - sigmoid(s_i)) ].
- Gate regularization:
  - L_gate = λ * sum_b ||g_b||_1, with λ = 1e-4.
- Total loss:
  - L = L_BCE + L_gate.

Notes on control and initialization:
- The parameter-matched channel-attention control uses per-scale scalar vectors with similar total parameter count (±2%) and the same L1 λ and init = 1.0 to isolate spectral specificity versus capacity.
- If gate collapse is observed, mitigations include freezing gates for the first 10 steps or increasing gate initialization (e.g., to 1.1).

Implementation-level constraints:
- Keep SSSA extra parameters ≤10% of baseline.
- Image input size = 384, batch_size = 8.
- Optimizer settings are in the Experimental Setup.

## 4. Experimental Setup
Dataset and paths
- Dataset: ISIC2018 Task3 (stratified small probe subset).
- Path: /data/ISIC2018/
- Splits: TBD (train/val/test split specifics to be set; ensure stratification for small lesions).

Hyperparameters and budget
- Backbone: EfficientNet-B0 (ImageNet pretrained).
- FPN: lightweight lateral projection to 128 channels for C3/C4/C5 (identical across conditions).
- img_size = 384, batch_size = 8.
- Learning rate = 3e-4, warmup_steps = 10.
- Max steps per run: 100 OR max_epochs = 1 (hard stop, whichever occurs first).
- Seeds: {0,1,2}.
- Stopping/abort rules: abort on NaN/Inf loss, zero gradient norm for 5 consecutive steps, or OOM; save debug artifacts on abort.

Baselines and ablations
- Baseline: EfficientNet-B0 + lightweight FPN (no SSSA).
- Control: EfficientNet-B0 + FPN + parameter-matched channel-attention.
- Novelty: EfficientNet-B0 + FPN + SSSA.

Metrics
- Primary: ROC AUC (validation).
- Secondary: small-lesion recall (lesions with max dimension ≤64 px at img_size = 384, validation).
- Additional: val_accuracy, test_accuracy (provided in run records as 0.0 placeholders).
- Seeds and statistical test plan: compare mean deltas across seeds; use nonparametric tests where appropriate.

Runs (provided run records)
- A list of experimental run names was provided with val_accuracy and test_accuracy values (all listed as 0.0). These placeholders are reported in Results exactly as given.

Compute and logging
- Hardware: TBD.
- Logging: per-step loss, gate norms, validation metrics at checkpoint intervals; store runs under /runs/{exp}/{cond}/{seed}/.
- Hard resource constraint: no run to exceed 100 gradient steps.

## 5. Results
Reported run metrics (as provided). Values are reported exactly as given (placeholders present in input):

- baseline: val_accuracy = 0.0, test_accuracy = 0.0
- baseline_mbv3: val_accuracy = 0.0, test_accuracy = 0.0
- novelty: val_accuracy = 0.0, test_accuracy = 0.0
- novelty_mbv3: val_accuracy = 0.0, test_accuracy = 0.0
- ablation: val_accuracy = 0.0, test_accuracy = 0.0
- ablation_mbv3: val_accuracy = 0.0, test_accuracy = 0.0
- novelty_lr_up: val_accuracy = 0.0, test_accuracy = 0.0
- novelty_inp_up: val_accuracy = 0.0, test_accuracy = 0.0
- novelty_sgd: val_accuracy = 0.0, test_accuracy = 0.0
- novelty_dropout_high: val_accuracy = 0.0, test_accuracy = 0.0

Other requested target metrics (ROC AUC, small-lesion recall) are not supplied in the provided run records and are therefore TBD.

Notes
- The provided run records appear to be placeholders. No numerical ROC AUC, small-lesion recall, or seed-wise performance numbers were provided; therefore statistical comparisons against the success criteria cannot be computed from the supplied data.

## 6. Discussion
- Status: The plan, model specification, loss formulation, and constrained experimental protocol are fully specified. Provided run records contain placeholder accuracy values (0.0) and do not include the primary metrics required to judge hypotheses (ROC AUC, small-lesion recall). As a result, claims about SSSA performance relative to baseline/control are not supported by supplied numeric evidence.
- Practical interpretation under tight compute: the design emphasizes a low-overhead, testable spectral module (SSSA) with explicit falsification criteria and a parameter-matched control. This enforces parameter matching and limited steps to enable cheap, reproducible probes.
- Failure modes to monitor during runs: gate collapse (all gates → 0), NaN losses, zero gradients; mitigations are included in methodology and stopping rules.
- Reproducibility: seeds {0,1,2}, hard step cap (100), dataset path, and optimizer hyperparameters are provided to ensure runs are reproducible within the same compute constraints.

## 7. Conclusion
This compact report specifies SSSA and a constrained experimental protocol for a cheap, falsifiable probe of spectral attention for skin lesion classification. The implementation details (DCT block sizes, band splits, gate initialization, L1 regularization), the loss formulation, and strict stopping rules are provided. Provided run records are placeholders (val_accuracy/test_accuracy = 0.0); definitive evaluation on the stated target metrics (ROC AUC and small-lesion recall) remains TBD pending execution of the constrained runs.

## 8. Future Work
- Execute the planned probe runs (baseline, control, SSSA) for seeds {0,1,2} under the 100-step cap and report ROC AUC and small-lesion recall.
- If gate collapse or unstable training occurs, apply mitigations: freeze gates for first 10 steps or increase gate init to 1.1 and re-run.
- Budget-permitting ablations:
  - Parameter-matched spatial channel-attention control (already planned).
  - Remove one band (e.g., test low+mid only) to test band importance.
  - Test alternative regularizers (e.g., L2) only if budget permits.
- Expand the small-lesion evaluation to explicit IoU or localization recall if segmentation/crops are available.

## 9. Limitations
- Compute and probe budget: experiments are limited to ≤1 epoch or ≤100 gradient steps; results will be preliminary and may not reflect full convergence behavior.
- Dataset/split details: train/val/test split specifications are TBD; without precise splits, reported metrics may not be comparable.
- Provided runs currently contain placeholder metrics (0.0) and do not include ROC AUC or small-lesion recall; substantive claims cannot be made until these metrics are produced.
- No external citations or additional datasets were introduced; this report relies only on supplied inputs.

## 10. References
None provided.