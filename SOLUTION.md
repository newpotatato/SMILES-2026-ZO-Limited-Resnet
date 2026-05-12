# Solution: Zero-Order Fine-Tuning of ResNet18 on CIFAR100

## Reproducibility Instructions

### Environment

```
Python  3.11
torch       2.10.0
torchvision 0.25.0
tqdm        4.67.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Reproducing results.json

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --seed 42 \
    --output results.json
```

`256 × 32 = 8192` samples — the full allowed compute budget. CIFAR100 is downloaded automatically on first run. Reproducibility is guaranteed by `--seed 42` (fixed in `validate.py` via `seed_everything`). Expected deviation is within ±0.5% across runs.

---

## Final Solution Description

Four files were modified: `zo_optimizer.py`, `head_init.py`, `augmentation.py`, `train_data.py`.

### zo_optimizer.py — SPSA + Adam with curriculum

**Gradient estimator: SPSA** (Simultaneous Perturbation Stochastic Approximation).

The baseline skeleton uses a 2-point central-difference estimator that perturbs each parameter individually:

```
grad_i ≈ (f(x + ε·eᵢ) - f(x - ε·eᵢ)) / (2ε)
```

This costs `2d` forward passes per step, where `d` is the number of parameters. For the classification head alone `d = 512×100 + 100 = 51300`, which is prohibitively expensive within the 8192-sample budget.

SPSA perturbs all parameters simultaneously with a single Bernoulli ±1 vector `δ`:

```
grad_i ≈ (f(x + ε·δ) - f(x - ε·δ)) / (2ε) · δ_i
```

Cost: exactly **2 forward passes per estimate**, regardless of `d`. This makes it practical to run many steps and to tune deeper layers. Five independent estimates (`k_estimates=5`) are averaged per step to reduce variance at the cost of 10 forward passes total per step — still vastly cheaper than central difference.

**Update rule: Adam** with `lr=1e-2`, `β₁=0.9`, `β₂=0.999`. Adam's adaptive per-parameter learning rates stabilize training compared to plain SGD, especially since SPSA gradient estimates are noisy.

**Curriculum (layer schedule):**

- Steps 1–99: tune only `fc.weight` and `fc.bias` (the classification head, ~51K parameters).
- Steps 100+: additionally unlock the last residual block `layer4.1` (conv1, conv2, bn1, bn2 — ~2.4M parameters).

Focusing on the head first lets Adam accumulate reliable moment estimates before gradient signal from deep layers is introduced. Since SPSA cost is independent of parameter count, unlocking `layer4.1` adds no extra forward-pass cost.

### head_init.py — Xavier uniform × 0.1

Xavier uniform initialization preserves signal variance across layers. Scaling weights by 0.1 keeps initial logits close to zero, so the initial cross-entropy is near `log(100) ≈ 4.6`. This prevents saturated softmax outputs at step 0, which would produce near-zero SPSA gradient estimates and stall the optimizer in the first steps.

### augmentation.py — Extended augmentation pipeline

Added to the training pipeline (validation pipeline unchanged):

- `T.RandomCrop(224, padding=28)` — translation invariance via padding + crop.
- `T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)` — colour robustness.
- `T.RandomErasing(p=0.2)` — occlusion robustness applied after normalization.

These reduce overfitting to specific image positions and colour statistics, which matters because ZO optimization explores the loss landscape through noisy perturbations and benefits from a smoother training surface.

### train_data.py — Full training set

The full CIFAR100 training split (50,000 images) is used. The infinite iterator in `validate.py` cycles through it continuously, so each batch is a fresh random sample from the full distribution.

### What contributed most

In order of impact:

1. **SPSA** — switching from central difference was the prerequisite for any meaningful optimization within the budget. Without it, each step would consume thousands of samples on gradient estimation alone.
2. **Curriculum learning** — tuning only the head for the first 100 steps dramatically improves the starting point before the harder task of adjusting convolutional weights begins.
3. **Adam** — noticeably more stable than SGD given the high variance of SPSA estimates.
4. **Head initialization** — the ×0.1 scaling avoids a degenerate start where gradient estimates carry no useful signal.

---

## Experiments and Failed Attempts

### Central-difference estimator

The skeleton's 2-point per-parameter estimator was tested with `n_batches=256, batch_size=32`. With 51,300 head parameters, a single step would require 102,600 forward passes — far beyond the budget. Even restricting to bias-only updates (100 parameters, 200 FPs/step) yielded near-zero improvement because 256 steps on bias alone cannot shift the decision boundary meaningfully. Discarded in favour of SPSA.

### SGD update rule

Replacing Adam with SGD (momentum 0.9, same lr) produced slower and less stable convergence. SPSA estimates have high variance per step; without per-parameter adaptive scaling, a single large gradient estimate on one parameter could dominate the update and destabilize others. Adam's `v̂` term effectively clips noisy directions. Discarded.

### Aggressive layer expansion from step 1

Unlocking the full backbone (`layer4`, `layer3`) from the beginning was tested. SPSA estimates averaged over all those parameters had too much variance relative to the useful signal at early steps; head accuracy did not improve above checkpoint-2 baseline. The two-phase curriculum (head first, then one block) addresses this by letting the head reach a reasonable state before the backbone is touched.

### Large k_estimates

`k_estimates=20` was tested to reduce gradient variance further. The improvement in estimate quality was marginal compared to k=5, while the cost (40 FPs/step vs. 10) reduced the number of achievable steps within the budget by 4×. k=5 gives a better accuracy-budget trade-off.

### Orthogonal head initialization

`nn.init.orthogonal_` was tested as an alternative to Xavier. It produced slightly higher `val_accuracy_top1_init_head` (checkpoint 2) but identical or marginally lower fine-tuned accuracy (checkpoint 3). The small-scale Xavier initialization appears to give a more conservative and stable starting loss for ZO optimization. Discarded.
