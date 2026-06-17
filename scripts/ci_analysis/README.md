# CI module post-hoc analysis (paper §5.3 + the "what CI learned" figure)

What the class-interaction (CI) residual module **actually learns**. The finding: not
co-occurrence, but a reproducible **per-class domain recalibration** — on the soundscape
domain it systematically boosts the under-represented non-avian taxa (insects / amphibians
/ mammals), and independently trained models, **across architectures**, converge to the
same calibration.

## Method (key point: real data, not synthetic)

Run the 66 labeled soundscapes (`train_soundscapes_labels.csv`, 75 classes that occur)
through a model's 4 folds, attach a forward hook on `head.class_interaction`, and capture:
- **z** = the pre-CI clip logits (first C dims of the hook input)
- **Δ** = the CI residual (the hook output)

Each model is run over **all 66 files × its own 4 folds**, so the logit scale is consistent
within a model. We deliberately **do not stitch OOF predictions**: OOF concatenates the
predictions of different fold-models, which mixes logit scales and artificially inflates
within-class variance (measured: stitched-OOF between% 0.34–0.68 vs same-scale per-fold
0.82–0.97). AUC is a ranking metric and hides this scale issue, but Δ is an absolute
quantity and must be measured at a single scale.

## The 7 models (5× EfficientNetV2-S + 2× eca_nfnet_l0)

| id  | label  | backbone            | note |
|-----|--------|---------------------|------|
| 524 | N0     | tf_efficientnetv2_s | clean ancestor, shallowest CI training → outlier (ρ≈0.6) |
| 607 | N2     | tf_efficientnetv2_s | |
| 676 | N5     | tf_efficientnetv2_s | |
| 804 | X0     | tf_efficientnetv2_s | |
| 817 | X2     | tf_efficientnetv2_s | |
| 681 | N5$^e$ | eca_nfnet_l0        | cross-architecture check |
| 822 | X2$^e$ | eca_nfnet_l0        | cross-architecture check |

We use 5+2 rather than all 10 models: the cross-model panel argues that *independent models
converge to the same calibration*, which rests on **architecture diversity**, not count. The
two eca_nfnet models close off the "maybe it's just an EfficientNet artifact" objection;
additional same-family effv2 models would only be redundant.

### Main results (`analyze_ci.py`)
- The 6 mature models: pairwise Spearman mean **0.97** (min 0.94).
- **Cross-architecture pairs (eca vs effv2): mean ρ=0.96, min 0.94** — the strongest evidence.
- N0 (524), the clean ancestor, is an outlier (ρ≈0.63); CI is trained least on it.
- Non-avian minus avian mean correction is +0.9 to +1.7 logit, consistent across all 7 models.
- Between/(between+within) variance fraction is 0.82–0.97, i.e. the correction is **per-class**,
  not scene-dependent.

## Files

| File | What it does |
|------|--------------|
| `probe_ci_allfiles.py` | The probe: `python3 probe_ci_allfiles.py <exp_dir> <model_id>` → `/tmp/ci_real_<id>.npz` |
| `analyze_ci.py` | Numerical analysis over the 7 models (all numbers in the table above) |
| `make_fig.py` | Generates `../figures/fig_ci_learned.{pdf,png}` (the paper figure) |
| `data/ci_real_<id>.npz` | Per-model 4-fold z/d/y. **Not distributed with the repo**; regenerate with `probe_ci_allfiles.py` over your checkpoints and drop into `data/`, after which `analyze_ci.py` / `make_fig.py` run. |
