# BirdCLEF+ 2026 — Private 9th | Public 2nd Solution

Reference code for our BirdCLEF+ 2026 CLEF working note,
**"Class Interaction as Domain Recalibration: A Variance-Aware Sound-Event-Detection
Pipeline for BirdCLEF+ 2026"** (Yannan Chen, Independent Researcher).
Solo solution, **2nd public / 9th private** among 4,000+ teams.

**Links:** [Competition](https://www.kaggle.com/competitions/birdclef-2026/) · [Solution write-up](https://www.kaggle.com/competitions/birdclef-2026/writeups/private-9th-public-2nd-solution)

The system is a Xeno-Canto–pretrained SED stack (EfficientNetV2-S and eca\_nfnet\_l0,
GeM pooling, attention head with clip + frame-max logits) trained at 10-second
windows split into two 5-second sub-segments, with:

- **raw-waveform MixUp** as a focal→soundscape bridge,
- **external in-domain data** (AnuraSet / InsectSet459 / iNaturalist / Xeno-Canto) to close the 98%-avian taxonomic gap,
- **iterated noisy-student distillation** with a deliberately diversity-preserving teacher ensemble,
- a **class-interaction (CI) module** — our one architectural addition — that learns a reproducible *per-class domain recalibration*,
- **peak-preserving, taxon-aware** post-processing,
- and a **CPU-only inference** kernel (single-threaded ONNX Runtime workers, shared Mel front-end) that meets the competition's compute budget.

> **Scope.** This is *reference* code, not a turnkey pipeline: the dataset, the
> external corpora, and the trained checkpoints are **not** included, and the
> paths are relative placeholders for your own data layout. Like most competition repositories, it is
> meant to be read and adapted rather than run end-to-end.

## Layout

| File | What it is |
|------|------------|
| `train_folds.py` | The full training script — SED model, the **CI module**, raw-waveform MixUp, EMA, the 4-fold setup, and ONNX export of each trained fold (`export_onnx_for_fold`, run at the end of training). The run configuration is defined inline in the `CFG` dict near the top of the file. |
| `CV_splits/` | The 4-fold cross-validation split for the labeled soundscapes (`sc_fold_split_B.json`, mapping fold → file list). `train_folds.py` reads its `sc_fold_split_path` from here for reproducible folds. |
| `infer_kernel_folds.py` | The CPU-only inference kernel: four single-threaded ONNX Runtime workers, shared Mel front-end, taxon-aware temporal smoothing + file-post. Consumes the per-fold ONNX that `train_folds.py` exports at the end of training. |
| `kernel-metadata.json` | Kaggle kernel metadata for the inference script. |
| `scripts/infer_distill_unlabeled_sc_ensemble.py` | Generates the noisy-student teacher soft-labels over the unlabeled soundscapes. Each model group's four folds are equal-weight averaged, then groups are prob-domain weighted-blended (`MODEL_GROUPS` + `OUT_DIR` set per run). One group → a single-model teacher; several → a weighted ensemble. This is the teacher-label format `train_folds.py` consumes. |
| `scripts/distill_similarity.py` | Pseudo-label similarity across the distillation chain (Pearson over the teacher soft-labels): shows the ancestors are least similar to their descendants, the EfficientNetV2 descendants converge, and the eca\_nfnet models stay distinct — i.e. the teacher ensemble keeps its diversity. |
| `scripts/ci_analysis/` | Analysis of what the CI module learns (per-class domain recalibration) — the probe (`probe_ci_allfiles.py`), the statistics (`analyze_ci.py`), and the figure generator (`make_fig.py`). See its own `README.md`. The probed residuals are not committed; regenerate them with `probe_ci_allfiles.py` over your trained checkpoints. |

## Notes
- Paths are relative (e.g. `birdclef-2026/`, `distill_models/`); point them at your own data / checkpoint layout.
- The competition data is under Kaggle's rules; download it from the BirdCLEF+ 2026 competition page.
- Trained weights are not distributed.

## Citation
```
@inproceedings{chen2026classinteraction,
  title  = {Class Interaction as Domain Recalibration: A Variance-Aware
            Sound-Event-Detection Pipeline for BirdCLEF+ 2026},
  author = {Chen, Yannan},
  booktitle = {Working Notes of CLEF 2026 -- Conference and Labs of the Evaluation Forum},
  year   = {2026}
}
```
