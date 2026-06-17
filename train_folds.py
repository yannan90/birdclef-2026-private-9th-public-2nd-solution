"""
BirdCLEF 2026 Baseline Training Script
=======================================


Usage:
    python train.py          # normal training
    python train.py --debug  # debug mode (200 samples, 2 epochs)

Train / Val modes
-----------------
Determined by the combination of `duration` / `n_segments` / `head_type`.
Val dispatch rule (in the train loop, `n_seg = cfg.get("n_segments", cfg["duration"]//5)`):
    if n_seg > 1:                     → val_epoch_clip_overlap  (clip-level sliding window over 60s files)
    else:                             → val_epoch               (regular val_ds)

Mode table:

  Mode A — 5s short-window baseline
    config: duration=5, n_segments=1
    train:  5s wav → head(n_seg=1) → (B, n_cls)
    val:    val_ds yields 5s wav → val_epoch

  Mode B — multi-segment sliding-window training
    config: duration=10/15/20, n_segments=duration//5
    train:  long-window wav → head(n_seg>1) → (B, n_seg, n_cls)
    val:    val_epoch_clip_overlap (sliding window per file + n_seg aggregation, val_ds unused)

  Mode C — long-window single-segment training + automatic 5s short-window val
    config: duration=10+, n_segments=1
    train:  long-window wav → head(n_seg=1) → (B, n_cls) (whole-segment single-clip prediction)
    val:    val_ds automatically crops to 5s wav (see val_cfg construction) → val_epoch (aligns to 5s granularity)
    note:   the Attention head softmax has a different distribution at train T'≈1024 vs val T'≈512; if val is unstable, fix img_size=(256,256) or switch to the LSE head (more robust to T)

See CFG for orthogonal feature switches: aux head / species weight / mixup / specaug / bg_noise, etc.
"""

import os
import sys
import time
import json
import argparse
import random
import csv
import ast
import re
from pathlib import Path

import math
import functools

# Force print flush to avoid delayed output under nohup
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import timm
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from scipy.sparse import coo_matrix
from scipy.stats import rankdata
import gc
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
CFG = {
    # Paths
    "data_dir":    os.path.join(os.path.dirname(os.path.abspath(__file__)), "birdclef-2026"),
    "output_dir":  os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"),

    # Fold split (focal = stratified by primary_label, SC = GroupKFold by filename)
    "n_folds":     4,
    "fold_ids":    [0, 1, 2, 3],   # all 4 folds
    # Fixed SC fold split file: None = compute GroupKFold on the fly (sklearn-version sensitive, results differ across environments);
    # set to a json path → skip GroupKFold and split by file directly (reproducible across environments).
    # File format: {"0":[fname...], "1":[...], ...}, must cover all SC files & n_splits == n_folds
    "sc_fold_split_path": "CV_splits/sc_fold_split_B.json",

    # ── External ext_df: add extra samples for the BC2026 234 classes from the pretrain pool into train (not val) ──
    # CSV format is compatible with BC2026 train.csv (primary_label + secondary_labels in BC2026 code format).
    # Set None to disable; ext_df rows get _fold=-1, the main CV is untouched, and every fold's train automatically includes all ext.
    # Build script: scripts/build_unified_train_csv.py + bc2026_ext.csv selecting the non-empty primary_label subset
    "ext_df_path":  None,
    # ext_neotropical (colombia/peru) selective-BCE: SC segments with true annotations for 36 overlapping species (only 36 supervised, 198 masked)
    "ext_sc_labels_path": None,   # default None disables it; set the path only when running ext experiments (loading is triggered if this value is set)
    "ext_sc_audio_dir":   "ext_neotropical_sc",   # directory of 60s ogg segments (ext_neotropical_sc/)
    "ext_overlap_path":   "ext_overlap_species.txt",   # list of 36 overlapping species (one per line: code\ttaxonomy_index)
    "ext_sc_cap":         5000,   # cap on ext positive segments (subsample to this count to control the ratio; None = all 7393 positive segments)
    "ext_df_root":  None,
    "ext_df_cap":   None,

    # Audio
    "sample_rate": 32000,
    "duration":    10,
    "n_segments":  2,
    "n_mels":      128,
    "n_fft":       2048,
    "win_length":  2048,
    "hop_length":  512,
    "fmin":        20,
    "fmax":        16000,

    "top_db":      80,
    "img_size":    0,    # natural shape (no resize)
    "mel_scale":   "htk",
    "norm":        "slaney",
    "mel_norm":    "zscore_minmax",   # two-stage mel normalization (zscore → minmax)
    "mel_normalized": True,           # STFT normalized=True (no actual effect after zscore_minmax, kept for strict alignment)
    "db_scope_train": "sample", # channel-agnostic amplitude-to-dB forces 4D input → per-sample ref_max

    # Model (Attention head + mean freq pool + dual-path SEDLoss)
    "head_type":        "Attention",         # SED AttentionHead; "gap"=GAPHead; "LSE"=LSEHead
    "freq_pool_type":     "gem",      # GeMFreq (p=3 learnable)
    "timewise_weight":    0.5,         # dual-path SEDLoss
    # AttentionHead tunable parameters (only effective when head_type="Attention", ignored under LSE)
    "att_hidden_dim":     1024,
    "att_dropout":        0.5,
    "att_temperature":    1.0,
    "att_activation":     "softmax",
    "per_class_att_temperature": False,   # no per-class τ (shared temperature=1.0)
    "att_conv_kernel":   1,             # att_conv time-dim kernel size (>1 = local temporal smoothing within a segment)

    # Class interaction module: residual class-class linear
    # Enabled only for "SC samples not affected by mixup" (is_sc & is_unmixed), so the W matrix only learns true co-occurrence.
    # focal (global data) and post-mixup samples (pseudo co-occurrence) skip class interaction to avoid polluting the prior.
    # init: zero (residual init=0, equivalent to no-op early in training, gradually learns the true co-occurrence prior)
    "use_class_interaction": True,
    # interaction position switch (mutually exclusive):
    # "clipwise":  refinement on clipwise (B,n_cls) after the attention pool
    # "frame_cls": per-frame refinement on cls (B,n_cls,T) (across class, not across frame; W gets ×T more training data)
    "class_interaction_position": "clipwise",
    # Class interaction structures (mask-controlled: trained only on SC unmixed samples / fully enabled at inference):
    # "linear":   Linear(n_cls, n_cls)        zero init        ~55K params
    # "mlp":      Linear→GELU→Linear           last layer zero init   ~60K params
    # "selfattn": class_embed → self-attn → proj  last layer zero init  ~150K params
    "class_interaction_type":     "mlp",
    "class_interaction_hidden":   128,       # mlp first hidden dim / selfattn class embed dim
    "class_interaction_hidden2":  None,      # mlp second hidden dim (None=1 hidden layer original behavior, set a value=2 hidden layers bottleneck/wider)
    "class_interaction_n_heads":  4,         # heads in selfattn mode
    # cross-segment "communication" mode for mlp ci when n_seg>1 + clipwise (linear always uses flat joint, mlp defaults to none):
    #   "none": independent mlp per seg (per-seg broadcast, current default)
    #   "flat": flatten (B, n_seg*n_cls) → MLP → reshape (analogous to linear joint)
    #   "pool": global=mean(logits, seg-dim), each seg cat[self, global] → per-seg MLP, symmetric
    "class_interaction_cross_seg_mode": "pool",


    "log_temp_wd":       0.1,
    "loss_type":        "bce",          # bce/ce/auc/bce_auc(BCE+λ·SoftAUC aux); pure auc plateaus around 0.81 → prefer bce_auc
    "focal_gamma":      0.0,
    "auc_aux_weight":   0.5,      # bce_auc mode: loss = BCE + λ·SoftAUC, added to both the clip and frame paths (λ=0.5 to start)
    "label_smoothing":  0.0,   # disabled (caused collapse after ep20)
    "asl_gamma_neg":    0.0,
    "asl_gamma_pos":    0.0,      # ASL positive-sample gamma (default 0 keeps positive-sample gradients)
    "head_dropout":     0.2,    # LSEHead dense-layer dropout
    "backbone":    "hgnetv2_b4",      # paired with the self-trained XC backbone
    "pretrained":  False,    # new machine cannot reach HF; the XC pretrained weights overwrite backbone 774/774, fully replacing ImageNet weights, so =True is equivalent
    "in_chans":    1,     # single-mel 1 channel
    "drop_path_rate": 0.15,
    # features_only=True makes the backbone output the last stage features (V2-S=256ch)
    # instead of num_classes=0+global_pool="" mode (which includes conv_head outputting 1280ch). The head's first-layer dim differs significantly as a result.
    "use_features_only": True,    # True = drop conv_head/bn2, output the backbone's last stage

    # Xeno-Canto pretrained backbone weights from a public BirdCLEF release.
    # Using this init reaches single-model LB around 0.941.
    # File location: models/pretrain_xc_2025/
    # None = do not load (default); path = strict=False load into model.backbone
    "xc_pretrained_path": "pretrain_ckpt_2026/backbone_hgnetv2b4_ep42_val_roc_auc0.993.pth",  # self-trained hgnetv2b4 backbone

    # Data
    "use_secondary_labels": True,
    "secondary_label_weight": 1.0,
    "merge_sc_segments": True,       # True = merge consecutive same-label segments; False = keep the original 5s grid (auto-ignored when n_segments>1)
    "sc_oversample":    2,
    "min_sample":       100,         # upsample rare focal classes to 100 segments
    # SC-side rare-species upsample (file level): covers a min_sample gap (min_sample only looks at species present in focal, so sonotypes like 47158son* with focal=0 are missed entirely)
    # Group by file: take the union of GT species across all segments in a file, take the max copy factor, and duplicate the whole file's segments (factor-1) times.
    # Compatible with n_seg>1 (the dataset picks a random slot, so segment-level duplication is ineffective; file-level duplication makes "files containing rare species" be sampled more often).
    # 0 = disabled; 50 = duplicate files containing e.g. sonotype 47158son12 (appearing 10 times) ×5
    "sc_min_sample":    0,

    # Batch-level MixUp (in train_epoch, samples within a batch mix with each other in the raw-audio domain)
    "use_batch_mixup":    True,      # True = enable batch-level MixUp (can stack with NoisyPool MixUp); False = disable
    "mixup_main_weight":  0.0,       # 0 = random by lam-Beta (per-sample diversity); >0 = fixed main weight (overrides alpha mode)
    "mixup_alpha":         0.4,      # per-sample Beta(α,α)/Dirichlet(α,...,α) shape parameter; also the mixup switch (0=off, >0=on). <1 U-shaped toward extremes, =1 uniform, >1 centered
    "mixup_n_max":         2,        # max number of sources to mix
    # Stage-based mixup: switch to a larger n_max after the peak
    "mixup_n_max_late":          2,
    "mixup_n_max_switch_epoch":  0,  # disabled
    "mixup_warmup_epoch":  2,        # 0 = mixup the entire run (default); >0 = no mixup for the first N epochs
    "mixup_prob":          0.5,
    "mixup_sc_participate": False,   # True = labeled SC segments enter the left side of mixup

    # Background noise augmentation
    "use_bg_noise":       False,
    "bg_noise_dir":       "background-noise",
    "bg_noise_prob":      0.3,
    "bg_noise_snr_min":   15.0,
    "bg_noise_snr_max":   30.0,

    # Shared trigger probability for three augmentations (each rolls independently)
    # Set 1.0 to disable the gating (always trigger when enabled)
    "aug_prob":           0.5,
    # Gain augmentation (±6dB)
    "use_gain_aug":       False,
    "gain_min_db":        -6.0,
    "gain_max_db":        6.0,
    # Gaussian noise augmentation (SNR 10-30dB)
    "use_gaussian_noise":  True,
    "gn_snr_min":          10.0,
    "gn_snr_max":           30.0,
    # Time shift augmentation (circular shift in time domain)
    "use_time_shift":      False,
    "time_shift_max_frac": 0.1,

    # Pink noise augmentation (1/f spectrum, more natural than white)
    "use_pink_noise":      False,
    "pn_snr_min":          10.0,
    "pn_snr_max":          30.0,

    # Band-limited pink noise (targets the SC-XC spectral gap, where SC is 3-10dB stronger than XC at 8-16kHz)
    # Only adds 1/sqrt(f)-shaped noise within the band [fmin, fmax], narrowing the train/test domain gap
    "use_bl_pink_noise":   False,
    "bl_pink_fmin":        8000.0,
    "bl_pink_fmax":        16000.0,
    "bl_pink_snr_min":     15.0,
    "bl_pink_snr_max":     25.0,
    "bl_pink_prob":        0.3,


    # SpecAugment (freq/time masking in the mel domain, randomly masking frequency and time bands during training)
    "spec_aug":           True,
    "freq_mask_param":    10,
    "time_mask_param":    20,
    "num_freq_masks":     1,
    "num_time_masks":     2,

    # RandomFiltering: wave-level batch EQ aug (multiply a random EQ curve interpolated from 4 control points in the STFT domain, then iSTFT)
    # Applied before mixup, each sample samples a different EQ independently. Disabled by default.
    "use_random_filter":    False,
    "random_filter_min_db": -20.0,  # dB attenuation floor (attenuation only, no gain)
    "random_filter_n_bands": 4,     # number of control points (linearly interpolated along the frequency axis)

    # Per-species BCE pos_weight: weight rare classes (the 8 Mammalia classes) to counter late-training decay
    "use_class_pos_weight":   False,
    "class_pos_weight_alpha": 0.15,

    # Debug
    "debug":         False,
    "debug_samples": 200,

    # Training
    "epochs":      30,
    "save_epochs": [],
    "early_stop_epoch": 15,
    "batch_size":  64,
    "grad_accum_steps": 1,
    "lr":          5e-4,
    "min_lr":      1e-6,
    "weight_decay": 1e-4,
    "optimizer":   "adamw",
    "scheduler":   "flat_cosine",
    "warmup_epochs": 2,
    "flat_epochs":   20,            # flat_cosine(2+20+8)=30ep
    "num_workers": 6,
    "seed":        42,

    # EMA (Exponential Moving Average of weights) — smooths val oscillation and locks in stable weights.
    # Targets val jitter during the overfitting phase of long-duration training.
    "use_ema":          True,
    "ema_decay":        0.999,
    "ema_start_epoch":  3,       # start tracking from epoch N (skipping the noisy warmup weights, N=warmup_epochs is reasonable)
    "ema_val":          True,    # True = use EMA weights for val; False = EMA used only for the final save
    "ema_save_best":    True,    # True = best_model.pth stores EMA weights; False = stores online weights

    # ======

    # Shared teacher npz/meta source for the NoisyPool unlabeled SC pseudo-labels
    # npz shape supports (N, 234) ensemble teacher or (4, N, 234) per-fold zero-leak
    "teacher_npz":         "distill_input/student_labels/ensemble_v3-1/student_labels_prob.npz",
    # Weighted blend of multiple npz (raw single-model npz → blend → smooth, equivalent to the internal ensemble logic):
    # when set to a non-empty list it takes priority over teacher_npz, e.g. [("single_607_fm/...npz", 0.6), ("single_524_fm/...npz", 0.4)]
    # Mathematically equivalent to the prebuilt ensemble (assuming the single npz are raw unsmoothed + teacher_smooth_after_blend=True)
    "teacher_npz_list": [
        ("distill_input/student_labels/single_645_fm/student_labels_prob.npz", 0.5),
        ("distill_input/student_labels/single_672_fm/student_labels_prob.npz", 0.3),
        ("distill_input/student_labels/single_524_fm/student_labels_prob.npz", 0.2),
        ],
    "teacher_smooth_after_blend":   True,    # run time_smooth_v2 per-file after the blend
    # ── Extra post-processing step, aligned with infer_kernel_folds, disabled by default (baking into the pseudo-labels is riskier than smoothing) ──
    "teacher_file_post":            False,
    "teacher_meta_csv":    "distill_input/student_labels/single_645_fm/student_meta.csv",


    # Sample-level MixUp / NoisyPool (in Dataset __getitem__, each sample mixes with a random soundscape pseudo-labeled segment)
    # Based on Noisy Student. Can stack with batch-level MixUp (double MixUp).
    "use_noisy_pool":      True,
    # The npz/meta paths are merged into the top-level teacher_npz / teacher_meta_csv
    "noisy_mixup_prob":    0.5,
    "noisy_mixup_clip_only": False,
    "noisy_mixup_alpha":   0.0,
    "noisy_mixup_weight":  0.5,
    "noisy_min_conf":      0.0,    # no filtering
    "noisy_conf_weighted": True,   # weighted sampling by each segment's max probability (high-confidence segments are sampled more often)
    # Nonlinear label transform (suppress low-confidence noise, enhance high-confidence peaks)
    "noisy_power":          1.3,
    "noisy_label_sharpen": False,   # p_label = p_label * (p_label > thr).float() + p_label ** 2
    "noisy_sharpen_thresh": 0.1,    # only effective when noisy_label_sharpen=True
    "noisy_sharpen_power": 2,        # only effective when noisy_label_sharpen=True

    # ======
}


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_for_deploy(state_dict):
    """Strip from deploy checkpoints (best_model.pth / model_epXX.pth):
    - the `_orig_mod.` prefix (left behind by torch.compile)
    Note: the checkpoint.pth used for resume does not call this function and keeps the full state.
    """
    out = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("distill."):
            continue
        out[k] = v
    return out


def _build_ckpt_payload(cfg, model):
    """Build the archive dict for best_model.pth / model_epXX.pth (state_dict + all architecture/front-end parameters needed to rebuild the model for inference)."""
    return {
        "state_dict":        _strip_for_deploy(model.state_dict()),
        "head_type":         cfg.get("head_type", "LSE"),
        "freq_pool_type":    cfg.get("freq_pool_type", "mean"),
        # AttentionHead architecture parameters (the head must be rebuilt with identical values at inference)
        "att_hidden_dim":            cfg.get("att_hidden_dim", 512),
        "att_dropout":               cfg.get("att_dropout", 0.5),
        "att_temperature":           cfg.get("att_temperature", 1.0),
        "att_activation":            cfg.get("att_activation", "softmax"),
        "per_class_att_temperature": cfg.get("per_class_att_temperature", False),
        "att_conv_kernel":           cfg.get("att_conv_kernel", 1),
        "use_class_interaction":     cfg.get("use_class_interaction", False),
        "class_interaction_type":    cfg.get("class_interaction_type", "linear"),
        "class_interaction_hidden":  cfg.get("class_interaction_hidden", 128),
        "class_interaction_hidden2": cfg.get("class_interaction_hidden2", None),
        "class_interaction_n_heads": cfg.get("class_interaction_n_heads", 4),
        "class_interaction_cross_seg_mode": cfg.get("class_interaction_cross_seg_mode", "none"),
        "class_interaction_position": cfg.get("class_interaction_position", "clipwise"),
        "backbone":          cfg["backbone"],
        "duration":          cfg["duration"],
        "n_mels":             cfg["n_mels"],
        "n_fft":              cfg["n_fft"],
        "win_length":         cfg.get("win_length"),
        "hop_length":         cfg["hop_length"],
        "img_size":           cfg["img_size"],
        "fmin":               cfg["fmin"],
        "fmax":               cfg["fmax"],
        "mel_scale":          cfg.get("mel_scale"),
        "norm":               cfg.get("norm"),
        "mel_norm":           cfg.get("mel_norm", "minmax"),
        "mel_normalized":     cfg.get("mel_normalized", False),
        "top_db":             cfg.get("top_db", 80),
        "in_chans":           cfg.get("in_chans", 3),
        "use_features_only":  cfg.get("use_features_only", False),
    }


class EMAWrapper:
    """EMA (Exponential Moving Average) weight tracker.
    Applies an exponential moving average to all float tensors in model.state_dict().
    Compatible with torch.compile (the _orig_mod. prefix is tracked as-is).

    Usage:
        ema = EMAWrapper(model, decay=0.999)
        # after each step (after optimizer.step())
        ema.update(model)
        # swap to EMA weights before val
        ema.apply_to(model)
        val_auc = val_epoch(...)
        ema.restore(model)  # restore online weights and continue training
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._init_shadow(model)

    @torch.no_grad()
    def _init_shadow(self, model):
        """Initialize the shadow as a snapshot of the current model weights"""
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def reinit(self, model):
        """At ema_start_epoch, reinitialize the shadow from the post-warmup online weights to remove cold-start contamination"""
        self._init_shadow(model)

    @torch.no_grad()
    def update(self, model):
        """EMA update from online weights → shadow weights"""
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def apply_to(self, model):
        """swap: back up online weights, load EMA weights into the model."""
        self.backup = {}
        sd = model.state_dict()
        for k, ema_v in self.shadow.items():
            if k in sd:
                self.backup[k] = sd[k].detach().clone()
                sd[k].copy_(ema_v)

    def restore(self, model):
        """swap back: restore online weights from the backup."""
        sd = model.state_dict()
        for k, v in self.backup.items():
            if k in sd:
                sd[k].copy_(v)
        self.backup = {}


def get_exp_dir(output_dir, debug=False, resume=False, exp_name=None):
    """Normal training creates exp001, exp002, ...; debug mode uses a fixed debug/; resume mode returns the latest exp directory.
    exp_name: manually specified experiment name, used for cloud training to avoid numbering conflicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        exp_dir = output_dir / "debug"
        exp_dir.mkdir(exist_ok=True)
        return exp_dir
    if exp_name:
        exp_dir = output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        return exp_dir
    existing = sorted(output_dir.glob("exp???"))
    if resume:
        assert existing, "No exp directory found to resume from"
        return existing[-1]  # return the latest exp, do not create a new directory
    last_idx = int(existing[-1].name[3:]) if existing else 0
    exp_dir = output_dir / f"exp{last_idx + 1:03d}"
    exp_dir.mkdir()
    return exp_dir


def parse_time_to_sec(t_str):
    """'HH:MM:SS' → seconds (integer)"""
    parts = t_str.strip().split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def build_species_to_taxon(taxonomy, species_list):
    """Return an (n_classes, n_taxa) float tensor.
    n_taxa is fixed by the {Aves, Amphibia, Insecta, Mammalia, Reptilia} present in taxonomy["class_name"], in fixed order.
    The row for species i is one-hot (each species belongs to one taxon).
    """
    taxon_order = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
    taxon_to_idx = {t: i for i, t in enumerate(taxon_order)}
    sp_to_class  = dict(zip(taxonomy["primary_label"].astype(str),
                            taxonomy["class_name"].astype(str)))
    mat = torch.zeros(len(species_list), len(taxon_order), dtype=torch.float32)
    missing = []
    for i, sp in enumerate(species_list):
        cls = sp_to_class.get(str(sp))
        if cls is None or cls not in taxon_to_idx:
            missing.append((sp, cls))
            continue
        mat[i, taxon_to_idx[cls]] = 1.0
    if missing:
        print(f"  [taxon_map] WARN: {len(missing)} species have no taxon mapping, defaulting to Aves. Examples: {missing[:3]}")
        for sp, _ in missing:
            i = species_list.index(sp)
            mat[i, 0] = 1.0   # fallback: assign to Aves
    return mat, taxon_order


def compute_species_pos_weight(train_df, species_to_idx, alpha=0.3):
    """Per-species BCE pos_weight = (max_count / class_count) ** alpha
    Rare species (the 8 Mammalia classes) get a larger pos_weight, common species (mostly Aves) ≈ 1.
    alpha=0.3 => Mammalia ~ 1.5-2x, a gentle weighting.
    """
    n_cls = len(species_to_idx)
    counts = np.zeros(n_cls, dtype=np.float64)
    for s in train_df["labels"].astype(str):
        primary = s.split(";")[0].strip()
        i = species_to_idx.get(primary)
        if i is not None:
            counts[i] += 1
    counts = np.maximum(counts, 1.0)
    max_count = counts.max()
    pw = (max_count / counts) ** alpha
    return torch.tensor(pw, dtype=torch.float32)


def compute_auc(labels, preds):
    """Macro ROC-AUC, skipping classes with no positive samples (matches the competition metric)"""
    aucs = []
    for i in range(labels.shape[1]):
        if labels[:, i].sum() == 0:
            continue
        aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
    return float(np.mean(aucs)) if aucs else 0.0


# Global per-taxon labeling info (assigned in main(); when unset, compute_per_taxon_auc returns an empty dict)
_SPECIES_TAXON_IDX = None   # np.ndarray (n_cls,), the taxon index of each species
_TAXON_NAMES       = None   # list[str], taxon order


def compute_per_taxon_auc(labels_np, preds_np):
    """Compute the mean per-class AUC grouped by taxon.
    Depends on _SPECIES_TAXON_IDX / _TAXON_NAMES (initialized in main()).
    Returns {taxon_name: mean_auc}; the value is None if a taxon has no positive-sample class in val.
    """
    if _SPECIES_TAXON_IDX is None or _TAXON_NAMES is None:
        return {}
    result = {}
    for t, name in enumerate(_TAXON_NAMES):
        cls_idx = np.where(_SPECIES_TAXON_IDX == t)[0]
        aucs = []
        for i in cls_idx:
            if labels_np[:, i].sum() == 0:
                continue
            aucs.append(roc_auc_score(labels_np[:, i], preds_np[:, i]))
        result[name] = float(np.mean(aucs)) if aucs else None
    return result


def build_mel_transform(cfg):
    """Return (mel, db): a single MelSpectrogram with parameters taken from the top-level cfg."""
    db = T.AmplitudeToDB(top_db=cfg.get("top_db", 80))

    def _make(n_fft, hop, win, n_mels, fmin, fmax, mel_scale, mel_filterbank_norm):
        kw = {}
        if mel_scale is not None:           kw["mel_scale"] = mel_scale
        if mel_filterbank_norm is not None: kw["norm"]      = mel_filterbank_norm
        if win is not None:                 kw["win_length"] = win
        # normalized=True (STFT divided by sqrt(sum(window²)))
        kw["normalized"] = cfg.get("mel_normalized", False)
        return T.MelSpectrogram(
            sample_rate=cfg["sample_rate"],
            n_fft=n_fft, hop_length=hop, f_min=fmin, f_max=fmax, n_mels=n_mels,
            **kw,
        )

    return _make(
        cfg["n_fft"], cfg["hop_length"], cfg.get("win_length"), cfg["n_mels"],
        cfg["fmin"], cfg["fmax"], cfg.get("mel_scale"), cfg.get("norm"),
    ), db


def pad_or_crop(wav, n_samples, pad_type="left"):
    """Zero-pad short audio, crop long audio. Used only for short training recordings (XC/iNat train_audio).
    pad_type:
      "left"   — pad zeros on the left, audio flush to the right end. For Stage 1 supervised training:
                 ensures that during MixUp the two audio clips always overlap on the right, giving attention a contrast signal.
      "random" — random-position crop/pad. For Pseudo-label Stage 2:
                 audio drifts randomly within the window, adding time-shift diversity and preventing the model from overfitting a position prior.
    """
    n = len(wav)
    if n < n_samples:
        padded = torch.zeros(n_samples)
        if pad_type == "left":
            content_start = n_samples - n            # left zero-pad: audio flush to the right end
        else:  # "random"
            content_start = random.randint(0, n_samples - n)  # random position
        padded[content_start : content_start + n] = wav
        return padded, content_start
    if n == n_samples:
        return wav, 0
    if pad_type == "random":
        start = random.randint(0, n - n_samples)  # random crop
    else:
        start = 0  # head crop (both "left" and "head" take the head)
    return wav[start : start + n_samples], 0


# ============================================================
# Data merging + fold splitting
# ============================================================

def build_combined_df(cfg, species_to_idx):
    """One-time preprocessing + fold splitting (called once at main() startup, shared across folds):
       - Focal:  StratifiedKFold by primary_label
       - SC:     GroupKFold by filename
    Returns a single DataFrame with a _fold field per row. SC segments keep the original 5s grid (unmerged).
    Each fold does its train/val split inside the fold loop, then decides whether to merge train SC segments based on n_segments.
    """
    n_folds      = cfg.get("n_folds", 5)
    random_state = cfg.get("seed", 42)
    rows = []

    # ── Focal (train_audio): StratifiedKFold by primary_label ──
    train_csv = pd.read_csv(os.path.join(cfg["data_dir"], "train.csv"))
    train_csv = train_csv[train_csv["primary_label"].isin(species_to_idx)].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_csv["_fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_csv, train_csv["primary_label"])):
        train_csv.loc[val_idx, "_fold"] = fold

    for _, r in train_csv.iterrows():
        filename = r["filename"]                   # "species_id/XC12345.ogg"
        audio_id = Path(filename).stem
        labels   = r["primary_label"]
        if cfg.get("use_secondary_labels"):
            sec_raw = r.get("secondary_labels", "[]")
            try:
                sec_list = ast.literal_eval(sec_raw) if isinstance(sec_raw, str) else []
            except Exception:
                sec_list = []
            if sec_list:
                labels = labels + ";" + ";".join(str(s).strip() for s in sec_list)
        rows.append({
            "file_path": os.path.join(cfg["data_dir"], "train_audio", filename),
            "audio_id":  audio_id,
            "labels":    labels,
            "start_sec": -1,
            "end_sec":   -1,
            "data_type": "clip",
            "is_ext":    False,                  # official train_audio, not ext
            "_fold":     int(r["_fold"]),
        })

    # ── SC (train_soundscapes labeled segments): GroupKFold by filename ──
    sc_labels_path = os.path.join(cfg["data_dir"], "train_soundscapes_labels.csv")
    sc_labels = pd.read_csv(sc_labels_path).drop_duplicates().reset_index(drop=True)
    sc_labels["start_sec"] = sc_labels["start"].apply(parse_time_to_sec)
    sc_labels["end_sec"]   = sc_labels["end"].apply(parse_time_to_sec)
    sc_labels = sc_labels.sort_values(["filename", "start_sec"]).reset_index(drop=True)
    soundscape_dir = os.path.join(cfg["data_dir"], "train_soundscapes")

    sc_files_df = sc_labels[["filename"]].drop_duplicates().reset_index(drop=True)
    # SC fold split: prefer the cfg["sc_fold_split_path"] file (reproducible across environments);
    # otherwise fall back to computing GroupKFold on the fly (sklearn-version sensitive, results may differ across environments).
    sc_split_path = cfg.get("sc_fold_split_path", None)
    if sc_split_path:
        # support relative paths (relative to the train_folds.py directory, i.e. the project root)
        if not os.path.isabs(sc_split_path):
            sc_split_path = os.path.join(str(Path(__file__).resolve().parent), sc_split_path)
        if not os.path.exists(sc_split_path):
            raise FileNotFoundError(f"sc_fold_split_path does not exist: {sc_split_path}")
        with open(sc_split_path) as _f:
            _split = json.load(_f)
        # validate: covers all SC files + fold count matches
        _file_to_fold = {fn: int(fid) for fid, fns in _split.items() for fn in fns}
        _expected = set(sc_files_df["filename"])
        _got = set(_file_to_fold.keys())
        _missing = _expected - _got
        _extra   = _got - _expected
        if _missing:
            raise ValueError(f"sc_fold_split file is missing {len(_missing)} SC files (first 3): {list(_missing)[:3]}")
        if _extra:
            print(f"  WARN: sc_fold_split file contains {len(_extra)} files not in the CSV (will be ignored)")
        _n_split_folds = len(set(_file_to_fold.values()))
        if _n_split_folds != n_folds:
            raise ValueError(f"sc_fold_split file contains {_n_split_folds} folds, does not match cfg n_folds={n_folds}")
        sc_files_df["_fold"] = sc_files_df["filename"].map(_file_to_fold).astype(int)
        file_to_fold = {fn: int(fid) for fn, fid in zip(sc_files_df["filename"], sc_files_df["_fold"])}
        print(f"  SC fold split loaded from file: {sc_split_path} ({len(file_to_fold)} files, {n_folds} folds)")
    else:
        gkf = GroupKFold(n_splits=n_folds)
        sc_files_df["_fold"] = -1
        for fold, (_, val_idx) in enumerate(gkf.split(sc_files_df, groups=sc_files_df["filename"])):
            sc_files_df.loc[sc_files_df.index[val_idx], "_fold"] = fold
        file_to_fold = dict(zip(sc_files_df["filename"], sc_files_df["_fold"]))
        print(f"  SC fold split via on-the-fly GroupKFold (n_splits={n_folds}) — note the sklearn-version dependency")

    for _, r in sc_labels.iterrows():
        rows.append({
            "file_path": os.path.join(soundscape_dir, r["filename"]),
            "audio_id":  Path(r["filename"]).stem,
            "labels":    r["primary_label"],
            "start_sec": r["start_sec"],
            "end_sec":   r["end_sec"],
            "data_type": "soundscape",
            "is_ext":    False,                  # SC labeled segments, not ext
            "_fold":     int(file_to_fold[r["filename"]]),
        })

    # ── ext_df: external supplementary data (StratifiedKFold by primary_label, same fold split as focal) ──
    # Enabled by setting cfg["ext_df_path"]; CSV fields file/primary_label/secondary_labels (BC2026 code format)
    # Fold assignment:
    #   - species with sample count >= n_folds → StratifiedKFold into 0..n-1
    #   - sample count <  n_folds → _fold=-1 (always in train, since stratified requires ≥n_folds per class)
    # val still filters data_type=="soundscape", so even folded ext never enters val (same as focal)
    # Net effect: each fold sees 3/4 of ext, increasing training-data diversity across folds
    ext_csv = cfg.get("ext_df_path")
    if ext_csv:
        ext_root = cfg.get("ext_df_root") or os.path.dirname(ext_csv).replace(os.path.basename(ext_csv), "")
        ext_df = pd.read_csv(ext_csv)

        # first filter valid rows + collect the primary list, then do fold assignment
        valid_rows = []
        primaries  = []
        for _, r in ext_df.iterrows():
            primary = r["primary_label"]
            if primary not in species_to_idx:
                continue
            valid_rows.append(r)
            primaries.append(primary)

        # ext_df_cap: cap per species again at runtime (independent of the offline cap)
        # reproduced with the same seed=random_state; shuffles only within a group, preserving cross-species proportions
        ext_cap = cfg.get("ext_df_cap")
        if ext_cap is not None and ext_cap > 0:
            from collections import defaultdict as _dd
            import random as _rand
            _rng = _rand.Random(random_state)
            by_pl_idx = _dd(list)
            for i, p in enumerate(primaries):
                by_pl_idx[p].append(i)
            kept_idx = []
            n_capped_species = 0
            for pl, idxs in by_pl_idx.items():
                if len(idxs) > ext_cap:
                    _rng.shuffle(idxs)
                    kept_idx.extend(idxs[:ext_cap])
                    n_capped_species += 1
                else:
                    kept_idx.extend(idxs)
            kept_idx = sorted(kept_idx)
            valid_rows = [valid_rows[i] for i in kept_idx]
            primaries  = [primaries[i]  for i in kept_idx]
            print(f"    ext_df_cap={ext_cap}: reduced to {len(valid_rows)} rows ({n_capped_species} species capped)")

        # split into two groups by sample count: rare (<n_folds) → -1, others → StratifiedKFold
        from collections import Counter
        cls_cnt = Counter(primaries)
        rare = {lbl for lbl, c in cls_cnt.items() if c < n_folds}
        ext_fold = [-1] * len(valid_rows)
        strat_idx  = [i for i, lbl in enumerate(primaries) if lbl not in rare]
        if strat_idx:
            strat_lbl = [primaries[i] for i in strat_idx]
            from sklearn.model_selection import StratifiedKFold as _SKF
            _ext_skf = _SKF(n_splits=n_folds, shuffle=True, random_state=random_state)
            for fold_k, (_, val_idx) in enumerate(_ext_skf.split(strat_idx, strat_lbl)):
                for vi in val_idx:
                    ext_fold[strat_idx[vi]] = fold_k
        n_rare_kept = sum(1 for f in ext_fold if f == -1)

        # emit rows
        for i, r in enumerate(valid_rows):
            primary = primaries[i]
            labels = primary
            if cfg.get("use_secondary_labels"):
                sec_raw = r.get("secondary_labels", "[]")
                try:
                    sec_list = ast.literal_eval(sec_raw) if isinstance(sec_raw, str) else []
                except Exception:
                    sec_list = []
                sec_list = [s for s in sec_list if s in species_to_idx]
                if sec_list:
                    labels = labels + ";" + ";".join(str(s).strip() for s in sec_list)
            rel_file = r["file"]
            file_path = os.path.join(ext_root, rel_file) if ext_root else rel_file
            rows.append({
                "file_path": file_path,
                "audio_id":  Path(rel_file).stem,
                "labels":    labels,
                "start_sec": -1,
                "end_sec":   -1,
                "data_type": "clip",
                "is_ext":    True,
                "_fold":     ext_fold[i],   # 0..n-1 stratified, or -1 for rare (<n_folds)
            })
        print(f"  ext_df loaded: {len(valid_rows)} rows from {ext_csv} (root={ext_root})")
        print(f"    stratified {n_folds}-fold: {len(valid_rows)-n_rare_kept} rows; "
              f"rare(_fold=-1 always-train): {n_rare_kept} rows (classes <{n_folds} samples)")

    # ext_neotropical SC (colombia/peru): selective-BCE SC segments with true annotations
    #   one row per 5s segment; data_type="soundscape" + is_ext_neo=True + _fold=-1 (always train, never val,
    #   since val filters data_type=="soundscape" & _fold==fk, and _fold=-1 is excluded); labels in Dataset.ext_label_lookup
    #   precomputed as 234-dim with a -1 mask (only 36 overlapping species supervised, 198 non-overlapping + unlabeled dims masked)
    ext_sc_csv = cfg.get("ext_sc_labels_path")
    if ext_sc_csv and os.path.exists(ext_sc_csv):
        ext_sc = pd.read_csv(ext_sc_csv)
        ext_sc["start_sec"] = ext_sc["start"].apply(parse_time_to_sec)
        ext_sc["end_sec"]   = ext_sc["end"].apply(parse_time_to_sec)
        ext_audio_dir = cfg.get("ext_sc_audio_dir", "")
        # only positive segments (containing an overlapping species) become training rows, anchored as the start → the first of the n_seg consecutive segments carries the positive label;
        #    negative segments are not rows, but all segments (positive + negative) enter ext_label_lookup (multi-segment reads look up neighboring-segment labels, see Dataset init)
        _pl = ext_sc["primary_label"].astype(str).str.strip()
        ext_pos = ext_sc[(_pl != "") & (_pl != "nan")].reset_index(drop=True)
        ext_cap = cfg.get("ext_sc_cap")              # cap: subsample positive segments to control the ext ratio
        if ext_cap and ext_cap > 0 and len(ext_pos) > ext_cap:
            ext_pos = ext_pos.sample(n=ext_cap, random_state=random_state).reset_index(drop=True)
        for _, r in ext_pos.iterrows():
            rows.append({
                "file_path": os.path.join(ext_audio_dir, r["filename"]),
                "audio_id":  Path(r["filename"]).stem,
                "labels":    str(r["primary_label"]),
                "start_sec": int(r["start_sec"]), "end_sec": int(r["end_sec"]),
                "data_type": "soundscape", "is_ext": False,
                "is_ext_neo": True, "has_unidentified": int(r["has_unidentified"]),
                "_fold": -1,
            })
        print(f"  ext_neotropical SC: {len(ext_pos)} positive-segment rows (anchored start, cap={ext_cap}); lookup contains all segments")

    df = pd.DataFrame(rows).reset_index(drop=True)
    # ext_neotropical marker columns fillna (other rows lacking these two columns → False/0)
    if "is_ext_neo" in df.columns:
        df["is_ext_neo"] = df["is_ext_neo"].fillna(False).astype(bool)
        df["has_unidentified"] = df["has_unidentified"].fillna(0).astype(int)
    gc.collect()
    return df


def merge_sc_train_segments(train_df):
    """Merge consecutive same-label soundscape segments in train_df (segments >5s are randomly cropped, acting as data augmentation).
    Rows with other data_type are kept. Returns a new DataFrame.
    Only called when n_segments == 1 (multi-segment mode needs a fixed length per segment and cannot merge).
    """
    sc_mask = train_df["data_type"] == "soundscape"
    sc_df = (train_df[sc_mask]
             .sort_values(["file_path", "start_sec"])
             .reset_index(drop=True))
    other_df = train_df[~sc_mask].reset_index(drop=True)

    merged = []
    tmp_fp = tmp_pl = tmp_aid = tmp_start = tmp_end = None
    for _, r in sc_df.iterrows():
        fp, pl, aid, s, e = r["file_path"], r["labels"], r["audio_id"], r["start_sec"], r["end_sec"]
        if fp == tmp_fp and pl == tmp_pl:
            tmp_end = e
        else:
            if tmp_fp is not None:
                merged.append({
                    "file_path": tmp_fp, "audio_id": tmp_aid, "labels": tmp_pl,
                    "start_sec": tmp_start, "end_sec": tmp_end, "data_type": "soundscape",
                    "is_ext": False,
                })
            tmp_fp, tmp_pl, tmp_aid, tmp_start, tmp_end = fp, pl, aid, s, e
    if tmp_fp is not None:
        merged.append({
            "file_path": tmp_fp, "audio_id": tmp_aid, "labels": tmp_pl,
            "start_sec": tmp_start, "end_sec": tmp_end, "data_type": "soundscape",
            "is_ext": False,
        })
    return pd.concat([other_df, pd.DataFrame(merged)], ignore_index=True)


# ============================================================
# Datasets
# ============================================================

class BirdDataset(Dataset):
    """Unified dataset:
    - start_sec == -1: train_audio, randomly cropped during training, center-cropped during validation
    - start_sec >= 0: soundscape segment, read 5s at a fixed offset
    """

    def __init__(self, df, species_to_idx, cfg, is_train=True, noisy_pool=None):
        self.df             = df.reset_index(drop=True)
        self.species_to_idx = species_to_idx
        self.n_classes      = len(species_to_idx)
        self.cfg            = cfg
        self.is_train       = is_train
        self.n_samples      = cfg["sample_rate"] * cfg["duration"]
        # When duration>5 we need to look up labels of adjacent segments
        self.sc_label_lookup = {}
        if cfg["duration"] > 5:
            sc_rows = df[df["start_sec"] >= 0]
            for _, r in sc_rows.iterrows():
                key = (Path(r["file_path"]).name, int(r["start_sec"]))
                self.sc_label_lookup[key] = r["labels"]
        # ext_neotropical (colombia/peru) selective-BCE: precompute a 234-dim label per segment (with a -1 mask)
        #   36 overlapping species: labeled=1 / unlabeled=0 (absence, guaranteed by full annotation); in has_unidentified segments, unlabeled=-1 (may be an unrecognized overlapping species);
        #   198 non-overlapping species=-1 (colombia/peru never label these, so always masked / unsupervised). In SEDLoss only label>=0 counts toward the loss.
        self.ext_label_lookup = {}
        ext_csv = cfg.get("ext_sc_labels_path")
        _has_ext = bool(df["is_ext_neo"].any()) if "is_ext_neo" in df.columns else False
        # Build the lookup from all segments (positive + negative) in the csv → multi-segment reads can look up surrounding segment labels (train_df only contains positive segment rows, see build_combined_df)
        if _has_ext and ext_csv and os.path.exists(ext_csv):
            ov_path = cfg.get("ext_overlap_path", "")
            ov_idx = [int(ln.split("\t")[1]) for ln in open(ov_path)] if (ov_path and os.path.exists(ov_path)) else []
            ext_full = pd.read_csv(ext_csv)
            for _, r in ext_full.iterrows():
                key = (str(r["filename"]), int(parse_time_to_sec(r["start"])))
                lab = torch.full((self.n_classes,), -1.0)          # default: fully masked
                for ci in ov_idx: lab[ci] = 0.0                    # 36 overlapping species default to absence
                pl = str(r["primary_label"])
                if pl and pl != "nan":
                    for sp in pl.split(";"):
                        si = self.species_to_idx.get(sp.strip())
                        if si is not None: lab[si] = 1.0           # labeled=1
                if int(r["has_unidentified"]) == 1:
                    for ci in ov_idx:
                        if lab[ci] == 0.0: lab[ci] = -1.0          # also mask unlabeled entries in has_unidentified segments
                self.ext_label_lookup[key] = lab
            print(f"  ext_label_lookup: {len(self.ext_label_lookup)} segments (all positive+negative, {len(ov_idx)} overlapping species supervised)")
        self.noisy_pool           = noisy_pool
        self.noisy_mixup_prob     = cfg.get("noisy_mixup_prob", 1.0)
        self.noisy_mixup_weight   = cfg.get("noisy_mixup_weight", 0.5)
        self.noisy_label_sharpen  = cfg.get("noisy_label_sharpen", False)
        self.noisy_sharpen_thresh = cfg.get("noisy_sharpen_thresh", 0.3)

        # Background noise pool
        self.bg_noise_pool = None
        if cfg.get("use_bg_noise", False):
            import os as _os
            bg_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), cfg.get("bg_noise_dir", "background-noise"))
            if _os.path.isdir(bg_dir):
                self.bg_noise_pool = BackgroundNoisePool(bg_dir, sr=cfg["sample_rate"], duration=cfg["duration"])
            else:
                print(f"WARNING: bg_noise_dir not found: {bg_dir}")



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        path      = row["file_path"]
        start_sec = row["start_sec"]

        n_segments = self.cfg.get("n_segments") or self.cfg["duration"] // 5

        if start_sec >= 0:
            # soundscape segment
            info = sf.info(str(path))
            sr   = info.samplerate
            seg_dur = row["end_sec"] - start_sec
            duration = self.cfg["duration"]

            if n_segments == 1:
                # Single-segment mode: support random cropping of merged segments
                if self.is_train and seg_dur > duration:
                    start_sec = start_sec + random.random() * (seg_dur - duration)
            else:
                # Multi-segment mode: pick a random start and read n_segments consecutive 5s segments
                # start_sec must be on the 5s grid (ignoring merges)
                start_sec = int(row["start_sec"])
                # A 60s file has at most 12 segments; start range is [0, 12-n_segments] * 5
                max_start_slot = max(0, 12 - n_segments)
                if self.is_train and not bool(row.get("is_ext_neo", False)):
                    # ext_neotropical skips the random slot (start_sec is the anchor);
                    #   other SC rows (labeled soundscapes) use a random slot for data augmentation
                    slot = random.randint(0, max_start_slot)
                    start_sec = slot * 5
                elif bool(row.get("is_ext_neo", False)):
                    # ext: place the positive segment at a random position within the n_seg window (first/middle/last), guaranteeing the window contains the positive label while maximizing crop diversity
                    #   window start_slot ∈ [pos in last segment, pos in first segment]: lo=pos-(n_seg-1), hi=pos, both clipped to [0, max_start_slot]
                    pos_slot = int(row["start_sec"]) // 5
                    lo = max(0, pos_slot - (n_segments - 1))
                    hi = min(pos_slot, max_start_slot)
                    start_sec = (random.randint(lo, hi) if (self.is_train and hi >= lo) else min(pos_slot, max_start_slot)) * 5
                # start_sec is now either a random start on the 5s grid (labeled SC) or the ext anchor

            frame_offset  = int(start_sec * sr)
            n_samples_src = int(self.n_samples * sr / self.cfg["sample_rate"])
            data, _ = sf.read(str(path), start=frame_offset, frames=n_samples_src,
                              dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.mean(axis=1))
            if sr != self.cfg["sample_rate"]:
                wav = torchaudio.functional.resample(wav, sr, self.cfg["sample_rate"])
            wav, _ = pad_or_crop(wav, self.n_samples, pad_type="left" if self.is_train else "head")
        else:
            # train_audio: random crop during training, center crop (duration length) during validation
            info          = sf.info(path)
            total_frames  = info.frames
            sr            = info.samplerate
            n_samples_src = int(self.n_samples * sr / self.cfg["sample_rate"])
            if total_frames <= n_samples_src:
                data, _ = sf.read(path, dtype="float32", always_2d=True)
            else:
                if self.is_train:
                    start = random.randint(0, total_frames - n_samples_src)
                    data, _ = sf.read(path, start=start, frames=n_samples_src,
                                      dtype="float32", always_2d=True)
                else:
                    start = (total_frames - n_samples_src) // 2     # center crop
                    data, _ = sf.read(path, start=start, frames=n_samples_src,
                                      dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.mean(axis=1))
            if sr != self.cfg["sample_rate"]:
                wav = torchaudio.functional.resample(wav, sr, self.cfg["sample_rate"])
            wav, _ = pad_or_crop(wav, self.n_samples, pad_type="left" if self.is_train else "head")

        # Labels
        def _parse_label(labels_str):
            lbl = torch.zeros(self.n_classes)
            for sp in str(labels_str).split(";"):
                sp = sp.strip()
                sp_idx = self.species_to_idx.get(sp)
                if sp_idx is not None:
                    lbl[sp_idx] = 1.0
            return lbl

        _is_ext = bool(row.get("is_ext_neo", False))
        if n_segments == 1:
            if _is_ext:
                # ext_neotropical: precomputed 234-dim label (with -1 mask), selective-BCE
                label = self.ext_label_lookup.get((Path(path).name, int(start_sec)), torch.full((self.n_classes,), -1.0))
            else:
                label = _parse_label(row["labels"])
        elif start_sec >= 0:
            # soundscape multi-segment: a separate label per 5s (looked up from sc_labels)
            labels_list = []
            fname = Path(path).name
            for i in range(n_segments):
                seg_sec = int(start_sec + i * 5)
                key = (fname, seg_sec)
                if _is_ext:
                    # ext_neotropical: use the precomputed 234-dim label (with -1 mask) for selective-BCE
                    labels_list.append(self.ext_label_lookup.get(key, torch.full((self.n_classes,), -1.0)))
                else:
                    # Look up the label from build_combined_df (the row with the same file and start_sec in train_df)
                    lbl_str = self.sc_label_lookup.get(key, "")
                    labels_list.append(_parse_label(lbl_str))
            label = torch.stack(labels_list)  # (n_segments, n_classes)
        else:
            # train_audio multi-segment: all share the same label
            lbl = _parse_label(row["labels"])
            label = lbl.unsqueeze(0).expand(n_segments, -1).clone()

        # NoisyPool MixUp (sample level: mix the current sample with a NoisyPool soundscape).
        # noisy_mixup_clip_only=True: only clips participate; False: all samples participate.
        noisy_mix_ok = (not self.cfg.get("noisy_mixup_clip_only", False)
                    or row.get("data_type", "clip") == "clip")
        # ext_neotropical samples are excluded from noisy_pool mixup: label max(ext, noisy) would
        #   break the selective mask (the -1 mask entries and absence-0 entries get overwritten by
        #   the noisy pseudo-label, corrupting the true-annotation semantics); protect ext labels + mask.
        if _is_ext:
            noisy_mix_ok = False
        if self.is_train and self.noisy_pool is not None and noisy_mix_ok and random.random() < self.noisy_mixup_prob:
            # In multi-segment mode let NoisyPool also assemble labels for n_seg consecutive 5s segments, avoiding broadcast sharing.
            p_wavs, p_labels = self.noisy_pool.sample_batch(1, n_seg=n_segments)
            p_wav   = torch.from_numpy(p_wavs[0])
            p_label = torch.from_numpy(p_labels[0])
            if self.noisy_label_sharpen:
                thr     = self.noisy_sharpen_thresh
                power   = float(self.cfg.get("noisy_sharpen_power", 2.0))
                p_label = p_label * (p_label > thr).float() + p_label ** power
                p_label = p_label.clamp(0.0, 1.0)
            # Main-side weight: alpha>0 uses a per-sample random Beta(alpha, alpha); otherwise a fixed noisy_mixup_weight
            alpha = self.cfg.get("noisy_mixup_alpha", 0.0)
            if alpha > 0:
                w = float(torch.distributions.Beta(torch.tensor(float(alpha)),
                                                     torch.tensor(float(alpha))).sample().item())
            else:
                w = self.noisy_mixup_weight
            wav_max = wav.abs().amax().clamp(min=1e-7)
            p_max   = p_wav.abs().amax().clamp(min=1e-7)
            wav     = w * (wav / wav_max) + (1 - w) * (p_wav / p_max)
            # Label merge uses element-wise max (the maximum between targets is taken in all MixUps).
            label = torch.maximum(label, p_label)

        # Gain augmentation
        _aug_prob = self.cfg.get("aug_prob", 0.5)   # probabilistic gating (each sample rolls independently)
        if self.is_train and self.cfg.get("use_gain_aug", False) and random.random() < _aug_prob:
            gain_db = random.uniform(self.cfg.get("gain_min_db", -6.0), self.cfg.get("gain_max_db", 6.0))
            wav = wav * (10 ** (gain_db / 20))

        # Background noise augmentation: overlay nocall/environment audio (no label change)
        if self.is_train and self.bg_noise_pool is not None and random.random() < self.cfg.get("bg_noise_prob", 0.5):
            noise = self.bg_noise_pool.sample()
            if len(noise) != len(wav):
                noise = noise[:len(wav)] if len(noise) > len(wav) else F.pad(noise, (0, len(wav) - len(noise)))
            snr_db = random.uniform(self.cfg.get("bg_noise_snr_min", 3.0), self.cfg.get("bg_noise_snr_max", 15.0))
            wav_rms = wav.pow(2).mean().sqrt().clamp(min=1e-7)
            noise_rms = noise.pow(2).mean().sqrt().clamp(min=1e-7)
            target_noise_rms = wav_rms / (10 ** (snr_db / 20))
            wav = wav + noise * (target_noise_rms / noise_rms)

        # Gaussian noise augmentation (additive white noise)
        if self.is_train and self.cfg.get("use_gaussian_noise", False) and random.random() < _aug_prob:
            snr_db = random.uniform(self.cfg.get("gn_snr_min", 10.0), self.cfg.get("gn_snr_max", 30.0))
            wav_rms = wav.pow(2).mean().sqrt().clamp(min=1e-7)
            target_noise_rms = wav_rms / (10 ** (snr_db / 20))
            noise = torch.randn_like(wav) * target_noise_rms
            wav = wav + noise

        # Pink noise augmentation (1/f spectrum)
        if self.is_train and self.cfg.get("use_pink_noise", False):
            snr_db = random.uniform(self.cfg.get("pn_snr_min", 10.0), self.cfg.get("pn_snr_max", 30.0))
            N = len(wav)
            # FFT-based pink noise: white noise shaped by 1/sqrt(f)
            white = torch.randn(N)
            spectrum = torch.fft.rfft(white)
            freqs = torch.arange(len(spectrum), dtype=torch.float32)
            freqs[0] = 1.0  # avoid div-by-zero at DC
            spectrum = spectrum / freqs.sqrt()
            pink = torch.fft.irfft(spectrum, n=N)
            pink = pink / pink.abs().amax().clamp(min=1e-7)  # normalize to [-1, 1]
            wav_rms = wav.pow(2).mean().sqrt().clamp(min=1e-7)
            target_noise_rms = wav_rms / (10 ** (snr_db / 20))
            pink_rms = pink.pow(2).mean().sqrt().clamp(min=1e-7)
            wav = wav + pink * (target_noise_rms / pink_rms)

        # Band-limited pink noise augmentation (targets the SC-XC spectral gap)
        # Adds 1/sqrt(f) pink noise only within the [fmin, fmax] band, simulating the 8-16kHz insect/broadband ambient sound in SC
        if self.is_train and self.cfg.get("use_bl_pink_noise", False):
            if random.random() < self.cfg.get("bl_pink_prob", 0.5):
                fmin = self.cfg.get("bl_pink_fmin", 8000.0)
                fmax = self.cfg.get("bl_pink_fmax", 16000.0)
                snr_db = random.uniform(
                    self.cfg.get("bl_pink_snr_min", 5.0),
                    self.cfg.get("bl_pink_snr_max", 15.0))
                N = len(wav)
                sr = self.cfg.get("sample_rate", 32000)
                # FFT-domain band-limited pink: active only within [fmin, fmax], weighted by 1/sqrt(f)
                white = torch.randn(N)
                spectrum = torch.fft.rfft(white)
                n_bins = len(spectrum)
                freqs = torch.arange(n_bins, dtype=torch.float32) * sr / N
                mask = (freqs >= fmin) & (freqs <= fmax)
                pink_weights = torch.zeros(n_bins)
                # In-band 1/sqrt(f) shape (div-by-zero protection near DC; the mask already excludes f<fmin so freqs[mask] >= fmin > 0)
                pink_weights[mask] = 1.0 / freqs[mask].sqrt()
                spectrum = spectrum * pink_weights
                pink = torch.fft.irfft(spectrum, n=N)
                # Normalize, then add according to SNR
                pink_max = pink.abs().amax().clamp(min=1e-7)
                pink = pink / pink_max
                wav_rms = wav.pow(2).mean().sqrt().clamp(min=1e-7)
                target_noise_rms = wav_rms / (10 ** (snr_db / 20))
                pink_rms = pink.pow(2).mean().sqrt().clamp(min=1e-7)
                wav = wav + pink * (target_noise_rms / pink_rms)

        # Time shift augmentation (circular shift)
        if self.is_train and self.cfg.get("use_time_shift", False) and random.random() < _aug_prob:
            max_shift = int(len(wav) * self.cfg.get("time_shift_max_frac", 0.5))
            shift = random.randint(-max_shift, max_shift)
            wav = torch.roll(wav, shifts=shift, dims=0)

        # Mark whether this is an XC clip (soundscape segments do not take part in batch-level MixUp)
        is_clip         = torch.tensor(row.get("data_type", "clip") == "clip", dtype=torch.bool)
        is_distill_only = torch.tensor(False, dtype=torch.bool)
        return wav, label, is_clip, is_distill_only


# ============================================================
# GPU Batch Mel
# ============================================================

def normalize_mel(mel, mel_norm="zscore_minmax"):
    """mel (B, n_mels, T') → normalized mel, per-sample operation.
    - "zscore_minmax": first z-score to unify contrast, then min-max into [0,1]
    - "minmax": direct min-max → [0,1]
    - "zscore": direct z-score (legacy, unbounded range)
    """
    flat = mel.reshape(mel.size(0), -1)
    if mel_norm == "zscore_minmax":
        # Step 1: per-sample z-score (unify contrast)
        mean = flat.mean(dim=1)[:, None, None]
        std  = flat.std(dim=1)[:, None, None] + 1e-6
        mel  = (mel - mean) / std
        # Step 2: per-sample min-max → [0, 1] (unify value range)
        flat2    = mel.reshape(mel.size(0), -1)
        norm_min = flat2.min(dim=1)[0][:, None, None]
        norm_max = flat2.max(dim=1)[0][:, None, None]
        mel = (mel - norm_min) / (norm_max - norm_min + 1e-7)
    elif mel_norm == "minmax":
        mel_min = flat.min(dim=1)[0][:, None, None]
        mel_max = flat.max(dim=1)[0][:, None, None]
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
    else:
        # zscore
        mean = flat.mean(dim=1)[:, None, None]
        std  = flat.std(dim=1)[:, None, None] + 1e-6
        mel  = (mel - mean) / std
    return mel


def batch_wav_to_image(wavs, mel_tr, db_tr, img_size, in_chans=3, mel_norm="zscore_minmax", db_scope="sample"):
    """wavs (B, T) → normalized spectrogram (B, in_chans, H, W), computed in batch on the GPU
    db_scope:
      - "sample": db_tr takes 4D input → per-sample ref_max (deterministic, default for val/infer)
      - "batch":  db_tr takes 3D input → batch-global ref_max (acts as implicit augmentation during training)
    mel_tr: a single MelSpectrogram → standard 1 channel (then repeated to in_chans)
    """
    # Single-mel path
    mel = mel_tr(wavs)                          # (B, n_mels, T')
    # Fake-channel trick: torchaudio T.AmplitudeToDB does a batch-global reduce on 3D (B,F,T) input;
    # adding a dim to make it 4D (B,1,F,T) takes the per-sample branch → db reference = each sample's own max (per-duration scale)
    if db_scope == "batch":
        mel = db_tr(mel)                        # 3D → batch-global ref_max (training augmentation)
    else:
        mel = db_tr(mel.unsqueeze(1)).squeeze(1)   # 4D → per-sample ref_max (deterministic)
    mel = normalize_mel(mel, mel_norm)
    mel = mel.unsqueeze(1)                      # (B, 1, n_mels, T')
    # img_size: int=square resize, tuple=(H,W), 0=no resize
    if isinstance(img_size, (list, tuple)):
        mel = torch.nn.functional.interpolate(
            mel, size=tuple(img_size), mode="bilinear", align_corners=False)
    elif img_size > 0:
        mel = torch.nn.functional.interpolate(
            mel, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return mel.repeat(1, in_chans, 1, 1)        # (B, in_chans, H, W)


# ============================================================
# SpecAugment (mel-domain data augmentation during training)
# ============================================================

def spec_augment(mel, freq_mask_param=30, time_mask_param=40, num_freq_masks=2, num_time_masks=2):
    """SpecAugment: randomly mask frequency and time bands on the mel spectrogram.
    mel: (B, C, n_mels, T) — normalized mel spectrogram
    Called during training, not during inference. Masked value is set to 0 (the minimum after min-max normalization).
    """
    B, C, n_mels, T = mel.shape
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        mel[:, :, f0:f0+f, :] = 0
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, T - t))
        mel[:, :, :, t0:t0+t] = 0
    return mel


def random_filter_wav(wavs, min_db=-20.0, n_bands=4, n_fft=1024, hop_length=512):
    """Equivalent implementation of RandomFiltering (wave-level random EQ).
    Pipeline: STFT → multiply along the frequency axis by a random EQ curve linearly interpolated from 4 points (each point dB in [min_db, 0]) → iSTFT → peak-normalize.
    Each sample samples a different EQ curve independently. Called before mixup.
    wavs: (B, N) float32. Returns the same shape.
    """
    device = wavs.device
    window = torch.hann_window(n_fft, device=device)
    # STFT: (B, F, T_frames)
    spec = torch.stft(wavs, n_fft=n_fft, hop_length=hop_length,
                      win_length=None, window=window, center=True,
                      pad_mode="reflect", normalized=False, return_complex=True)
    B, F, _ = spec.shape
    # 4 random dB control points → linearly interpolated along the F dimension to all bins
    filter_points = torch.rand(B, n_bands, device=device) * min_db        # dB ∈ [min_db, 0]
    filter_coeffs = nn.functional.interpolate(
        filter_points.unsqueeze(1), mode="linear", size=F
    ).squeeze(1)                                                          # (B, F)
    filter_coeffs = 10 ** (filter_coeffs / 20)                            # dB → amplitude
    spec = spec * filter_coeffs.unsqueeze(-1)                             # (B, F, T_frames)
    # iSTFT → wave; pass length explicitly to match the input wav length
    out = torch.istft(spec, n_fft=n_fft, hop_length=hop_length,
                      win_length=None, window=window, center=True,
                      normalized=False, length=wavs.shape[-1],
                      return_complex=False)
    # peak normalize
    peak = out.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    return out / peak


# ============================================================
# MixUp
# ============================================================

def mixup_wav_batch(wavs, labels, mixup_eligible=None,
                    exclude_as_partner=None, prob=1.0, n_max=2,
                    main_weight=0.0, alpha=1.0):
    """MixUp in the raw-audio domain (with multi-species extension).
    - Absmax-normalizes each wav before mixing (so all sources have comparable energy).
    - Mixes before the mel transform, preserving frame-level species energy contrast so the attention head can learn discriminative features.
    - mixup_eligible: (B,) bool; rows that are False get perm[i]=i (left exclusion: the row is not mixed).
    - exclude_as_partner: (B,) bool; True indices never appear in the perm values (right exclusion: not used as anyone's partner).
                          Typical use: doubly excluded rows set both mixup_eligible=False and exclude_as_partner=True.
    - prob: probability that each sample triggers MixUp (1.0 = mix everything).
    - n_max: maximum number of sources mixed (2 = classic two-species, 3+ = multi-species, narrowing the domain gap to 4+ species soundscapes).
    - main_weight: weight of the main species. 0 = random by lam; >0 = main gets main_weight, the rest split (1-main_weight)/(n-1).
    - alpha: per-sample Beta/Dirichlet shape parameter (only active when main_weight=0).
             alpha<1 (e.g. 0.4) is U-shaped toward extremes; alpha=1 is uniform; alpha>1 is centered.
    Returns: (wavs_mixed, labels_mixed, perms_list, weights_list)
      perms_list:   list of length n_mix-1, each element a (B,) perm tensor
      weights_list: list of length n_mix; [0] = main weight, [1:] = per-perm weights
        - lam mode: each element is a (B,) tensor (per-sample random)
        - main_weight mode: each element is a scalar
    """
    # absmax normalization (each wav normalized independently)
    wav_max = wavs.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
    wavs_norm = wavs / wav_max

    B        = wavs.size(0)
    identity = torch.arange(B, device=wavs.device)

    # Randomly choose how many sources to mix for this batch (uniform across the batch to avoid inconsistency)
    n_mix = random.randint(2, n_max)

    # Weights (priority: main_weight > per-sample lam-Beta/Dirichlet)
    if main_weight > 0 and n_mix > 1:
        # Fixed unequal weights (batch level, shared across the whole batch)
        w_main  = main_weight
        w_other = (1.0 - main_weight) / (n_mix - 1)
        weights_list = [w_main] + [w_other] * (n_mix - 1)
    else:
        # Per-sample random weights: n=2 uses Beta(alpha, alpha), n>2 uses Dirichlet(alpha,...,alpha); maximizes diversity
        if n_mix == 2:
            alpha_t = torch.tensor(float(alpha), device=wavs.device)
            lam = torch.distributions.Beta(alpha_t, alpha_t).sample((B,))   # (B,)
            weights_list = [lam, 1.0 - lam]
        else:
            conc = torch.full((n_mix,), float(alpha), device=wavs.device)
            w = torch.distributions.Dirichlet(conc).sample((B,))            # (B, n_mix)
            weights_list = [w[:, i].contiguous() for i in range(n_mix)]

    # Generate n_mix-1 perms (the first source is the sample itself)
    # Right exclusion: rows with exclude_as_partner=True must not appear in the perm values → do a without-replacement randperm within the ~exclude subset,
    # keeping perm[i] = identity for excluded rows. With no right exclusion this reduces to a randperm over the whole batch.
    perms = []
    has_right_exclude = exclude_as_partner is not None and exclude_as_partner.any()
    if has_right_exclude:
        allowed_idx = torch.where(~exclude_as_partner)[0]   # (M,)
    for _ in range(n_mix - 1):
        if has_right_exclude:
            # Without-replacement permutation within the subset: sub_perm reshuffles allowed_idx internally and writes back to original positions
            perm = identity.clone()
            if len(allowed_idx) >= 1:
                sub_perm = torch.randperm(len(allowed_idx), device=wavs.device)
                perm[allowed_idx] = allowed_idx[sub_perm]
        else:
            perm = torch.randperm(B, device=wavs.device)
        if prob < 1.0:
            # Each sample independently decides with probability prob whether to take part in MixUp; the rest keep the identity permutation
            do_mix = torch.rand(B, device=wavs.device) < prob
            perm   = torch.where(do_mix, perm, identity)
        if mixup_eligible is not None:
            # Left exclusion: rows with mixup_eligible=False keep the identity permutation (paired with themselves)
            perm = torch.where(mixup_eligible, perm, identity)
        perms.append(perm)

    # Mix + max label: scalar weights multiply directly, (B,) tensor weights broadcast to (B, 1)
    def _wmul(w, x):
        if torch.is_tensor(w) and w.dim() >= 1:
            return w.view(-1, *([1] * (x.dim() - 1))) * x
        return w * x

    wavs_mixed   = _wmul(weights_list[0], wavs_norm)
    labels_mixed = labels.clone()
    for w, perm in zip(weights_list[1:], perms):
        wavs_mixed   = wavs_mixed + _wmul(w, wavs_norm[perm])
        # Max label: any species present in any clip gets a hard label of 1, avoiding a soft 0.5 that keeps the model "uncertain"
        labels_mixed = torch.maximum(labels_mixed, labels[perm])

    return wavs_mixed, labels_mixed, perms, weights_list


# ============================================================
# Soft-label pools (for distillation)
# ============================================================


class BackgroundNoisePool:
    """Background noise pool: randomly sample nocall/environment audio to overlay on training samples.
    Does not change labels (noise has no species annotation).
    """
    def __init__(self, noise_dir, sr=32000, duration=5):
        self.sr = sr
        self.n_samples = sr * duration
        noise_dir = Path(noise_dir)
        self.files = []
        for ext in ["*.ogg", "*.wav", "*.mp3", "*.flac"]:
            self.files.extend(list(noise_dir.rglob(ext)))
        print(f"BackgroundNoisePool: {len(self.files)} noise files from {noise_dir}")

    def sample(self):
        """Sample a random noise segment, return (n_samples,) float32 tensor"""
        path = random.choice(self.files)
        try:
            data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.mean(axis=1))
            if native_sr != self.sr:
                wav = torchaudio.functional.resample(wav, native_sr, self.sr)
        except Exception:
            return torch.zeros(self.n_samples)
        if len(wav) >= self.n_samples:
            start = random.randint(0, len(wav) - self.n_samples)
            wav = wav[start:start + self.n_samples]
        else:
            wav = F.pad(wav, (0, self.n_samples - len(wav)))
        return wav


# ---- time_smooth_v2 helpers ----
# Same formula as the distillation side, used to apply smooth_v2 after a multi-npz blend, matching the "blend → smooth" behavior.
def _load_texture_mask_for_smooth(cfg, n_classes):
    """load taxonomy → return (n_classes,) bool: True for Insecta/Amphibia, False for Aves/Mammalia/Reptilia"""
    tax_path = os.path.join(cfg["data_dir"], "taxonomy.csv")
    tax = pd.read_csv(tax_path)
    tax["primary_label"] = tax["primary_label"].astype(str)
    label_cols = tax["primary_label"].tolist()
    assert len(label_cols) == n_classes, f"taxonomy {len(label_cols)} != n_classes {n_classes}"
    tex_set = set(tax[tax["class_name"].isin(["Insecta", "Amphibia"])]["primary_label"])
    return np.array([l in tex_set for l in label_cols], dtype=bool)


def _smooth_v2_texture_np(p, mask, alpha):
    if alpha <= 0 or not mask.any(): return p
    x = p[:, mask]
    prev_x = np.concatenate([x[:1], x[:-1]], axis=0)
    next_x = np.concatenate([x[1:], x[-1:]], axis=0)
    out = p.copy()
    out[:, mask] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return out


def _smooth_v2_event_np(p, mask, alpha):
    if alpha <= 0 or not mask.any(): return p
    x = p[:, mask]
    prev_x = np.concatenate([x[:1], x[:-1]], axis=0)
    next_x = np.concatenate([x[1:], x[-1:]], axis=0)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    out = p.copy()
    out[:, mask] = (1.0 - alpha) * x + alpha * local_max
    return out


def _smooth_v2_adaptive_delta_np(p, base_alpha):
    if base_alpha <= 0: return p
    out = p.copy()
    n = len(p)
    for i in range(1, n - 1):
        conf = p[i].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p[i-1] + p[i+1]) / 2.0
        out[i] = (1.0 - a) * p[i] + a * neighbor_avg
    return out


def _time_smooth_v2_per_file_np(probs_per_file, texture_mask,
                                 texture_alpha=0.35, event_alpha=0.15, delta_alpha=0.20):
    """(12, n_cls) per-file → smooth_v2 (same formula as the distillation script's time_smooth_v2)"""
    out = _smooth_v2_texture_np(probs_per_file, texture_mask, texture_alpha)
    out = _smooth_v2_event_np(out, ~texture_mask, event_alpha)
    out = _smooth_v2_adaptive_delta_np(out, delta_alpha)
    return out


def _load_teacher_scores(legacy_npz_path, cfg, fold_id=0, meta_df=None):
    """Main entry for loading teacher prob scores, supporting single-npz / multi-npz blend.

    Prefers cfg["teacher_npz_list"] (e.g. [("rel_path1", 0.6), ("rel_path2", 0.4)]):
      - multiple npz, weighted blend in prob space
      - when cfg["teacher_smooth_after_blend"]=True (default), runs time_smooth_v2 per-file after the blend
        (mathematically equivalent to the internal "blend → overlap-avg → smooth")
    If the list is empty/None → fall back to a legacy single npz (legacy_npz_path) with no extra smoothing.

    Returns: (N, n_cls) float32 scores
    """
    npz_list = cfg.get("teacher_npz_list", None)
    if not npz_list:
        # legacy single npz
        _all = np.load(legacy_npz_path)["scores"]
        if _all.ndim == 3:
            return _all[fold_id].astype(np.float32)
        return _all.astype(np.float32)

    # multi-npz blend
    base_dir = Path(cfg["data_dir"]).parent
    weights = np.array([float(w) for _, w in npz_list], dtype=np.float64)
    weights = weights / weights.sum()
    print(f"Teacher: weighted blend of {len(npz_list)} npz "
          f"(normalized weights={[round(float(w), 4) for w in weights]})")

    scores = None
    for (rel_path, _), w in zip(npz_list, weights):
        arr = np.load(base_dir / rel_path)["scores"]
        if arr.ndim == 3:
            arr = arr[fold_id]
        arr = arr.astype(np.float32)
        scores = (w * arr) if scores is None else (scores + w * arr)
    scores = scores.astype(np.float32)

    if cfg.get("teacher_smooth_after_blend", True):
        N, n_cls = scores.shape
        assert N % 12 == 0, f"scores N={N} not divisible by 12 (expected n_files * 12 segments)"
        n_files = N // 12
        texture_mask = _load_texture_mask_for_smooth(cfg, n_cls)
        scores_3d = scores.reshape(n_files, 12, n_cls)
        for fi in range(n_files):
            scores_3d[fi] = _time_smooth_v2_per_file_np(scores_3d[fi], texture_mask)
        scores = scores_3d.reshape(N, n_cls).astype(np.float32)
        print(f"Teacher: time_smooth_v2 applied after blend ({n_files} files × 12 seg)")

    # === Extra post (off by default) ===
    use_file_post = bool(cfg.get("teacher_file_post", False))
    if use_file_post:
        N, n_cls = scores.shape
        assert N % 12 == 0, f"scores N={N} not divisible by 12 (expected n_files * 12 segments)"
        n_files = N // 12
        scores_3d = scores.reshape(n_files, 12, n_cls).astype(np.float32)

        # File-level top-prob multiplication: per file, 12 chunks * max(chunk, axis=time)
        top_prob = scores_3d.max(axis=1, keepdims=True)                # (n_files, 1, n_cls)
        scores_3d = scores_3d * top_prob
        print(f"Teacher: file-level top-prob mult applied ({n_files} files × 12 seg)")

        scores = scores_3d.reshape(N, n_cls).astype(np.float32)

    return scores


class NoisyPool:
    """Precomputed soft-label pool, used for Noisy Student MixUp.
    Each sample_batch() call randomly samples B soundscape 5s windows and their soft labels.
    Files with existing expert annotations are excluded to avoid soft labels interfering with supervised samples.
    """

    def __init__(self, npz_path, meta_csv, soundscape_dir, labeled_stems, cfg,
                 conf_weighted=True, min_conf=0.0, fold_id=0):
        meta   = pd.read_csv(meta_csv)
        # Load teacher prob via the helper: cfg["teacher_npz_list"] takes priority (multi blend + smooth);
        # otherwise use npz_path (legacy single, same as cfg["teacher_npz"]).
        scores = _load_teacher_scores(npz_path, cfg, fold_id=fold_id, meta_df=meta)
        print(f"NoisyPool: teacher scores loaded, shape={scores.shape}")

        # Exclude files with existing expert annotations
        mask        = ~meta["stem"].isin(labeled_stems)
        meta        = meta[mask].reset_index(drop=True)
        scores      = scores[mask.values]

        # Optional: filter out low-confidence segments
        if min_conf > 0.0:
            keep   = scores.astype(np.float32).max(axis=1) >= min_conf
            meta   = meta[keep].reset_index(drop=True)
            scores = scores[keep]

        # Power transform: labels = labels ** power (exponent>1 suppresses noise)
        power = cfg.get("noisy_power", 1.0)
        if power != 1.0:
            scores = np.clip(scores.astype(np.float32), 0, 1) ** power
            print(f"NoisyPool: applied power transform (exponent={power:.2f})")

        self.scores         = scores.astype(np.float32)  # (N, 234) float32
        self.soundscape_dir = Path(soundscape_dir)
        self.n_samples      = cfg["sample_rate"] * cfg["duration"]   # 160000
        self.sr             = cfg["sample_rate"]
        self.n              = len(meta)

        # Convert meta to numpy to avoid row-by-row pandas iloc access
        self.filenames   = meta["filename"].values       # (N,) str
        self.window_idxs = meta["window_idx"].values.astype(np.int32)  # (N,)

        # (filename, window_idx) → row idx lookup, used to assemble labels from adjacent 5s segments when n_seg>1
        self.lookup = {(self.filenames[i], int(self.window_idxs[i])): i for i in range(self.n)}

        # Cache the sample rate of all soundscape files once at init, eliminating per-call sf.info()
        print(f"NoisyPool: caching soundscape sample rates...")
        unique_fns = np.unique(self.filenames)
        sr_map = {}
        for fn in unique_fns:
            try:
                sr_map[fn] = sf.info(str(self.soundscape_dir / fn)).samplerate
            except Exception:
                sr_map[fn] = self.sr
        self.native_srs = np.array([sr_map[fn] for fn in self.filenames], dtype=np.int32)

        # Confidence-weighted sampling
        # - "segment" (legacy): w_i = max_class_prob_i, independent per-segment weight
        # - "soundscape": the 12 segments of a filename share weight = sum_over_windows(max_class_prob)
        #   Rationale: file-level sum-of-max-probs reflects the overall labeling quality of the soundscape,
        #              so almost-unlabeled segments (low sum) are down-weighted to avoid amplifying noise
        if conf_weighted:
            # File-level sum-of-max-probs weighting
            # The 12 segments of a file share one weight; the file's labeling quality determines its sampling probability
            seg_max      = self.scores.max(axis=1).clip(min=1e-6).astype(np.float64)
            w_per_file   = pd.Series(seg_max).groupby(pd.Series(self.filenames)).transform("sum").to_numpy()
            w            = np.asarray(w_per_file, dtype=np.float64).copy()   # the pandas groupby view may be read-only, so copy first
            w           /= w.sum()
            w[-1]       += 1.0 - w.sum()   # correct floating-point error to ensure exact normalization
            self.weights = w
            print(f"NoisyPool: conf_weighted=soundscape "
                  f"(max weight={w.max():.6f}, min={w.min():.6f}, ratio={w.max()/w.min():.1f}x)")
        else:
            self.weights = None

        print(f"NoisyPool: {self.n} segments available (excluded {len(labeled_stems)} already-labeled files)")

    def sample_batch(self, batch_size, n_seg=1):
        """Randomly sample batch_size (wav, soft_label) pairs, returned as numpy float32 arrays.
        n_seg=1: labels shape (B, n_classes)
        n_seg>1: read a cfg.duration-second wav (≈ n_seg × 5s); labels shape (B, n_seg, n_classes),
                 concatenating soft labels from adjacent 5s segments. The start is base_window_idx;
                 if it exceeds the file range [0, 11] it falls back to max(0, 12-n_seg).
        """
        n_classes = self.scores.shape[1]
        if self.weights is not None:
            indices = np.random.choice(self.n, size=batch_size, replace=True, p=self.weights)
        else:
            indices = np.random.randint(0, self.n, size=batch_size)

        wavs = np.zeros((batch_size, self.n_samples), dtype=np.float32)
        if n_seg == 1:
            labels = self.scores[indices].copy()                        # (B, n_classes)
        else:
            labels = np.zeros((batch_size, n_seg, n_classes), dtype=np.float32)

        for i, idx in enumerate(indices):
            filename   = self.filenames[idx]
            native_sr  = int(self.native_srs[idx])
            base_win   = int(self.window_idxs[idx])
            # In multi-segment mode, adjust the start so [base_win, base_win + n_seg) stays within the file range [0, 11]
            if n_seg > 1 and base_win + n_seg > 12:
                base_win = max(0, 12 - n_seg)
            start_sec  = base_win * 5
            path       = self.soundscape_dir / filename

            try:
                frame_offset  = int(start_sec * native_sr)
                n_frames_src  = int(self.n_samples * native_sr / self.sr)
                data, _       = sf.read(str(path), start=frame_offset,
                                        frames=n_frames_src, dtype="float32", always_2d=True)
                wav = data.mean(axis=1)
                if native_sr != self.sr:
                    wav = torchaudio.functional.resample(
                        torch.from_numpy(wav), native_sr, self.sr).numpy()
                n = len(wav)
                if n < self.n_samples:
                    wav = np.pad(wav, (0, self.n_samples - n))
                else:
                    wav = wav[:self.n_samples]
                wavs[i] = wav
            except Exception:
                pass   # On read failure, keep zero frames (equivalent to a silent segment)

            # Multi-segment label: look up base_win + j to assemble (n_seg, n_classes)
            if n_seg > 1:
                for j in range(n_seg):
                    k = self.lookup.get((filename, base_win + j), -1)
                    if k >= 0:
                        labels[i, j] = self.scores[k]
                    # else: not in pool (filtered out by min_conf), keep 0 placeholder

        return wavs, labels


# ============================================================
# Model
# ============================================================

class GEMFreqPool(nn.Module):
    """Generalized Mean Pooling along the frequency axis (dim=2), mapping (B,C,F',T') → (B,C,T').
    p is learnable, initialized to 3 (empirical); clamp(min=1) prevents collapse to max-pool.
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)


class LSEFreqPool(nn.Module):
    """LogSumExp Pooling along the frequency axis (dim=2), mapping (B,C,F',T') → (B,C,T').
    r is a learnable temperature, clamp(min=0.1) for numerical stability; r→∞ approaches max, r→0 approaches mean, r=1 is smooth aggregation.
    Symmetric to LSEHead.lse_pool (only the aggregation dim differs). Safe for negatives, no ReLU/clamp needed beforehand.
    """
    def __init__(self, r=1.0):
        super().__init__()
        self.r = nn.Parameter(torch.ones(1) * r)

    def forward(self, x):
        r = self.r.clamp(min=0.1)
        F_dim = x.size(2)
        return r * (torch.logsumexp(x / r, dim=2) - math.log(F_dim))


class AttentionHead(nn.Module):
    """Temporal attention aggregation head: (B, C, T') → (B, n_classes) or (B, n_segments, n_classes)
    - dense: nonlinear transform (reduce to hidden_dim)
    - att_conv: per-class attention over time frames (tanh + temperature + softmax/sigmoid over time)
    - cls_conv: per-class per-frame score
    - n_segments > 1: split the time dim into n_segments segments, attention-pooled independently within each
    - output: clip-level logits (or segment-level logits) + timewise logits (B, T', n_classes)

    Tunable parameters (all passed through from SEDModel):
    - hidden_dim: intermediate hidden dim (default 512)
    - dropout: dense-layer dropout (default 0.5, heavy)
    - temperature: attention softmax temperature; >1 more uniform, <1 more peaky (only under softmax)
    - activation: "softmax" (competitive, sums to 1, default) or "sigmoid" (independent, multiple frames can attend at once)
    """
    def __init__(self, in_dim, n_classes, n_segments,
                 hidden_dim=512, dropout=0.5, temperature=1.0, activation="softmax",
                 per_class_temperature=False,
                 att_conv_kernel=1,
                 use_class_interaction=False,
                 class_interaction_type="linear",
                 class_interaction_hidden=128,
                 class_interaction_hidden2=None,
                 class_interaction_n_heads=4,
                 class_interaction_position="clipwise",
                 class_interaction_cross_seg_mode="none"):
        super().__init__()
        assert activation in ("softmax", "sigmoid"), f"activation must be softmax|sigmoid, got {activation}"
        self.n_segments  = n_segments
        self.activation  = activation
        self.per_class_temperature = per_class_temperature
        if per_class_temperature:
            # per-class τ_c (learn log_τ to keep τ>0). Initial log_τ = log(temperature) → all classes start at temperature, equivalent to the scalar case
            self.log_temperature = nn.Parameter(torch.full((n_classes,), math.log(float(temperature))))
        else:
            self.temperature = float(temperature)
        # reduce to hidden_dim + heavy dropout (ReLU activation)
        self.dense = nn.Sequential(
            nn.Dropout(dropout / 2),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # att_conv kernel_size is configurable: >1 gives local temporal smoothing within a segment
        # padding keeps the time dim unchanged (so chunk splits stay equal-length)
        _att_pad = (att_conv_kernel - 1) // 2
        self.att_conv = nn.Conv1d(hidden_dim, n_classes, kernel_size=att_conv_kernel, padding=_att_pad)

        self.cls_conv = nn.Conv1d(hidden_dim, n_classes, kernel_size=1)

        # Class interaction: residual class-class refinement added after the clipwise pool, three variants:
        # "linear":   Linear(n_cls, n_cls) zero init                     ~55K params
        # "mlp":      Linear→GELU→Linear (last layer zero init)          ~60K params
        # "selfattn": class_embed → inject scalar → self-attn → proj    ~150K params
        # Only enabled for SC samples that were not mixed up (controlled by a mask in forward), so W learns only real co-occurrence
        self.use_class_interaction = use_class_interaction
        self.class_interaction_type = class_interaction_type
        self.class_interaction_position = class_interaction_position
        self.class_interaction_cross_seg_mode = class_interaction_cross_seg_mode
        # _ci_use_joint: whether to enable cross-segment joint mode (requires both clipwise and n_seg>1)
        _ci_use_joint = (n_segments > 1 and class_interaction_position == "clipwise")
        if use_class_interaction:
            if class_interaction_type == "linear":
                # With n_seg>1 + clipwise position, enable joint cross-segment: Linear(n_seg*n_cls, n_seg*n_cls)
                # otherwise (n_seg=1 or frame_cls position) it degenerates to the older Linear(n_cls, n_cls) behavior
                if _ci_use_joint:
                    _ci_dim = n_classes * n_segments
                else:
                    _ci_dim = n_classes
                self.class_interaction = nn.Linear(_ci_dim, _ci_dim, bias=False)
                nn.init.zeros_(self.class_interaction.weight)
            elif class_interaction_type == "mlp":
                # Three cross-seg modes (only effective when joint is enabled, else all degenerate to standard mlp):
                #   "none": independent mlp per seg, input/output n_cls
                #   "flat": flatten input n_seg*n_cls → mlp → output n_seg*n_cls (~120K params, does not preserve symmetry)
                #   "pool": input cat[self, mean_pool] = 2*n_cls → mlp → output n_cls (~90K params, preserves symmetry)
                if _ci_use_joint and class_interaction_cross_seg_mode == "flat":
                    _in, _out = n_classes * n_segments, n_classes * n_segments
                elif _ci_use_joint and class_interaction_cross_seg_mode == "pool":
                    _in, _out = n_classes * 2, n_classes
                else:
                    _in, _out = n_classes, n_classes
                # hidden2=None: 1 hidden layer, older behavior Linear→GELU→Linear
                # hidden2 set: 2 hidden layers Linear→GELU→Linear→GELU→Linear (bottleneck/wider depending on hidden2 vs hidden)
                if class_interaction_hidden2 is None:
                    self.class_interaction = nn.Sequential(
                        nn.Linear(_in, class_interaction_hidden),
                        nn.GELU(),
                        nn.Linear(class_interaction_hidden, _out),
                    )
                else:
                    self.class_interaction = nn.Sequential(
                        nn.Linear(_in, class_interaction_hidden),
                        nn.GELU(),
                        nn.Linear(class_interaction_hidden, class_interaction_hidden2),
                        nn.GELU(),
                        nn.Linear(class_interaction_hidden2, _out),
                    )
                # last layer zero init → residual starts at 0, initially a no-op; the nonlinear co-occurrence pattern is learned gradually during training
                nn.init.zeros_(self.class_interaction[-1].weight)
                nn.init.zeros_(self.class_interaction[-1].bias)
            elif class_interaction_type == "selfattn":
                # transformer variant: one learnable embedding per class, inject the scalar logit, run self-attention, then project back to a scalar delta
                # only mask=True samples go through self-attn refinement, so the self-attn weights are trained only on SC unmixed samples
                qd = class_interaction_hidden
                self.ci_class_embed = nn.Parameter(torch.randn(n_classes, qd) * 0.02)
                self.ci_inject = nn.Linear(1, qd)
                self.ci_self_attn = nn.MultiheadAttention(qd, num_heads=class_interaction_n_heads, batch_first=True)
                self.ci_proj = nn.Linear(qd, 1)
                nn.init.zeros_(self.ci_proj.weight)
                nn.init.zeros_(self.ci_proj.bias)
            else:
                raise ValueError(f"unknown class_interaction_type: {class_interaction_type}")

    def _apply_attention(self, att_raw, cls):
        """Attention pooling over the last (time) dim: softmax/sigmoid + weighted sum."""
        if self.activation == "softmax":
            if self.per_class_temperature:
                # broadcast per-class τ_c to (1, n_classes, 1) and divide (B, n_classes, T)
                tau = torch.exp(self.log_temperature).view(1, -1, 1)
                att = F.softmax(att_raw / tau, dim=-1)
            else:
                att = F.softmax(att_raw / self.temperature, dim=-1)
            return (att * cls).sum(dim=-1)       # softmax sums to 1, weighted sum
        else:
            # sigmoid: each frame independent in (0,1), not forced to sum to 1; use mean to keep scale stable
            att = torch.sigmoid(att_raw)
            return (att * cls).mean(dim=-1)

    def _apply_class_interaction(self, logits):
        """Unified forward for the three class-interaction variants; input/output shapes match.
        logits: (k, n_cls) or (k, n_seg, n_cls), returns a delta of the same shape.
        """
        if self.class_interaction_type == "linear":
            # clipwise + n_seg>1: flatten across clips → joint Linear → reshape back to (k, n_seg, n_cls)
            # makes the (i_seg0, j_seg1) entries of W learnable, capturing genuine cross-clip class co-occurrence
            # only the clipwise position uses joint; at the frame_cls position logits[1] is T', not n_seg, so it uses the older Linear
            if (self.n_segments > 1 and logits.dim() == 3
                and self.class_interaction_position == "clipwise"):
                B = logits.shape[0]
                flat = logits.reshape(B, -1)                                # (k, n_seg*n_cls)
                delta = self.class_interaction(flat)                        # (k, n_seg*n_cls)
                return delta.reshape(B, self.n_segments, -1)                # (k, n_seg, n_cls)
            return self.class_interaction(logits)
        elif self.class_interaction_type == "mlp":
            # cross_seg_mode controls joint behavior, only effective for clipwise + n_seg>1:
            #   "flat": flatten → mlp → reshape (analogous to linear joint, does not preserve symmetry)
            #   "pool": global=mean(seg), per seg cat[self, global] → mlp (symmetric)
            #   "none": per-seg broadcast (default, degenerates to Linear(n_cls,n_cls))
            if (self.n_segments > 1 and logits.dim() == 3
                and self.class_interaction_position == "clipwise"):
                if self.class_interaction_cross_seg_mode == "pool":
                    g = logits.mean(dim=1, keepdim=True).expand(-1, self.n_segments, -1)  # (B, n_seg, n_cls)
                    mlp_in = torch.cat([logits, g], dim=-1)                       # (B, n_seg, 2*n_cls)
                    return self.class_interaction(mlp_in)                          # (B, n_seg, n_cls)
                if self.class_interaction_cross_seg_mode == "flat":
                    B = logits.shape[0]
                    flat = logits.reshape(B, -1)                                  # (B, n_seg*n_cls)
                    delta = self.class_interaction(flat)                          # (B, n_seg*n_cls)
                    return delta.reshape(B, self.n_segments, -1)                  # (B, n_seg, n_cls)
            return self.class_interaction(logits)
        elif self.class_interaction_type == "selfattn":
            # Handle 3D input: flatten the batch+seg dims for self-attn
            original_shape = logits.shape
            if logits.dim() == 3:
                logits_flat = logits.reshape(-1, original_shape[-1])
            else:
                logits_flat = logits
            bsz = logits_flat.shape[0]
            # Turn each sample into a (n_cls, qd) class-token sequence: embed + injected scalar logit
            tokens = self.ci_class_embed.unsqueeze(0).expand(bsz, -1, -1)         # (k, n_cls, qd)
            tokens = tokens + self.ci_inject(logits_flat.unsqueeze(-1))            # inject scalar
            # classes attend to each other
            tokens = tokens + self.ci_self_attn(tokens, tokens, tokens, need_weights=False)[0]
            # project back to a scalar delta
            delta = self.ci_proj(tokens).squeeze(-1)                               # (k, n_cls)
            if logits.dim() == 3:
                delta = delta.reshape(original_shape)
            return delta

    def _apply_ci_with_mask(self, x, mask):
        """Apply class interaction to x (last dim = n_cls), gated by mask.
        mask=None → the whole batch passes through; mask=tensor → only mask=True samples pass, the rest skip the forward.
        """
        if mask is None:
            return x + self._apply_class_interaction(x)
        if mask.any():
            sel = mask
            refine = self._apply_class_interaction(x[sel])
            x = x.clone()
            x[sel] = x[sel] + refine
        return x

    def forward(self, x, class_interaction_mask=None):
        """class_interaction_mask: (B,) bool tensor or None
            - None: inference / val (the whole batch passes through class interaction)
            - tensor (True entries): at train time, only SC unmixed samples pass (is_sc & is_unmixed)
        """
        # x: (B, T', C)
        x = self.dense(x)                   # (B, T', hidden_dim)
        x = x.permute(0, 2, 1)              # (B, hidden_dim, T')
        att_raw = torch.tanh(self.att_conv(x))  # (B, n_classes, T')
        cls     = self.cls_conv(x)              # (B, n_classes, T')

        # P1: frame_cls interaction (per-frame W refinement on cls (B, n_cls, T) before attention pool)
        # permute so the last dim is class → nn.Linear applies per frame automatically → permute back
        if self.use_class_interaction and self.class_interaction_position == "frame_cls":
            cls_t = cls.permute(0, 2, 1)                # (B, T', n_classes), last dim = n_cls
            cls_t = self._apply_ci_with_mask(cls_t, class_interaction_mask)
            cls   = cls_t.permute(0, 2, 1)              # back to (B, n_classes, T')

        if self.n_segments > 1:
            att_chunks = att_raw.chunk(self.n_segments, dim=-1)       # n_segments × (B, n_cls, T'/n)
            cls_chunks = cls.chunk(self.n_segments, dim=-1)
            seg_logits = [self._apply_attention(att_c, cls_c)
                          for att_c, cls_c in zip(att_chunks, cls_chunks)]
            logits = torch.stack(seg_logits, dim=1)                    # (B, n_segments, n_classes)
        else:
            logits = self._apply_attention(att_raw, cls)               # (B, n_classes)

        # P0: clipwise interaction (refinement on (B, n_cls) after attention pool)
        if self.use_class_interaction and self.class_interaction_position == "clipwise":
            logits = self._apply_ci_with_mask(logits, class_interaction_mask)

        frame_logits = cls.permute(0, 2, 1)                            # (B, T', n_classes)
        return logits, frame_logits


class LSEHead(nn.Module):
    """LogSumExp aggregation head: (B, C, T') → (B, n_classes) or (B, n_segments, n_classes)
    - dense: two Linear layers (C→C→n_classes) with ReLU + Dropout in between
    - n_segments > 1: split the time dim into n_segments segments, lse_pool independently within each
    - output: clip/segment logits + frame_logits (B, T', n_classes), interface symmetric with AttentionSEDHead
    """
    def __init__(self, in_dim, n_classes, n_segments, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.n_segments = n_segments
        hidden_dim = hidden_dim or in_dim
        self.dense = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    @staticmethod
    def lse_pool(x, dim=1, r=1.0):
        """LogSumExp pooling (temperature=1.0):
        r→∞ approaches max, r→0 approaches mean; r=1.0 is smooth aggregation.
        """
        T = x.size(dim)
        return r * (torch.logsumexp(x / r, dim=dim) - math.log(T))

    def forward(self, x, class_interaction_mask=None):
        # x: (B, T', C); class_interaction_mask is unused by LSEHead, kept for signature compatibility with AttentionHead
        frame_logits = self.dense(x)         # (B, T', n_classes)

        if self.n_segments > 1:
            chunks = frame_logits.chunk(self.n_segments, dim=1)
            logits = torch.stack([self.lse_pool(c, dim=1) for c in chunks], dim=1)  # (B, n_segments, n_classes)
        else:
            logits = self.lse_pool(frame_logits, dim=1)                              # (B, n_classes)

        return logits, frame_logits


class GAPHead(nn.Module):
    """Global average pooling head: dense → frame_logits; clip = mean over time (GAP, equal-weight average).
    Interface symmetric with LSEHead/AttentionHead (dual output clip + frame_logits), no loss/distillation changes.
    Most orthogonal to attention: clip uses a parameter-free mean pool (all frames equal-weight) rather than attention peak-finding or an LSE soft-max.
    """
    def __init__(self, in_dim, n_classes, n_segments, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.n_segments = n_segments
        hidden_dim = hidden_dim or in_dim
        self.dense = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, class_interaction_mask=None):
        # x: (B, T', C); class_interaction_mask is unused by GAPHead, kept for signature compatibility with AttentionHead
        frame_logits = self.dense(x)         # (B, T', n_classes)
        if self.n_segments > 1:
            chunks = frame_logits.chunk(self.n_segments, dim=1)
            logits = torch.stack([c.mean(dim=1) for c in chunks], dim=1)  # (B, n_segments, n_classes) per-segment GAP
        else:
            logits = frame_logits.mean(dim=1)                              # (B, n_classes) global GAP
        return logits, frame_logits


class SEDModel(nn.Module):
    """Unified model: backbone → freq_pool → head.
    head_type:
      - "Attention": AttentionHead (tanh+softmax time-weighted aggregation)
      - "LSE":       LSEHead (lse_pool time aggregation)
    freq_pool_type: "mean"=mean / "gem"=GEMFreqPool(p=3) / "lse"=LSEFreqPool(r=1.0)
    """
    def __init__(self, backbone_name, n_classes,
                 head_type="LSE",
                 freq_pool_type="mean",
                 pretrained=True, in_chans=3, drop_path_rate=0.0,
                 head_dropout=0.1, n_segments=1,
                 att_hidden_dim=512, att_dropout=0.5,
                 att_temperature=1.0, att_activation="softmax",
                 per_class_att_temperature=False,
                 att_conv_kernel=1,
                 use_class_interaction=False,
                 class_interaction_type="linear",
                 class_interaction_hidden=128,
                 class_interaction_hidden2=None,
                 class_interaction_n_heads=4,
                 class_interaction_position="clipwise",
                 class_interaction_cross_seg_mode="none",
                 use_features_only=False):
        super().__init__()
        assert head_type in ("Attention", "LSE", "gap"), f"head_type must be 'Attention'/'LSE'/'gap', got {head_type}"
        self.head_type         = head_type
        self.n_segments        = n_segments

        # With features_only=True the backbone outputs a list of stage features, [-1] is the last stage.
        # Otherwise use num_classes=0+global_pool="" (keeps conv_head/bn2, giving a larger output dim).
        self.use_features_only = use_features_only
        if use_features_only:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained,
                features_only=True,
                in_chans=in_chans,
                drop_path_rate=drop_path_rate,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained,
                num_classes=0, global_pool="",
                in_chans=in_chans,
                drop_path_rate=drop_path_rate,
            )

        # freq pool: mean (None) / GEMFreqPool(p=3 learnable) / LSEFreqPool(r=1.0 learnable)
        assert freq_pool_type in ("mean", "gem", "lse"), f"freq_pool_type must be mean/gem/lse, got {freq_pool_type}"
        self.freq_pool_type = freq_pool_type

        if freq_pool_type == "gem":
            self.pool = GEMFreqPool()
        elif freq_pool_type == "lse":
            self.pool = LSEFreqPool()
        else:
            self.pool = None

        # Dummy shape: use the real mel input shape (128, 313) to infer the backbone output channel dim
        # (5s @ 128 mel @ hop=512 → time=313, backbone 5x downsample → F=4, T=10)
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, 128, 313)
            bb_out = self.backbone(dummy)
            # features_only=True → list of feature tensors
            if isinstance(bb_out, (list, tuple)):
                actual_features = bb_out[-1].shape[1]
            else:
                actual_features = bb_out.shape[1]

        if head_type == "Attention":
            self.head = AttentionHead(actual_features, n_classes, n_segments,
                                       hidden_dim=att_hidden_dim, dropout=att_dropout,
                                       temperature=att_temperature, activation=att_activation,
                                       per_class_temperature=per_class_att_temperature,
                                       att_conv_kernel=att_conv_kernel,
                                       use_class_interaction=use_class_interaction,
                                       class_interaction_type=class_interaction_type,
                                       class_interaction_hidden=class_interaction_hidden,
                                       class_interaction_hidden2=class_interaction_hidden2,
                                       class_interaction_n_heads=class_interaction_n_heads,
                                       class_interaction_position=class_interaction_position,
                                       class_interaction_cross_seg_mode=class_interaction_cross_seg_mode)
        elif head_type == "gap":
            self.head = GAPHead(actual_features, n_classes, n_segments, dropout=head_dropout)
        else:
            self.head = LSEHead(actual_features, n_classes, n_segments, dropout=head_dropout)

    def _pool_concat_stages(self, stages_4d):
        """list of 4D backbone features → freq pool → time-align → channel concat → (B, sum_C, T_deepest)
        - Time-dimension alignment: shallow stages are downsampled to the deepest stage's T via adaptive_avg_pool1d.
        """
        target_T = stages_4d[-1].shape[-1]
        pooled = []
        for i, feat in enumerate(stages_4d):
            pool_module = self.pool
            if self.freq_pool_type == "gem":
                f_pool = pool_module(F.relu(feat))
            elif self.freq_pool_type == "lse":
                f_pool = pool_module(feat)
            else:
                f_pool = feat.mean(dim=2)
            if f_pool.shape[-1] != target_T:
                f_pool = F.adaptive_avg_pool1d(f_pool, target_T)
            pooled.append(f_pool)
        return torch.cat(pooled, dim=1)

    def forward(self, x, class_interaction_mask=None):
        """Unified forward: supports features_only=True / num_classes=0+global_pool="" modes.
        Freq pool + time-align + concat feeds the head.
        class_interaction_mask: (B,) bool, True means the sample runs class interaction (only is_sc & is_unmixed).
        """
        bb_out = self.backbone(x)

        # Uniformly extract stages (list of 4D tensors)
        if isinstance(bb_out, (list, tuple)):
            stages = [bb_out[-1]]
        else:
            stages = [bb_out]                                                    # num_classes=0 mode

        feats_3d = self._pool_concat_stages(stages)
        logits, frame_logits = self.head(feats_3d.transpose(1, 2),
                                         class_interaction_mask=class_interaction_mask)

        return (logits, frame_logits)



# ============================================================
# SED Loss
# ============================================================

def soft_ce_loss(logits, labels):
    """Multi-label soft CrossEntropy:
    -sum(target * log_softmax(logits)), without normalizing the target.
    Samples with more positive labels get a larger loss, implicitly upweighting hard samples.
    """
    return -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()



# ============================================================
# Soft AUC Loss (directly optimizes the AUC metric)
# Supports soft labels; used for distillation and semi-supervised learning.
# ============================================================
class SoftAUCLoss(nn.Module):
    """Pairwise AUC loss: for each (pos, neg) pair compute log(1+exp(-(pos-neg)*margin)).
    Supports soft labels: weighted by |label - 0.5| (the more confident, the higher the weight).
    Note: complexity is O(N_pos * N_neg); can be slow when a batch has many pos/neg.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, preds, labels):
        # preds: (B, C) logits, labels: (B, C) soft targets
        # flatten over all samples × all classes
        preds_flat = preds.reshape(-1)
        labels_flat = labels.reshape(-1)

        pos_mask = labels_flat > 0.5
        neg_mask = labels_flat < 0.5

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        pos_preds = preds_flat[pos_mask]
        neg_preds = preds_flat[neg_mask]
        pos_weights = (labels_flat[pos_mask] - 0.5)   # higher weight for more confident labels
        neg_weights = (0.5 - labels_flat[neg_mask])

        # OOM guard: only subsample negatives (positives are usually scarce, kept in full).
        # max_pairs=2M → diff/loss/weight ~8 MB each, ~24 MB total.
        max_pairs = 2_000_000
        n_pos, n_neg = len(pos_preds), len(neg_preds)
        if n_pos * n_neg > max_pairs:
            n_neg_keep = max_pairs // max(n_pos, 1)
            idx_n = torch.randperm(n_neg, device=preds.device)[:n_neg_keep]
            neg_preds = neg_preds[idx_n]
            neg_weights = neg_weights[idx_n]

        # pairwise diff: (N_pos, N_neg)
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
        # softplus(-diff*margin) ≡ log(1+exp(-diff*margin)) but numerically stable
        loss_matrix = F.softplus(-diff * self.margin)

        # weighting
        weight_matrix = pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        weighted_loss = loss_matrix * weight_matrix

        return weighted_loss.mean()


# ============================================================
# Focal / Asymmetric BCE losses
# ============================================================
def focal_bce(logits, targets, gamma=2.5):
    """BCE with focal modulation: (1-pt)^gamma * BCE; gamma=0 reduces to plain BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma > 0:
        pt = torch.exp(-bce)
        bce = ((1 - pt) ** gamma) * bce
    return bce.mean()


def asl_bce(logits, targets, gamma_neg=4.0, gamma_pos=0.0):
    """Asymmetric Loss: different gamma for positives vs negatives.
    gamma_pos=0 preserves positive gradients; gamma_neg>0 suppresses the many easy negatives.
    Better suited to multi-label tasks than symmetric Focal Loss (each sample has only 1-2 positives out of 234 classes).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    gamma = targets * gamma_pos + (1 - targets) * gamma_neg
    return (((1 - pt) ** gamma) * bce).mean()


class SEDLoss(nn.Module):
    """SED dual-path loss: clip-level + timewise max/lse, weighted by timewise_weight.
    loss_type supports "bce" / "ce" / "auc" (same names as the top-level criterion branches).
    """
    def __init__(self, timewise_weight=0.5, loss_type="bce", pos_weight=None, auc_aux_weight=0.0):
        super().__init__()
        self.timewise_weight = timewise_weight
        self.auc_loss = SoftAUCLoss(margin=1.0)
        self.loss_type = loss_type
        self.auc_aux_weight = auc_aux_weight                         # bce_auc mode: loss = BCE + auc_aux_weight·SoftAUC
        self.pos_weight = pos_weight                                 # cached so the sample_weight path can call F.bce directly
        if loss_type in ("bce", "bce_auc"):
            # optional per-class pos_weight to upweight rare classes
            self.base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    def _compute_loss(self, logits, label, sample_weight=None):
        # sample_weight: (B,) or None. Only effective on the bce/bce_auc path; ignored by ce/auc.
        if self.loss_type == "ce":
            return soft_ce_loss(logits, label)
        if self.loss_type == "auc":
            return self.auc_loss(logits, label)
        # class-mask (selective-BCE): dimensions with label=-1 are not supervised (used by ext_neotropical).
        # Ordinary samples have label∈{0,1} with no -1 → original path unchanged; ext samples contain -1 → masked mean over class.
        has_class_mask = (label < 0).any()
        if sample_weight is None and not has_class_mask:
            bce_loss = self.base_loss(logits, label)                 # original path unchanged (reduction='mean')
        else:
            # per-sample weight / class-mask path:
            # reduction='none' element-wise → (weighted by class_mask) mean over class → mul sample_weight → mean over batch
            class_mask = (label >= 0).float()                        # (B, n_cls); -1→0 (mask), 0/1→1
            bce = F.binary_cross_entropy_with_logits(
                logits, label.clamp(0.0, 1.0), pos_weight=self.pos_weight, reduction='none')   # (B, n_classes)
            bce = bce * class_mask
            per_sample = bce.sum(dim=-1) / class_mask.sum(dim=-1).clamp(min=1.0)   # (B,) masked mean over class
            bce_loss = (per_sample * sample_weight).mean() if sample_weight is not None else per_sample.mean()
        # bce_auc: BCE + auc_aux_weight·SoftAUC (same logits/label; forward calls this once for clip and once for frame → both paths add aux)
        if self.loss_type == "bce_auc":
            return bce_loss + self.auc_aux_weight * self.auc_loss(logits, label.clamp(0.0, 1.0))
        return bce_loss

    def forward(self, logits, timewise_logits, label, sample_weight=None):
        # logits         : (B, n_classes) or (B, n_segments, n_classes)
        # timewise_logits: (B, T', n_classes)
        # label          : (B, n_classes), shared across all segments
        if logits.dim() == 3:
            # Multi-segment: run dual loss once per segment (clip + matching timewise chunk), then average.
            # label may be (B, n_cls) [shared] or (B, n_seg, n_cls) [per-seg, returned by Dataset when n_segments>1]
            n_seg = logits.size(1)
            tw_chunks = timewise_logits.chunk(n_seg, dim=1)     # n_seg × (B, T'/n, n_cls)
            lab_i = lambda i: label[:, i, :] if label.dim() == 3 else label
            loss_clip = sum(self._compute_loss(logits[:, i, :], lab_i(i),     sample_weight=sample_weight) for i in range(n_seg)) / n_seg
            loss_time = sum(self._compute_loss(tw.max(dim=1)[0], lab_i(i), sample_weight=sample_weight) for i, tw in enumerate(tw_chunks)) / n_seg
        else:
            loss_clip = self._compute_loss(logits, label, sample_weight=sample_weight)
            agg = timewise_logits.max(dim=1)[0]                  # max pool aggregation over timewise
            loss_time = self._compute_loss(agg, label, sample_weight=sample_weight)
        return (1 - self.timewise_weight) * loss_clip + self.timewise_weight * loss_time


# ============================================================
# Train / Val loops
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device, scaler, cfg, mel_tr, db_tr,
                epoch=1, scheduler=None, ema=None):
    """epoch: current epoch (1-based), used to decide MixUp warmup.
    scheduler: if provided, step() is called every batch (OneCycleLR style).
    ema: EMAWrapper instance; if provided, EMA weights are updated after every step.
    """
    model.train()

    total_loss = 0.0
    in_chans   = cfg.get("in_chans", 3)
    mel_norm   = cfg.get("mel_norm", "minmax")

    # MixUp warmup: MixUp is disabled when epoch < mixup_warmup_epoch.
    use_mixup    = (cfg.get("mixup_alpha", 0) > 0 and
                    epoch >= cfg.get("mixup_warmup_epoch", 1))
    # Gradient accumulation: used when long-duration runs hit OOM (e.g. batch=16 accum=4 ≈ batch=64).
    accum_steps = max(1, int(cfg.get("grad_accum_steps", 1)))
    n_batches_total = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for _i, (wavs, labels, is_clips, is_distill_only) in enumerate(tqdm(loader, desc="  train", leave=False)):
        wavs, labels = wavs.to(device), labels.to(device)
        # Label smoothing
        ls_eps = cfg.get("label_smoothing", 0.0)
        if ls_eps > 0:
            labels = labels * (1 - ls_eps) + ls_eps / 2
        is_clips        = is_clips.to(device)
        is_distill_only = is_distill_only.to(device)
        # label_mask selects rows that contribute to the BCE / aux loss.
        label_mask = ~is_distill_only
        has_label  = bool(label_mask.any().item())
        # Mixup role matrix:
        #   left  (self gets mixed, perm[i] != i): mixup_eligible
        #   right (mixed into by others, perm[j]=i): ~exclude_as_partner
        #   - clip            : left ✓ right ✓
        #   - labeled SC      : left ✓ when mixup_sc_participate=True; right ✓ always
        #   - unlabeled SC    : left ✗ right ✗ (excluded both ways)
        if cfg.get("mixup_sc_participate", False):
            mixup_eligible = ~is_distill_only      # clip + labeled SC
        else:
            mixup_eligible = is_clips & ~is_distill_only   # clip only
        exclude_as_partner = is_distill_only                     # unsupervised rows never appear in perm values
        # ext_neotropical (label contains a -1 mask) is excluded from batch mixup both ways: label max(ext, partner) would break the selective mask
        #   (the -1 and absence-0 dims get overwritten by the partner label); excluded both ways like the unsupervised rows.
        is_ext_batch = (labels < 0).flatten(1).any(dim=1)        # (B,) contains -1 → ext_neotropical
        if is_ext_batch.any():
            mixup_eligible     = mixup_eligible & ~is_ext_batch  # ext not mixed (left exclusion)
            exclude_as_partner = exclude_as_partner | is_ext_batch  # ext not a partner (right exclusion)

        # RandomFiltering (wave-level batch EQ aug): applied before mixup.
        if cfg.get("use_random_filter", False):
            wavs = random_filter_wav(wavs,
                                     min_db=float(cfg.get("random_filter_min_db", -20.0)),
                                     n_bands=int(cfg.get("random_filter_n_bands", 4)))

        # Batch-level MixUp (raw-audio domain, before the mel transform).
        # The Dataset may already have applied NoisyPool MixUp (sample level, clip+soundscape mix).
        # use_batch_mixup=True: stack both (double MixUp).
        # use_batch_mixup=False: rely only on NoisyPool MixUp.
        # Stage-based mixup_n_max: switch to a larger n_max later in training (single-species focus early, more species mid/late).
        _switch = cfg.get("mixup_n_max_switch_epoch", 0)
        _n_max_current = cfg.get("mixup_n_max_late", 2) if (_switch > 0 and epoch >= _switch) else cfg.get("mixup_n_max", 2)
        mixup_perms = None
        mix_weights = None
        if use_mixup and cfg.get("use_batch_mixup", True):
            wavs, labels, mixup_perms, mix_weights = mixup_wav_batch(wavs, labels,
                                           mixup_eligible=mixup_eligible,
                                           exclude_as_partner=exclude_as_partner,
                                           prob=cfg.get("mixup_prob", 1.0),
                                           n_max=_n_max_current,
                                           main_weight=cfg.get("mixup_main_weight", 0.0),
                                           alpha=cfg.get("mixup_alpha", 1.0))

        # GPU batch mel (img_size=0 keeps the natural shape; nonzero resizes; works for SED/LSE/GAP).
        _img_size = cfg.get("img_size", 0)
        imgs = batch_wav_to_image(wavs, mel_tr, db_tr, _img_size, in_chans, mel_norm,
                                  db_scope=cfg.get("db_scope_train", "sample"))
        # SpecAugment (mel-domain freq/time masking during training)
        if cfg.get("spec_aug", False):
            imgs = spec_augment(imgs,
                                freq_mask_param=cfg.get("freq_mask_param", 30),
                                time_mask_param=cfg.get("time_mask_param", 40),
                                num_freq_masks=cfg.get("num_freq_masks", 2),
                                num_time_masks=cfg.get("num_time_masks", 2))

        use_dual_loss = cfg.get("timewise_weight", 0.0) > 0

        # Class interaction mask: enabled only for SC samples that were not mixed up.
        # is_sc = ~is_clips (clip → False, soundscape → True)
        # is_unmixed = after mixup all perms[k][i] == i (identity permutation → not mixed)
        class_interaction_mask = None
        if cfg.get("use_class_interaction", False):
            B = is_clips.shape[0]
            is_sc = ~is_clips                                          # (B,) bool
            is_unmixed = torch.ones(B, dtype=torch.bool, device=device)
            if mixup_perms is not None and len(mixup_perms) > 0:
                identity = torch.arange(B, device=device)
                for p in mixup_perms:
                    is_unmixed = is_unmixed & (p == identity)
            class_interaction_mask = is_sc & is_unmixed

        # Single forward; mask is non-None only when use_class_interaction is set.
        out = model(imgs, class_interaction_mask=class_interaction_mask)
        logits, timewise_logits = out[0], out[1]

        # Main loss: dual-path SEDLoss or single-path criterion (unsupervised rows are excluded via label_mask).
        sample_weight = None
        if has_label:
            if use_dual_loss:
                loss = criterion(logits[label_mask], timewise_logits[label_mask], labels[label_mask],
                                  sample_weight=sample_weight)
            else:
                if logits.dim() == 3:
                    l = logits[label_mask].reshape(-1, logits.size(-1))
                    y = labels[label_mask].reshape(-1, labels.size(-1))
                    loss = criterion(l, y)
                else:
                    loss = criterion(logits[label_mask], labels[label_mask])
        else:
            # Entire batch is label-free (very rare): skip the classification loss.
            loss = torch.zeros((), device=device)

        # Gradient accumulation: scale loss by 1/accum, step every accum batches; the last batch forces a step.
        if accum_steps > 1:
            (loss / accum_steps).backward()
        else:
            loss.backward()
        total_loss += loss.item()

        is_step_boundary = ((_i + 1) % accum_steps == 0) or ((_i + 1) == n_batches_total)
        if is_step_boundary:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # OneCycleLR needs a step every batch (here at the optimizer-step frequency under grad_accum)
            if ema is not None:
                ema.update(model)  # EMA shadow weight moving update
            optimizer.zero_grad(set_to_none=True)

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    return avg_loss


@torch.no_grad()
def val_epoch(model, loader, criterion, device, cfg, mel_tr, db_tr):
    """Returns (val_loss, auc_clip, auc_framemax, per_taxon, per_taxon_fm, labels_np, preds_np).
    - labels_np/preds_np are attn raw (sample order matches loader.dataset.df),
      used to concatenate across val_df folds and compute the combined AUC.
    - auc_clip: AUC based on clip logits.
    - auc_framemax: AUC based on sigmoid(frame_logits.max(dim=1)[0]).
      When n_seg>1, first chunk the time axis into n_seg, max per segment, then reshape to align with the flattened labels.
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []
    all_preds_framemax = []   # framewise-max aggregated predictions
    in_chans = cfg.get("in_chans", 3)
    mel_norm        = cfg.get("mel_norm", "minmax")
    for wavs, labels, _, _ in tqdm(loader, desc="  val  ", leave=False):
        wavs, labels = wavs.to(device), labels.to(device)
        # Scale img_size by val_duration/duration to keep the same ms/pixel time density.
        # Mode C: single-segment long-window training crops val to 5s → shrink the time axis by 5/duration.
        _img_size = cfg.get("img_size", 0)
        if _img_size and _img_size != 0 and cfg.get("n_segments", 1) == 1 and cfg["duration"] != 5:
            ratio = 5 / cfg["duration"]
            if isinstance(_img_size, (list, tuple)):
                _img_size = [_img_size[0], int(_img_size[1] * ratio)]
            else:
                _img_size = int(_img_size * ratio) if ratio < 1 else _img_size
        imgs = batch_wav_to_image(wavs, mel_tr, db_tr, _img_size, in_chans, mel_norm)
        # No amp: some files overflow to NaN under float16, which corrupts the AUC computation.
        # Always obtain (clip_logits, frame_logits)
        logits, timewise_logits = model(imgs)
        if cfg.get("timewise_weight", 0.0) > 0:
            loss = criterion(logits, timewise_logits, labels)     # dual-path SEDLoss (3D handled inside the loss)
        else:
            if logits.dim() == 3:
                # Multi-segment: flatten to compute loss (each segment evaluated independently)
                logits = logits.reshape(-1, logits.size(-1))
                labels = labels.reshape(-1, labels.size(-1))
            loss = criterion(logits, labels)
        # Flatten for the AUC convention (shared by both loss branches) to avoid feeding 3D labels/preds to compute_auc
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1, labels.size(-1))
        preds = torch.sigmoid(logits)
        # framemax aggregation (computed for both n_seg==1 and n_seg>1)
        # Follow the head's current segmentation (the val path may temporarily set head.n_segments to 1);
        # do not read cfg directly, or tw_max's B axis would not align with labels.
        n_seg = getattr(getattr(model, "head", None), "n_segments", None) \
                or cfg.get("n_segments") or cfg["duration"] // 5
        if n_seg > 1:
            # Chunk the time axis into n_seg, max per segment, stack → reshape to align with the flattened labels
            tw_chunks = timewise_logits.chunk(n_seg, dim=1)   # n_seg × (B, T'/n, n_cls)
            tw_max = torch.stack([c.max(dim=1)[0] for c in tw_chunks], dim=1)  # (B, n_seg, n_cls)
            tw_max = tw_max.reshape(-1, tw_max.size(-1))       # (B*n_seg, n_cls)
        else:
            tw_max = timewise_logits.max(dim=1)[0]             # (B, n_cls)
        preds_fm = torch.sigmoid(tw_max)
        all_preds_framemax.append(preds_fm.cpu().numpy())
        total_loss += loss.item()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
    labels_np = np.concatenate(all_labels)
    preds_np  = np.concatenate(all_preds)
    auc = compute_auc(labels_np, preds_np)
    per_taxon    = compute_per_taxon_auc(labels_np, preds_np)
    auc_framemax = 0.0
    per_taxon_fm = {}
    if all_preds_framemax:
        preds_fm_np  = np.concatenate(all_preds_framemax)
        auc_framemax = compute_auc(labels_np, preds_fm_np)
        per_taxon_fm = compute_per_taxon_auc(labels_np, preds_fm_np)
    return total_loss / len(loader), auc, auc_framemax, per_taxon, per_taxon_fm, labels_np, preds_np


@torch.no_grad()
def val_epoch_clip_overlap(model, val_df, species_to_idx, mel_tr, db_tr, device, cfg):
    """Overlap sliding-window validation (matching the infer_kernel n_segments>1 branch):
    a 60s soundscape is cut into n_windows × duration windows (stride 5s); the model outputs (n_windows, n_segments, n_cls) clip-level,
    and each 5s slot = average of the clip predictions from the windows covering it. Only active when n_segments>1 (duration>5s).
    """
    model.eval()
    N_SEGS      = 12
    SEG_SEC     = 5
    duration    = cfg["duration"]
    sr          = cfg["sample_rate"]
    n_segments  = duration // SEG_SEC                  # 20s → 4
    n_windows   = N_SEGS - n_segments + 1              # 9
    n_per_seg   = sr * SEG_SEC
    n_per_dur   = sr * duration
    total_samp  = N_SEGS * n_per_seg                    # 60s
    in_chans    = cfg.get("in_chans", 3)
    n_classes   = len(species_to_idx)

    sc_df = val_df[val_df["start_sec"] >= 0].copy()
    all_labels, all_preds, all_preds_fm = [], [], []

    for file_path, group in tqdm(sc_df.groupby("file_path"), desc="  val  ", leave=False):
        data, native_sr = sf.read(str(file_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))
        if native_sr != sr:
            wav = torchaudio.functional.resample(wav, native_sr, sr)
        if len(wav) < total_samp:
            wav = F.pad(wav, (0, total_samp - len(wav)))

        # n_windows duration-long windows (stride 5s)
        segs = torch.stack([wav[i * n_per_seg : i * n_per_seg + n_per_dur]
                            for i in range(n_windows)]).to(device)         # (n_windows, n_per_dur)
        _img_size = cfg.get("img_size", 0)
        imgs = batch_wav_to_image(segs, mel_tr, db_tr, _img_size, in_chans, cfg.get("mel_norm", "minmax"))
        clip_logits, frame_logits = model(imgs)                              # (n_win, n_seg, n_cls), (n_win, T', n_cls)
        clip_probs = torch.sigmoid(clip_logits.float()).cpu().numpy()
        # framemax: within each window, chunk frame_logits into n_seg and max per segment, then run the same overlap aggregation as clip
        fm_chunks  = frame_logits.chunk(n_segments, dim=1)
        fm_stack   = torch.stack([c.max(dim=1)[0] for c in fm_chunks], dim=1)   # (n_win, n_seg, n_cls)
        fm_probs   = torch.sigmoid(fm_stack.float()).cpu().numpy()

        # Overlap averaging: window t's n_segments predictions are assigned to slots [t, t+1, ..., t+n_seg-1]
        probs     = np.zeros((N_SEGS, n_classes), dtype=np.float32)
        probs_fm  = np.zeros((N_SEGS, n_classes), dtype=np.float32)
        counts    = np.zeros((N_SEGS, 1),         dtype=np.float32)
        for t in range(n_windows):
            probs[t:t+n_segments]    += clip_probs[t]
            probs_fm[t:t+n_segments] += fm_probs[t]
            counts[t:t+n_segments]   += 1
        sc_probs    = probs    / counts.clip(min=1)                          # (12, n_cls)
        sc_probs_fm = probs_fm / counts.clip(min=1)                          # (12, n_cls)

        for _, row in group.iterrows():
            slot = int(row["start_sec"]) // SEG_SEC
            if 0 <= slot < N_SEGS:
                label = np.zeros(n_classes, dtype=np.float32)
                for sp in str(row["labels"]).split(";"):
                    sp = sp.strip()
                    sp_idx = species_to_idx.get(sp)
                    if sp_idx is not None:
                        label[sp_idx] = 1.0
                all_labels.append(label)
                all_preds.append(sc_probs[slot])
                all_preds_fm.append(sc_probs_fm[slot])

    labels_np    = np.stack(all_labels)
    preds_np     = np.stack(all_preds)
    preds_fm_np  = np.stack(all_preds_fm)
    auc          = compute_auc(labels_np, preds_np)
    auc_fm       = compute_auc(labels_np, preds_fm_np)
    per_taxon    = compute_per_taxon_auc(labels_np, preds_np)
    per_taxon_fm = compute_per_taxon_auc(labels_np, preds_fm_np)
    return 0.0, auc, auc_fm, per_taxon, per_taxon_fm, labels_np, preds_np


# ============================================================
# ONNX Export (per fold)
# ============================================================

def export_onnx_for_fold(cfg, exp_dir, fold_id, n_classes):
    """Load best_model_fold{k}_*.pth → rebuild a clean SEDModel (dropping the aux head) →
    export sed_fold{k}_ep{N}_{auc|fm}_{S}.onnx for both best_auc and best_fm → sanity check.
    Sanity-check results are appended to exp_dir/onnx_summary.txt.
    Output: (clip_logits, frame_logits) — the kernel side blends them itself.
    """
    # Collect this fold's best_auc and best_fm checkpoints and export each to ONNX
    auc_ckpts = sorted(exp_dir.glob(f"best_model_fold{fold_id}_ep*_auc_*.pth"))
    fm_ckpts  = sorted(exp_dir.glob(f"best_model_fold{fold_id}_ep*_fm_*.pth"))
    # Backward compat for old naming (no _auc_/_fm_ suffix): treat as auc
    legacy_ckpts = [p for p in sorted(exp_dir.glob(f"best_model_fold{fold_id}_*.pth"))
                    if "_auc_" not in p.name and "_fm_" not in p.name]
    if not auc_ckpts and legacy_ckpts:
        auc_ckpts = legacy_ckpts

    targets = []
    if auc_ckpts:
        targets.append(("auc", auc_ckpts[-1]))
    if fm_ckpts:
        targets.append(("fm",  fm_ckpts[-1]))
    if not targets:
        print(f"  [fold {fold_id}] no best ckpt found, skip ONNX export")
        return

    summary_path = exp_dir / "onnx_summary.txt"
    summary_lines = []  # accumulated for this call

    for kind, ckpt_path in targets:
        # Extract epoch + score from the ckpt filename → carry them into the ONNX name
        m_meta = re.search(r"_ep(\d+)_(auc|fm)_([0-9.]+)", ckpt_path.stem)
        if m_meta:
            ep_tag, sc_tag = m_meta.group(1), m_meta.group(3)
            onnx_path = exp_dir / f"sed_fold{fold_id}_ep{ep_tag}_{kind}_{sc_tag}.onnx"
        else:
            # Old-naming fallback: epoch+score
            m_old = re.search(r"_epoch(\d+)_score([0-9.]+)", ckpt_path.stem)
            if m_old:
                onnx_path = exp_dir / f"sed_fold{fold_id}_ep{m_old.group(1)}_{kind}_{m_old.group(2)}.onnx"
            else:
                onnx_path = exp_dir / f"sed_fold{fold_id}_{kind}.onnx"

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}

        # Rebuild the SEDModel for inference.
        backbone   = ckpt.get("backbone",          cfg["backbone"])
        head_type  = ckpt.get("head_type",         cfg.get("head_type", "LSE"))
        # Backward compat for old ckpts: map use_gem_freq_pool to freq_pool_type when only the former is present
        if "freq_pool_type" in ckpt:
            pool_type = ckpt["freq_pool_type"]
        elif ckpt.get("use_gem_freq_pool", False):
            pool_type = "gem"
        else:
            pool_type = cfg.get("freq_pool_type", "mean")
        in_chans   = ckpt.get("in_chans",          cfg.get("in_chans", 3))
        n_seg      = ckpt.get("n_segments") or (ckpt.get("duration", cfg["duration"]) // 5)

        model = SEDModel(
            backbone, n_classes,
            head_type=head_type, freq_pool_type=pool_type,
            pretrained=False, in_chans=in_chans, n_segments=n_seg,
            att_hidden_dim=ckpt.get("att_hidden_dim", 512),
            att_dropout=ckpt.get("att_dropout", 0.5),
            att_temperature=ckpt.get("att_temperature", 1.0),
            att_activation=ckpt.get("att_activation", "softmax"),
            per_class_att_temperature=ckpt.get("per_class_att_temperature", False),
            att_conv_kernel=ckpt.get("att_conv_kernel", 1),
            use_class_interaction=ckpt.get("use_class_interaction", False),
            class_interaction_type=ckpt.get("class_interaction_type", "linear"),
            class_interaction_hidden=ckpt.get("class_interaction_hidden", 128),
            class_interaction_hidden2=ckpt.get("class_interaction_hidden2", None),
            class_interaction_n_heads=ckpt.get("class_interaction_n_heads", 4),
            class_interaction_position=ckpt.get("class_interaction_position", "clipwise"),
            class_interaction_cross_seg_mode=ckpt.get("class_interaction_cross_seg_mode", "none"),
            use_features_only=ckpt.get("use_features_only", False),
        )
        model.load_state_dict(state_dict, strict=False)
        model.cpu().eval()

        # Dummy input shape (H, W): from ckpt/cfg img_size; otherwise W = sample_rate * duration / hop + 1.
        img_size = ckpt.get("img_size", cfg.get("img_size", 0))
        if isinstance(img_size, int) and img_size > 0:
            H = W = img_size
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            H, W = img_size
        else:
            H = cfg.get("n_mels", 128)
            _hop = cfg["hop_length"]
            W = (cfg["sample_rate"] * cfg["duration"]) // _hop + 1
        dummy = torch.zeros(1, in_chans, H, W)

        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"],
            output_names=["clip_logits", "frame_logits"],
            dynamic_axes={"input":         {0: "batch"},
                          "clip_logits":   {0: "batch"},
                          "frame_logits":  {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )

        # onnxsim post-processing: remove dead initializers + graph dedup.
        # Reduces model size (NFNet's doubly-baked weight-standardized weights get eliminated) at equivalent accuracy (<1e-6 sigmoid diff).
        try:
            import onnx as _onnx
            from onnxsim import simplify as _onnxsim_simplify
            _m = _onnx.load(str(onnx_path))
            _sim, _check = _onnxsim_simplify(_m)
            if _check:
                _onnx.save(_sim, str(onnx_path))
        except ImportError:
            print(f"  [onnxsim not installed, skipping size/speed optimization] install via pip install onnxsim")
        except Exception as _e:
            print(f"  [onnxsim failed {type(_e).__name__}: {_e}], using the original ONNX")

        # Sanity check: CPU torch vs CPU ONNX (compare clip_logits) → written to summary
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            with torch.no_grad():
                torch_clip = model(dummy)[0].numpy()
            onnx_clip = sess.run(None, {"input": dummy.numpy()})[0]
            diff = float(np.abs(torch_clip - onnx_clip).max())
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            status = "WARN diff>1e-3" if diff > 1e-3 else "OK"
            line = (f"fold {fold_id} [{kind}]: {onnx_path.name}  size={size_mb:.1f}MB  "
                    f"shape={tuple(dummy.shape)}  diff={diff:.6f}  {status}")
        except Exception as e:
            line = f"fold {fold_id} [{kind}]: sanity check skipped: {type(e).__name__}: {e}"
        summary_lines.append(line)
        print(f"  {line}")

    # Append to onnx_summary.txt (write a header on first write)
    write_header = not summary_path.exists()
    with open(summary_path, "a") as f:
        if write_header:
            f.write("# ONNX export sanity check (CPU torch vs CPU ONNX, max abs diff on clip_logits)\n")
        for line in summary_lines:
            f.write(line + "\n")


# ============================================================
# Main
# ============================================================

def main(cfg):
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    exp_dir = get_exp_dir(cfg["output_dir"], debug=cfg["debug"], resume=cfg.get("resume", False),
                          exp_name=cfg.get("exp_name"))
    print(f"Experiment: {exp_dir.name}{'  [RESUME]' if cfg.get('resume') else ''}")
    with open(exp_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    # Save a snapshot of the training code for full reproducibility
    import shutil
    shutil.copy(__file__, exp_dir / "train_folds.py")

    # Species list (taxonomy order = submission column order)
    taxonomy       = pd.read_csv(os.path.join(cfg["data_dir"], "taxonomy.csv"))
    species_list   = taxonomy["primary_label"].tolist()
    species_to_idx = {sp: i for i, sp in enumerate(species_list)}
    n_classes      = len(species_list)
    print(f"Classes: {n_classes}")

    # Per-taxon AUC grouping info (always computed and written to log.csv, for taxon-grouped val reporting)
    global _SPECIES_TAXON_IDX, _TAXON_NAMES
    _s2t_mat, _TAXON_NAMES = build_species_to_taxon(taxonomy, species_list)
    _SPECIES_TAXON_IDX = _s2t_mat.argmax(dim=1).numpy()
    _taxon_counts = [int((_SPECIES_TAXON_IDX == t).sum()) for t in range(len(_TAXON_NAMES))]
    print(f"  Taxon breakdown: " + ", ".join(f"{n}={c}" for n, c in zip(_TAXON_NAMES, _taxon_counts)))

    # ─── Shared log.csv (appended across folds, first column is fold) ───
    log_path = exp_dir / "log.csv"
    if not cfg.get("resume") or not log_path.exists():
        per_taxon_cols = [f"auc_{t}"    for t in _TAXON_NAMES] + \
                         [f"auc_fm_{t}" for t in _TAXON_NAMES]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["fold", "epoch", "train_loss", "val_loss", "val_auc", "val_auc_fm"]
                + per_taxon_cols + ["lr", "elapsed_s"])

    # ─── One-time preprocessing + fold splitting (focal Stratified + SC GroupKFold; shared across folds) ───
    combined_df = build_combined_df(cfg, species_to_idx)
    print(f"  combined_df: {len(combined_df)} rows  "
          f"(clip={(combined_df['data_type']=='clip').sum()}, "
          f"sc={(combined_df['data_type']=='soundscape').sum()})")

    # ─── Fold loop ───
    _orig_resume = cfg.get("resume", False)
    fold_ids = cfg.get("fold_ids") or [cfg.get("fold_id", 0)]
    # OOF accumulation (concatenated across folds to compute macro AUC):
    #   _oof_sc_y/p_list:        each fold's best-epoch SC raw (assembles the full-set OOF)
    #   _per_fold_epoch_raw[fk]: {epoch: (sc_y, sc_p)} per-fold per-epoch SC raw (per-epoch pooled OOF)
    _oof_sc_y_list, _oof_sc_p_list = [], []
    _per_fold_epoch_raw = {}
    _fold_bests = {}     # {fold_id: (best_auc, best_fm)} for writing oof_summary
    print(f"\nFold loop: {fold_ids}")
    for _fk in fold_ids:
        cfg["fold_id"] = _fk
        # Already-trained check: best_model_fold{k}_*.pth exists + no checkpoint_fold{k}.pth → skip
        existing = sorted(exp_dir.glob(f"best_model_fold{_fk}_*.pth"))
        ckpt_fold_path = exp_dir / f"checkpoint_fold{_fk}.pth"
        if existing and not ckpt_fold_path.exists():
            print(f"\n[fold {_fk}] best already exists, skipping: {existing[-1].name}")
            continue
        # resume only when this fold has a checkpoint
        cfg["resume"] = _orig_resume and ckpt_fold_path.exists()

        print(f"\n{'='*60}\n  FOLD {_fk}\n{'='*60}")

        # ---- Split train/val + merge SC train segments + SC oversample ----
        fold_id  = cfg.get("fold_id", 0)
        n_folds  = cfg.get("n_folds", 5)
        train_df = combined_df[combined_df["_fold"] != _fk].drop(columns=["_fold"]).reset_index(drop=True)
        # val uses only SC segments (the SC main metric + per-taxon are all based on these)
        val_df       = combined_df[combined_df["_fold"] == _fk].drop(columns=["_fold"]).reset_index(drop=True)
        val_df       = val_df[val_df["data_type"] == "soundscape"].reset_index(drop=True)

        # In single-segment training (n_segments==1), merge consecutive same-label train SC segments; multi-segment keeps the 5s grid
        n_seg_train = cfg.get("n_segments") or (cfg["duration"] // 5)
        if cfg.get("merge_sc_segments", True) and n_seg_train == 1:
            train_df = merge_sc_train_segments(train_df)

        # Rare focal-class upsample: species with count<min_sample are replicated to at least min_sample rows (clip rows of train_df only).
        # is_ext filter: upsample applies only to the official train_audio, not ext_df (ext does not get floor protection).
        min_sample = cfg.get("min_sample", 0)
        if min_sample > 0:
            clip_mask = (train_df["data_type"] == "clip") & (~train_df["is_ext"].astype(bool))
            if clip_mask.any():
                clip_df  = train_df[clip_mask]
                # focal labels format: primary;sec1;sec2, where primary = labels.split(";")[0]
                primary  = clip_df["labels"].apply(lambda s: str(s).split(";")[0].strip())
                counts   = primary.value_counts()
                rare_sp  = counts[counts < min_sample].index.tolist()
                extra    = []
                for sp in rare_sp:
                    sp_rows  = clip_df[primary == sp]
                    n_copies = int(np.ceil(min_sample / len(sp_rows))) - 1
                    for _ in range(n_copies):
                        extra.append(sp_rows)
                if extra:
                    n_before = len(train_df)
                    train_df = pd.concat([train_df] + extra, ignore_index=True)
                    print(f"  Upsample rare focal species (min_sample={min_sample}): "
                          f"{len(rare_sp)} species  rows {n_before} → {len(train_df)}")

        # SC rare-species upsample (file level): targets species absent from focal data / scarce in SC segments (e.g. sonotypes).
        # min_sample only sees species present in focal (pd.value_counts skips count=0), so sonotypes are missed entirely.
        # Here we group by file and replicate whole files (compatible with n_seg>1: the dataset randomizes the slot at read time,
        # so single-row replication is ineffective; file-level replication makes "that file appear multiple times", and the dataset's
        # repeated random slots can at least hit the time spans containing rare species).
        sc_min_sample = cfg.get("sc_min_sample", 0)
        if sc_min_sample > 0:
            sc_mask = train_df["data_type"] == "soundscape"
            if sc_mask.any():
                sc_df = train_df[sc_mask]
                # Multi-label split: count the number of segments each species appears in (each segment contributes to all GT species' counts)
                from collections import defaultdict as _dd
                species_count = _dd(int)
                for s in sc_df["labels"].astype(str):
                    for sp in s.split(";"):
                        sp = sp.strip()
                        if sp:
                            species_count[sp] += 1
                # Compute the copy factor needed for each rare species
                rare_target_copies = {sp: int(np.ceil(sc_min_sample / max(cnt, 1)))
                                      for sp, cnt in species_count.items() if cnt < sc_min_sample}
                if rare_target_copies:
                    extra_rows = []
                    n_files_upsampled = 0
                    n_total_extra_segs = 0
                    # Group by file, union the GT species over the whole file → take the max copy factor → replicate all segments of the file together
                    for fpath, group_df in sc_df.groupby("file_path", sort=False):
                        all_gt = set()
                        for s in group_df["labels"].astype(str):
                            for sp in s.split(";"):
                                sp = sp.strip()
                                if sp:
                                    all_gt.add(sp)
                        copy_factor = max(
                            (rare_target_copies.get(sp, 1) for sp in all_gt),
                            default=1,
                        )
                        if copy_factor > 1:
                            n_files_upsampled += 1
                            n_total_extra_segs += len(group_df) * (copy_factor - 1)
                            for _ in range(copy_factor - 1):
                                extra_rows.append(group_df)
                    if extra_rows:
                        n_before = len(train_df)
                        train_df = pd.concat([train_df] + extra_rows, ignore_index=True)
                        print(f"  Upsample rare SC species (sc_min_sample={sc_min_sample}, file-level): "
                              f"{len(rare_target_copies)} species below threshold, "
                              f"{n_files_upsampled} files upsampled, "
                              f"{n_total_extra_segs} extra segs, rows {n_before} → {len(train_df)}")

        # SC oversample (train SC segments only)
        sc_oversample = cfg.get("sc_oversample", 1)
        if sc_oversample > 1:
            train_sc = train_df[train_df["data_type"] == "soundscape"]
            if len(train_sc) > 0:
                extra = pd.concat([train_sc] * (sc_oversample - 1), ignore_index=True)
                train_df = pd.concat([train_df, extra], ignore_index=True)


        # ---- NoisyPool (sample-level pseudo-label SC mixup) ----
        noisy_pool = None
        if cfg.get("use_noisy_pool", False):
            sc_labels_df  = pd.read_csv(os.path.join(cfg["data_dir"], "train_soundscapes_labels.csv"))
            labeled_stems = set(Path(fn).stem for fn in sc_labels_df["filename"].unique())
            base_dir      = Path(cfg["data_dir"]).parent
            # npz scores shape (4, N, 234) — NoisyPool indexes the first dim by fold_id internally for zero leak
            noisy_pool    = NoisyPool(
                npz_path       = base_dir / cfg["teacher_npz"],
                meta_csv       = base_dir / cfg["teacher_meta_csv"],
                soundscape_dir = os.path.join(cfg["data_dir"], "train_soundscapes"),
                labeled_stems  = labeled_stems,
                cfg            = cfg,
                conf_weighted  = cfg.get("noisy_conf_weighted", True),
                min_conf       = cfg.get("noisy_min_conf", 0.0),
                fold_id        = cfg.get("fold_id", 0),
            )


        n_train_clip = (train_df["data_type"] == "clip").sum()
        n_train_sc   = (train_df["data_type"] == "soundscape").sum()
        n_val_clip   = (val_df["data_type"] == "clip").sum()
        n_val_sc     = (val_df["data_type"] == "soundscape").sum()
        # File counts: clip deduped by audio_id, soundscape also deduped by audio_id
        n_train_clip_files    = train_df[train_df["data_type"] == "clip"]["audio_id"].nunique()
        n_train_sc_files = train_df[train_df["data_type"] == "soundscape"]["audio_id"].nunique()
        n_val_clip_files     = val_df[val_df["data_type"] == "clip"]["audio_id"].nunique()
        n_val_sc_files = val_df[val_df["data_type"] == "soundscape"]["audio_id"].nunique()
        # Species count: expand the labels column (;-separated multi-label) and count uniques
        train_sp = set(sp.strip() for lbls in train_df["labels"] for sp in str(lbls).split(";"))
        val_sp   = set(sp.strip() for lbls in val_df["labels"]   for sp in str(lbls).split(";"))
        print(f"Fold {fold_id}/{n_folds-1}")
        print(f"  Train: {len(train_df)} segs, {n_train_clip_files} audio files, {n_train_sc_files} sc files, {len(train_sp)} species"
              f"  (clip={n_train_clip} segs, sc={n_train_sc} segs)")
        print(f"  Val:   {len(val_df)} segs, {n_val_clip_files} audio files, {n_val_sc_files} sc files, {len(val_sp)} species"
              f"  (clip={n_val_clip} segs, sc={n_val_sc} segs)")

        if cfg["debug"]:
            train_df = train_df.sample(cfg["debug_samples"], random_state=cfg["seed"]).reset_index(drop=True)
            val_df   = val_df.sample(min(50, len(val_df)),   random_state=cfg["seed"]).reset_index(drop=True)
            cfg["epochs"] = 2
            # In debug, switch to a short scheduler to avoid flat_cosine(warmup=3+flat=30) requiring epoch≥34
            cfg["scheduler"] = "cosine"
            cfg["warmup_epochs"] = 0
            print(f"[DEBUG] train={len(train_df)}, val={len(val_df)}, epochs={cfg['epochs']}, scheduler=cosine")

        # Aux taxon head label mapping + pos_weight

        train_ds = BirdDataset(train_df, species_to_idx, cfg, is_train=True,  noisy_pool=noisy_pool)
        # Main SC val: single-segment long-window training auto-crops val to 5s (mode C); other modes use val=duration
        val_cfg     = {**cfg, "duration": 5} if (cfg.get("n_segments", 1) == 1 and cfg["duration"] != 5) else cfg
        val_ds      = BirdDataset(val_df,      species_to_idx, val_cfg, is_train=False)


        # DataLoader (shuffled)
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            pin_memory=True, drop_last=True,
            persistent_workers=True, prefetch_factor=2,
        )
        val_loader      = DataLoader(
            val_ds, batch_size=cfg["batch_size"],
            shuffle=False, num_workers=cfg["num_workers"],
            pin_memory=True,
            persistent_workers=True, prefetch_factor=2,
        )

        # Unified SEDModel: head_type + freq_pool_type control behavior (LSE/Attention × mean/gem/lse, six combinations)
        drop_path_rate = cfg.get("drop_path_rate", 0.0)
        model = SEDModel(cfg["backbone"], n_classes,
                         head_type=cfg.get("head_type", "LSE"),
                         freq_pool_type=cfg.get("freq_pool_type", "mean"),
                         pretrained=cfg["pretrained"],
                         in_chans=cfg.get("in_chans", 3),
                         drop_path_rate=drop_path_rate,
                         head_dropout=cfg.get("head_dropout", 0.1),
                         n_segments=cfg.get("n_segments") or cfg["duration"] // 5,
                         att_hidden_dim=cfg.get("att_hidden_dim", 512),
                         att_dropout=cfg.get("att_dropout", 0.5),
                         att_temperature=cfg.get("att_temperature", 1.0),
                         att_activation=cfg.get("att_activation", "softmax"),
                         per_class_att_temperature=cfg.get("per_class_att_temperature", False),
                         att_conv_kernel=cfg.get("att_conv_kernel", 1),
                         use_class_interaction=cfg.get("use_class_interaction", False),
                         class_interaction_type=cfg.get("class_interaction_type", "linear"),
                         class_interaction_hidden=cfg.get("class_interaction_hidden", 128),
                         class_interaction_hidden2=cfg.get("class_interaction_hidden2", None),
                         class_interaction_n_heads=cfg.get("class_interaction_n_heads", 4),
                         class_interaction_position=cfg.get("class_interaction_position", "clipwise"),
                         class_interaction_cross_seg_mode=cfg.get("class_interaction_cross_seg_mode", "none"),
                         use_features_only=cfg.get("use_features_only", False)).to(device)
        print(f"  SEDModel: head_type={cfg.get('head_type', 'LSE')}, freq_pool_type={cfg.get('freq_pool_type', 'mean')}")

        # Load the XC-pretrained backbone (public weights)
        # Overwrites after ImageNet pretrained=True; relaxes shape mismatch via strict=False
        xc_path = cfg.get("xc_pretrained_path")
        if xc_path:
            from pathlib import Path as _P
            if _P(xc_path).is_file():
                print(f"  Loading XC pretrained backbone: {xc_path}")
                xc_state = torch.load(xc_path, map_location=device, weights_only=False)
                # The XC ECA-NFNet uses stem_conv/stages_X naming (NFNet custom); timm uses stem.conv/stages.X.
                # When features_only=True, timm also uses underscore naming (stem_conv1), so prefer existing keys without remapping.
                _bb_keys = set((model.backbone if hasattr(model, "backbone") else model).state_dict().keys())
                def _remap_xc(k):
                    if k in _bb_keys:
                        return k  # direct match, no remap (e.g. features_only=True with eca_nfnet)
                    if k.startswith("stem_conv"):
                        return "stem." + k[len("stem_"):]
                    if k.startswith("stages_"):
                        return "stages." + k[len("stages_"):]
                    return k
                xc_state = {_remap_xc(k): v for k, v in xc_state.items()}
                # channel adaptation: when conv_stem.weight channel count mismatches, replicate the XC 1ch pretrain to model in_chans
                if hasattr(model, "backbone"):
                    target_stem = model.backbone.state_dict().get("conv_stem.weight")
                else:
                    target_stem = model.state_dict().get("conv_stem.weight")
                if target_stem is not None and "conv_stem.weight" in xc_state:
                    src = xc_state["conv_stem.weight"]
                    if src.shape != target_stem.shape and src.shape[1] != target_stem.shape[1]:
                        new_in = target_stem.shape[1]
                        old_in = src.shape[1]
                        # Simple strategy: copy channel 0 to all new channels (each mel view uses the same pretrained filters)
                        expanded = src[:, :1, :, :].repeat(1, new_in, 1, 1) / new_in * old_in
                        xc_state["conv_stem.weight"] = expanded
                        print(f"  XC conv_stem adapter: {tuple(src.shape)} → {tuple(expanded.shape)} (channel repeat × {new_in / old_in:.2f})")
                # eca_nfnet uses stem.conv1.weight (not conv_stem.weight), which also needs channel adaptation
                if hasattr(model, "backbone"):
                    target_stem_nf = model.backbone.state_dict().get("stem.conv1.weight")
                else:
                    target_stem_nf = model.state_dict().get("stem.conv1.weight")
                if target_stem_nf is not None and "stem.conv1.weight" in xc_state:
                    src_nf = xc_state["stem.conv1.weight"]
                    if src_nf.shape != target_stem_nf.shape and src_nf.shape[1] != target_stem_nf.shape[1]:
                        new_in_nf = target_stem_nf.shape[1]
                        old_in_nf = src_nf.shape[1]
                        expanded_nf = src_nf[:, :1, :, :].repeat(1, new_in_nf, 1, 1) / new_in_nf * old_in_nf
                        xc_state["stem.conv1.weight"] = expanded_nf
                        print(f"  XC stem.conv1 adapter (eca_nfnet): {tuple(src_nf.shape)} → {tuple(expanded_nf.shape)} (channel repeat × {new_in_nf / old_in_nf:.2f})")
                if hasattr(model, "backbone"):
                    missing, unexpected = model.backbone.load_state_dict(xc_state, strict=False)
                    matched = len(xc_state) - len(unexpected)
                    print(f"  XC loaded: {matched}/{len(xc_state)} matched, missing={len(missing)}, unexpected={len(unexpected)}")
                    if len(unexpected) > 0:
                        print(f"  unexpected keys (first 5): {unexpected[:5]}")
                else:
                    # The GAP timm model is the backbone directly (no .backbone attribute)
                    missing, unexpected = model.load_state_dict(xc_state, strict=False)
                    matched = len(xc_state) - len(unexpected)
                    print(f"  XC loaded into GAP model: {matched}/{len(xc_state)} matched, missing={len(missing)}, unexpected={len(unexpected)}")
            else:
                print(f"  WARN: xc_pretrained_path not found: {xc_path}, skipping")

        # torch.compile disabled (caused a continuous recompile loop with GPU near 0% utilization on some machines)
        # if hasattr(torch, "compile"):
        #     model = torch.compile(model)
        #     print("  torch.compile enabled")
        print("  torch.compile DISABLED (recompile loop on new instance)")

        # EMA: exponential moving average of weights (optional, smooths val fluctuations)
        ema = None
        if cfg.get("use_ema", False):
            ema = EMAWrapper(model, decay=cfg.get("ema_decay", 0.999))
            print(f"  EMA enabled: decay={cfg.get('ema_decay', 0.999)}, start_epoch={cfg.get('ema_start_epoch', 3)}, ema_val={cfg.get('ema_val', True)}")

        # GPU mel transforms (batched mel computation on GPU, much faster than serial CPU)
        mel_tr, db_tr = build_mel_transform(cfg)
        mel_tr = mel_tr.to(device)
        db_tr  = db_tr.to(device)

        # Optional: set a separate weight decay for the per-class log_temperature (strong L2 to suppress Aves τ drift)
        log_temp_wd = cfg.get("log_temp_wd", None)
        _has_log_tau = cfg.get("per_class_att_temperature", False)
        # Optimizer choice: adamw (default) / radam (recommended for BN-free backbones like NFNet)
        _opt_name = str(cfg.get("optimizer", "adamw")).lower()
        assert _opt_name in ("adamw", "radam"), f"unknown optimizer: {_opt_name}"
        _opt_cls = torch.optim.RAdam if _opt_name == "radam" else torch.optim.AdamW
        if log_temp_wd is not None and _has_log_tau:
            log_temp_params, other_params = [], []
            for name, p in model.named_parameters():
                (log_temp_params if "log_temperature" in name else other_params).append(p)
            optimizer = _opt_cls(
                [{"params": other_params, "weight_decay": cfg.get("weight_decay", 0)},
                 {"params": log_temp_params, "weight_decay": float(log_temp_wd)}],
                lr=cfg["lr"],
            )
            print(f"  log_temp_wd={log_temp_wd} separate wd applied to {len(log_temp_params)} log_temperature parameters")
        else:
            optimizer = _opt_cls(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg.get("weight_decay", 0))
        print(f"  optimizer={_opt_name} lr={cfg['lr']} wd={cfg.get('weight_decay', 0)}")

        # Scheduler: onecycle / cosine / flat_cosine
        sched_type = cfg.get("scheduler", "cosine")
        if sched_type == "onecycle":
            # OneCycleLR: per-batch step；pct_start = warmup_epochs / epochs
            warmup_ep  = cfg.get("warmup_epochs", 3)
            scheduler  = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr        = cfg["lr"],
                epochs        = cfg["epochs"],
                steps_per_epoch = len(train_loader),
                pct_start     = min(warmup_ep / cfg["epochs"], 0.99),  # clamp to guard against debug mode where epochs < warmup_ep
                div_factor    = 25,                           # initial_lr = max_lr/25
                final_div_factor = 4.0,                       # min_lr = initial_lr/4
            )
            per_batch_sched = scheduler
            per_epoch_sched = None
        elif sched_type == "flat_cosine":
            # warmup → constant LR → cosine decay (per-batch step; same style as onecycle)
            warmup_ep = cfg.get("warmup_epochs", 2)
            flat_ep   = cfg.get("flat_epochs", 22)
            tail_ep   = cfg["epochs"] - warmup_ep - flat_ep
            assert tail_ep > 0, f"flat_cosine: epochs({cfg['epochs']}) - warmup({warmup_ep}) - flat({flat_ep}) = {tail_ep} <= 0"
            steps_per_ep = len(train_loader)
            warmup_steps = warmup_ep * steps_per_ep
            flat_steps   = flat_ep   * steps_per_ep
            tail_steps   = tail_ep   * steps_per_ep
            warmup_sched  = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
            flat_sched    = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=flat_steps)
            cosine_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tail_steps, eta_min=cfg.get("min_lr", 0.0))
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, flat_sched, cosine_sched],
                milestones=[warmup_steps, warmup_steps + flat_steps])
            print(f"  Scheduler: flat_cosine (per-batch, {steps_per_ep} steps/ep)  "
                  f"warmup={warmup_ep}ep  flat={flat_ep}ep  cosine_tail={tail_ep}ep")
            per_batch_sched = scheduler
            per_epoch_sched = None
        else:
            warmup_ep = cfg.get("warmup_epochs", 0)
            if warmup_ep > 0:
                # warmup + cosine: linearly ramp to lr over the first warmup_ep, then cosine decay (per-batch step)
                steps_per_ep = len(train_loader)
                warmup_steps = warmup_ep * steps_per_ep
                tail_steps   = (cfg["epochs"] - warmup_ep) * steps_per_ep
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=tail_steps, eta_min=cfg.get("min_lr", 0.0))
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
                print(f"  Scheduler: cosine (per-batch, {steps_per_ep} steps/ep)  "
                      f"warmup={warmup_ep}ep  cosine_tail={cfg['epochs'] - warmup_ep}ep")
                per_batch_sched = scheduler
                per_epoch_sched = None
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg["epochs"], eta_min=cfg.get("min_lr", 0.0))
                per_batch_sched = None
                per_epoch_sched = scheduler

        # Loss choice: loss_type="bce"/"ce"/"auc"; focal_gamma>0 → focal; timewise_weight>0 → dual-path SEDLoss
        loss_type = cfg.get("loss_type", "bce")
        focal_gamma = cfg.get("focal_gamma", 0.0)
        timewise_weight = cfg.get("timewise_weight", 0.0)

        # per-species BCE pos_weight (upweight rare classes)
        species_pos_weight = None
        if cfg.get("use_class_pos_weight", False) and loss_type == "bce":
            alpha = cfg.get("class_pos_weight_alpha", 0.3)
            species_pos_weight = compute_species_pos_weight(train_df, species_to_idx, alpha=alpha).to(device)
            print(f"  Class pos_weight (alpha={alpha}): max={species_pos_weight.max().item():.2f} min={species_pos_weight.min().item():.2f}")

        if timewise_weight > 0:
            # SEDLoss supports bce/ce/auc/bce_auc base; focal/ASL are not consumed by SEDLoss, so reject them rather than silently ignore
            if focal_gamma > 0 or cfg.get("asl_gamma_neg", 0.0) > 0:
                raise ValueError(
                    f"timewise_weight={timewise_weight} + focal/ASL is incompatible"
                    f" (SEDLoss only supports loss_type in {{bce,ce,auc,bce_auc}})"
                )
            criterion = SEDLoss(timewise_weight=timewise_weight, loss_type=loss_type, pos_weight=species_pos_weight,
                                auc_aux_weight=cfg.get("auc_aux_weight", 0.0))
            _aux = f", auc_aux_weight={cfg.get('auc_aux_weight', 0.0)}" if loss_type == "bce_auc" else ""
            print(f"  Loss: SEDLoss (timewise_weight={timewise_weight}, loss_type={loss_type}{_aux})")
        elif loss_type == "ce":
            criterion = soft_ce_loss
        elif focal_gamma > 0:
            criterion = lambda logits, labels: focal_bce(logits, labels, gamma=focal_gamma)
            print(f"  Loss: Focal BCE (gamma={focal_gamma})")
        elif cfg.get("asl_gamma_neg", 0.0) > 0:
            gn, gp = cfg["asl_gamma_neg"], cfg.get("asl_gamma_pos", 0.0)
            criterion = lambda logits, labels: asl_bce(logits, labels, gamma_neg=gn, gamma_pos=gp)
            print(f"  Loss: ASL (gamma_neg={gn}, gamma_pos={gp})")
        else:
            criterion = nn.BCEWithLogitsLoss()
        scaler    = None  # amp disabled; variable kept to avoid interface changes

        # Resume: load checkpoint_fold{k}.pth (independent per fold)
        start_epoch = 1
        best_auc    = 0.0
        best_fm     = 0.0
        # Early-stop counters: auc / fm tracked independently; stop after N epochs without a new best
        no_improve_auc = 0
        no_improve_fm  = 0
        early_stop_n   = cfg.get("early_stop_epoch", None)
        ckpt_path   = exp_dir / f"checkpoint_fold{cfg['fold_id']}.pth"
        # Track the current best ckpt path (auc and fm independent; delete the old one on a new best)
        best_auc_ckpt_path = None
        best_fm_ckpt_path  = None
        # Track the best epoch's SC raw (for OOF concatenation) + per-epoch SC raw (per-epoch pooled OOF)
        best_sc_y, best_sc_p = None, None
        _epoch_raw = {}
        if cfg.get("resume") and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            # scaler disabled, skip loading
            start_epoch = ckpt["epoch"] + 1
            best_auc    = ckpt.get("best_auc", 0.0)
            best_fm     = ckpt.get("best_fm",  0.0)
            print(f"  Resumed from epoch {ckpt['epoch']}, best_auc={best_auc:.4f}, best_fm={best_fm:.4f}")

        # Shared log.csv (written across folds): the header is written before the fold loop in main(); here we only append
        log_path = exp_dir / "log.csv"

        t0 = time.time()

        for epoch in range(start_epoch, cfg["epochs"] + 1):
            ep_t0      = time.time()
            current_lr = optimizer.param_groups[0]["lr"]  # read param_groups directly (get_last_lr is awkward with OneCycleLR)

            # Re-seed any batch sampler that supports per-epoch reseeding.
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            # EMA starts tracking from ema_start_epoch (skipping the noisy warmup weights).
            ema_active = ema is not None and epoch >= cfg.get("ema_start_epoch", 3)
            # At epoch == start_epoch, rebuild the shadow from the current online weights to remove pretrained cold-start contamination.
            if ema is not None and epoch == cfg.get("ema_start_epoch", 3):
                ema.reinit(model)
                print(f"  [EMA] shadow reinit @ epoch {epoch} (start tracking)")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg, mel_tr, db_tr,
                                     epoch=epoch, scheduler=per_batch_sched,
                                     ema=ema if ema_active else None)
            # The val path follows the head's actual segmentation (cfg["n_segments"]), not duration // 5.
            # e.g. duration=10 + n_segments=1 → n_seg=1, use val_epoch (val_ds auto-crops to 5s wav)
            n_seg = cfg.get("n_segments", cfg["duration"] // 5)
            # EMA val swap: run val with EMA weights (smooths fluctuations)
            swapped = ema_active and cfg.get("ema_val", True)
            if swapped:
                ema.apply_to(model)
            # Main SC val: multi-seg → val_epoch_clip_overlap; otherwise → val_epoch
            if n_seg > 1:
                val_loss, val_auc, val_auc_fm, val_pt, val_pt_fm, _sc_y, _sc_p = val_epoch_clip_overlap(model, val_df, species_to_idx, mel_tr, db_tr, device, cfg)
            else:
                val_loss, val_auc, val_auc_fm, val_pt, val_pt_fm, _sc_y, _sc_p = val_epoch(model, val_loader, criterion, device, cfg, mel_tr, db_tr)
            # Cache this epoch's SC raw (for per-epoch pooled OOF, concatenated across folds at the same epoch)
            if _sc_y is not None:
                _epoch_raw[epoch] = (_sc_y, _sc_p)
            if per_epoch_sched is not None:
                per_epoch_sched.step()

            elapsed = time.time() - ep_t0
            # per-taxon abbreviated display (first 3 chars), showing only taxa with a value
            def _pt_str(d):
                return " ".join(f"{t[:3]}={d[t]:.3f}" for t in _TAXON_NAMES if d.get(t) is not None)
            print(
                f"Epoch {epoch:02d}/{cfg['epochs']} | "
                f"train={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_auc={val_auc:.4f} | val_auc_fm={val_auc_fm:.4f} | "
                f"lr={current_lr:.2e} | {elapsed:.0f}s"
            )
            if val_pt:    print(f"    per_taxon_clip: {_pt_str(val_pt)}")
            if val_pt_fm: print(f"    per_taxon_fm:   {_pt_str(val_pt_fm)}")

            def _fmt(v): return f"{v:.4f}" if v is not None else ""
            with open(log_path, "a", newline="") as f:
                row = [cfg["fold_id"], epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                       f"{val_auc:.4f}", f"{val_auc_fm:.4f}"]
                row += [_fmt(val_pt.get(t))    for t in _TAXON_NAMES]
                row += [_fmt(val_pt_fm.get(t)) for t in _TAXON_NAMES]
                row += [f"{current_lr:.2e}", f"{elapsed:.0f}"]
                csv.writer(f).writerow(row)

            # EMA save strategy:
            # - ema_save_best=True (default): best/snapshot stores EMA weights (model still in swap state)
            # - ema_save_best=False: restore first, best/snapshot stores online weights
            if swapped and not cfg.get("ema_save_best", True):
                ema.restore(model)
                swapped = False

            if val_auc > best_auc:
                best_auc = val_auc
                no_improve_auc = 0
                # Also cache the current best epoch's SC val raw (for cross-fold OOF concatenation, keyed on auc)
                best_sc_y, best_sc_p = _sc_y, _sc_p
                # Naming: best_model_fold{k}_ep{N}_auc_{S:.4f}.pth; delete the old one on a new best
                if best_auc_ckpt_path is not None and best_auc_ckpt_path.exists():
                    best_auc_ckpt_path.unlink()
                best_auc_ckpt_path = exp_dir / f"best_model_fold{cfg['fold_id']}_ep{epoch}_auc_{best_auc:.4f}.pth"
                torch.save(_build_ckpt_payload(cfg, model), best_auc_ckpt_path)
                print(f"  → best val_auc={best_auc:.4f}  saved: {best_auc_ckpt_path.name}")
            else:
                no_improve_auc += 1

            if val_auc_fm > best_fm:
                best_fm = val_auc_fm
                no_improve_fm = 0
                if best_fm_ckpt_path is not None and best_fm_ckpt_path.exists():
                    best_fm_ckpt_path.unlink()
                best_fm_ckpt_path = exp_dir / f"best_model_fold{cfg['fold_id']}_ep{epoch}_fm_{best_fm:.4f}.pth"
                torch.save(_build_ckpt_payload(cfg, model), best_fm_ckpt_path)
                print(f"  → best val_auc_fm={best_fm:.4f}  saved: {best_fm_ckpt_path.name}")
            else:
                no_improve_fm += 1

            # Save an extra model snapshot at specified epochs (same format as best_model, directly submittable)
            if epoch in cfg.get("save_epochs", []):
                save_path = exp_dir / f"model_ep{epoch}_fold{cfg['fold_id']}.pth"
                torch.save(_build_ckpt_payload(cfg, model), save_path)
                print(f"  → epoch {epoch} snapshot saved: {save_path.name}")

            # EMA restore: online weights must be restored before saving the checkpoint (resume uses online; EMA is tracked separately via shadow)
            if swapped:
                ema.restore(model)
                swapped = False

            # Save a full checkpoint each epoch for resuming
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                # scaler disabled
                "best_auc":   best_auc,
                "best_fm":    best_fm,
                "ema_shadow": ema.shadow if ema is not None else None,  # EMA state for resume
            }, ckpt_path)

            # Early stop: exit this fold after N consecutive epochs where neither val_auc nor val_auc_fm improves the best
            if early_stop_n is not None and no_improve_auc >= early_stop_n and no_improve_fm >= early_stop_n:
                print(f"  [EarlyStop] no_improve_auc={no_improve_auc} no_improve_fm={no_improve_fm} ≥ {early_stop_n} → stop fold @ ep {epoch}")
                break

        # Training finished normally; delete checkpoint.pth to save disk (only needed on interruption)
        if ckpt_path.exists():
            ckpt_path.unlink()

        total = time.time() - t0
        print(f"\nFold {cfg['fold_id']} done. Best val_auc={best_auc:.4f} | val_auc_fm={best_fm:.4f} | {total/60:.1f} min")
        if best_auc_ckpt_path is not None:
            print(f"Saved (auc): {best_auc_ckpt_path.name}")
        if best_fm_ckpt_path is not None:
            print(f"Saved (fm):  {best_fm_ckpt_path.name}")

        # Accumulate this fold's best-epoch SC raw into the global OOF (concatenated for macro AUC at the end of the fold loop)
        if best_sc_y is not None and best_sc_p is not None:
            _oof_sc_y_list.append(best_sc_y)
            _oof_sc_p_list.append(best_sc_p)
        # Accumulate per-epoch SC raw (for per-epoch pooled OOF)
        if _epoch_raw:
            _per_fold_epoch_raw[_fk] = _epoch_raw
        # Accumulate this fold's best val_auc / val_auc_fm (for oof_summary)
        _fold_bests[_fk] = (best_auc, best_fm)

        # ─── Export ONNX + CPU sanity check immediately after this fold finishes ───
        # Done atomically per fold (so earlier folds still have ONNX even if training later fails)
        print(f"\n  --- ONNX export fold {cfg['fold_id']} ---")
        try:
            export_onnx_for_fold(cfg, exp_dir, cfg['fold_id'], n_classes)
        except Exception as e:
            print(f"  fold {cfg['fold_id']}: ONNX export failed: {type(e).__name__}: {e}")

    # ─── OOF evaluation: concatenate every fold's best-epoch SC val raw → full-set macro AUC ───
    _oof_lines = []

    # First write each fold's best val_auc / val_auc_fm and their mean (at the top of the summary for a quick glance)
    if _fold_bests:
        _aucs = [v[0] for v in _fold_bests.values()]
        _fms  = [v[1] for v in _fold_bests.values()]
        mean_auc = sum(_aucs) / len(_aucs)
        mean_fm  = sum(_fms)  / len(_fms)
        print(f"\n{'='*60}\n  Per-fold best val ({len(_fold_bests)} folds)\n{'='*60}")
        _oof_lines.append(f"Per-fold best val ({len(_fold_bests)} folds):")
        for fid in sorted(_fold_bests):
            ba, bf = _fold_bests[fid]
            print(f"  fold {fid}: val_auc={ba:.4f}  val_auc_fm={bf:.4f}")
            _oof_lines.append(f"  fold {fid}: val_auc={ba:.4f}  val_auc_fm={bf:.4f}")
        print(f"  mean   : val_auc={mean_auc:.4f}  val_auc_fm={mean_fm:.4f}")
        _oof_lines.append(f"  mean   : val_auc={mean_auc:.4f}  val_auc_fm={mean_fm:.4f}")
        _oof_lines.append("")

    if _oof_sc_y_list:
        all_y = np.concatenate(_oof_sc_y_list)
        all_p = np.concatenate(_oof_sc_p_list)
        oof_macro    = compute_auc(all_y, all_p)
        oof_per_taxon = compute_per_taxon_auc(all_y, all_p)
        # rank-norm OOF: re-score each fold by percentile (removes cross-fold prob-scale differences, closer to real ensemble behavior)
        p_rn_list  = [rankdata(p, axis=0) / max(len(p), 1) for p in _oof_sc_p_list]
        all_p_rn   = np.concatenate(p_rn_list)
        oof_macro_rn = compute_auc(all_y, all_p_rn)
        oof_per_taxon_rn = compute_per_taxon_auc(all_y, all_p_rn)
        print(f"\n{'='*60}\n  OOF RESULTS (best-attn ckpts on SC val, {len(_oof_sc_y_list)} folds)\n{'='*60}")
        print(f"  Total samples: {len(all_y)}")
        print(f"  Macro AUC (raw)       : {oof_macro:.4f}")
        print(f"  Macro AUC (rank-norm) : {oof_macro_rn:.4f}  (per-fold percentile, removes calibration mismatch)")
        _oof_lines.append(f"OOF (best-attn ckpts, SC val, {len(_oof_sc_y_list)} folds, n_samples={len(all_y)})")
        _oof_lines.append(f"Macro AUC (raw)       : {oof_macro:.4f}")
        _oof_lines.append(f"Macro AUC (rank-norm) : {oof_macro_rn:.4f}")
        for t in _TAXON_NAMES:
            v    = oof_per_taxon.get(t)
            v_rn = oof_per_taxon_rn.get(t)
            if v is not None:
                print(f"    {t:<12}: raw={v:.4f}  rank-norm={v_rn:.4f}")
                _oof_lines.append(f"  {t}: raw={v:.4f}  rank-norm={v_rn:.4f}")
    else:
        print("\n[OOF] no fold has best_sc raw (all folds were skipped) — skipping OOF best evaluation")

    # ─── Per-epoch pooled OOF: at each epoch, concatenate SC val across folds → macro AUC (to find the overall best epoch) ───
    if _per_fold_epoch_raw:
        all_folds = list(_per_fold_epoch_raw.keys())
        common_eps = set.intersection(*[set(_per_fold_epoch_raw[fk]) for fk in all_folds])
        if common_eps:
            print(f"\n{'='*60}\n  Per-epoch pooled OOF macro AUC ({len(all_folds)} folds)\n{'='*60}")
            _oof_lines.append("")
            _oof_lines.append(f"Per-epoch pooled OOF ({len(all_folds)} folds):")
            for ep in sorted(common_eps):
                ys = [_per_fold_epoch_raw[fk][ep][0] for fk in all_folds]
                ps = [_per_fold_epoch_raw[fk][ep][1] for fk in all_folds]
                macro = compute_auc(np.concatenate(ys), np.concatenate(ps))
                print(f"  Ep{ep:02d}: macro={macro:.4f}")
                _oof_lines.append(f"  Ep{ep:02d}: {macro:.4f}")

    if _oof_lines:
        with open(exp_dir / "oof_summary.txt", "w") as f:
            f.write("\n".join(_oof_lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",  action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume from the latest exp's checkpoint.pth")
    parser.add_argument("--exp_name", type=str, default=None, help="specify the experiment name, skipping auto-numbering")
    args = parser.parse_args()
    if args.debug:
        CFG["debug"] = True
    if args.resume:
        CFG["resume"] = True
    if args.exp_name:
        CFG["exp_name"] = args.exp_name
    main(CFG)
