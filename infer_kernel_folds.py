"""dataloader
BirdCLEF 2026 Multi-Fold Inference Kernel
==========================================
Loads the multiple sed_fold{k}_epoch{N}_score{S}.onnx files exported by training
and performs a fold ensemble. Runs in a Kaggle notebook environment.

Data sources:
  - Competition data:  /kaggle/input/competitions/birdclef-2026/
  - Multi-fold ONNX:   /kaggle/input/birdclef2026-folds-models/
                       should contain: sed_fold0_*.onnx, sed_fold1_*.onnx, ..., config.json
  - onnxruntime wheel: /kaggle/input/datasets/yannan90/onnxruntime-and-openvino-wheels/

ONNX input:  (B, C, H, W) mel image
ONNX output: (clip_logits, frame_logits)
  - clip_logits:  (B, n_classes)
  - frame_logits: (B, T', n_classes)

Ensemble order:
  per segment -> per fold inference -> sigmoid + fm_blend -> fold-mean -> TTA blend
"""

# Offline install of onnxruntime (the Kaggle competition env runs with enable_internet=false,
# so a local wheel is required). Must happen before importing onnxruntime / ort.
# Guard: install only when the Kaggle wheel actually exists; skip when this file is merely
# imported in another environment for tagging.
import subprocess, sys, os, glob as _glob

# Inference backend switch: "ort"=ONNX Runtime CPU EP; "ov_fp32"=OpenVINO native FP32
# INFERENCE_BACKEND = "ov_fp32"
INFERENCE_BACKEND = "ort"

_KAGGLE_WHEEL = ("/kaggle/input/datasets/yannan90/onnxruntime-and-openvino-wheels/"
                  "onnxruntime-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
if os.path.exists(_KAGGLE_WHEEL):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _KAGGLE_WHEEL], check=True)

# When INFERENCE_BACKEND="ov_*", install OpenVINO (wheel already present in the same dataset);
# install telemetry first, then openvino, with --no-deps to avoid any network access.
if INFERENCE_BACKEND.startswith("ov_"):
    _ov_tel  = _glob.glob("/kaggle/input/**/openvino_telemetry-*.whl", recursive=True)
    _ov_main = [p for p in _glob.glob("/kaggle/input/**/openvino-*.whl", recursive=True)
                if "telemetry" not in os.path.basename(p)]
    for _whl in _ov_tel + _ov_main:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", _whl], check=False)
    print(f"[boot] Installed openvino for INFERENCE_BACKEND={INFERENCE_BACKEND}")

import os

# ============================================================
# Multi-group model ensemble config (each group has its own cfg / mel / ONNX / fm_blend / weight)
# ============================================================
# Each group is an independent model directory holding its own sed_fold*.onnx + config.json.
# At inference each group runs its full mel + ONNX -> (12, 234) prob, and the groups are then
# combined with a weighted prob-mean across groups.
# Post-processing (smoothing / mirror) is applied uniformly after the ensemble.


MODEL_GROUPS = [
    {
        "name":           "main",
        "model_dir":      "/kaggle/input/birdclef2026-folds-models",
        "overlap":        True,
        "ensemble_folds": None,   # None = use all folds found in model_dir
        "best_type":      "fm",
        "fm_blend_w":     None,
        "weight":         1.0,
    },
]

# ---- Throughput ----
# Each DataLoader worker subprocess does I/O + mel + ONNX forward + ensemble; the main thread
# only does post-processing. Using 4 workers x 1 thread instead of 1 main thread x 4 threads:
# same total CPU, but adds cross-file parallelism and hides mel/IO latency.
NUM_WORKERS     = 4                                                     # OpenVINO must run in the main process: destroying an OV session inside a worker can segfault, and os._exit only covers main; num_workers=0 keeps OV in main so os._exit fully controls teardown
WORKER_INTRA_OP = 1                                                     # OpenVINO single process saturates 4 cores (INFERENCE_NUM_THREADS=4)

# Non-overlapping windows: each group specifies "overlap" in MODEL_GROUPS (True=11 overlapping sliding windows / False=6 non-overlapping windows); no global default, required per group. Inter-window aggregation is fixed to overlap-avg (mean).

# Cross-group (cross-model) fusion: "mean"=arithmetic weighted (default); "gaus"=RMS; "vlom"=(geometric mean + RMS)/2
ENSEMBLE_AGG = "mean"

# ---- Pipeline: wave -> slice -> batched mel -> db -> norm ----
# DB_SCOPE controls the max-reference scope of AmplitudeToDB:
#   "duration": per-segment db, reference = max within the duration length (aligned with per-duration training)
#   "file":     all segments share db, reference = max over the full 60s file
DB_SCOPE = "duration"

# ---- Local test switch ----
DEBUG_FALLBACK_N = 600                                                    # when test_soundscapes is empty, use the first N train_soundscapes files instead (on Kaggle submission test is non-empty, so this fallback does not trigger)

# ---- Post-processing (inference side only) ----
USE_TIME_SMOOTH = True                                                  # master switch for time smoothing (False=no smoothing)
SMOOTH_MODE     = "v2"                                                  # "texture"=3-tap; "gaussian"=5-tap Gaussian; "v2"=two-layer cascade
SMOOTH_EVENT    = (0.20, 0.60, 0.20)                                    # texture mode: event-class 3-tap (Aves/Mammalia/Reptilia, preserve temporal precision)
SMOOTH_TEXTURE  = (0.35, 0.30, 0.35)                                    # texture mode: texture-class 3-tap (Insecta/Amphibia, strong diffusion)
SMOOTH_GAUSSIAN = (0.1, 0.2, 0.4, 0.2, 0.1)                             # gaussian mode: 5-tap, all classes treated equally
SMOOTH_V2_TEXTURE_ALPHA = 0.35                                          # v2 Layer1: texture-class mean smoothing alpha
SMOOTH_V2_EVENT_ALPHA   = 0.15                                          # v2 Layer1: event-class local-max smoothing alpha
SMOOTH_V2_DELTA_ALPHA   = 0.20                                          # v2 Layer2: adaptive delta-shift base alpha

USE_FILE_POST   = True                                                  # file-level top probability multiplication (each file's predictions x that file's per-species top probability)

# ---- Frame-shift TTA (applies for both n_seg=1 and n_seg>1) ----
# Main keeps the A path (max-then-avg); the TTA branch: frame_probs overlap-avg -> long timeline -> per-slot +/-d frame shift max.
# Weighted fusion: w * main + (1-w)/2 * (left + right). No extra forward pass.
TTA             = False
TTA_SHIFT_SEC   = 1.0       # shift in seconds (frames = round(TTA_SHIFT_SEC x T'/duration))
TTA_MAIN_W      = 0.5       # main weight (left/right each get (1-w)/2)

# ---- Mirror pairs (take max across same inat_taxon=47158 sonotype classes to keep them in sync) ----
USE_MIRROR_PAIRS = False                                                # default off: heuristic merge, decide via A/B comparison
MIRROR_PAIRS = (                                                        # 4 groups / 10 sons, all Insecta sonotypes under the 47158 parent taxon
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
)


import os
import io
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import timm
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG defaults (newer-format checkpoints override n_mels/n_fft/backbone)
# ============================================================
CFG = {
    "data_dir":    "/kaggle/input/competitions/birdclef-2026",
    "sample_rate": 32000,                # global: all groups must use 32k mono input
}

SEGMENT_SEC = 5
N_SEGMENTS  = 12  # 1 minute / 5 seconds




# ============================================================
# Audio -> image
# ============================================================

def build_mel_transform(cfg):
    """Build the Mel front-end from the (checkpoint) config; returns (MelSpectrogram, AmplitudeToDB)."""
    db = T.AmplitudeToDB(top_db=cfg.get("top_db", 80))

    def _make(n_fft, hop, win, n_mels, fmin, fmax, mel_scale, mel_filterbank_norm):
        kw = {}
        if mel_scale is not None:           kw["mel_scale"] = mel_scale
        if mel_filterbank_norm is not None: kw["norm"]      = mel_filterbank_norm
        if win is not None:                 kw["win_length"] = win
        # newer checkpoints carry a mel_normalized field for STFT normalization
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


def normalize_mel(mel, mel_norm="minmax"):
    """mel normalization (vectorized, reduce over the last two dims, keepdim=True).
    Works for a single sample (n_mels, T') and batched (N, n_mels, T') -- batched input is
    normalized per sample.
    - "zscore_minmax": z-score then min-max -> [0,1]
    - "minmax": direct min-max -> [0,1]
    - "zscore": direct z-score (unbounded)
    """
    dims = (-2, -1)
    if mel_norm == "zscore_minmax":
        mean = mel.mean(dim=dims, keepdim=True)
        std  = mel.std(dim=dims, keepdim=True) + 1e-6
        mel  = (mel - mean) / std
        norm_min = torch.amin(mel, dim=dims, keepdim=True)
        norm_max = torch.amax(mel, dim=dims, keepdim=True)
        mel = (mel - norm_min) / (norm_max - norm_min + 1e-7)
    elif mel_norm == "minmax":
        mel_min = torch.amin(mel, dim=dims, keepdim=True)
        mel_max = torch.amax(mel, dim=dims, keepdim=True)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-7)
    else:
        mean = mel.mean(dim=dims, keepdim=True)
        std  = mel.std(dim=dims, keepdim=True) + 1e-6
        mel = (mel - mean) / std
    return mel


def wav_to_images(
        wav, starts_sec, dur_sec, sample_rate,
        img_size, in_chans, mel_tr, db_tr, mel_norm,
    ):
    """wave -> slice -> batched mel -> db -> norm -> resize -> channel repeat -> (N, in_chans, H, W).
    DB_SCOPE:
      - "duration": mel.unsqueeze(1) adds a fake channel before torchaudio -> per-sample branch (each segment's own max)
      - "file":     3D mel fed directly to torchaudio -> batch-global branch (all segments in this batch share max = full-file max)
    norm is always per-sample (reduce over the last two dims, keep the batch dim).
    mel_tr is a single MelSpectrogram -> 1 channel (then repeated to in_chans).
    """
    seg_samples = int(round(dur_sec * sample_rate))
    segs = []
    for s_sec in starts_sec:
        start_s = int(round(s_sec * sample_rate))
        seg = wav[start_s:start_s + seg_samples]
        if len(seg) < seg_samples:
            seg = F.pad(seg, (0, seg_samples - len(seg)))
        segs.append(seg)
    wavs = torch.stack(segs, 0)                              # (N, T_seg)

    # single-mel path
    mel = mel_tr(wavs)                                       # (N, n_mels, T')
    if DB_SCOPE == "duration":
        mel = db_tr(mel.unsqueeze(1)).squeeze(1)             # fake channel → per-sample db
    else:  # "file"
        mel = db_tr(mel)                                     # 3D -> batch-global = this batch shares max
    mel = normalize_mel(mel, mel_norm)                       # per-sample norm

    mel = mel.unsqueeze(1)                                   # (N, 1, n_mels, T')
    if isinstance(img_size, (list, tuple)):
        mel = torch.nn.functional.interpolate(
            mel, size=tuple(img_size),
            mode="bilinear", align_corners=False)
    elif img_size and img_size > 0:
        mel = torch.nn.functional.interpolate(
            mel, size=(img_size, img_size),
            mode="bilinear", align_corners=False)
    return mel.repeat(1, in_chans, 1, 1)                     # (N, in_chans, H, W)


# ============================================================
# Post-processing helpers
# ============================================================

def load_texture_mask(label_cols):
    """Read class_name from taxonomy.csv and return a boolean mask for texture classes (Insecta/Amphibia)."""
    tax = pd.read_csv(os.path.join(CFG["data_dir"], "taxonomy.csv"))
    tax["primary_label"] = tax["primary_label"].astype(str)
    texture_set = set(tax[tax["class_name"].isin(["Insecta", "Amphibia"])]["primary_label"])
    return np.array([l in texture_set for l in label_cols], dtype=bool)


# Module-level texture_mask (used for taxon aggregation): loaded once in main and assigned to
# the global, inherited by forked workers (same mechanism as CFG/ENSEMBLE_AGG globals).
_TEXTURE_MASK = None


def _smooth_v2_texture(p, mask, alpha):
    """Per-file mean smoothing (mask: bool (n_cls,)).
    Formula: (1-a)x + 0.5a*(prev + next), edge-clamped boundaries.
    Used for texture classes (Amphibia/Insecta).
    """
    if alpha <= 0 or not mask.any():
        return p
    x      = p[:, mask]                                                 # (N_SEGMENTS, k)
    prev_x = np.concatenate([x[:1], x[:-1]], axis=0)                    # edge clamp (first window prev=itself)
    next_x = np.concatenate([x[1:], x[-1:]], axis=0)                    # edge clamp (last window next=itself)
    out = p.copy()
    out[:, mask] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return out


def _smooth_v2_event(p, mask, alpha):
    """Per-file local-max smoothing.
    Formula: (1-a)x + a*max(x, prev, next)  <- 3-tap max including x itself.
    One-directional lift: peaks stay (max=itself), troughs are filled up by the neighbor max.
    Used for event classes (Aves/Mammalia/Reptilia).
    """
    if alpha <= 0 or not mask.any():
        return p
    x      = p[:, mask]
    prev_x = np.concatenate([x[:1], x[:-1]], axis=0)
    next_x = np.concatenate([x[1:], x[-1:]], axis=0)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))               # max(x, prev, next), not max(prev, next)
    out = p.copy()
    out[:, mask] = (1.0 - alpha) * x + alpha * local_max
    return out


def _smooth_v2_adaptive_delta(p, base_alpha):
    """Per-file adaptive delta smoothing.
    Formula: a = base_alpha * (1 - max_class_conf_per_window)
             new[i] = (1-a)*old[i] + a*(old[i-1] + old[i+1])/2
    Boundaries: i=0 / i=N-1 keep their original value (for i in range(1, n_windows-1)).
    All classes treated equally; no texture/event distinction.
    Important: the loop only reads p (original) and writes to out (a copy) to avoid iterative contamination.
    """
    if base_alpha <= 0:
        return p
    out = p.copy()
    n   = len(p)
    for i in range(1, n - 1):
        conf         = p[i].max(axis=-1, keepdims=True)                 # (1,) highest class confidence in this window
        a            = base_alpha * (1.0 - conf)                        # (1,) high confidence -> small alpha
        neighbor_avg = (p[i-1] + p[i+1]) / 2.0                          # (n_cls,)
        out[i]       = (1.0 - a) * p[i] + a * neighbor_avg              # broadcast (1,) over (n_cls,)
    return out


def time_smooth(preds, is_texture):
    """Time smoothing (preds: (N_SEGMENTS, n_classes), edge padding at boundaries).
    SMOOTH_MODE:
      - "texture":  3-tap; texture classes (Insecta/Amphibia) use SMOOTH_TEXTURE, event classes use SMOOTH_EVENT
      - "gaussian": 5-tap Gaussian, all classes equal
      - "v2":       two-layer cascade
                    Layer1: texture mean (a=0.35) + event local-max (a=0.15)
                    Layer2: adaptive delta-shift (base_a=0.20, alpha scaled by the highest in-window confidence)
                    Note: layer1 runs in prob space here; the formula and hyperparameters are
                          identical but math behavior differs slightly from a logit-space version.
    """
    if N_SEGMENTS <= 2:
        return preds
    if SMOOTH_MODE == "v2":
        # Layer 1: texture mean / event local-max
        out = _smooth_v2_texture(preds,  is_texture, SMOOTH_V2_TEXTURE_ALPHA)
        out = _smooth_v2_event  (out,   ~is_texture, SMOOTH_V2_EVENT_ALPHA)
        # Layer 2: adaptive delta-shift (run again on the layer1 output)
        out = _smooth_v2_adaptive_delta(out, SMOOTH_V2_DELTA_ALPHA)
        return out
    if SMOOTH_MODE == "gaussian":
        # 5-tap Gaussian: all classes treated equally
        w = np.array(SMOOTH_GAUSSIAN, dtype=np.float64)
        w = w / w.sum()
        pad = np.pad(preds, ((2, 2), (0, 0)), mode="edge")
        return (w[0]*pad[:-4] + w[1]*pad[1:-3] + w[2]*pad[2:-2]
                + w[3]*pad[3:-1] + w[4]*pad[4:])
    # default texture-aware 3-tap
    def _smooth3(p, w):
        pad = np.pad(p, ((1, 1), (0, 0)), mode="edge")
        return w[0] * pad[:-2] + w[1] * pad[1:-1] + w[2] * pad[2:]
    out = preds.copy()
    if is_texture.any():
        out[:, is_texture]  = _smooth3(preds[:, is_texture],  np.array(SMOOTH_TEXTURE))
    if (~is_texture).any():
        out[:, ~is_texture] = _smooth3(preds[:, ~is_texture], np.array(SMOOTH_EVENT))
    return out


def apply_mirror_pairs(probs, mirror_idx_groups):
    """Mirror pairs: take the max across a same-sonotype group to keep them in sync.
    probs:             (n_seg, n_cls) per-file probabilities
    mirror_idx_groups: list of int lists, each inner list is the cls column indices of one group (invalid labels already filtered)
    return: a copy with every column in a group overwritten by the group max
    """
    if not mirror_idx_groups:
        return probs
    out = probs.copy()
    for idx in mirror_idx_groups:
        if len(idx) >= 2:
            mx = out[:, idx].max(axis=1, keepdims=True)                     # (n_seg, 1)
            out[:, idx] = mx                                                # broadcast to (n_seg, len(idx))
    return out


# ============================================================
# DataLoader: parallel loading of .ogg files (I/O prefetch)
# ============================================================

class SoundscapeDataset(Dataset):
    """Each worker subprocess does I/O + full-group forward + ensemble -> returns (stem, probs).
    Each worker lazily sets up the ONNX sessions of all groups; the main thread only receives
    probs and does post-processing.
    """
    def __init__(self, sc_files, sample_rate, total_samples,
                 groups_cfg, weights, n_classes, intra_op=1):
        self.sc_files      = sc_files
        self.sample_rate   = sample_rate
        self.total_samples = total_samples
        self.groups_cfg    = groups_cfg          # raw MODEL_GROUPS dict list (picklable)
        self.weights       = weights             # list[float], already normalized
        self.n_classes     = n_classes
        self.intra_op      = intra_op
        self._groups       = None                # lazily built in the worker subprocess

    def __len__(self):
        return len(self.sc_files)

    def _build_groups(self):
        """Set up all groups on the worker's first __getitem__; verbose only on worker 0 to avoid log spam."""
        try:
            info = torch.utils.data.get_worker_info()
            verbose = (info is None or info.id == 0)
        except Exception:
            verbose = True
        self._groups = [setup_group(gc, intra_op=self.intra_op, verbose=verbose)
                        for gc in self.groups_cfg]

    def __getitem__(self, idx):
        sc_path = self.sc_files[idx]
        wav, sr = torchaudio.load(str(sc_path))
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        # mono fast path (BC2026 ogg is mono by default; .mean(0) costs an extra reduce)
        wav = wav[0] if wav.shape[0] == 1 else wav.mean(0)
        if len(wav) < self.total_samples:
            wav = F.pad(wav, (0, self.total_samples - len(wav)))

        if self._groups is None:
            self._build_groups()
        # cross-group fusion: ENSEMBLE_AGG in {mean, gaus(RMS), vlom((geometric mean + RMS)/2)}; weights already normalized
        ensemble_probs = np.zeros((N_SEGMENTS, self.n_classes), dtype=np.float64)
        gaus_acc  = np.zeros_like(ensemble_probs) if ENSEMBLE_AGG in ("gaus", "vlom") else None
        geo_acc   = np.zeros_like(ensemble_probs) if ENSEMBLE_AGG == "vlom" else None
        mel_cache = {}
        for g, w in zip(self._groups, self.weights):
            probs_g = predict_one_file_per_group(wav, g, self.n_classes, mel_cache=mel_cache)
            ensemble_probs += w * probs_g                              # arithmetic weighted (mean)
            if gaus_acc  is not None: gaus_acc  += w * probs_g ** 2    # weighted sum of squares -> RMS
            if geo_acc   is not None: geo_acc   += w * np.log(probs_g + 1e-9)  # weighted log -> geometric mean
        if ENSEMBLE_AGG == "mean":
            out = ensemble_probs
        elif ENSEMBLE_AGG == "gaus":
            out = np.sqrt(gaus_acc)
        elif ENSEMBLE_AGG == "vlom":
            out = 0.5 * (np.exp(geo_acc) + np.sqrt(gaus_acc))
        else:
            raise ValueError(f"unknown ENSEMBLE_AGG={ENSEMBLE_AGG!r}")
        return sc_path.stem, out.astype(np.float32)


def _sc_collate(batch):
    """batch_size=1 case: return the single (stem, mel_full) directly, so the default collate does not stack stem into a list."""
    return batch[0]


# ============================================================
# OpenVINO compiled-model wrapper (mimic ORT InferenceSession API)
# ============================================================
# Constructed when INFERENCE_BACKEND='ov_*' so downstream code like predict_with_folds needs no changes.
class _OVSession:
    class _Inp:
        __slots__ = ("name",)
        def __init__(self, name): self.name = name

    def __init__(self, onnx_path, intra_op=1, precision="f32"):
        import openvino as _ov
        core = _ov.Core()
        model = core.read_model(onnx_path)
        config = {"INFERENCE_NUM_THREADS": int(max(1, intra_op)),
                  "PERFORMANCE_HINT": "LATENCY",
                  "INFERENCE_PRECISION_HINT": precision}
        self._compiled = core.compile_model(model, "CPU", config)
        self._in_name = self._compiled.input(0).get_any_name()
        self._n_outs  = len(self._compiled.outputs)

    def get_inputs(self):
        return [self._Inp(self._in_name)]

    def run(self, output_names, input_dict):
        arr = next(iter(input_dict.values()))
        outs = self._compiled([arr])
        return [outs[i] for i in range(self._n_outs)]


# ============================================================
# Multi-fold ONNX session
# ============================================================

def load_fold_sessions(model_dir, pattern="sed_fold*.onnx", selected_folds=None,
                       best_type="auc", providers=None,
                       intra_op=None, verbose=True):
    """Load all ONNX under model_dir/**/sed_fold*.onnx, return [(fold_id, session, path), ...].
    selected_folds: None=load all; list of int=load only these folds (e.g. [0,2,3]).
    best_type: "auc"=keep only ONNX whose filename contains _auc_ (older unlabeled ONNX default to auc); "fm"=keep only _fm_; "all"=load both.
    intra_op: None=auto N_CPU (best when running ONNX on the main thread); int=explicit (usually 1 when running ONNX inside a worker)
    verbose: when False, suppress all prints (avoids multi-copy log spam during in-worker build)
    Each ONNX input name comes from sess.get_inputs()[0].name; output order is fixed [clip_logits, frame_logits].
    """
    import re as _re
    import onnxruntime as ort
    paths = sorted(glob.glob(os.path.join(model_dir, "**", pattern), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No ONNX matching {pattern} under {model_dir}")

    # Filter by best_type: _fm_ files are "fm"; _auc_ or unlabeled (older naming) are "auc"
    def _kind(name):
        if "_fm_" in name:
            return "fm"
        return "auc"
    if best_type not in ("auc", "fm", "all"):
        raise ValueError(f"BEST_TYPE must be 'auc'/'fm'/'all', got {best_type!r}")
    if best_type != "all":
        kept = [p for p in paths if _kind(os.path.basename(p)) == best_type]
        skipped = [p for p in paths if _kind(os.path.basename(p)) != best_type]
        if verbose:
            for p in skipped:
                print(f"  Skip [{_kind(os.path.basename(p))}]: {os.path.basename(p)} (BEST_TYPE={best_type})")
        if not kept:
            raise FileNotFoundError(f"BEST_TYPE={best_type} matched no ONNX")
        paths = kept

    # Parse fold_id and filter by selected_folds
    paths_with_fold = []
    for p in paths:
        m = _re.search(r"sed_fold(\d+)", os.path.basename(p))
        fid = int(m.group(1)) if m else len(paths_with_fold)
        paths_with_fold.append((fid, p))
    if selected_folds is not None:
        sel = set(selected_folds)
        all_available_fids = sorted({fid for fid, _ in paths_with_fold})        # record available folds before filtering, for error messages
        skipped = [(fid, p) for fid, p in paths_with_fold if fid not in sel]
        paths_with_fold = [(fid, p) for fid, p in paths_with_fold if fid in sel]
        if verbose:
            for fid, p in skipped:
                print(f"  Skip fold {fid}: {os.path.basename(p)} (not in ensemble_folds)")
        if not paths_with_fold:
            raise ValueError(f"ensemble_folds={selected_folds} matched no ONNX "
                             f"(available fold ids: {all_available_fids})")
        missing = sel - {fid for fid, _ in paths_with_fold}
        if missing and verbose:
            print(f"  WARN: ensemble_folds entries {sorted(missing)} have no matching ONNX")

    sess_opts = ort.SessionOptions()
    # intra_op: use full N_CPU when running ONNX on the main thread
    # pass 1 explicitly when running ONNX in workers (4 workers x 1 thread = 4 vCPU saturated; cross-file parallelism replaces intra-op parallelism)
    if intra_op is None:
        n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
        sess_opts.intra_op_num_threads = max(1, n_cpu)
    else:
        sess_opts.intra_op_num_threads = max(1, intra_op)
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _providers = providers if providers is not None else ['CPUExecutionProvider']
    _intra_used = sess_opts.intra_op_num_threads
    sessions = []
    for fold_id, p in paths_with_fold:
        if INFERENCE_BACKEND == "ort":
            sess = ort.InferenceSession(p, sess_opts, providers=_providers)
        elif INFERENCE_BACKEND in ("ov_fp32", "ov_fp16"):
            _prec = "f16" if INFERENCE_BACKEND == "ov_fp16" else "f32"
            sess = _OVSession(p, intra_op=_intra_used, precision=_prec)
        else:
            raise ValueError(f"unknown INFERENCE_BACKEND={INFERENCE_BACKEND!r}")
        size_mb = os.path.getsize(p) / 1024 / 1024
        sessions.append((fold_id, sess, p))
        if verbose:
            print(f"  Loaded fold {fold_id}: {os.path.basename(p)} ({size_mb:.1f}MB)")
    if verbose:
        print(f"  Ensemble: {len(sessions)} folds {sorted(fid for fid, _, _ in sessions)}  "
              f"intra_op={_intra_used}  backend={INFERENCE_BACKEND}")
    return sessions


def predict_with_folds(sessions, imgs, fm_blend_w, n_segments=1,
                       shift_sec=0.0, duration_sec=None):
    """For each segment, run every fold's ONNX -> sigmoid + fm_blend -> fold-mean.
    imgs: (N, C, H, W) torch tensor on CPU
    n_segments=1: clip_logits (N, n_classes), frame_logits (N, T', n_classes); fm=frame.max(t)
    n_segments>1: clip_logits (N, n_segments, n_classes), frame_logits (N, T', n_classes);
                  split fm evenly along time into n_segments -> per-segment max -> stack

    When shift_sec > 0, enable frame-shift TTA (shares one forward with main, no extra compute):
        - main output matches shift_sec=0 (caller still does overlap-avg)
        - also stitch frame_probs -> long timeline (n_seg=1 degenerates to concat) -> per-slot +/-d frame max
        - returns a 3-tuple (main_per_win, fm_left_per_slot, fm_right_per_slot)
        - left/right do not mix clip; caller weights main with TTA_MAIN_W (when fm_blend_w<1.0, clip only flows through main)
        - d = round(shift_sec x T'/duration_sec); when d not in (0, step) it degrades to main only
    shift_sec = 0 (default) behavior unchanged:
        - single segment returns (N, n_classes); multi segment (N, n_segments, n_classes) numpy probs
    """
    arr = imgs.cpu().numpy()
    enable_shift = shift_sec > 0 and duration_sec is not None and duration_sec > 0
    fold_probs, fold_fm_l, fold_fm_r = [], [], []

    for fold_id, sess, _ in sessions:
        in_name = sess.get_inputs()[0].name
        clip_logits, frame_logits = sess.run(None, {in_name: arr})
        N_win, T, n_cls = frame_logits.shape
        step = T // n_segments                                              # number of frames per 5s

        # -- Main path (per-window blend) --
        if n_segments > 1:
            fm = np.stack(
                [frame_logits[:, s*step:(s+1)*step].max(axis=1) for s in range(n_segments)], axis=1)
        else:
            fm = frame_logits.max(axis=1)        # (N, n_classes)
        p_clip = 1.0 / (1.0 + np.exp(-clip_logits))
        p_fm   = 1.0 / (1.0 + np.exp(-fm))
        if fm_blend_w == 0.0:    probs = p_clip
        elif fm_blend_w == 1.0:  probs = p_fm
        else:                    probs = (1.0 - fm_blend_w) * p_clip + fm_blend_w * p_fm
        fold_probs.append(probs)

        # -- TTA: stitch long timeline + per-slot +/-d frame shift max --
        if enable_shift:
            d = int(round(shift_sec * T / duration_sec))
            if 0 < d < step:
                # frame_probs -> per-frame overlap-avg -> long timeline (when n_seg=1 the windows do not overlap, equivalent to concat)
                p_frame   = 1.0 / (1.0 + np.exp(-frame_logits))             # (N_win, T, n_cls)
                total_len = T + step * (N_win - 1)
                timeline  = np.zeros((total_len, n_cls), dtype=np.float32)
                tl_ct     = np.zeros((total_len, 1),     dtype=np.float32)
                for i in range(N_win):
                    timeline[i*step : i*step + T] += p_frame[i]
                    tl_ct   [i*step : i*step + T] += 1
                timeline = timeline / tl_ct.clip(min=1)
                # per-slot max over the shifted range; slot k covers timeline[k*step:(k+1)*step]
                n_slots = N_win + n_segments - 1                            # = N_SEGMENTS = 12
                fm_l = np.empty((n_slots, n_cls), dtype=np.float32)
                fm_r = np.empty((n_slots, n_cls), dtype=np.float32)
                for k in range(n_slots):
                    lo, hi = k*step, (k+1)*step
                    lo_l, hi_l = lo - d, hi - d
                    lo_r, hi_r = lo + d, hi + d
                    fm_l[k] = timeline[lo_l:hi_l].max(axis=0) if lo_l >= 0       else timeline[lo:hi].max(axis=0)
                    fm_r[k] = timeline[lo_r:hi_r].max(axis=0) if hi_r <= total_len else timeline[lo:hi].max(axis=0)
                fold_fm_l.append(fm_l)
                fold_fm_r.append(fm_r)

    main_avg = np.mean(fold_probs, axis=0)
    # return a tuple only if every fold produced a shift; otherwise degrade to main only (compatible with shift_sec=0)
    if enable_shift and len(fold_fm_l) == len(fold_probs):
        return (main_avg, np.mean(fold_fm_l, axis=0), np.mean(fold_fm_r, axis=0))
    return main_avg


# ============================================================
# Group setup (multi-group ensemble: each group has its own cfg / mel / ONNX / fm_blend / weight)
# ============================================================

def setup_group(g_cfg, intra_op=None, verbose=True):
    """Load all resources for one group: config.json + ONNX sessions + mel transform.
    Returns a dict with all state needed for inference.
    intra_op / verbose: passed through to load_fold_sessions, see its docstring.
    """
    import json as _json
    name      = g_cfg["name"]
    model_dir = g_cfg["model_dir"]
    if verbose:
        print(f"\n=== Setup group: {name} ===")
        print(f"  model_dir: {model_dir}")

    # 1. config.json (must exist; holds the training-side mel parameters)
    cfg_paths = sorted(glob.glob(os.path.join(model_dir, "**", "config.json"), recursive=True))
    if not cfg_paths:
        raise FileNotFoundError(f"config.json missing under {model_dir}")
    train_cfg = _json.load(open(cfg_paths[0]))
    if verbose:
        print(f"  config.json: {os.path.relpath(cfg_paths[0], model_dir)}")

    # default fallbacks (consistent with the global CFG)
    cfg = {
        "sample_rate": CFG["sample_rate"],
        "n_mels":      128, "n_fft": 1024, "hop_length": 512,
        "fmin": 0, "fmax": 16000,
        "mel_scale": "htk", "norm": None,
        "img_size": 0, "in_chans": 1,
        "mel_norm": "minmax", "top_db": 80,
        "duration": 5,
    }
    for k in ("duration", "n_segments", "n_mels", "n_fft", "win_length", "hop_length",
              "fmin", "fmax", "mel_scale", "norm",
              "mel_norm", "top_db", "img_size", "in_chans"):
        if k in train_cfg:
            cfg[k] = train_cfg[k]
    duration   = cfg["duration"]
    n_segments = cfg.get("n_segments") or max(1, duration // 5)
    if "overlap" not in g_cfg:
        raise KeyError(f"group {name!r} missing required 'overlap' (True=11 overlapping sliding windows / False=6 non-overlapping windows)")
    overlap = bool(g_cfg["overlap"])                                       # per group, no global default
    if verbose:
        print(f"  Mel: n_mels={cfg['n_mels']} n_fft={cfg['n_fft']} hop={cfg['hop_length']} "
              f"in_chans={cfg['in_chans']} duration={duration}s n_segments={n_segments}  "
              f"img_size={cfg.get('img_size')} mel_norm={cfg.get('mel_norm')} overlap={overlap}")

    # 2. ONNX sessions (best_type decides which checkpoints to pick)
    best_type = g_cfg["best_type"]
    sessions  = load_fold_sessions(model_dir, pattern="sed_fold*.onnx",
                                    selected_folds=g_cfg.get("ensemble_folds"),
                                    best_type=best_type,
                                    intra_op=intra_op, verbose=verbose)

    # 3. fm_blend_w: a group may give an explicit float override; None/absent -> auto from best_type (auc=0.0 clip head, fm=1.0 frame_max)
    fm_override = g_cfg.get("fm_blend_w", None)
    if fm_override is None:
        fm_blend_w = 0.0 if best_type == "auc" else 1.0
        head_name  = "clip" if best_type == "auc" else "frame_max"
        if verbose:
            print(f"  Head: {head_name}  fm_blend_w={fm_blend_w} (auto from best_type={best_type})  ensemble_weight={g_cfg.get('weight', 1.0)}")
    else:
        fm_blend_w = float(fm_override)
        assert 0.0 <= fm_blend_w <= 1.0, f"fm_blend_w must be in [0,1], got {fm_blend_w}"
        head_name  = f"clip+fm_blend"
        if verbose:
            print(f"  Head: {head_name}  fm_blend_w={fm_blend_w} (override; best_type={best_type})  ensemble_weight={g_cfg.get('weight', 1.0)}")

    # 4. mel transform (per group, not shared)
    mel_tr, db_tr = build_mel_transform(cfg)
    mel_tr = mel_tr.cpu()
    db_tr = db_tr.cpu()

    return {
        "name":       name,
        "cfg":        cfg,
        "sessions":   sessions,
        "mel_tr":     mel_tr,
        "db_tr":      db_tr,
        "duration":   duration,
        "n_segments": n_segments,
        "overlap":    overlap,
        "in_chans":   cfg["in_chans"],
        "fm_blend_w": fm_blend_w,
        "img_size":   cfg.get("img_size", 0),
        "mel_norm":   cfg.get("mel_norm", "minmax"),
        "weight":     float(g_cfg.get("weight", 1.0)),
    }


def predict_at_group(wav, starts_sec, group):
    """slice -> the group's own mel -> the group's own ONNX -> sigmoid + fm_blend -> fold-mean
    returns: single segment (N, n_classes); multi segment (N, n_segments, n_classes) numpy probs
    """
    imgs = wav_to_images(
        wav, starts_sec, group["duration"], CFG["sample_rate"],
        group["img_size"], group["in_chans"],
        group["mel_tr"], group["db_tr"], group["mel_norm"],
    )
    return predict_with_folds(group["sessions"], imgs, group["fm_blend_w"], n_segments=group["n_segments"])


def _mel_cache_key(group):
    """Build a cache key for shareable mel output:
    groups with the same key produce identical imgs from wav_to_images (same mel cfg + slice starts),
    so mel can be computed once and reused across their ONNX runs. Different checkpoints do not affect mel output.
    """
    cfg = group["cfg"]
    return (
        group["duration"],
        group["n_segments"],
        group["overlap"],                                                  # different window strategy -> different slice starts -> different imgs, not shareable
        group["in_chans"],
        group["img_size"],
        group["mel_norm"],
        cfg.get("n_fft"),
        cfg.get("hop_length"),
        cfg.get("n_mels"),
        cfg.get("fmin"),
        cfg.get("fmax"),
        cfg.get("win_length"),
        cfg.get("mel_scale"),
        cfg.get("norm"),
        cfg.get("top_db"),
    )


def _compute_mel_imgs(wav, group):
    """Compute mel imgs from the group cfg (includes wav slicing).
    n_seg=1 uses 12 5s starts; n_seg>1 uses sliding windows with (N_SEGMENTS - n_seg + 1) starts.
    """
    n_seg = group["n_segments"]
    if n_seg > 1:
        if not group["overlap"]:
            # non-overlapping: N_SEGMENTS/n_seg windows, stride n_seg*SEGMENT_SEC (n_seg=2 -> 6 windows @ 0,10,..,50)
            n_windows = N_SEGMENTS // n_seg
            starts = [i * n_seg * SEGMENT_SEC for i in range(n_windows)]
        else:
            # overlapping sliding windows: N_SEGMENTS-n_seg+1 windows, stride SEGMENT_SEC
            n_windows = N_SEGMENTS - n_seg + 1
            starts = [i * SEGMENT_SEC for i in range(n_windows)]
    else:
        starts = [i * SEGMENT_SEC for i in range(N_SEGMENTS)]
    return wav_to_images(
        wav, starts, group["duration"], CFG["sample_rate"],
        group["img_size"], group["in_chans"],
        group["mel_tr"], group["db_tr"], group["mel_norm"],
    )


def _predict_with_imgs(group, imgs_main, n_classes):
    """Run ONNX on precomputed imgs, return (N_SEGMENTS, n_classes) probs.
    n_seg>1: imgs_main is sliding windows (n_windows, ...) -> ONNX -> overlap-averaged to N_SEGMENTS slots
    n_seg=1: imgs_main is 12 segments -> ONNX -> directly (N_SEGMENTS, n_classes)
    When TTA=True, predict_with_folds returns a 3-tuple and the frame-shift TTA is weighted into the ensemble.
    """
    n_seg = group["n_segments"]

    # per-window/chunk probs -> per-slot
    def _to_slot(per_win):
        if per_win.ndim == 2:                                   # n_seg=1: (N_win, n_cls) -> (N_win, 1, n_cls)
            per_win = per_win[:, None, :]
        N_win = per_win.shape[0]
        if not group["overlap"] and n_seg > 1:
            # non-overlapping: window i -> slots [i*n_seg : (i+1)*n_seg], laid out directly (N_win*n_seg = N_SEGMENTS), no overlap-avg
            return per_win.reshape(N_SEGMENTS, n_classes).astype(np.float32)
        # overlapping: overlap-avg to N_SEGMENTS slots
        out = np.zeros((N_SEGMENTS, n_classes), dtype=np.float32)
        ct  = np.zeros((N_SEGMENTS, 1),         dtype=np.float32)
        for i in range(N_win):
            out[i:i+n_seg] += per_win[i]
            ct [i:i+n_seg] += 1
        return out / ct

    if TTA:
        result = predict_with_folds(group["sessions"], imgs_main, group["fm_blend_w"],
                                     n_segments=n_seg,
                                     shift_sec=TTA_SHIFT_SEC, duration_sec=group["duration"])
    else:
        result = predict_with_folds(group["sessions"], imgs_main, group["fm_blend_w"], n_segments=n_seg)

    if isinstance(result, tuple):                               # TTA succeeded (d valid)
        main_perwin, fm_left, fm_right = result
        main_perslot = _to_slot(main_perwin)
        w = TTA_MAIN_W
        return w * main_perslot + (1.0 - w) * 0.5 * (fm_left + fm_right)

    # single ndarray: TTA off or d out of range fallback -> main only
    return _to_slot(result)


def predict_one_file_per_group(wav, group, n_classes, mel_cache=None):
    """One 60s wav -> this group's (N_SEGMENTS, n_classes) prob.

    mel_cache: shares mel output imgs across groups (computed once per mel cfg key).
    """
    if mel_cache is None:
        imgs_main = _compute_mel_imgs(wav, group)
    else:
        key = _mel_cache_key(group)
        imgs_main = mel_cache.get(key)
        if imgs_main is None:
            imgs_main = _compute_mel_imgs(wav, group)
            mel_cache[key] = imgs_main
    return _predict_with_imgs(group, imgs_main, n_classes)


# ============================================================
# Main inference
# ============================================================

def main():
    print(f"ONNX runs on CPU (sessions only support CPUExecutionProvider)")
    # boot print: show the thread / CPU settings actually in effect
    _n_cpu_detected = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or "?")
    _n_cpu_total    = os.cpu_count()
    _intra_op_actual = max(1, _n_cpu_detected) if isinstance(_n_cpu_detected, int) else 4
    print(f"[boot] N_CPU detected: {_n_cpu_detected} (sched_getaffinity), {_n_cpu_total} (os.cpu_count)")
    print(f"[boot] ONNX intra_op_num_threads = {_intra_op_actual}")
    try:
        import onnxruntime as _ort
        print(f"[boot] onnxruntime version: {_ort.__version__}")
    except Exception:
        pass

    # species list (taxonomy order = submission column order, globally unique)
    taxonomy     = pd.read_csv(os.path.join(CFG["data_dir"], "taxonomy.csv"))
    species_list = taxonomy["primary_label"].tolist()
    n_classes    = len(species_list)
    print(f"Classes: {n_classes}")

    # --- global post-processing resources (independent of any group) ---
    global _TEXTURE_MASK
    need_texture_mask = USE_TIME_SMOOTH
    _TEXTURE_MASK = load_texture_mask(species_list) if need_texture_mask else None   # loaded once into the global, inherited by forked workers
    texture_mask  = _TEXTURE_MASK                                          # alias used below in main
    mirror_idx_groups = []
    if USE_MIRROR_PAIRS:
        _l2i = {l: i for i, l in enumerate(species_list)}
        for grp in MIRROR_PAIRS:
            idx = [_l2i[s] for s in grp if s in _l2i]
            if len(idx) >= 2:
                mirror_idx_groups.append(idx)

    if USE_TIME_SMOOTH:
        if SMOOTH_MODE == "gaussian":
            print(f"  Time smooth: ON  mode=gaussian (5-tap {SMOOTH_GAUSSIAN})")
        elif SMOOTH_MODE == "v2":
            print(f"  Time smooth: ON  mode=v2 (two-layer cascade)  "
                  f"L1: texture a={SMOOTH_V2_TEXTURE_ALPHA}({int(texture_mask.sum())}cls) + event-localmax a={SMOOTH_V2_EVENT_ALPHA}({int((~texture_mask).sum())}cls)  "
                  f"L2: adaptive delta-shift base_a={SMOOTH_V2_DELTA_ALPHA}")
        else:
            print(f"  Time smooth: ON  mode=texture (3-tap)  "
                  f"texture={texture_mask.sum()}  event={(~texture_mask).sum()}")
    if USE_MIRROR_PAIRS:
        n_mirror_cls = sum(len(g) for g in mirror_idx_groups)
        print(f"  Mirror pairs: ON  groups={len(mirror_idx_groups)}  classes_covered={n_mirror_cls}  "
              f"(default 47158sonXX 4 groups)")
    if TTA:
        print(f"  Frame-shift TTA: ON  shift_sec={TTA_SHIFT_SEC}s  main_w={TTA_MAIN_W} "
              f"(left/right each {(1-TTA_MAIN_W)/2:.2f}, no extra forward)")

    # --- multi-group ensemble setup: each group has its own cfg / mel / ONNX ---
    if not MODEL_GROUPS:
        raise RuntimeError("MODEL_GROUPS is empty -- configure at least one group")
    groups = [setup_group(g_cfg) for g_cfg in MODEL_GROUPS]

    # normalize ensemble weights
    raw_weights = np.array([g["weight"] for g in groups], dtype=np.float64)
    if (raw_weights <= 0).any():
        raise ValueError(f"group weights cannot be <=0: {raw_weights}")
    weights = raw_weights / raw_weights.sum()
    print(f"\n=== Groups: {len(groups)} ===")
    for g, w in zip(groups, weights):
        print(f"  {g['name']}: duration={g['duration']}s n_seg={g['n_segments']} "
              f"in_chans={g['in_chans']} fm_blend={g['fm_blend_w']}  norm_weight={w:.4f}")

    # --- DataLoader ---
    n_per_seg = CFG["sample_rate"] * SEGMENT_SEC
    total_samples = n_per_seg * N_SEGMENTS    # 60s soundscape

    sub = pd.read_csv(os.path.join(CFG["data_dir"], "sample_submission.csv"))
    # class order consistency check: taxonomy.primary_label order must == sample_submission column order, otherwise columns misalign
    assert species_list == sub.columns[1:].tolist(), \
        "species_list (taxonomy) != sample_submission.columns[1:], columns would misalign"
    test_dir = Path(CFG["data_dir"]) / "test_soundscapes"
    sc_files = sorted(test_dir.glob("*.ogg")) if test_dir.is_dir() else []
    if len(sc_files) == 0:
        fallback = Path(CFG["data_dir"]) / "train_soundscapes"
        if fallback.is_dir():
            sc_files = sorted(fallback.glob("*.ogg"))[:DEBUG_FALLBACK_N]
            print(f"test_soundscapes empty -> using first {len(sc_files)} train_soundscapes for a dry run")
    print(f"Soundscapes to infer: {len(sc_files)}")

    # Workers do setup + forward + ensemble in their subprocess; the main thread only post-processes.
    # Release the main-process sessions / mel_tr (each worker loads its own); must release before the
    # DataLoader's first iter (workers fork copy-on-write, so the main-process ONNX data does not enter worker memory).
    for g in groups:
        g["sessions"] = None
        g["mel_tr"]   = None
        g["db_tr"]    = None
    import gc as _gc; _gc.collect()
    print(f"  Worker ONNX: NUM_WORKERS={NUM_WORKERS} x intra_op={WORKER_INTRA_OP}, main thread post-processing only")

    dataset = SoundscapeDataset(
        sc_files, CFG["sample_rate"], total_samples,
        groups_cfg=MODEL_GROUPS, weights=weights.tolist(),
        n_classes=n_classes, intra_op=WORKER_INTRA_OP,
    )
    loader  = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=_sc_collate,
        pin_memory=False,
        persistent_workers=(NUM_WORKERS > 0),
    )
    print(f"DataLoader: num_workers={NUM_WORKERS}  DB_SCOPE={DB_SCOPE}\n")

    # Limit the thread count of PyTorch ops (mel/STFT) so they do not compete with ONNX sessions for CPU cores.
    # With persistent_workers=True, workers fork on the first iter, at which point set_num_threads is already in effect -> each worker uses 1 thread.
    import torch as _torch
    _torch.set_num_threads(1)

    rows = []
    for sc_stem, probs in tqdm(loader, desc="Inference", total=len(dataset)):
        # probs: the worker already did I/O + full-group forward + ensemble -> (N_SEGMENTS, n_classes) float32

        # post-processing order: time_smooth -> mirror_pairs
        if USE_TIME_SMOOTH:
            probs = time_smooth(probs, texture_mask)
        if USE_MIRROR_PAIRS and mirror_idx_groups:
            # take max across columns within a sonotype group to sync them (last step)
            probs = apply_mirror_pairs(probs, mirror_idx_groups)

        for i, prob in enumerate(probs):
            end_time = (i + 1) * SEGMENT_SEC
            rows.append([f"{sc_stem}_{end_time}"] + prob.tolist())

    result = pd.DataFrame(rows, columns=["row_id"] + species_list)

    # --- DEBUG output: during a dry run sample_submission only has placeholder rows, so the merge
    #     below would drop the 30-file predictions. Dump the full predictions before the merge
    #     (.npy for exact numeric diff, .csv for eyeballing), useful for verification + before/after comparison.
    #     On a real submission sample_submission aligns with test, so this dump just writes two harmless extra files.
    _dbg_vals = result.iloc[:, 1:].to_numpy(np.float32)
    np.save("/kaggle/working/debug_raw_preds.npy", _dbg_vals)
    result.to_csv("/kaggle/working/debug_raw_preds.csv", index=False)
    print(f"[debug] raw preds (pre-merge): shape={_dbg_vals.shape}  "
          f"mean={_dbg_vals.mean():.5f}  max={_dbg_vals.max():.5f}  "
          f"nonzero(>1e-4)_frac={(_dbg_vals > 1e-4).mean():.4f}  rows={len(result)}")

    result = sub[["row_id"]].merge(result, on="row_id", how="left").fillna(0.0)

    # --- file-level top-K post-processing inside the SED branch ---
    # each window's prediction x that file's max prob (boosts consistent species, suppresses inconsistent ones)
    if USE_FILE_POST:
        probs_arr = result.iloc[:, 1:].values
        N, F = probs_arr.shape
        n_files = N // 12
        if N == n_files * 12:
            probs_arr = probs_arr.reshape((n_files, 12, F))
            top_prob = np.sort(probs_arr, axis=1)[:, -1:, :]                    # (n_files, 1, F)
            probs_arr = probs_arr * top_prob
            result.iloc[:, 1:] = probs_arr.reshape((N, F))
            print(f"Applied file-level top probability postprocessing (SED branch)")

    result.to_csv("/kaggle/working/submission.csv", index=False)
    print(f"Saved: submission.csv  shape={result.shape}")


if __name__ == "__main__":
    main()
    # OpenVINO C++ objects can segfault when destroyed at interpreter shutdown (inference is done,
    # the submission is already written). Call os._exit(0) immediately to skip the Python destructor
    # chain + DataLoader worker teardown -> Kaggle sees exit 0 + the correct csv.
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)