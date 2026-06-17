"""Distill teacher soft-label generation -- ensemble_v2: weighted fusion of two fm 4-fold models

Differences from the single-model distill inference:
  1. Multi-model weighted fusion, not per-fold output.
     - Full ensemble: weighted fusion over the 8 ONNX (two models x 4 folds) -> a single (N, 234)
     - weights: model A 0.6 / 4 = 0.15 per fold, model B 0.4 / 4 = 0.10 per fold (sum = 1.0)
     - equivalent to 0.6 * mean(A 4 folds) + 0.4 * mean(B 4 folds)
  2. Output npz shape (N, 234) float16; the training side supports the `_kd_all.ndim == 2` branch.

Prerequisite:
  the two models' cfg must match (backbone / duration / n_segments / mel / sample_rate). Verified:
  - tf_efficientnetv2_s_in21k / 10s / n_seg=2 / mel_norm=zscore_minmax / sr=32000
  - n_fft=2048 / hop=512 / n_mels=128 / fmin=20 / fmax=16000 / win=2048 / top_db=80

Input:  birdclef-2026/train_soundscapes/*.ogg
        excluding the labeled files listed in train_soundscapes_labels.csv
Models: distill_models/exp554_fm/sed_fold*_fm_*.onnx (4 folds)
        distill_models/exp524_fm/sed_fold*_fm_*.onnx (4 folds)
Post-processing: time_smooth v2 only

Output:
  distill_input/student_labels/ensemble_v2/
    student_labels_prob.npz   key="scores", shape (N, 234) float16
    student_meta.csv          fields filename / stem / window_idx

Training side usage:
  cfg["soft_kd_npz"]   = ".../ensemble_v2/student_labels_prob.npz"
  cfg["soft_kd_meta_csv"] = ".../ensemble_v2/student_meta.csv"
  # _kd_all.ndim == 2 -> takes the ensemble branch (not per-fold)
"""
import os, sys, glob, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import onnxruntime as ort

# ---- Paths ----
DATA_DIR        = "birdclef-2026"
SOUNDSCAPE_DIR  = f"{DATA_DIR}/train_soundscapes"
LABEL_CSV       = f"{DATA_DIR}/train_soundscapes_labels.csv"
TAXONOMY        = f"{DATA_DIR}/taxonomy.csv"

# ---- Ensemble model group config (name, dir, weight) ----
# weights applied in prob domain; within each group the 4 folds are equally averaged, then groups are weighted
MODEL_GROUPS = [
    ("exp554", "distill_models/exp554_fm", 0.6),
    ("exp524", "distill_models/exp524_fm", 0.4),
]
OUT_DIR         = "distill_input/student_labels/ensemble_v2"

# ---- Inference parameters ----
SAMPLE_RATE     = 32000
SEGMENT_SEC     = 5
N_SEGMENTS      = 12                   # 60s / 5s
TOTAL_SAMPLES   = SAMPLE_RATE * SEGMENT_SEC * N_SEGMENTS

# ---- Post-processing (same defaults as the inference kernel; smooth v2 only) ----
USE_TIME_SMOOTH         = True
SMOOTH_V2_TEXTURE_ALPHA = 0.35
SMOOTH_V2_EVENT_ALPHA   = 0.15
SMOOTH_V2_DELTA_ALPHA   = 0.20


def load_cfg(model_dir):
    return json.load(open(f"{model_dir}/config.json"))


def build_mel(cfg, device="cpu"):
    db = T.AmplitudeToDB(top_db=cfg.get("top_db", 80))
    mel = T.MelSpectrogram(
        sample_rate=cfg["sample_rate"],
        n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        win_length=cfg.get("win_length"), n_mels=cfg["n_mels"],
        f_min=cfg["fmin"], f_max=cfg["fmax"],
        mel_scale=cfg.get("mel_scale", "htk"),
        norm=cfg.get("norm"),
    )
    return mel.to(device), db.to(device)


def normalize_mel(mel, mode="zscore_minmax"):
    dims = (-2, -1)
    if mode == "zscore_minmax":
        m = mel.mean(dim=dims, keepdim=True)
        s = mel.std(dim=dims, keepdim=True) + 1e-6
        mel = (mel - m) / s
        mn = torch.amin(mel, dim=dims, keepdim=True)
        mx = torch.amax(mel, dim=dims, keepdim=True)
        mel = (mel - mn) / (mx - mn + 1e-7)
    elif mode == "minmax":
        mn = torch.amin(mel, dim=dims, keepdim=True)
        mx = torch.amax(mel, dim=dims, keepdim=True)
        mel = (mel - mn) / (mx - mn + 1e-7)
    return mel


def wav_to_imgs(wav, starts_sec, dur_sec, in_chans, mel_tr, db_tr, mel_norm, device="cpu"):
    seg_samples = int(round(dur_sec * SAMPLE_RATE))
    segs = []
    for s in starts_sec:
        a = int(round(s * SAMPLE_RATE))
        seg = wav[a:a+seg_samples]
        if len(seg) < seg_samples:
            seg = F.pad(seg, (0, seg_samples - len(seg)))
        segs.append(seg)
    wavs = torch.stack(segs, 0).to(device, non_blocking=True)
    mel = mel_tr(wavs)
    mel = db_tr(mel.unsqueeze(1)).squeeze(1)
    mel = normalize_mel(mel, mel_norm)
    mel = mel.unsqueeze(1).repeat(1, in_chans, 1, 1)
    return mel


def _ort_session_options(use_gpu):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if use_gpu:
        gpu_id = int(os.environ.get("GPU_ID", "0"))
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" not in avail:
            print(f"CUDAExecutionProvider unavailable, falling back to CPU.")
            providers = ["CPUExecutionProvider"]
            n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
            so.intra_op_num_threads = max(1, n_cpu)
        else:
            providers = [
                ("CUDAExecutionProvider", {"device_id": gpu_id, "cudnn_conv_algo_search": "DEFAULT"}),
                "CPUExecutionProvider",
            ]
    else:
        providers = ["CPUExecutionProvider"]
        n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
        so.intra_op_num_threads = max(1, n_cpu)
    return so, providers


def load_ensemble_sessions(model_groups):
    """Load all 4-fold ONNX of every model group, return a list of (group_name, fold_weight, sess).
    fold_weight = group_weight / n_folds_in_group, normalized so the sum of weights = 1.0.
    """
    use_gpu = os.environ.get("USE_GPU", "1") == "1"
    so, providers = _ort_session_options(use_gpu)
    print(f"  ORT providers: {providers[0] if isinstance(providers[0], str) else providers[0][0]}")

    sessions = []   # list of (label, fold_weight, sess)
    total_w = 0.0
    for gname, gdir, gw in model_groups:
        paths = sorted(glob.glob(os.path.join(gdir, "sed_fold*_fm_*.onnx")))
        assert len(paths) > 0, f"no onnx under {gdir}"
        fold_w = gw / len(paths)
        for p in paths:
            sess = ort.InferenceSession(p, so, providers=providers)
            actual = sess.get_providers()[0]
            sessions.append((f"{gname}:{Path(p).stem}", fold_w, sess))
            print(f"  loaded {gname} {Path(p).name}  fold_w={fold_w:.4f}  provider={actual}")
            total_w += fold_w
    assert abs(total_w - 1.0) < 1e-6, f"weights not normalized: total={total_w}"
    print(f"  total sessions = {len(sessions)}, weights sum = {total_w:.4f}")
    return sessions


def predict_one_file_ensemble(wav, sessions, n_seg, duration, in_chans, mel_tr, db_tr, mel_norm, mel_device="cpu"):
    """One 60s wav -> a single (12, 234) prob (after weighted ensemble fusion, same shape as single-model output).
    Flow:
      1. slice wav into n_windows x duration sliding windows, compute mel in one shot -> arr (numpy on cpu)
      2. for each session, run ONNX -> frame_logits -> n_seg max-chunk -> sigmoid -> prob_per_window (n_win, n_seg, n_cls)
      3. weighted sum of prob across sessions (each session carries its fold_w, total = 1.0)
      4. overlap-average to 12 slots
    """
    n_windows = N_SEGMENTS - n_seg + 1
    starts = [i * SEGMENT_SEC for i in range(n_windows)]
    imgs = wav_to_imgs(wav, starts, duration, in_chans, mel_tr, db_tr, mel_norm, device=mel_device)
    arr = imgs.detach().cpu().numpy()

    p_blend = None   # (n_windows, n_seg, n_cls) prob, weighted sum across sessions
    for _label, w, sess in sessions:
        in_name = sess.get_inputs()[0].name
        clip_logits, frame_logits = sess.run(None, {in_name: arr})
        T_ = frame_logits.shape[1]
        chunk_size = (T_ + n_seg - 1) // n_seg
        fm = np.stack([
            frame_logits[:, s*chunk_size : min((s+1)*chunk_size, T_)].max(axis=1)
            for s in range(n_seg)
        ], axis=1)
        p_fm = 1.0 / (1.0 + np.exp(-fm))
        if p_blend is None:
            p_blend = w * p_fm
        else:
            p_blend = p_blend + w * p_fm

    # overlap-average to 12 slots (sum of weights is already 1.0, no need to divide by total weight)
    n_classes = p_blend.shape[-1]
    probs  = np.zeros((N_SEGMENTS, n_classes), dtype=np.float64)
    counts = np.zeros((N_SEGMENTS, 1),        dtype=np.float64)
    for t in range(n_windows):
        probs[t:t+n_seg]  += p_blend[t]
        counts[t:t+n_seg] += 1
    return probs / counts                                        # (12, n_cls)


# ---- Post-processing helpers (identical to the single-model version) ----
def load_texture_mask(label_cols):
    tax = pd.read_csv(TAXONOMY)
    tax["primary_label"] = tax["primary_label"].astype(str)
    tex_set = set(tax[tax["class_name"].isin(["Insecta", "Amphibia"])]["primary_label"])
    return np.array([l in tex_set for l in label_cols], dtype=bool)


def _smooth_v2_texture(p, mask, alpha):
    if alpha <= 0 or not mask.any():
        return p
    x = p[:, mask]
    prev_x = np.concatenate([x[:1], x[:-1]], axis=0)
    next_x = np.concatenate([x[1:], x[-1:]], axis=0)
    out = p.copy()
    out[:, mask] = (1.0 - alpha) * x + 0.5 * alpha * (prev_x + next_x)
    return out


def _smooth_v2_event(p, mask, alpha):
    if alpha <= 0 or not mask.any():
        return p
    x = p[:, mask]
    prev_x = np.concatenate([x[:1], x[:-1]], axis=0)
    next_x = np.concatenate([x[1:], x[-1:]], axis=0)
    local_max = np.maximum(x, np.maximum(prev_x, next_x))
    out = p.copy()
    out[:, mask] = (1.0 - alpha) * x + alpha * local_max
    return out


def _smooth_v2_adaptive_delta(p, base_alpha):
    if base_alpha <= 0:
        return p
    out = p.copy()
    n = len(p)
    for i in range(1, n - 1):
        conf = p[i].max(axis=-1, keepdims=True)
        a = base_alpha * (1.0 - conf)
        neighbor_avg = (p[i-1] + p[i+1]) / 2.0
        out[i] = (1.0 - a) * p[i] + a * neighbor_avg
    return out


def time_smooth_v2(probs, texture_mask):
    out = _smooth_v2_texture(probs,  texture_mask, SMOOTH_V2_TEXTURE_ALPHA)
    out = _smooth_v2_event  (out,   ~texture_mask, SMOOTH_V2_EVENT_ALPHA)
    out = _smooth_v2_adaptive_delta(out, SMOOTH_V2_DELTA_ALPHA)
    return out


def main():
    print("=" * 80)
    print(f"[boot] N_CPU = {len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()}")
    print(f"[boot] onnxruntime version: {ort.__version__}")
    print(f"[boot] available providers: {ort.get_available_providers()}")
    print(f"[boot] USE_GPU = {os.environ.get('USE_GPU', '1')} | GPU_ID = {os.environ.get('GPU_ID', '0')}")
    print(f"[boot] torch num_threads: {torch.get_num_threads()}")
    torch.set_num_threads(1)
    print(f"[boot] set torch num_threads = 1 (mel should not compete with ONNX/GPU)")
    print(f"[boot] MODEL_GROUPS:")
    for gname, gdir, gw in MODEL_GROUPS:
        print(f"   - {gname}: {gdir}  weight={gw}")
    print(f"[boot] OUT_DIR = {OUT_DIR}")
    print("=" * 80)

    # use the first group's cfg (all groups must have identical config, sanity-checked below)
    cfg = load_cfg(MODEL_GROUPS[0][1])
    duration   = cfg["duration"]
    n_segments = cfg.get("n_segments") or max(1, duration // 5)
    in_chans   = cfg.get("in_chans", 1)
    mel_norm   = cfg.get("mel_norm", "zscore_minmax")
    print(f"\n[cfg] (from {MODEL_GROUPS[0][0]}) duration={duration}s n_seg={n_segments} in_chans={in_chans} mel_norm={mel_norm}")
    print(f"[cfg] mel: n_fft={cfg['n_fft']} hop={cfg['hop_length']} n_mels={cfg['n_mels']} fmin={cfg['fmin']} fmax={cfg['fmax']}")

    # cross-group sanity: the key mel/duration parameters must match
    cfg_keys_must_match = ["duration", "n_segments", "sample_rate", "n_fft", "hop_length",
                           "n_mels", "fmin", "fmax", "win_length", "mel_norm", "in_chans"]
    for gname, gdir, _gw in MODEL_GROUPS[1:]:
        c2 = load_cfg(gdir)
        for k in cfg_keys_must_match:
            v1, v2 = cfg.get(k), c2.get(k)
            assert v1 == v2, f"cfg mismatch: {gname}.{k}={v2} vs base.{k}={v1}"
    print(f"[cfg] all {len(MODEL_GROUPS)} groups cfg consistent")

    # species order
    tax = pd.read_csv(TAXONOMY)
    species_list = tax["primary_label"].astype(str).tolist()
    n_classes = len(species_list)
    print(f"\n[cfg] n_classes = {n_classes}")

    texture_mask = load_texture_mask(species_list) if USE_TIME_SMOOTH else None
    if texture_mask is not None:
        print(f"[cfg] texture (Insecta+Amphibia) = {int(texture_mask.sum())} / {n_classes}")

    # unlabeled soundscape files
    sc_labels = pd.read_csv(LABEL_CSV)
    labeled_stems = set(Path(fn).stem for fn in sc_labels["filename"].unique())
    all_files = sorted(glob.glob(os.path.join(SOUNDSCAPE_DIR, "*.ogg")))
    unlabeled = [p for p in all_files if Path(p).stem not in labeled_stems]
    print(f"\n[data] all SC files: {len(all_files)} | labeled: {len(labeled_stems)} | unlabeled: {len(unlabeled)}")

    # multi-machine sharding
    n_workers = int(os.environ.get("N_WORKERS", "1"))
    worker_id = int(os.environ.get("WORKER_ID", "0"))
    assert 0 <= worker_id < n_workers, f"WORKER_ID {worker_id} out of range for N_WORKERS {n_workers}"
    if n_workers > 1:
        unlabeled = unlabeled[worker_id::n_workers]
        print(f"[shard] worker {worker_id}/{n_workers} → {len(unlabeled)} files")

    use_gpu = os.environ.get("USE_GPU", "1") == "1" and "CUDAExecutionProvider" in ort.get_available_providers()
    mel_device = "cuda" if use_gpu else "cpu"
    print(f"\n[model] loading ensemble sessions ...")
    print(f"[model] mel_device = {mel_device}")
    mel_tr, db_tr = build_mel(cfg, device=mel_device)
    sessions = load_ensemble_sessions(MODEL_GROUPS)

    # DataLoader prefetch
    from torch.utils.data import Dataset, DataLoader

    class WavDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            fp = self.paths[idx]
            data, native_sr = sf.read(fp, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.mean(axis=1))
            if native_sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, native_sr, SAMPLE_RATE)
            if len(wav) < TOTAL_SAMPLES:
                wav = F.pad(wav, (0, TOTAL_SAMPLES - len(wav)))
            else:
                wav = wav[:TOTAL_SAMPLES]
            return idx, wav

    def _collate(batch):
        idx, wav = batch[0]
        return idx, wav

    n_dl_workers = int(os.environ.get("N_DL_WORKERS", "8"))
    dataset = WavDataset(unlabeled)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=n_dl_workers, collate_fn=_collate,
        pin_memory=use_gpu, prefetch_factor=4 if n_dl_workers > 0 else None,
        persistent_workers=(n_dl_workers > 0),
    )
    print(f"[loader] DataLoader: num_workers={n_dl_workers}, pin_memory={use_gpu}")

    # inference: single ensemble output (merged together, unlike the per-fold variant)
    os.makedirs(OUT_DIR, exist_ok=True)
    all_probs = [None] * len(unlabeled)        # [idx] = (12, 234) f16

    t0 = time.time()
    for idx, wav in loader:
        p = predict_one_file_ensemble(
            wav, sessions, n_segments, duration,
            in_chans, mel_tr, db_tr, mel_norm, mel_device=mel_device,
        )                                                # (12, 234) float64
        p = p.astype(np.float32)
        if USE_TIME_SMOOTH:
            p = time_smooth_v2(p, texture_mask)
        all_probs[idx] = p.astype(np.float16)

        if (idx + 1) % 200 == 0 or idx == 0:
            done = sum(1 for p in all_probs if p is not None)
            elapsed = time.time() - t0
            eta = elapsed / done * (len(unlabeled) - done) if done > 0 else 0
            print(f"  [{done}/{len(unlabeled)}] elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min, "
                  f"speed {done/elapsed:.2f} file/s ({len(sessions)} model ensemble)")

    # build meta + write to disk
    meta_rows = []
    for fp in unlabeled:
        fname = Path(fp).name
        stem  = Path(fp).stem
        for w in range(N_SEGMENTS):
            meta_rows.append({"filename": fname, "stem": stem, "window_idx": w})
    meta_df = pd.DataFrame(meta_rows)

    scores = np.concatenate(all_probs, axis=0).astype(np.float16)   # (N, 234)
    assert scores.shape == (len(meta_df), n_classes), f"shape mismatch: scores={scores.shape}, meta={len(meta_df)}"

    npz_path = os.path.join(OUT_DIR, "student_labels_prob.npz")
    csv_path = os.path.join(OUT_DIR, "student_meta.csv")
    np.savez(npz_path, scores=scores)
    meta_df.to_csv(csv_path, index=False)
    print(f"\n[save] {npz_path}  ({os.path.getsize(npz_path)/1024/1024:.1f} MB)  shape={scores.shape}")
    print(f"[save] {csv_path}  ({len(meta_df)} rows)")

    total = time.time() - t0
    print(f"\n[done] total {total/60:.1f} min, {len(unlabeled)/total:.2f} file/s")
    print(f"\n training-side config:")
    print(f"  cfg['soft_kd_npz']      = '{npz_path}'")
    print(f"  cfg['soft_kd_meta_csv'] = '{csv_path}'")
    print(f"  # _kd_all.ndim == 2 -> takes the ensemble branch (not per-fold)")


if __name__ == "__main__":
    main()
