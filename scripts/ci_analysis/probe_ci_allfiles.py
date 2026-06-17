# Real-SC CI probe: run the 66 labeled soundscapes through a model's 4 folds, use a
# forward hook to capture the CI layer's real input (pre-CI clip logits z) and residual
# (Δ), and save to npz. No synthetic vectors.
# Usage: python3 probe_ci_allfiles.py <exp_dir> <model_id> [n_files]
import sys, os, json, glob, re
sys.path.insert(0, sys.argv[1])          # the model's own train_folds.py
import numpy as np, torch, soundfile as sf
import torch.nn.functional as F
import torchaudio.transforms as T
import train_folds as tf
from train_folds import SEDModel

EXP = sys.argv[1]; MID = sys.argv[2]
N_FILES = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # 0 = all files
DATA = "birdclef-2026"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = json.load(open(f"{EXP}/config.json"))
SR = cfg["sample_rate"]; DUR = cfg["duration"]; NSEG = cfg.get("n_segments") or DUR // 5
import pandas as pd
tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype=str); species = tax.primary_label.tolist()
C = len(species); idx = {s: i for i, s in enumerate(species)}

# ── labeled soundscape labels: (file, start_sec) -> set(class_idx) ──
lab = pd.read_csv(f"{DATA}/train_soundscapes_labels.csv", dtype=str)
def t2s(t): h, m, s = t.split(":"); return int(h)*3600 + int(m)*60 + int(s)
seg_lab = {}
for _, r in lab.iterrows():
    ii = [idx[s] for s in str(r.primary_label).split(";") if s in idx]
    seg_lab[(r.filename, t2s(r.start))] = ii
files = sorted(lab.filename.unique())
if N_FILES: files = files[:N_FILES]

# ── mel (copied from similarity_analysis.py, symmetric with training) ──
def build_mel(cfg):
    db = T.AmplitudeToDB(top_db=cfg.get("top_db", 80))
    kw = {"normalized": cfg.get("mel_normalized", False)}
    if cfg.get("mel_scale"): kw["mel_scale"] = cfg["mel_scale"]
    if cfg.get("norm"): kw["norm"] = cfg["norm"]
    if cfg.get("win_length"): kw["win_length"] = cfg["win_length"]
    return T.MelSpectrogram(sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
                            f_min=cfg["fmin"], f_max=cfg["fmax"], n_mels=cfg["n_mels"], **kw), db
def norm_mel(mel, mn):
    d = (-2, -1)
    if mn == "zscore_minmax":
        me = mel.mean(d, keepdim=True); s = mel.std(d, keepdim=True)+1e-6; mel = (mel-me)/s
        lo = torch.amin(mel, d, keepdim=True); hi = torch.amax(mel, d, keepdim=True); return (mel-lo)/(hi-lo+1e-7)
    if mn == "minmax":
        lo = torch.amin(mel, d, keepdim=True); hi = torch.amax(mel, d, keepdim=True); return (mel-lo)/(hi-lo+1e-7)
    me = mel.mean(d, keepdim=True); s = mel.std(d, keepdim=True)+1e-6; return (mel-me)/s
MT, DT = build_mel(cfg); MN = cfg.get("mel_norm", "minmax"); IMG = cfg.get("img_size", 0); INCH = cfg.get("in_chans", 3)

def wav_to_imgs(wav, starts):
    ns = int(round(DUR*SR)); segs = []
    for st in starts:
        a = int(round(st*SR)); seg = wav[a:a+ns]
        if len(seg) < ns: seg = F.pad(seg, (0, ns-len(seg)))
        segs.append(seg)
    mel = MT(torch.stack(segs, 0)); mel = DT(mel.unsqueeze(1)).squeeze(1)
    mel = norm_mel(mel, MN).unsqueeze(1)
    if isinstance(IMG, int) and IMG > 0: mel = F.interpolate(mel, size=(IMG, IMG), mode="bilinear", align_corners=False)
    return mel.repeat(1, INCH, 1, 1)

import inspect
SIG = set(inspect.signature(SEDModel.__init__).parameters)   # SEDModel signature differs across exps; filter kwargs by signature
def build_model(ckpt):
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["state_dict"].items()}
    g = ckpt.get
    pool = ckpt["freq_pool_type"] if "freq_pool_type" in ckpt else ("gem" if ckpt.get("use_gem_freq_pool") else cfg.get("freq_pool_type", "mean"))
    kw = dict(head_type=g("head_type", cfg.get("head_type", "LSE")), freq_pool_type=pool,
              pretrained=False, in_chans=g("in_chans", cfg.get("in_chans", 3)),
              n_segments=g("n_segments") or (g("duration", cfg["duration"])//5),
              att_hidden_dim=g("att_hidden_dim", 512), att_dropout=g("att_dropout", 0.5),
              att_temperature=g("att_temperature", 1.0), att_activation=g("att_activation", "softmax"),
              per_class_att_temperature=g("per_class_att_temperature", False),
              att_conv_kernel=g("att_conv_kernel", 1),
              use_class_interaction=g("use_class_interaction", cfg.get("use_class_interaction", False)),
              class_interaction_type=g("class_interaction_type", cfg.get("class_interaction_type", "linear")),
              class_interaction_hidden=g("class_interaction_hidden", cfg.get("class_interaction_hidden", 128)),
              class_interaction_hidden2=g("class_interaction_hidden2", cfg.get("class_interaction_hidden2", None)),
              class_interaction_n_heads=g("class_interaction_n_heads", 4),
              class_interaction_position=g("class_interaction_position", cfg.get("class_interaction_position", "clipwise")),
              class_interaction_cross_seg_mode=g("class_interaction_cross_seg_mode", cfg.get("class_interaction_cross_seg_mode", "none")),
              use_features_only=g("use_features_only", cfg.get("use_features_only", False)))
    kw = {k: v for k, v in kw.items() if k in SIG}            # only pass params this version accepts
    m = SEDModel(g("backbone", cfg["backbone"]), C, **kw)
    m.load_state_dict(sd, strict=False); return m.to(DEV).eval()

starts = list(range(0, 60, DUR))          # nonoverlap 10s windows
out = {}
for f in range(4):
    cks = sorted(glob.glob(f"{EXP}/best_model_fold{f}_*_fm_*.pth")) or \
          [p for p in sorted(glob.glob(f"{EXP}/best_model_fold{f}_*.pth")) if "_auc_" not in p and "_fm_" not in p]
    if not cks: print(f"  fold{f}: no ckpt", flush=True); continue
    ckpt = torch.load(cks[-1], map_location="cpu", weights_only=False)
    model = build_model(ckpt)
    ci = next(mod for n, mod in model.named_modules() if n.endswith("class_interaction"))
    cap = {}
    h = ci.register_forward_hook(lambda m, i, o: cap.update(inp=i[0].detach(), out=o.detach()))
    Z, D, Y = [], [], []
    for fn in files:
        wav, sr = sf.read(f"{DATA}/train_soundscapes/{fn}", dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        wav = torch.from_numpy(wav)
        if len(wav) < 60*SR: wav = F.pad(wav, (0, 60*SR-len(wav)))
        with torch.no_grad():
            model(wav_to_imgs(wav, starts).to(DEV))
        z = cap["inp"][..., :C].reshape(-1, C).cpu().numpy()   # (n_win*n_seg, C) pre-CI logits
        d = cap["out"].reshape(-1, C).cpu().numpy()            # (n_win*n_seg, C) CI residual Δ
        for k in range(len(starts)*NSEG):
            t = starts[k//NSEG] + (k % NSEG)*5
            yi = seg_lab.get((fn, t))
            if yi is None: continue
            Z.append(z[k]); D.append(d[k]); Y.append(yi)
    h.remove()
    Z = np.array(Z, np.float32); D = np.array(D, np.float32)
    Ymat = np.zeros((len(Y), C), np.float32)
    for r, yi in enumerate(Y): Ymat[r, yi] = 1
    out[f"z{f}"] = Z; out[f"d{f}"] = D; out[f"y{f}"] = Ymat
    print(f"  fold{f}: {Z.shape[0]} segs | z mean/std {Z.mean():.2f}/{Z.std():.2f} | |Δ| median {np.median(np.abs(D)):.4f}", flush=True)
np.savez_compressed(f"/tmp/ci_real_{MID}.npz", **out)
print(f"saved /tmp/ci_real_{MID}.npz", flush=True)
