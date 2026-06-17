# What the CI module actually learns -- residual analysis over 7 models x 4 folds on
# real soundscapes.
# Data: data/ci_real_<id>.npz (produced by probe_ci_allfiles.py over the 66 labeled
# soundscapes; captures the CI layer's pre-CI logits z and residual Delta. Each model is
# run over "all 66 files x its own 4 folds" so scale stays consistent within a model and
# OOF stitching cannot introduce cross-model scale artifacts).
import numpy as np, pandas as pd, os, itertools
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "birdclef-2026"
# 5x EfficientNetV2-S + 2x eca_nfnet_l0 (^e); enough architecture diversity to test
# whether independent models converge to the same calibration
MODELS = ["524", "607", "676", "804", "817", "681", "822"]
NAME   = {"524": "N0", "607": "N2", "676": "N5", "804": "X0", "817": "X2",
          "681": "N5$^e$", "822": "X2$^e$"}
ARCH   = {"524": "effv2", "607": "effv2", "676": "effv2", "804": "effv2", "817": "effv2",
          "681": "eca", "822": "eca"}

tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype=str); cls = np.array(tax.class_name.tolist()); C = len(tax)
M = [m for m in MODELS if os.path.exists(f"{HERE}/data/ci_real_{m}.npz")]
Z = {m: np.load(f"{HERE}/data/ci_real_{m}.npz") for m in M}
Y = Z[M[0]]["y0"]; freq = Y.sum(0); P = np.where(freq > 0)[0]; lf = np.log(freq[P])
# Per-class mean correction Δ̄_i, averaged over the 4 folds (present classes only)
D     = {m: np.mean([Z[m][f"d{f}"] for f in range(4)], 0) for m in M}
dbar  = {m: D[m][:, P].mean(0) for m in M}
ml    = {m: np.mean([Z[m][f"z{f}"] for f in range(4)], 0)[:, P].mean(0) for m in M}

print(f"models: {[NAME[m].replace('$','') for m in M]}  (n={len(M)})  present classes={len(P)}")
print("\n=== 1) Δ̄_i vs frequency / mean logit;  variance between/(b+w);  |Δ̄| ===")
print("model  arch    ρ(logfreq)  ρ(logit)  between%  |Δ̄|med")
for m in M:
    d = D[m][:, P]; vb = np.var(d.mean(0)); vw = np.mean(np.var(d, 0))
    print(f"  {NAME[m].replace('$',''):>5} {ARCH[m]:>5}   {spearmanr(dbar[m],lf)[0]:+.2f}      "
          f"{spearmanr(dbar[m],ml[m])[0]:+.2f}     {vb/(vb+vw):.2f}     {np.median(np.abs(dbar[m])):.2f}")

print("\n=== 2) Cross-model Δ̄_i similarity (Spearman) ===")
print("        " + " ".join(f"{NAME[m].replace('$',''):>6}" for m in M))
S = np.array([[spearmanr(dbar[a], dbar[b])[0] for b in M] for a in M])
for i, a in enumerate(M):
    print(f"  {NAME[a].replace('$',''):>5} " + " ".join(f"{S[i,j]:+.2f}" for j in range(len(M))))

print("\n=== 3) Per-taxon mean correction Δ̄ (per model) ===")
taxa = ["Aves", "Amphibia", "Insecta", "Mammalia"]
TI = {t: [k for k, i in enumerate(P) if cls[i] == t] for t in taxa}
print("model   " + "  ".join(f"{t[:4]:>6}" for t in taxa) + "   nonAv-Av")
for m in M:
    mv = {t: dbar[m][TI[t]].mean() for t in taxa}
    nonav = [k for t in ["Amphibia", "Insecta", "Mammalia"] for k in TI[t]]
    print(f"  {NAME[m].replace('$',''):>5} " + "  ".join(f"{mv[t]:+6.2f}" for t in taxa) +
          f"    {dbar[m][nonav].mean()-dbar[m][TI['Aves']].mean():+.2f}")

# Cross-model consistency summary (all models + mature-only; 524 = clean ancestor, a known outlier)
prs_all = [S[i,j] for i,j in itertools.combinations(range(len(M)),2)]
mature = [m for m in M if m != "524"]
mi = [M.index(m) for m in mature]
prs_mat = [spearmanr(dbar[a], dbar[b])[0] for a,b in itertools.combinations(mature,2)]
cross = [spearmanr(dbar[a], dbar[b])[0] for a in mature for b in mature
         if ARCH[a]!=ARCH[b]]   # eca vs effv2 cross-architecture pairs
print(f"\nAll pairwise Spearman: mean={np.mean(prs_all):.2f}  range=[{min(prs_all):+.2f},{max(prs_all):+.2f}]")
print(f"Mature models (excl. 524) pairwise: mean={np.mean(prs_mat):.2f}  min={min(prs_mat):+.2f}")
print(f"Cross-architecture pairs (eca vs effv2): mean={np.mean(cross):.2f}  min={min(cross):+.2f}  (n={len(cross)})")
print(f"524 vs others: mean={np.mean([spearmanr(dbar['524'],dbar[b])[0] for b in mature]):.2f}")
