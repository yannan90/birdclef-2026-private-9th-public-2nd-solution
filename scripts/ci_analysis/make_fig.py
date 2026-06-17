# Figure (what the CI module learns): two panels, data = data/ci_real_<id>.npz
# (CI residuals on the real labeled soundscapes).
#  (a) per-taxon mean correction Δ̄_i -- systematically boosts the under-represented
#      non-avian taxa (averaged over the 6 mature models)
#  (b) Spearman similarity of Δ̄_i across the 7 models -- including 2 eca_nfnet, i.e.
#      independent architectures converge to the same calibration
import numpy as np, pandas as pd, os
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "savefig.dpi": 150})

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "birdclef-2026"
OUT  = "figures"
MODELS = ["524", "607", "676", "804", "817", "681", "822"]
NAME   = {"524": "N0", "607": "N2", "676": "N5", "804": "X0", "817": "X2",
          "681": "N5$^e$", "822": "X2$^e$"}
ARCH   = {"524": "effv2", "607": "effv2", "676": "effv2", "804": "effv2", "817": "effv2",
          "681": "eca", "822": "eca"}
MATURE = ["607", "676", "804", "817", "681", "822"]   # drop the clean ancestor N0 (524), an outlier

tax = pd.read_csv(f"{DATA}/taxonomy.csv", dtype=str); cls = np.array(tax.class_name.tolist())
Y = np.load(f"{HERE}/data/ci_real_524.npz")["y0"]; freq = Y.sum(0); P = np.where(freq > 0)[0]
dbar = {m: np.mean([np.load(f"{HERE}/data/ci_real_{m}.npz")[f"d{f}"] for f in range(4)], 0)[:, P].mean(0)
        for m in MODELS}

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
# ── (a) per-taxon correction (per-class mean over the 6 mature models, grouped by taxon) ──
mat = np.mean([dbar[m] for m in MATURE], 0)
taxa = ["Aves", "Amphibia", "Insecta", "Mammalia"]
col  = {"Aves": "#1f77b4", "Amphibia": "#2ca02c", "Insecta": "#ff7f0e", "Mammalia": "#9467bd"}
groups = [mat[[k for k, i in enumerate(P) if cls[i] == t]] for t in taxa]
bp = ax[0].boxplot(groups, labels=[f"{t}\n(n={len(g)})" for t, g in zip(taxa, groups)],
                   widths=.6, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
for patch, t in zip(bp["boxes"], taxa): patch.set_facecolor(col[t]); patch.set_alpha(.55)
rng = np.random.default_rng(0)
for k, (t, g) in enumerate(zip(taxa, groups)):
    ax[0].scatter(rng.normal(k+1, .05, len(g)), g, s=10, c=col[t], alpha=.7, zorder=3, linewidths=0)
ax[0].axhline(0, color="grey", lw=.6, ls="--")
ax[0].set_ylabel("Per-class correction  $\\bar\\Delta_i$  (logit)")
ax[0].set_title("(a) Boosts the under-represented non-avian taxa")

# ── (b) cross-model similarity: same palette as fig_distill_sim (RdYlGn_r, red=high), tight vmin for contrast ──
S = np.array([[spearmanr(dbar[a], dbar[b])[0] for b in MODELS] for a in MODELS])
im = ax[1].imshow(S, vmin=0.55, vmax=1.0, cmap="RdYlGn_r", interpolation="nearest")
n = len(MODELS); n_eff = sum(1 for m in MODELS if ARCH[m] == "effv2")
ax[1].set_xticks(range(n)); ax[1].set_yticks(range(n))
ax[1].set_xticklabels([NAME[m] for m in MODELS]); ax[1].set_yticklabels([NAME[m] for m in MODELS])
for i in range(n):
    for j in range(n):
        ax[1].text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                   color="white" if (S[i, j] > 0.92 or S[i, j] < 0.70) else "black")
# black lines separate the EfficientNetV2 block from the eca_nfnet block, highlighting cross-architecture agreement
ax[1].axhline(n_eff - 0.5, color="k", lw=1.3); ax[1].axvline(n_eff - 0.5, color="k", lw=1.3)
fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04, label="Spearman of $\\bar\\Delta_i$")
ax[1].set_title("(b) Same correction across architectures ($^e$ = eca\\_nfnet)")
fig.tight_layout(w_pad=3.0)
fig.savefig(f"{OUT}/fig_ci_learned.pdf")
fig.savefig(f"{OUT}/fig_ci_learned.png", dpi=300)   # paper uses PNG: imshow has a PDF-backend rendering quirk (color cells flattened); PNG is consistent across viewers
print("saved fig_ci_learned (7 models: 5 effv2 + 2 eca)")
print("taxon medians:", {t: round(float(np.median(g)), 2) for t, g in zip(taxa, groups)})
