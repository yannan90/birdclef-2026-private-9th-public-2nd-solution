# Pseudo-label similarity matrix across the distillation chain (Pearson correlation
# of the flattened soundscape predictions). It shows: (1) the ancestor models
# (N0/X0, no distillation yet) are the least similar to their descendants;
# (2) after distillation the EfficientNetV2 descendants converge toward each other;
# (3) the eca_nfnet models stay distinct across architectures. In other words, the
# teacher ensemble preserves diversity rather than collapsing to one solution.
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "distill_input/student_labels"
# Chain order: chain-1 effv2 (ancestor N0 -> ...) | chain-1 eca | chain-2 effv2 (ancestor X0 -> ...) | chain-2 eca
MODELS = ["524", "554", "607", "645", "676",  "672", "681",  "804", "813", "817",  "822"]
# internal experiment ids -> paper labels ("e" suffix = eca_nfnet variant)
NAME   = {"524": "N0", "554": "N1", "607": "N2", "645": "N3", "676": "N5",
          "672": "N4e", "681": "N5e",
          "804": "X0", "813": "X1", "817": "X2", "822": "X2e"}
ARCH   = {m: "eca" if m in {"672", "681", "822"} else "effv2" for m in MODELS}
ANC    = {"524", "804"}   # ancestors (not yet distilled)


def load(m):
    f = f"{BASE}/single_{m}_fm/student_labels_prob.npz"
    return np.load(f)["scores"].astype(np.float32).ravel()


print("loading...", flush=True)
X = np.stack([load(m) for m in MODELS])            # (M, n_segments * n_classes)
print(f"  X = {X.shape}", flush=True)
C = np.corrcoef(X)                                 # M x M Pearson

# print the matrix
print("\n=== Pearson similarity matrix ===")
print("      " + " ".join(f"{NAME[m]:>5}" for m in MODELS))
for i, m in enumerate(MODELS):
    tag = "*" if m in ANC else " "
    print(f"{NAME[m]:>4}{tag} " + " ".join(f"{C[i,j]:.3f}" for j in range(len(MODELS))))

# key statistics
def avg_off(idx_a, idx_b):
    vals = [C[i, j] for i in idx_a for j in idx_b if i != j]
    return np.mean(vals)

eff1       = [MODELS.index(m) for m in ["524", "554", "607", "645", "676"]]
eff1_noanc = [MODELS.index(m) for m in ["554", "607", "645", "676"]]
eca1       = [MODELS.index(m) for m in ["672", "681"]]
i524 = MODELS.index("524"); i804 = MODELS.index("804")
eff2_noanc = [MODELS.index(m) for m in ["813", "817"]]

print("\n=== claim checks ===")
print(f"(1) ancestor N0 vs its effv2 descendants mean sim: {np.mean([C[i524,j] for j in eff1_noanc]):.3f}")
print(f"    those descendants among themselves:            {avg_off(eff1_noanc, eff1_noanc):.3f}  <- should be higher (convergence)")
print(f"(1) ancestor X0 vs its descendants (X1/X2):        {np.mean([C[i804,j] for j in eff2_noanc]):.3f}")
print(f"    X1 vs X2:                                      {C[MODELS.index('813'),MODELS.index('817')]:.3f}")
print(f"(3) eca pair (N4e/N5e):                            {C[eca1[0],eca1[1]]:.3f}")
print(f"(3) eca vs effv2 (cross-architecture) mean sim:    {avg_off(eca1, eff1):.3f}  <- should be lower than same-arch")
print(f"    N0 vs eca:                                     {np.mean([C[i524,j] for j in eca1]):.3f}")

# heatmap
fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(C, vmin=max(0.80, C[C < 0.999].min()), vmax=1.0, cmap="RdYlGn_r")
ax.set_xticks(range(len(MODELS))); ax.set_yticks(range(len(MODELS)))
ax.set_xticklabels([f"{'*' if m in ANC else ''}{NAME[m]}" for m in MODELS], rotation=45, fontsize=8)
ax.set_yticklabels([f"{'*' if m in ANC else ''}{NAME[m]}\n{ARCH[m]}" for m in MODELS], fontsize=7)
for i in range(len(MODELS)):
    for j in range(len(MODELS)):
        ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=6,
                color="white" if C[i, j] > 0.95 else "black")
# group separators: chain-1 effv2 | eca | chain-2 effv2 | eca
for s in [5, 7, 10]:
    ax.axhline(s - 0.5, color="k", lw=1.2); ax.axvline(s - 0.5, color="k", lw=1.2)
fig.colorbar(im, fraction=0.046, pad=0.04, label="Pearson correlation of pseudo-labels")
ax.set_title("Pseudo-label similarity across the distillation chain\n"
             "(* = ancestor; blocks = chain-1 effv2 | eca | chain-2 effv2 | eca)", fontsize=9.5)
fig.tight_layout()
fig.savefig("fig_distill_sim.pdf"); fig.savefig("fig_distill_sim.png", dpi=150)
np.save("distill_corr.npy", C)
print("\nsaved fig_distill_sim.pdf / .png")
