import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

features = pd.read_csv("results/phase_c_feature_matrix.csv")
classification = pd.read_csv("results/phase_e_primary_classification.csv")
merged = features.merge(classification[["lnp_id", "category_primary"]], on="lnp_id", how="inner")

# Order categories meaningfully
cat_order = ["Potent + Developable", "Potent only", "Developable only", "Neither"]
colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a7a7"]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

for ax, feat, ylabel in zip(axes,
    ["num_rings", "num_tertiary_amines", "num_ester_bonds"],
    ["Ring count", "Tertiary amine count", "Ester bond count"]):

    data = [merged[merged["category_primary"] == cat][feat].dropna().values
            for cat in cat_order]
    ns = [len(d) for d in data]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(
        ["P+Dev", "Potent\nonly", "Dev\nonly", "Neither"], ns)],
        fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(feat.replace("_", " ").title(), fontsize=11, fontweight="bold")

    # Add p-value annotation for P+D vs Potent-only
    pd_vals = merged[merged["category_primary"] == "Potent + Developable"][feat].dropna()
    po_vals = merged[merged["category_primary"] == "Potent only"][feat].dropna()
    if len(pd_vals) >= 3 and len(po_vals) >= 3:
        stat, pval = mannwhitneyu(pd_vals, po_vals, alternative="two-sided")
        pstr = f"p={pval:.4f}" if pval >= 0.0001 else f"p<0.0001"
        ymax = max([d.max() if len(d) > 0 else 0 for d in data]) * 1.15
        ax.annotate(f"P+D vs PO\n{pstr}", xy=(1.5, ymax),
                    ha="center", fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.suptitle("Ionizable Lipid Structural Features:\nPotent+Developable vs. Potent-only LNPs",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/step18_structural_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> figures/step18_structural_comparison.png")

# Also save a simple summary table
summary = []
for cat in cat_order:
    sub = merged[merged["category_primary"] == cat]
    summary.append({
        "category": cat,
        "n": len(sub),
        "rings_median": sub["num_rings"].median(),
        "tert_amines_median": sub["num_tertiary_amines"].median(),
        "ester_bonds_median": sub["num_ester_bonds"].median(),
    })
pd.DataFrame(summary).to_csv("results/step18_summary_table.csv", index=False)
print("Saved -> results/step18_summary_table.csv")
