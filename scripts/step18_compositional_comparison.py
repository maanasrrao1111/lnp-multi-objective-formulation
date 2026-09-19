import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

features = pd.read_csv("results/phase_c_feature_matrix.csv")
classification = pd.read_csv("results/phase_e_primary_classification.csv")

# Merge on lnp_id - keep only category label from classification
merged = features.merge(
    classification[["lnp_id", "category_primary"]],
    on="lnp_id",
    how="inner"
)

print(f"Rows after merge: {len(merged)}")

pd_group = merged[merged["category_primary"] == "Potent + Developable"]
po_group = merged[merged["category_primary"] == "Potent only"]

print(f"Potent + Developable with features: {len(pd_group)}")
print(f"Potent only with features: {len(po_group)}")
print()

# All features to test
test_features = [
    "mol_pct_ionizable", "mol_pct_helper", "mol_pct_sterol", "mol_pct_peg",
    "log_il_to_sterol", "log_helper_to_sterol", "log_peg_to_sterol",
    "mw", "logp", "tpsa", "num_rotatable_bonds", "num_hbd", "num_hba",
    "num_ester_bonds", "num_tertiary_amines", "tail_length_carbons",
    "num_unsaturated_bonds", "num_branch_points", "num_rings",
    "fraction_csp3", "heavy_atom_count",
]
test_features = [f for f in test_features if f in merged.columns]

bonferroni = 0.05 / len(test_features)
print(f"Features tested: {len(test_features)}")
print(f"Bonferroni threshold: {bonferroni:.4f}")
print()
print(f"{'Feature':<30} {'P+D med':>10} {'PO med':>10} {'diff':>10} {'p':>10} {'sig':<5}")
print("-" * 80)

results = []
for feat in test_features:
    pd_vals = pd_group[feat].dropna()
    po_vals = po_group[feat].dropna()
    if len(pd_vals) < 3 or len(po_vals) < 3:
        continue
    stat, pval = mannwhitneyu(pd_vals, po_vals, alternative="two-sided")
    pd_med = pd_vals.median()
    po_med = po_vals.median()
    diff = pd_med - po_med
    sig = "***" if pval < bonferroni else ("*" if pval < 0.05 else "")
    print(f"{feat:<30} {pd_med:>10.2f} {po_med:>10.2f} {diff:>+10.2f} {pval:>10.4f} {sig:<5}")
    results.append({
        "feature": feat,
        "pd_median": round(pd_med, 3),
        "pd_n": len(pd_vals),
        "po_median": round(po_med, 3),
        "po_n": len(po_vals),
        "median_diff_pd_minus_po": round(diff, 3),
        "p_value": round(pval, 6),
        "sig_uncorrected": pval < 0.05,
        "sig_bonferroni": pval < bonferroni,
    })

res_df = pd.DataFrame(results).sort_values("p_value")
res_df.to_csv("results/step18_compositional_comparison.csv", index=False)

print()
print(f"Nominally significant (p<0.05): {res_df['sig_uncorrected'].sum()}/{len(res_df)}")
print(f"Bonferroni significant: {res_df['sig_bonferroni'].sum()}/{len(res_df)}")
print()
print("Top 5 by p-value:")
print(res_df[["feature","pd_median","po_median","median_diff_pd_minus_po","p_value","sig_uncorrected","sig_bonferroni"]].head(5).to_string(index=False))
print()
print("Saved -> results/step18_compositional_comparison.csv")
