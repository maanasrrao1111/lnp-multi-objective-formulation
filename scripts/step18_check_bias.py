import pandas as pd
from scipy.stats import chi2_contingency

features = pd.read_csv("results/phase_c_feature_matrix.csv")
classification = pd.read_csv("results/phase_e_primary_classification.csv")
merged = features.merge(classification[["lnp_id", "category_primary"]], on="lnp_id", how="inner")

merged["has_descriptors"] = merged["num_rings"].notna()

print("Missing descriptors by category:")
ct = pd.crosstab(merged["category_primary"], merged["has_descriptors"])
ct.columns = ["missing", "present"]
ct["pct_present"] = (ct["present"] / (ct["missing"] + ct["present"]) * 100).round(1)
print(ct)

chi2, p, dof, expected = chi2_contingency(ct[["missing", "present"]])
print(f"\nChi-square test: chi2={chi2:.3f}, p={p:.4f}")
print("p>0.05 = missingness random across categories (figure unbiased)")
print("p<0.05 = missingness non-random (figure may be biased)")
