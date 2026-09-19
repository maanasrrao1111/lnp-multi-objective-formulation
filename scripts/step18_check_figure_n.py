import pandas as pd

features = pd.read_csv("results/phase_c_feature_matrix.csv")
classification = pd.read_csv("results/phase_e_primary_classification.csv")
merged = features.merge(classification[["lnp_id", "category_primary"]], on="lnp_id", how="inner")

print(f"Total merged: {len(merged)}")

check_cols = ["num_rings", "num_tertiary_amines", "num_ester_bonds"]
complete = merged.dropna(subset=check_cols)
print(f"With all three figure features non-null: {len(complete)}")
print()
print("Category counts (full merge, n=93):")
print(merged["category_primary"].value_counts())
print()
print("Category counts (figure subset, non-null on all 3 features):")
print(complete["category_primary"].value_counts())
print()
print("Missing breakdown per feature:")
for col in check_cols:
    print(f"  {col}: {merged[col].isna().sum()} missing of {len(merged)}")
