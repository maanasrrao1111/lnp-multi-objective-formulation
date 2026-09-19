import pandas as pd
import numpy as np

df = pd.read_csv("data/atlas_cleaned.csv")
classification = pd.read_csv("results/phase_e_primary_classification.csv")

# Flag cancer-relevant rows using multiple criteria
cancer_keywords_title = ["tumor", "tumour", "cancer", "vaccine", "immune", "antigen"]
cancer_keywords_bio = ["tumor", "tumour", "carcinoma", "cancer", "immune"]

df["cancer_title"] = df["paper_title"].str.contains(
    "|".join(cancer_keywords_title), case=False, na=False)
df["cancer_bio"] = df["bioactivity_profile"].str.contains(
    "|".join(cancer_keywords_bio), case=False, na=False)
df["cancer_relevant"] = df["cancer_title"] | df["cancer_bio"]

print(f"Total formulations: {len(df)}")
print(f"Cancer-relevant (title or bio): {df['cancer_relevant'].sum()}")
print()

# Which papers are cancer-relevant?
cancer_papers = df[df["cancer_relevant"]][["paper_doi", "paper_title"]].drop_duplicates()
print(f"Cancer-relevant papers: {len(cancer_papers)}")
for _, row in cancer_papers.iterrows():
    n = len(df[(df["paper_doi"] == row["paper_doi"]) & df["cancer_relevant"]])
    print(f"  n={n:3d}  {str(row['paper_title'])[:90]}")

# Now check overlap with the 105 complete-case set
merged = classification.merge(
    df[["lnp_id", "cancer_relevant", "paper_title", "bioactivity_profile"]],
    on="lnp_id", how="left"
)
merged["cancer_relevant"] = merged["cancer_relevant"].fillna(False)

print(f"\n--- Complete-case overlap ---")
print(f"Total complete cases: {len(merged)}")
print(f"Cancer-relevant complete cases: {merged['cancer_relevant'].sum()}")
print()

if merged["cancer_relevant"].sum() > 0:
    cancer_sub = merged[merged["cancer_relevant"]]
    print("Category breakdown (cancer-relevant complete cases):")
    print(cancer_sub["category_primary"].value_counts().to_string())
    print()
    potent_cancer = cancer_sub[cancer_sub["category_primary"].isin(
        ["Potent + Developable", "Potent only"])]
    n_potent = len(potent_cancer)
    n_fail = len(potent_cancer[potent_cancer["category_primary"] == "Potent only"])
    if n_potent > 0:
        print(f"Cancer-relevant potent formulations: {n_potent}")
        print(f"Cancer-relevant potent failing physchem: {n_fail}")
        print(f"Cancer-relevant failure rate: {100*n_fail/n_potent:.1f}%")
    print()
    print("Papers contributing cancer-relevant complete cases:")
    for doi in cancer_sub["paper_doi"].unique():
        sub = cancer_sub[cancer_sub["paper_doi"] == doi]
        title = df[df["paper_doi"] == doi]["paper_title"].iloc[0]
        print(f"  n={len(sub):3d}  {str(title)[:90]}")
else:
    print("No cancer-relevant formulations in the complete-case set.")
    print()
    # Check the broader 464 set for cancer formulations with at least SOME endpoints
    cancer_all = df[df["cancer_relevant"]]
    has_size = cancer_all["size_mean"].notna().sum()
    has_pdi = cancer_all["pdi_mean"].notna().sum()
    has_ee = cancer_all["ee_mean"].notna().sum()
    has_tx = cancer_all["transfection_pctile"].notna().sum()
    print(f"Cancer-relevant in full 464: {len(cancer_all)}")
    print(f"  with size: {has_size}")
    print(f"  with PDI: {has_pdi}")
    print(f"  with EE: {has_ee}")
    print(f"  with transfection: {has_tx}")
