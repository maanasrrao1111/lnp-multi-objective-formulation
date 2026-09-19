import pandas as pd

df = pd.read_csv("data/atlas_cleaned.csv")

print("Columns that might indicate cancer relevance:")
for col in ["bioactivity_profile", "target_type", "nucleic_acid_sequence", "paper_title"]:
    if col in df.columns:
        print(f"\n--- {col} ---")
        vals = df[col].dropna().unique()
        for v in vals[:30]:
            print(f"  {str(v)[:120]}")
        if len(vals) > 30:
            print(f"  ... ({len(vals)} total unique values)")

print("\n\n--- Searching for cancer/tumor/oncology keywords ---")
cancer_terms = ["cancer", "tumor", "tumour", "oncol", "melanoma", "glioma",
                "carcinoma", "lymphoma", "leukemia", "immune", "vaccine",
                "antigen", "CAR", "immunotherapy"]

for col in ["bioactivity_profile", "paper_title", "nucleic_acid_sequence", "target_type"]:
    if col not in df.columns:
        continue
    for term in cancer_terms:
        hits = df[df[col].str.contains(term, case=False, na=False)]
        if len(hits) > 0:
            print(f"  '{term}' in {col}: {len(hits)} rows")
            print(f"    example: {hits[col].iloc[0][:100]}")
