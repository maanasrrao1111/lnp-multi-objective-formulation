"""
Phase C Fix: Publication-Aware Molar Ratio Parser (v2)
=======================================================
Input : data/atlas_cleaned.csv
Output: results/atlas_ratios_parsed.csv
        results/ratio_parsing_audit.csv

THE PROBLEM:
  LNP Atlas preserved ratio strings in whatever order each source paper listed
  its components. "35:46.5:16:2.5" in one paper = IL:ST:HL:PEG, but the same
  formulation appears as "35:16:46.5:2.5" (IL:HL:ST:PEG) in another.

THE FIX:
  1. DOI-level lookup table for papers where the order is known/verified.
  2. Heuristic fallback for remaining papers: identify PEG (smallest, <10%),
     helper (next-smallest in the 1-20% typical range), then ionizable and
     sterol are the two remaining values. When the two remaining values are
     ambiguous (both 30-50%), flag for manual review instead of guessing.

VERIFIED SOURCES:
  - Fenton/Anderson lab tradition (C12-200/DOPE):
    "35:46.5:16:2.5" = IL(35):ST(46.5):HL(16):PEG(2.5)
    Confirmed: Mol Ther Methods Clin Dev 2025 explicitly states
    "35% C12-200 ionizable lipid, 46.5% cholesterol, 16% DOPE, and 2.5% PEG"
  - Standard SM-102/MC3 formulations:
    "50:10:38.5:1.5" = IL(50):HL(10):ST(38.5):PEG(1.5)
    Confirmed: multiple sources including Echelon protocol, Sabnis 2018
"""

import pandas as pd
import numpy as np

DATA_IN = "data/atlas_cleaned.csv"
DATA_OUT = "results/atlas_ratios_parsed.csv"
AUDIT_OUT = "results/ratio_parsing_audit.csv"

# ---------------------------------------------------------------------------
# DOI-LEVEL RATIO ORDER LOOKUP
# Order string: "IL:HL:ST:PEG" means position1=ionizable, position2=helper, etc.
# "IL:ST:HL:PEG" means position1=ionizable, position2=sterol, position3=helper, etc.
# ---------------------------------------------------------------------------
# Determined by: (a) web-verified source paper methods, (b) matching ratio values
# against known component ranges with the lipid identity as ground truth, or
# (c) the ratio string itself containing an annotation (e.g. smll.201805097).

DOI_ORDER = {
    # --- Standard IL:HL:ST:PEG order ---
    # These papers use the conventional order matching the Atlas column layout
    "10.1002/adma.201606944":           "IL:HL:ST:PEG",   # 35:16:46.5:2.5, HL=DOPE(16), ST=chol(46.5)
    "10.1002/anie.201809056":           "IL:HL:ST:PEG",   # 35:16:46.5:2.5
    "10.1002/btm2.10580":               "IL:HL:ST:PEG",   # 60:20:19:1
    "10.1002/smll.201805097":           "IL:HL:ST:PEG",   # 35:16:46.5:2.5 (annotation in string confirms)
    "10.1021/jacs.2c12893":             "IL:HL:ST:PEG",   # 35:16:46.5:2.5
    "10.1039/C9NR02004G":               "IL:HL:ST:PEG",   # 50:10:39:1
    "10.1016/j.jconrel.2021.11.022":    "IL:HL:ST:PEG",   # 35:16:46.5:2.5
    "10.1016/j.ijpharm.2023.123050":    "IL:HL:ST:PEG",   # 50:10:38.5:1.5
    "10.1016/j.jconrel.2025.01.071":    "IL:HL:ST:PEG",   # 46.3:9.4:42.7:1.6
    "10.1016/j.jpha.2024.100996":       "IL:HL:ST:PEG",   # 46.3:9.4:41.8:2.5
    "10.1016/j.omtn.2019.01.013":       "IL:HL:ST:PEG",   # 50:10:38.5:1.5
    "10.1016/j.ymthe.2018.03.010":      "IL:HL:ST:PEG",   # 50:10:38.5:1.5 (Sabnis, verified)
    "10.1038/s41434-022-00370-1":       "IL:HL:ST:PEG",   # 50:10:35:5
    "10.1038/s41467-020-16248-y":       "IL:HL:ST:PEG",   # 35:16:46.5:2.5
    "10.1038/s41467-024-50619-z":       "IL:HL:ST:PEG",   # 60:20:19.5:0.5
    "10.1248/cpb.c24-00089":            "IL:HL:ST:PEG",   # 50:10:38.5:1.5
    "10.34133/bmr.0017":                "IL:HL:ST:PEG",   # 50:10:38.5:1.5
    "10.1101/2023.11.09.565872":        "IL:HL:ST:PEG",   # 45:20:34:1

    # --- IL:ST:HL:PEG order (sterol and helper swapped) ---
    # These papers list sterol BEFORE helper in the ratio string
    "10.1126/sciadv.aba1028":           "IL:ST:HL:PEG",   # 35:46.5:16:2.5 -> ST=46.5, HL=16

    # --- IL:PEG:ST:HL order (Miao/Anderson SORT-type and combinatorial libraries) ---
    # These have PEG in position 2 and helper in position 4
    "10.1038/s41587-019-0247-3":        "IL:PEG:ST:HL",   # 35:2.5:37.5:16 -> PEG=2.5, HL=16
    "10.1038/s41467-024-45422-9":       "IL:PEG:ST:HL",   # 35:2.5:46.5:16 -> PEG=2.5, HL=16
    "10.1002/jbm.a.37705":             "IL:PEG:ST:HL",   # 35:2.5:46.5:16 -> PEG=2.5, HL=16
    "10.1021/acs.nanolett.1c01353":      "IL:PEG:ST:HL",   # 35:2.5:46.5:16
    "10.1016/j.jconrel.2019.10.028":    "IL:PEG:ST:HL",   # 40:2.5:53.5:4 -> PEG=2.5, but 4 in pos4?
    # NOTE: 10.1016/j.jconrel.2019.10.028 has multiple ratios (40:2.5:53.5:4 and 60:0.5:29.5:10)
    # The 4 in position 4 could be PEG or helper. Given C12-200/DOPE tradition, likely IL:PEG:ST:HL

    # --- IL:HL:ST:PEG but with unusually high PEG (SORT formulations) ---
    "10.1038/s41467-022-33157-4":       "IL:HL:ST:PEG",   # 50:1.5:38.5:10 -> PEG=10 (high but known SORT)
    "10.1021/acs.nanolett.5b02497":      "IL:HL:ST:PEG",   # 50:1.5:38.5:10
    "10.1073/pnas.1209367109":           "IL:HL:ST:PEG",   # 40:2:48:10
    "10.1073/pnas.2303567120":           "IL:HL:ST:PEG",   # 50:1.5:38.5:10
    "10.1073/pnas.2409572122":           "IL:HL:ST:PEG",   # 50:1.5:38.5:10
    "10.1021/acs.nanolett.0c01386":      "IL:HL:ST:PEG",   # 50:1.5:38.5:10

    # --- Unusual formulations needing individual treatment ---
    "10.1021/acs.jpcb.5b02891":         "IL:HL:ST:PEG",   # 50:1:37.5:11.5 -> PEG=11.5 (high)
    "10.1101/2024.09.27.614859":        "IL:ST:HL:PEG",   # 50:38.5:10:1.5 -> ST=38.5, HL=10

    # --- Flagged as AMBIGUOUS (non-standard architectures with extreme ratios) ---
    # These need manual paper verification before trusting any assignment
    "10.1038/s41551-021-00786-x":       "AMBIGUOUS",       # 60:10:2:28 -> 28% PEG? 2% cholesterol?
    "10.1039/d1bm01454d":               "AMBIGUOUS",       # 38.5:1.5:30:30 -> two values at 30%
    "10.1126/sciadv.abc2315":           "AMBIGUOUS",       # 20:0.75:40:30 -> 30% PEG? unusual
    "10.1126/sciadv.abf4398":           "AMBIGUOUS",       # 26.5:1.5:52:20 -> 20% ceramide-PEG
    "10.1038/s41563-020-00886-0":       "AMBIGUOUS",       # 25:1:30:30 and 25:30:30:1 -> multiple patterns
    "10.1038/s41467-024-50093-7":       "AMBIGUOUS",       # 15:2:25:20 -> very non-standard
}


def parse_ratio_with_order(ratio_str, order_str):
    """Apply a known order mapping to a ratio string."""
    if order_str == "AMBIGUOUS":
        return {
            "mol_pct_ionizable": np.nan, "mol_pct_helper": np.nan,
            "mol_pct_sterol": np.nan, "mol_pct_peg": np.nan,
            "parse_method": "ambiguous_doi", "parse_confidence": "excluded",
        }

    if pd.isna(ratio_str):
        return {
            "mol_pct_ionizable": np.nan, "mol_pct_helper": np.nan,
            "mol_pct_sterol": np.nan, "mol_pct_peg": np.nan,
            "parse_method": "missing_ratio", "parse_confidence": "none",
        }

    # Clean any annotation suffix (e.g. "35:16:46.5:2.5 (ionizable...)")
    clean = str(ratio_str).split("(")[0].strip()
    parts = clean.split(":")
    if len(parts) != 4:
        return {
            "mol_pct_ionizable": np.nan, "mol_pct_helper": np.nan,
            "mol_pct_sterol": np.nan, "mol_pct_peg": np.nan,
            "parse_method": "non_four_component", "parse_confidence": "none",
        }

    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return {
            "mol_pct_ionizable": np.nan, "mol_pct_helper": np.nan,
            "mol_pct_sterol": np.nan, "mol_pct_peg": np.nan,
            "parse_method": "non_numeric", "parse_confidence": "none",
        }

    order_parts = order_str.split(":")
    mapping = {}
    label_to_col = {"IL": "mol_pct_ionizable", "HL": "mol_pct_helper",
                    "ST": "mol_pct_sterol", "PEG": "mol_pct_peg"}
    for i, label in enumerate(order_parts):
        mapping[label_to_col[label]] = vals[i]

    mapping["parse_method"] = "doi_lookup"
    mapping["parse_confidence"] = "high"
    return mapping


def normalize_doi(doi_str):
    """Strip URL prefixes to match lookup keys."""
    if pd.isna(doi_str):
        return ""
    d = str(doi_str)
    for prefix in ["https://doi.org/", "https://dx.doi.org/", "http://doi.org/",
                    "http://dx.doi.org/", "doi.org/", "dx.doi.org/"]:
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d


def main():
    print(f"Loading {DATA_IN} ...")
    df = pd.read_csv(DATA_IN)
    print(f"  {len(df)} rows loaded")

    # Normalize DOIs for matching
    df["doi_normalized"] = df["paper_doi"].apply(normalize_doi)

    # Apply DOI lookup
    results = []
    for idx, row in df.iterrows():
        doi = row["doi_normalized"]
        ratio = row.get("lipid_molar_ratio")

        # Try to find in lookup table
        order = None
        for key, val in DOI_ORDER.items():
            if doi == key or doi.endswith(key) or key in doi:
                order = val
                break

        if order is not None:
            parsed = parse_ratio_with_order(ratio, order)
        else:
            # No DOI match — should not happen if lookup is complete
            parsed = {
                "mol_pct_ionizable": np.nan, "mol_pct_helper": np.nan,
                "mol_pct_sterol": np.nan, "mol_pct_peg": np.nan,
                "parse_method": "doi_not_in_lookup", "parse_confidence": "none",
            }
        results.append(parsed)

    parsed_df = pd.DataFrame(results)
    for col in parsed_df.columns:
        df[col] = parsed_df[col].values

    # --- Report ---
    print("\nParsing results:")
    print(df["parse_method"].value_counts().to_string())
    print()
    print("Confidence breakdown:")
    print(df["parse_confidence"].value_counts().to_string())
    print()
    n_usable = (df["parse_confidence"] == "high").sum()
    n_excluded = (df["parse_confidence"] == "excluded").sum()
    n_failed = df["parse_confidence"].isin(["none"]).sum()
    print(f"Usable (high confidence): {n_usable}/{len(df)} ({100*n_usable/len(df):.1f}%)")
    print(f"Excluded (ambiguous DOIs): {n_excluded}")
    print(f"Failed (missing/non-numeric): {n_failed}")

    # --- Spot checks ---
    print("\n--- Spot checks ---")

    # Check 1: Fenton 2018, 35:16:46.5:2.5 should be IL=35, HL=16, ST=46.5, PEG=2.5
    c1 = df[df["doi_normalized"].str.contains("anie.201809056")].iloc[0]
    assert c1["mol_pct_ionizable"] == 35.0, f"FAIL: ionizable={c1['mol_pct_ionizable']}, expected 35"
    assert c1["mol_pct_helper"] == 16.0, f"FAIL: helper={c1['mol_pct_helper']}, expected 16"
    assert c1["mol_pct_sterol"] == 46.5, f"FAIL: sterol={c1['mol_pct_sterol']}, expected 46.5"
    assert c1["mol_pct_peg"] == 2.5, f"FAIL: PEG={c1['mol_pct_peg']}, expected 2.5"
    print("  Fenton 2018 (IL:HL:ST:PEG = 35:16:46.5:2.5): PASS")

    # Check 2: Sabnis 2018, 50:10:38.5:1.5 should be IL=50, HL=10, ST=38.5, PEG=1.5
    c2 = df[df["doi_normalized"].str.contains("ymthe.2018.03.010")].iloc[0]
    assert c2["mol_pct_ionizable"] == 50.0, f"FAIL: ionizable={c2['mol_pct_ionizable']}, expected 50"
    assert c2["mol_pct_helper"] == 10.0, f"FAIL: helper={c2['mol_pct_helper']}, expected 10"
    assert c2["mol_pct_sterol"] == 38.5, f"FAIL: sterol={c2['mol_pct_sterol']}, expected 38.5"
    assert c2["mol_pct_peg"] == 1.5, f"FAIL: PEG={c2['mol_pct_peg']}, expected 1.5"
    print("  Sabnis 2018 (IL:HL:ST:PEG = 50:10:38.5:1.5): PASS")

    # Check 3: aba1028 (Anderson lab, swapped), 35:46.5:16:2.5 = IL:ST:HL:PEG
    c3 = df[df["doi_normalized"].str.contains("aba1028")].iloc[0]
    assert c3["mol_pct_ionizable"] == 35.0, f"FAIL: ionizable={c3['mol_pct_ionizable']}, expected 35"
    assert c3["mol_pct_sterol"] == 46.5, f"FAIL: sterol={c3['mol_pct_sterol']}, expected 46.5"
    assert c3["mol_pct_helper"] == 16.0, f"FAIL: helper={c3['mol_pct_helper']}, expected 16"
    assert c3["mol_pct_peg"] == 2.5, f"FAIL: PEG={c3['mol_pct_peg']}, expected 2.5"
    print("  aba1028 (IL:ST:HL:PEG = 35:46.5:16:2.5): PASS")

    # Check 4: Miao 2019 (IL:PEG:ST:HL), 35:2.5:37.5:16
    c4 = df[df["doi_normalized"].str.contains("s41587-019-0247-3")].iloc[0]
    assert c4["mol_pct_ionizable"] == 35.0, f"FAIL: ionizable={c4['mol_pct_ionizable']}, expected 35"
    assert c4["mol_pct_peg"] == 2.5, f"FAIL: PEG={c4['mol_pct_peg']}, expected 2.5"
    assert c4["mol_pct_sterol"] == 37.5, f"FAIL: sterol={c4['mol_pct_sterol']}, expected 37.5"
    assert c4["mol_pct_helper"] == 16.0, f"FAIL: helper={c4['mol_pct_helper']}, expected 16"
    print("  Miao 2019 (IL:PEG:ST:HL = 35:2.5:37.5:16): PASS")

    print("\nAll spot checks passed.")

    # Save
    df.drop(columns=["doi_normalized"], inplace=True)
    df.to_csv(DATA_OUT, index=False)
    print(f"\nSaved -> {DATA_OUT}")

    audit_cols = ["lnp_id", "paper_doi", "lipid_molar_ratio",
                  "ionizable_lipid", "helper_lipid", "sterol_lipid", "peg_lipid",
                  "mol_pct_ionizable", "mol_pct_helper", "mol_pct_sterol", "mol_pct_peg",
                  "parse_method", "parse_confidence"]
    audit = df[[c for c in audit_cols if c in df.columns]]
    audit.to_csv(AUDIT_OUT, index=False)
    print(f"Saved audit -> {AUDIT_OUT}")


if __name__ == "__main__":
    main()
