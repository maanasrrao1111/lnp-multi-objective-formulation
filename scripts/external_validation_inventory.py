#!/usr/bin/env python3
"""
External Validation Inventory
=============================

Inputs
------
data/external_validation_cleaned.csv
results/external_validation_feature_coverage_audit.csv

Outputs
-------
results/external_validation_inventory.csv
results/external_validation_inventory_report.txt

Purpose
-------
Summarize the independent validation set by paper, DOI, lipid identity,
ratio strings, and SMILES coverage before any external prediction is run.
"""

from pathlib import Path
import pandas as pd

EXTERNAL_IN = Path("data/external_validation_cleaned.csv")
AUDIT_IN = Path("results/external_validation_feature_coverage_audit.csv")
TABLE_OUT = Path("results/external_validation_inventory.csv")
REPORT_OUT = Path("results/external_validation_inventory_report.txt")


def first_existing(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(
            f"None of these columns were found: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def main():
    if not EXTERNAL_IN.exists():
        raise FileNotFoundError(f"Missing: {EXTERNAL_IN}")
    if not AUDIT_IN.exists():
        raise FileNotFoundError(f"Missing: {AUDIT_IN}")

    ext = pd.read_csv(EXTERNAL_IN)
    audit = pd.read_csv(AUDIT_IN)

    paper_col = first_existing(ext, ["paper", "paper_title"], required=False)
    doi_col = first_existing(ext, ["doi", "paper_doi"])
    lipid_col = first_existing(ext, ["ionizable_lipid", "ionizable_lipid_name"])
    ratio_col = first_existing(ext, ["molar_ratio", "lipid_molar_ratio"])

    if paper_col is None:
        ext["paper_display"] = ext[doi_col]
        paper_col = "paper_display"

    resolved_col = first_existing(
        audit,
        ["ionizable_lipid_smiles_resolved", "ionizable_lipid_smiles"],
    )
    match_method_col = first_existing(
        audit,
        ["smiles_match_method"],
        required=False,
    )

    # Align using row order because audit was created directly from the external table.
    if len(ext) != len(audit):
        raise ValueError(
            f"Row-count mismatch: external={len(ext)}, audit={len(audit)}"
        )

    ext = ext.copy()
    ext["smiles_resolved"] = audit[resolved_col].notna().to_numpy()
    ext["resolved_smiles"] = audit[resolved_col].to_numpy()

    if match_method_col is not None:
        ext["smiles_match_method"] = audit[match_method_col].to_numpy()
    else:
        ext["smiles_match_method"] = "unknown"

    rows = []

    for (paper, doi), group in ext.groupby([paper_col, doi_col], dropna=False):
        unique_raw_lipids = group[lipid_col].dropna().astype(str).nunique()
        unique_resolved_structures = (
            group.loc[group["smiles_resolved"], "resolved_smiles"]
            .dropna()
            .astype(str)
            .nunique()
        )
        unique_ratio_strings = (
            group[ratio_col].dropna().astype(str).nunique()
        )

        rows.append(
            {
                "paper": paper,
                "doi": doi,
                "n_formulations": len(group),
                "n_unique_raw_lipid_names": unique_raw_lipids,
                "n_rows_with_resolved_smiles": int(group["smiles_resolved"].sum()),
                "n_unique_resolved_structures": unique_resolved_structures,
                "n_unique_ratio_strings": unique_ratio_strings,
                "raw_lipid_names": " | ".join(
                    sorted(group[lipid_col].dropna().astype(str).unique())
                ),
                "ratio_examples": " | ".join(
                    sorted(group[ratio_col].dropna().astype(str).unique())[:10]
                ),
            }
        )

    inventory = pd.DataFrame(rows).sort_values(
        ["n_formulations", "paper"],
        ascending=[False, True],
    )
    inventory.to_csv(TABLE_OUT, index=False)

    optional_component_columns = [
        col for col in [
            "helper_lipid",
            "sterol_lipid",
            "peg_lipid",
            "mol_pct_ionizable",
            "mol_pct_helper",
            "mol_pct_sterol",
            "mol_pct_peg",
            "ratio_order",
        ]
        if col in ext.columns
    ]

    unique_resolved_structures = (
        ext.loc[ext["smiles_resolved"], "resolved_smiles"]
        .dropna()
        .astype(str)
        .nunique()
    )

    report_lines = [
        "EXTERNAL VALIDATION INVENTORY",
        "=============================",
        "",
        f"Total formulations: {len(ext)}",
        f"Total papers: {ext[doi_col].nunique(dropna=True)}",
        f"Unique raw lipid labels: {ext[lipid_col].nunique(dropna=True)}",
        f"Rows with resolved SMILES: {int(ext['smiles_resolved'].sum())}",
        f"Unique resolved chemical structures: {unique_resolved_structures}",
        "",
        "Available component/order columns in external file:",
        (
            "  " + ", ".join(optional_component_columns)
            if optional_component_columns
            else "  None beyond the ratio string itself"
        ),
        "",
        "Per-paper inventory:",
        inventory.to_string(index=False),
        "",
        "Important:",
        "  MC3 and DLin-MC3-DMA may be separate raw labels but the same normalized",
        "  chemical structure. Validation breadth should be counted by unique",
        "  structures, not only by raw lipid names.",
        "",
        "Saved:",
        f"  {TABLE_OUT}",
        f"  {REPORT_OUT}",
    ]

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
