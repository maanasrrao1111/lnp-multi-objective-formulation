#!/usr/bin/env python3
"""
External Validation Feature-Coverage Audit
==========================================

Inputs
------
data/external_validation_cleaned.csv
results/phase_c_feature_matrix.csv

Outputs
-------
results/external_validation_feature_coverage_audit.csv
results/external_validation_missing_smiles.csv
results/external_validation_feature_coverage_report.txt

Purpose
-------
Audit whether the independent external validation formulations have:
1. complete four-component ratios;
2. ratios that sum approximately to 100;
3. an ionizable-lipid SMILES match in the Atlas-derived feature matrix;
4. publication DOI overlap with the training set.
"""

from pathlib import Path
import pandas as pd

EXTERNAL_IN = Path("data/external_validation_cleaned.csv")
TRAIN_IN = Path("results/phase_c_feature_matrix.csv")

AUDIT_OUT = Path("results/external_validation_feature_coverage_audit.csv")
MISSING_OUT = Path("results/external_validation_missing_smiles.csv")
REPORT_OUT = Path("results/external_validation_feature_coverage_report.txt")


def normalize_doi(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    prefixes = [
        "https://doi.org/",
        "https://dx.doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "dx.doi.org/",
    ]

    lowered = text.lower()

    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break

    return text.strip().lower()


def first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {label}. Tried columns: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def main() -> None:
    if not EXTERNAL_IN.exists():
        raise FileNotFoundError(f"Missing input: {EXTERNAL_IN}")

    if not TRAIN_IN.exists():
        raise FileNotFoundError(f"Missing input: {TRAIN_IN}")

    external = pd.read_csv(EXTERNAL_IN)
    train = pd.read_csv(TRAIN_IN)

    ext_lipid_col = first_existing_column(
        external,
        ["ionizable_lipid", "ionizable_lipid_name"],
        "external ionizable-lipid name column",
    )

    ext_ratio_col = first_existing_column(
        external,
        ["molar_ratio", "lipid_molar_ratio"],
        "external molar-ratio column",
    )

    ext_doi_col = first_existing_column(
        external,
        ["doi", "paper_doi"],
        "external DOI column",
    )

    train_lipid_col = first_existing_column(
        train,
        ["ionizable_lipid", "ionizable_lipid_name"],
        "training ionizable-lipid name column",
    )

    train_smiles_col = first_existing_column(
        train,
        ["ionizable_lipid_smiles", "smiles"],
        "training ionizable-lipid SMILES column",
    )

    train_doi_col = first_existing_column(
        train,
        ["paper_doi", "doi"],
        "training DOI column",
    )

    aliases = {
        "MC3": "DLin-MC3-DMA",
    }

    external = external.copy()
    external["atlas_lipid_name"] = external[ext_lipid_col].replace(aliases)

    ratio_parts = (
        external[ext_ratio_col]
        .astype("string")
        .str.split(":", expand=True)
    )

    # Keep exactly the first four components for this conventional four-component audit.
    ratio_parts = ratio_parts.iloc[:, :4].apply(pd.to_numeric, errors="coerce")

    external["ratio_component_count"] = (
        external[ext_ratio_col]
        .astype("string")
        .str.count(":")
        .add(1)
    )

    external["ratio_sum"] = ratio_parts.sum(axis=1, min_count=4)
    external["ratio_complete"] = (
        ratio_parts.shape[1] == 4
        and ratio_parts.notna().all(axis=1)
    )
    external["ratio_sum_near_100"] = external["ratio_sum"].between(
        99.0,
        101.0,
        inclusive="both",
    )

    lookup = (
        train[[train_lipid_col, train_smiles_col]]
        .dropna()
        .drop_duplicates()
        .rename(
            columns={
                train_lipid_col: "atlas_lipid_name",
                train_smiles_col: "ionizable_lipid_smiles",
            }
        )
    )

    # Keep one row per lipid name if the same name occurs repeatedly with the same SMILES.
    duplicate_name_counts = (
        lookup.groupby("atlas_lipid_name")["ionizable_lipid_smiles"]
        .nunique()
    )

    conflicting_names = duplicate_name_counts[duplicate_name_counts > 1].index.tolist()

    if conflicting_names:
        raise ValueError(
            "Conflicting SMILES assignments were found for these training lipid names: "
            + ", ".join(map(str, conflicting_names[:20]))
        )

    lookup = lookup.drop_duplicates(subset=["atlas_lipid_name"])

    external_audit = external.merge(
        lookup,
        on="atlas_lipid_name",
        how="left",
    )

    train_dois = {
        normalize_doi(value)
        for value in train[train_doi_col].dropna()
        if normalize_doi(value)
    }

    external_dois = {
        normalize_doi(value)
        for value in external[ext_doi_col].dropna()
        if normalize_doi(value)
    }

    overlap = sorted(train_dois.intersection(external_dois))

    useful_missing_cols = [
        col
        for col in [
            "paper",
            ext_doi_col,
            ext_lipid_col,
            ext_ratio_col,
            "size_nm",
            "particle_size_nm_mean",
            "ee_percent",
            "encapsulation_efficiency_percent_mean",
            "pdi",
            "pdi_mean",
        ]
        if col in external_audit.columns
    ]

    missing = (
        external_audit[
            external_audit["ionizable_lipid_smiles"].isna()
        ][useful_missing_cols]
        .drop_duplicates()
    )

    if "paper" in missing.columns and ext_lipid_col in missing.columns:
        missing = missing.sort_values(["paper", ext_lipid_col])
    elif ext_lipid_col in missing.columns:
        missing = missing.sort_values(ext_lipid_col)

    external_audit.to_csv(AUDIT_OUT, index=False)
    missing.to_csv(MISSING_OUT, index=False)

    matched_mask = external_audit["ionizable_lipid_smiles"].notna()

    matched_lipids = (
        external_audit.loc[
            matched_mask,
            [ext_lipid_col, "atlas_lipid_name"],
        ]
        .drop_duplicates()
        .sort_values(ext_lipid_col)
    )

    report_lines = [
        "EXTERNAL VALIDATION FEATURE-COVERAGE AUDIT",
        "==========================================",
        "",
        f"External formulations: {len(external)}",
        f"External papers: {external[ext_doi_col].nunique(dropna=True)}",
        f"Unique external ionizable lipids: {external[ext_lipid_col].nunique(dropna=True)}",
        f"Rows with exactly four ratio components: {int((external['ratio_component_count'] == 4).sum())}",
        f"Rows with complete numeric four-component ratios: {int(external['ratio_complete'].sum())}",
        f"Rows with ratios summing approximately to 100: {int(external['ratio_sum_near_100'].sum())}",
        f"Rows currently matched to an ionizable-lipid SMILES: {int(matched_mask.sum())}",
        (
            "Unique external lipid identities currently matched: "
            f"{external_audit.loc[matched_mask, ext_lipid_col].nunique(dropna=True)}"
        ),
        (
            "Unique external lipids still missing SMILES: "
            f"{external_audit.loc[~matched_mask, ext_lipid_col].nunique(dropna=True)}"
        ),
        f"Training/external DOI overlap: {overlap}",
        "",
        "Currently matched lipids:",
        (
            matched_lipids.to_string(index=False)
            if len(matched_lipids)
            else "None"
        ),
        "",
        "Saved:",
        f"  {AUDIT_OUT}",
        f"  {MISSING_OUT}",
    ]

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")

    print(report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
