#!/usr/bin/env python3
"""
External Validation Feature-Coverage Audit — v2
================================================

Inputs
------
data/external_validation_cleaned.csv
results/phase_c_feature_matrix.csv

Outputs
-------
results/external_validation_feature_coverage_audit.csv
results/external_validation_missing_smiles.csv
results/external_validation_conflicting_training_names.csv
results/external_validation_feature_coverage_report.txt

Why v2
------
The first audit stopped when a generic training label such as "Custom lipid"
mapped to more than one SMILES. That conflict is scientifically meaningful:
name-based matching is unsafe for that label.

This version:
  1. Uses an external SMILES directly if one is already present.
  2. Uses name-based matching only when a normalized training lipid name maps
     to exactly one unique SMILES.
  3. Excludes conflicting/generic names rather than guessing.
  4. Saves the conflicting names for manual review.
"""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

EXTERNAL_IN = Path("data/external_validation_cleaned.csv")
TRAIN_IN = Path("results/phase_c_feature_matrix.csv")

AUDIT_OUT = Path("results/external_validation_feature_coverage_audit.csv")
MISSING_OUT = Path("results/external_validation_missing_smiles.csv")
CONFLICT_OUT = Path("results/external_validation_conflicting_training_names.csv")
REPORT_OUT = Path("results/external_validation_feature_coverage_report.txt")


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
    required: bool = True,
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(
            f"Could not find {label}. Tried columns: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def normalize_doi(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    lowered = text.lower()

    prefixes = [
        "https://doi.org/",
        "https://dx.doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "doi.org/",
        "dx.doi.org/",
    ]

    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break

    return text.strip().lower()


def normalize_lipid_name(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    aliases = {
        "mc3": "dlin-mc3-dma",
        "dlin mc3 dma": "dlin-mc3-dma",
        "dlin-mc3 dma": "dlin-mc3-dma",
    }

    # Normalize Unicode dashes, whitespace, and capitalization conservatively.
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .lower()
    )
    text = re.sub(r"\s+", " ", text).strip()

    return aliases.get(text, text)


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
    ext_smiles_col = first_existing_column(
        external,
        ["ionizable_lipid_smiles", "smiles"],
        "external ionizable-lipid SMILES column",
        required=False,
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

    external = external.copy()
    train = train.copy()

    external["lipid_name_normalized"] = external[ext_lipid_col].map(
        normalize_lipid_name
    )
    train["lipid_name_normalized"] = train[train_lipid_col].map(
        normalize_lipid_name
    )

    # ---------- Ratio audit ----------
    raw_ratio = external[ext_ratio_col].astype("string")
    external["ratio_component_count"] = raw_ratio.str.count(":").add(1)

    ratio_parts = raw_ratio.str.split(":", expand=True)
    if ratio_parts.shape[1] < 4:
        for idx in range(ratio_parts.shape[1], 4):
            ratio_parts[idx] = pd.NA

    first_four = ratio_parts.iloc[:, :4].apply(pd.to_numeric, errors="coerce")

    external["ratio_sum"] = first_four.sum(axis=1, min_count=4)
    external["ratio_complete"] = (
        external["ratio_component_count"].eq(4)
        & first_four.notna().all(axis=1)
    )
    external["ratio_sum_near_100"] = external["ratio_sum"].between(
        99.0,
        101.0,
        inclusive="both",
    )

    # ---------- Conservative training name -> SMILES lookup ----------
    train_pairs = (
        train[
            [
                train_lipid_col,
                "lipid_name_normalized",
                train_smiles_col,
            ]
        ]
        .dropna(subset=[train_smiles_col])
        .drop_duplicates()
    )

    # Count unique SMILES per normalized name.
    name_counts = (
        train_pairs.groupby("lipid_name_normalized")[train_smiles_col]
        .nunique()
        .rename("n_unique_smiles")
        .reset_index()
    )

    conflicts = train_pairs.merge(
        name_counts[
            name_counts["n_unique_smiles"] > 1
        ],
        on="lipid_name_normalized",
        how="inner",
    ).sort_values(
        ["lipid_name_normalized", train_smiles_col]
    )

    conflicts.to_csv(CONFLICT_OUT, index=False)

    safe_names = name_counts[
        name_counts["n_unique_smiles"] == 1
    ][["lipid_name_normalized"]]

    safe_lookup = (
        train_pairs.merge(
            safe_names,
            on="lipid_name_normalized",
            how="inner",
        )[
            [
                "lipid_name_normalized",
                train_smiles_col,
            ]
        ]
        .drop_duplicates(subset=["lipid_name_normalized"])
        .rename(
            columns={
                train_smiles_col: "smiles_from_unique_name_match",
            }
        )
    )

    external_audit = external.merge(
        safe_lookup,
        on="lipid_name_normalized",
        how="left",
    )

    # Prefer an externally supplied SMILES if available.
    if ext_smiles_col is not None:
        external_audit["smiles_from_external"] = external_audit[
            ext_smiles_col
        ]
    else:
        external_audit["smiles_from_external"] = pd.NA

    external_audit["ionizable_lipid_smiles_resolved"] = (
        external_audit["smiles_from_external"]
        .where(
            external_audit["smiles_from_external"].notna()
            & external_audit["smiles_from_external"].astype("string").str.strip().ne(""),
            external_audit["smiles_from_unique_name_match"],
        )
    )

    conflict_name_set = set(
        conflicts["lipid_name_normalized"].dropna().unique()
    )

    def match_method(row) -> str:
        ext_smiles = row["smiles_from_external"]
        name_smiles = row["smiles_from_unique_name_match"]
        norm_name = row["lipid_name_normalized"]

        if pd.notna(ext_smiles) and str(ext_smiles).strip():
            return "external_smiles"
        if pd.notna(name_smiles) and str(name_smiles).strip():
            return "unique_name_lookup"
        if norm_name in conflict_name_set:
            return "unmatched_conflicting_training_name"
        return "unmatched_name"

    external_audit["smiles_match_method"] = external_audit.apply(
        match_method,
        axis=1,
    )

    # ---------- DOI overlap ----------
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

    external_audit.to_csv(AUDIT_OUT, index=False)

    matched_mask = external_audit[
        "ionizable_lipid_smiles_resolved"
    ].notna()

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
            "lipid_name_normalized",
            "smiles_match_method",
        ]
        if col in external_audit.columns
    ]

    missing = (
        external_audit.loc[
            ~matched_mask,
            useful_missing_cols,
        ]
        .drop_duplicates()
    )

    sort_cols = [
        col
        for col in ["paper", ext_lipid_col]
        if col in missing.columns
    ]
    if sort_cols:
        missing = missing.sort_values(sort_cols)

    missing.to_csv(MISSING_OUT, index=False)

    matched_lipids = (
        external_audit.loc[
            matched_mask,
            [
                ext_lipid_col,
                "lipid_name_normalized",
                "smiles_match_method",
            ],
        ]
        .drop_duplicates()
        .sort_values(ext_lipid_col)
    )

    method_counts = external_audit[
        "smiles_match_method"
    ].value_counts(dropna=False)

    report_lines = [
        "EXTERNAL VALIDATION FEATURE-COVERAGE AUDIT — v2",
        "===============================================",
        "",
        f"External formulations: {len(external)}",
        f"External papers: {external[ext_doi_col].nunique(dropna=True)}",
        (
            "Unique external ionizable lipids: "
            f"{external[ext_lipid_col].nunique(dropna=True)}"
        ),
        (
            "Rows with exactly four ratio components: "
            f"{int(external['ratio_component_count'].eq(4).sum())}"
        ),
        (
            "Rows with complete numeric four-component ratios: "
            f"{int(external['ratio_complete'].sum())}"
        ),
        (
            "Rows with ratios summing approximately to 100: "
            f"{int(external['ratio_sum_near_100'].sum())}"
        ),
        (
            "Rows with resolved ionizable-lipid SMILES: "
            f"{int(matched_mask.sum())}"
        ),
        (
            "Unique external lipid identities with resolved SMILES: "
            f"{external_audit.loc[matched_mask, ext_lipid_col].nunique(dropna=True)}"
        ),
        (
            "Unique external lipids still missing SMILES: "
            f"{external_audit.loc[~matched_mask, ext_lipid_col].nunique(dropna=True)}"
        ),
        (
            "Conflicting normalized training lipid names excluded from "
            f"name matching: {conflicts['lipid_name_normalized'].nunique(dropna=True)}"
        ),
        f"Training/external DOI overlap: {overlap}",
        "",
        "SMILES match methods:",
        method_counts.to_string(),
        "",
        "Currently matched lipids:",
        (
            matched_lipids.to_string(index=False)
            if len(matched_lipids)
            else "None"
        ),
        "",
        "Important:",
        'Generic/conflicting names such as "Custom lipid" are not assigned a',
        "SMILES by name. They remain unresolved until a structure is recovered",
        "from the originating paper or supplement.",
        "",
        "Saved:",
        f"  {AUDIT_OUT}",
        f"  {MISSING_OUT}",
        f"  {CONFLICT_OUT}",
    ]

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")

    print(report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
