#!/usr/bin/env python3
"""
Phase C Post-Parse Audit
========================

Reads:
    results/atlas_ratios_parsed.csv

Creates:
    results/phase_c_ratio_postparse_audit.csv
    results/phase_c_ratio_model_ready.csv
    results/phase_c_ratio_review_needed.csv
    results/phase_c_ratio_postparse_report.txt

Purpose:
    Separate rows that are safely usable as direct four-component percentage
    features from rows that were mapped by DOI but whose four values do not
    sum to approximately 100. The latter are retained for manual source-paper
    review and are not silently normalized or used as percentage features.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("results/atlas_ratios_parsed.csv")
AUDIT_OUT = Path("results/phase_c_ratio_postparse_audit.csv")
READY_OUT = Path("results/phase_c_ratio_model_ready.csv")
REVIEW_OUT = Path("results/phase_c_ratio_review_needed.csv")
REPORT_OUT = Path("results/phase_c_ratio_postparse_report.txt")

RATIO_COLS = [
    "mol_pct_ionizable",
    "mol_pct_helper",
    "mol_pct_sterol",
    "mol_pct_peg",
]

SUM_LOWER = 99.0
SUM_UPPER = 101.0


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    missing = [c for c in RATIO_COLS + ["parse_confidence", "parse_method"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["ratio_sum"] = df[RATIO_COLS].sum(axis=1, min_count=4)
    df["ratio_sum_deviation_from_100"] = (df["ratio_sum"] - 100.0).abs()

    df["ratio_values_complete"] = df[RATIO_COLS].notna().all(axis=1)
    df["ratio_values_nonnegative"] = (df[RATIO_COLS] >= 0).all(axis=1)
    df["ratio_values_at_most_100"] = (df[RATIO_COLS] <= 100).all(axis=1)

    df["ratio_sum_near_100"] = df["ratio_sum"].between(
        SUM_LOWER, SUM_UPPER, inclusive="both"
    )

    # Conservative eligibility:
    # - DOI mapping designated high-confidence by the parser
    # - all four values present
    # - no negative or >100 component
    # - total is approximately 100
    df["ratio_model_eligible"] = (
        df["parse_confidence"].eq("high")
        & df["ratio_values_complete"]
        & df["ratio_values_nonnegative"]
        & df["ratio_values_at_most_100"]
        & df["ratio_sum_near_100"]
    )

    df["ratio_review_needed"] = (
        df["parse_confidence"].eq("high")
        & ~df["ratio_model_eligible"]
    )

    audit_cols = [
        c for c in [
            "lnp_id",
            "paper_doi",
            "paper_title",
            "lipid_molar_ratio",
            "ionizable_lipid",
            "helper_lipid",
            "sterol_lipid",
            "peg_lipid",
            *RATIO_COLS,
            "ratio_sum",
            "ratio_sum_deviation_from_100",
            "parse_method",
            "parse_confidence",
            "ratio_values_complete",
            "ratio_values_nonnegative",
            "ratio_values_at_most_100",
            "ratio_sum_near_100",
            "ratio_model_eligible",
            "ratio_review_needed",
        ] if c in df.columns
    ]

    audit = df[audit_cols].copy()
    ready = df[df["ratio_model_eligible"]].copy()
    review = df[df["ratio_review_needed"]].copy()

    audit.to_csv(AUDIT_OUT, index=False)
    ready.to_csv(READY_OUT, index=False)
    review.to_csv(REVIEW_OUT, index=False)

    n_total = len(df)
    n_high = int(df["parse_confidence"].eq("high").sum())
    n_excluded = int(df["parse_confidence"].eq("excluded").sum())
    n_none = int(df["parse_confidence"].eq("none").sum())
    n_ready = int(df["ratio_model_eligible"].sum())
    n_review = int(df["ratio_review_needed"].sum())

    review_by_doi = (
        review.groupby("paper_doi", dropna=False)
        .size()
        .sort_values(ascending=False)
    )

    excluded_by_doi = (
        df[df["parse_confidence"].eq("excluded")]
        .groupby("paper_doi", dropna=False)
        .size()
        .sort_values(ascending=False)
    )

    report_lines = [
        "PHASE C RATIO POST-PARSE AUDIT",
        "==============================",
        "",
        f"Total formulations: {n_total}",
        f"Parser-designated high confidence: {n_high}",
        f"Parser-excluded ambiguous DOI rows: {n_excluded}",
        f"Parser failures/non-numeric rows: {n_none}",
        "",
        f"Conservative ratio-model eligible rows (sum {SUM_LOWER:g}-{SUM_UPPER:g}): {n_ready}",
        f"High-confidence rows requiring manual review: {n_review}",
        "",
        "IMPORTANT:",
        'The label "high confidence" comes from the DOI lookup in the parser.',
        "This audit does not independently prove every DOI assignment.",
        "Rows outside the approximately-100 total are not normalized or silently",
        "treated as mole percentages; they are retained for paper-level review.",
        "",
        "High-confidence review-needed rows by DOI:",
        review_by_doi.to_string() if len(review_by_doi) else "None",
        "",
        "Excluded ambiguous rows by DOI:",
        excluded_by_doi.to_string() if len(excluded_by_doi) else "None",
        "",
        "Saved:",
        f"  {AUDIT_OUT}",
        f"  {READY_OUT}",
        f"  {REVIEW_OUT}",
    ]

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")

    print(report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
