#!/usr/bin/env python3
"""
Phase C Finalization: Build Publication-Aware Feature Matrix
============================================================

Input:
    results/phase_c_ratio_model_ready.csv

Outputs:
    results/phase_c_feature_matrix.csv
    results/phase_c_descriptor_lookup.csv
    results/phase_c_descriptor_failures.csv
    results/phase_c_feature_matrix_report.txt

What this does:
    1. Uses only the conservative ratio-model-eligible rows produced by the
       post-parse audit.
    2. Computes ionizable-lipid RDKit descriptors from SMILES.
    3. Merges descriptors back to each formulation.
    4. Adds three non-redundant composition log-ratio features using sterol
       as the reference component.
    5. Reports usable sample sizes for each endpoint and for complete-case
       multi-objective modeling.

What this does NOT do:
    - It does not alter original CSV files.
    - It does not train models.
    - It does not impute missing endpoints.
    - It does not use the 29 review-needed or 52 ambiguous-ratio rows.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

INPUT = Path("results/phase_c_ratio_model_ready.csv")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

FEATURE_OUT = RESULTS / "phase_c_feature_matrix.csv"
LOOKUP_OUT = RESULTS / "phase_c_descriptor_lookup.csv"
FAILURES_OUT = RESULTS / "phase_c_descriptor_failures.csv"
REPORT_OUT = RESULTS / "phase_c_feature_matrix_report.txt"

ESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
TERT_AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H0;!$([NX3][CX3]=[OX1])]")


def longest_carbon_chain(mol):
    """Heuristic longest carbon-only graph path."""
    carbon_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    if not carbon_idx:
        return np.nan

    adj = {i: [] for i in carbon_idx}
    cset = set(carbon_idx)

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in cset and b in cset:
            adj[a].append(b)
            adj[b].append(a)

    def bfs_farthest(start):
        visited = {start: 0}
        frontier = [start]
        while frontier:
            nxt = []
            for node in frontier:
                for nb in adj[node]:
                    if nb not in visited:
                        visited[nb] = visited[node] + 1
                        nxt.append(nb)
            frontier = nxt
        far_node = max(visited, key=visited.get)
        return far_node, visited[far_node]

    a, _ = bfs_farthest(carbon_idx[0])
    _, dist = bfs_farthest(a)
    return dist + 1


def compute_descriptors(smiles):
    keys = [
        "mw",
        "logp",
        "tpsa",
        "num_rotatable_bonds",
        "num_hbd",
        "num_hba",
        "num_ester_bonds",
        "num_tertiary_amines",
        "tail_length_carbons",
        "num_unsaturated_bonds",
        "num_branch_points",
        "num_rings",
        "fraction_csp3",
        "heavy_atom_count",
    ]

    result = {k: np.nan for k in keys}

    if pd.isna(smiles) or not str(smiles).strip():
        return result, "missing_smiles"

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return result, "unparseable_smiles"

    try:
        result.update(
            {
                "mw": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "num_hbd": rdMolDescriptors.CalcNumHBD(mol),
                "num_hba": rdMolDescriptors.CalcNumHBA(mol),
                "num_ester_bonds": len(mol.GetSubstructMatches(ESTER_SMARTS)),
                "num_tertiary_amines": len(
                    mol.GetSubstructMatches(TERT_AMINE_SMARTS)
                ),
                "tail_length_carbons": longest_carbon_chain(mol),
                "num_unsaturated_bonds": sum(
                    1
                    for b in mol.GetBonds()
                    if b.GetBondTypeAsDouble() == 2.0 and not b.GetIsAromatic()
                ),
                "num_branch_points": sum(
                    1
                    for a in mol.GetAtoms()
                    if a.GetSymbol() == "C" and a.GetDegree() >= 3
                ),
                "num_rings": rdMolDescriptors.CalcNumRings(mol),
                "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
                "heavy_atom_count": mol.GetNumHeavyAtoms(),
            }
        )
        return result, None
    except Exception as exc:
        return {k: np.nan for k in keys}, f"descriptor_error: {exc}"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "lnp_id",
        "paper_doi",
        "ionizable_lipid_smiles",
        "mol_pct_ionizable",
        "mol_pct_helper",
        "mol_pct_sterol",
        "mol_pct_peg",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Defensive validation: these rows should already have passed the post-parse audit.
    ratio_cols = [
        "mol_pct_ionizable",
        "mol_pct_helper",
        "mol_pct_sterol",
        "mol_pct_peg",
    ]
    ratio_sum = df[ratio_cols].sum(axis=1, min_count=4)

    invalid_ratio_rows = (
        df[ratio_cols].isna().any(axis=1)
        | (df[ratio_cols] <= 0).any(axis=1)
        | ~ratio_sum.between(99.0, 101.0, inclusive="both")
    )

    if invalid_ratio_rows.any():
        bad_ids = df.loc[invalid_ratio_rows, "lnp_id"].head(10).tolist()
        raise ValueError(
            "Input contains rows that are not conservative model-ready ratios. "
            f"Example lnp_id values: {bad_ids}"
        )

    # Compute descriptors once per unique ionizable-lipid SMILES.
    unique_smiles = df["ionizable_lipid_smiles"].dropna().drop_duplicates()
    descriptor_rows = []
    failures = []

    for smiles in unique_smiles:
        desc, reason = compute_descriptors(smiles)
        desc["ionizable_lipid_smiles"] = smiles
        descriptor_rows.append(desc)

        if reason:
            failures.append(
                {
                    "ionizable_lipid_smiles": smiles,
                    "reason": reason,
                }
            )

    lookup = pd.DataFrame(descriptor_rows)
    merged = df.merge(lookup, on="ionizable_lipid_smiles", how="left")

    # Non-redundant composition representation.
    # All four percentages sum to ~100, so using all four as independent
    # linear-model predictors creates perfect compositional redundancy.
    # These log ratios preserve relative composition using sterol as reference.
    merged["log_il_to_sterol"] = np.log(
        merged["mol_pct_ionizable"] / merged["mol_pct_sterol"]
    )
    merged["log_helper_to_sterol"] = np.log(
        merged["mol_pct_helper"] / merged["mol_pct_sterol"]
    )
    merged["log_peg_to_sterol"] = np.log(
        merged["mol_pct_peg"] / merged["mol_pct_sterol"]
    )

    descriptor_cols = [
        "mw",
        "logp",
        "tpsa",
        "num_rotatable_bonds",
        "num_hbd",
        "num_hba",
        "num_ester_bonds",
        "num_tertiary_amines",
        "tail_length_carbons",
        "num_unsaturated_bonds",
        "num_branch_points",
        "num_rings",
        "fraction_csp3",
        "heavy_atom_count",
    ]

    merged["descriptor_complete"] = merged[descriptor_cols].notna().all(axis=1)

    endpoints = {
        "transfection": "transfection_pctile",
        "particle_size": "size_mean",
        "pdi": "pdi_mean",
        "encapsulation_efficiency": "ee_mean",
    }

    for label, col in endpoints.items():
        if col in merged.columns:
            merged[f"eligible_{label}_model"] = (
                merged[col].notna() & merged["descriptor_complete"]
            )

    endpoint_cols = [c for c in endpoints.values() if c in merged.columns]
    merged["eligible_complete_four_endpoint_model"] = (
        merged[endpoint_cols].notna().all(axis=1)
        & merged["descriptor_complete"]
    )

    lookup.to_csv(LOOKUP_OUT, index=False)
    pd.DataFrame(failures).to_csv(FAILURES_OUT, index=False)
    merged.to_csv(FEATURE_OUT, index=False)

    n_rows = len(merged)
    n_studies = merged["paper_doi"].nunique(dropna=True)
    n_unique_smiles = len(unique_smiles)
    n_descriptor_complete = int(merged["descriptor_complete"].sum())

    report_lines = [
        "PHASE C FEATURE MATRIX REPORT",
        "=============================",
        "",
        f"Conservative ratio-model rows: {n_rows}",
        f"Studies represented: {n_studies}",
        f"Unique non-missing ionizable-lipid SMILES: {n_unique_smiles}",
        f"Unique SMILES descriptor failures: {len(failures)}",
        (
            "Formulations with complete ionizable-lipid descriptors: "
            f"{n_descriptor_complete}/{n_rows} "
            f"({100*n_descriptor_complete/n_rows:.1f}%)"
        ),
        "",
        "Endpoint modeling sample sizes with complete descriptors:",
    ]

    for label, col in endpoints.items():
        flag = f"eligible_{label}_model"
        if flag in merged.columns:
            report_lines.append(
                f"  {label}: {int(merged[flag].sum())}"
            )

    report_lines.extend(
        [
            (
                "  complete four-endpoint subset: "
                f"{int(merged['eligible_complete_four_endpoint_model'].sum())}"
            ),
            "",
            "Composition features:",
            "  Raw percentages retained: ionizable, helper, sterol, PEG",
            "  Non-redundant log-ratio features added using sterol as reference:",
            "    log_il_to_sterol",
            "    log_helper_to_sterol",
            "    log_peg_to_sterol",
            "",
            "Modeling cautions:",
            "  - Keep paper_doi as the grouping variable for cross-validation.",
            "  - Do not randomly split formulations across train/test folds.",
            "  - Do not impute missing outcome values.",
            "  - Compare descriptor-based models with identity-only baselines.",
            "  - For linear models, do not include all four raw percentages together.",
            "  - Tail length and branch-point counts are heuristic RDKit-derived features.",
            "",
            "Saved:",
            f"  {FEATURE_OUT}",
            f"  {LOOKUP_OUT}",
            f"  {FAILURES_OUT}",
        ]
    )

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")
    print(report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
