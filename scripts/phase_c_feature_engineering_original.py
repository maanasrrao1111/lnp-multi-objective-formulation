"""
Phase C: Feature Engineering
=============================
Input : data/atlas_cleaned.csv (464 x 41, from Phase B)
Output: results/atlas_features.csv

What this does, biologically/chemically:
  1. Splits the lipid_molar_ratio string ("60:20:19.5:0.5") into four numeric
     mole-% columns: ionizable / helper / sterol / PEG-lipid. This is the
     "recipe" of the formulation, and it's a first-order driver of both
     potency and physical properties (e.g. too much PEG-lipid shrinks
     particle size but can hurt endosomal escape; ionizable-lipid mole %
     tracks with N:P charge ratio and thus encapsulation efficiency).
  2. Computes RDKit molecular descriptors for each unique ionizable lipid
     SMILES. Ionizable lipid identity is the single strongest known driver
     of transfection potency in the LNP literature, so its structural
     properties (logP, amine count, tail length, etc.) are the features
     most likely to explain variance in Step 11 (SHAP) downstream.
  3. Merges compositional + molecular features into one feature matrix,
     keeping the original targets (size_mean, pdi_mean, ee_mean,
     transfection_pctile) alongside so Phase D can pull straight from
     this file without re-joining anything.

Assumptions / pitfalls flagged inline:
  - 70/464 rows have no ionizable_lipid_smiles (394/464 present per Phase B
    profiling) -> those rows get NaN molecular descriptors, NOT dropped.
    Decide at modeling time (Phase D) whether to drop or impute; don't
    silently drop here.
  - pKa is NOT computed. No robust open-source pKa predictor ships with
    RDKit; a real estimate needs a trained model (e.g. an ML pKa predictor)
    or literature values. Left as NaN with a TODO -- do not fabricate a
    number here.
  - "Tail length" and "branching points" are structural heuristics computed
    via graph traversal of sp3/sp2 carbon chains off the ionizable lipid's
    amine core, not database lookups -- treat as approximate, and sanity
    check a handful by hand before trusting SHAP rankings that lean on them.
"""

import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence RDKit's SMILES-parsing warnings; we
                                  # handle failures explicitly below instead

DATA_IN = "data/atlas_cleaned.csv"
DATA_OUT = "results/atlas_features.csv"
FAILED_SMILES_LOG = "results/phase_c_failed_smiles.csv"


# ---------------------------------------------------------------------------
# STEP 1: compositional features from lipid_molar_ratio
# ---------------------------------------------------------------------------
def parse_molar_ratio(ratio_str):
    """
    '60:20:19.5:0.5' -> (60.0, 20.0, 19.5, 0.5)
    Order in LNP Atlas is: ionizable, helper, sterol, PEG-lipid.
    Returns (nan, nan, nan, nan) if the string doesn't split into exactly
    4 numeric parts -- flagged, not guessed.
    """
    if pd.isna(ratio_str):
        return (np.nan, np.nan, np.nan, np.nan)
    parts = str(ratio_str).split(":")
    if len(parts) != 4:
        return (np.nan, np.nan, np.nan, np.nan)
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        return (np.nan, np.nan, np.nan, np.nan)


# ---------------------------------------------------------------------------
# STEP 2: RDKit molecular descriptors for the ionizable lipid
# ---------------------------------------------------------------------------
# SMARTS patterns used for substructure counts:
ESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")       # ester linkage
AMIDE_SMARTS = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")          # to exclude from amine count
TERT_AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H0;!$([NX3][CX3]=[OX1])]")  # tertiary N, not amide N


def longest_carbon_chain(mol):
    """
    Approximate 'tail length': longest simple path through carbon atoms
    only (ignores heteroatoms), via BFS from every carbon and taking the
    max eccentricity. This is a heuristic proxy for acyl/alkyl tail length,
    NOT a validated cheminformatics descriptor -- spot-check before relying
    on it heavily in downstream SHAP interpretation.
    """
    carbon_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]
    if not carbon_idx:
        return np.nan
    adj = {i: [] for i in carbon_idx}
    cset = set(carbon_idx)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
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

    # two-pass BFS to find the graph diameter (standard longest-path-in-tree trick;
    # LNP tails are branched but rarely cyclic, so this is a reasonable approximation)
    start = carbon_idx[0]
    a, _ = bfs_farthest(start)
    b, dist = bfs_farthest(a)
    return dist + 1  # +1 to count atoms, not bonds


def compute_descriptors(smiles):
    """Returns a dict of descriptors, or a dict of NaNs + records the failure."""
    keys = [
        "mw", "logp", "tpsa", "num_rotatable_bonds", "num_hbd", "num_hba",
        "num_ester_bonds", "num_tertiary_amines", "tail_length_carbons",
        "num_unsaturated_bonds", "num_branch_points", "num_rings",
    ]
    nan_result = {k: np.nan for k in keys}

    if pd.isna(smiles):
        return nan_result, "missing_smiles"

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return nan_result, "unparseable_smiles"

    try:
        result = {
            "mw": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_hbd": rdMolDescriptors.CalcNumHBD(mol),
            "num_hba": rdMolDescriptors.CalcNumHBA(mol),
            "num_ester_bonds": len(mol.GetSubstructMatches(ESTER_SMARTS)),
            "num_tertiary_amines": len(mol.GetSubstructMatches(TERT_AMINE_SMARTS)),
            "tail_length_carbons": longest_carbon_chain(mol),
            "num_unsaturated_bonds": sum(
                1 for b in mol.GetBonds()
                if b.GetBondTypeAsDouble() == 2.0 and not b.GetIsAromatic()
            ),
            "num_branch_points": sum(
                1 for a in mol.GetAtoms()
                if a.GetSymbol() == "C" and a.GetDegree() >= 3
            ),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
        }
        return result, None
    except Exception as e:
        return nan_result, f"descriptor_error: {e}"


def main():
    print(f"Loading {DATA_IN} ...")
    df = pd.read_csv(DATA_IN)
    print(f"  {len(df)} rows loaded")

    # --- Step 1: compositional features ---
    ratios = df["lipid_molar_ratio"].apply(parse_molar_ratio)
    df["mol_pct_ionizable"] = [r[0] for r in ratios]
    df["mol_pct_helper"] = [r[1] for r in ratios]
    df["mol_pct_sterol"] = [r[2] for r in ratios]
    df["mol_pct_peg"] = [r[3] for r in ratios]
    n_ratio_parsed = df["mol_pct_ionizable"].notna().sum()
    print(f"  Parsed molar ratio for {n_ratio_parsed}/{len(df)} rows")

    # --- Step 2: RDKit descriptors, computed ONCE per unique SMILES then joined ---
    unique_smiles = df["ionizable_lipid_smiles"].dropna().unique()
    print(f"  Computing descriptors for {len(unique_smiles)} unique ionizable lipid SMILES ...")

    desc_rows = []
    failures = []
    for smi in unique_smiles:
        desc, fail_reason = compute_descriptors(smi)
        desc["ionizable_lipid_smiles"] = smi
        desc_rows.append(desc)
        if fail_reason:
            failures.append({"smiles": smi, "reason": fail_reason})

    desc_df = pd.DataFrame(desc_rows)

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_df.to_csv(FAILED_SMILES_LOG, index=False)
        print(f"  WARNING: {len(failures)} SMILES failed descriptor computation "
              f"-> logged to {FAILED_SMILES_LOG}")

    # --- Step 3: merge ---
    merged = df.merge(desc_df, on="ionizable_lipid_smiles", how="left")

    n_with_full_desc = merged["mw"].notna().sum()
    print(f"  {n_with_full_desc}/{len(merged)} rows have full molecular descriptors")
    print(f"  ({len(merged) - n_with_full_desc} rows lack descriptors -- missing or "
          f"unparseable SMILES; left as NaN, not dropped or imputed)")

    merged.to_csv(DATA_OUT, index=False)
    print(f"Saved -> {DATA_OUT}  ({merged.shape[0]} x {merged.shape[1]})")

    # --- quick completeness summary for the log ---
    print("\nFeature completeness summary:")
    check_cols = ["mol_pct_ionizable", "mw", "logp", "tpsa", "tail_length_carbons"]
    for c in check_cols:
        pct = 100 * merged[c].notna().sum() / len(merged)
        print(f"  {c:24s} {pct:5.1f}% complete")


if __name__ == "__main__":
    main()
