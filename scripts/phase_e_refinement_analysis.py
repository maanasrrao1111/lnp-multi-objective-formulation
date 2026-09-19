from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/phase_e_primary_classification.csv")
RESULTS = Path("results")
FIGURES = Path("figures")

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

REQUIRED = [
    "lnp_id",
    "paper_doi",
    "paper_title",
    "transfection_pctile",
    "size_mean",
    "pdi_mean",
    "ee_mean",
    "potent_primary",
    "size_pass_primary",
    "pdi_pass_primary",
    "ee_pass_primary",
    "developable_primary",
    "category_primary",
]


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    """Return non-dominated rows when every objective is minimized."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if not np.isfinite(values).all():
        raise ValueError("Pareto objective matrix contains non-finite values")

    keep = np.ones(values.shape[0], dtype=bool)
    for i in range(values.shape[0]):
        dominated = (
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return keep


def size_deviation(series: pd.Series, lower: float, upper: float) -> np.ndarray:
    return np.where(
        series < lower,
        lower - series,
        np.where(series > upper, series - upper, 0.0),
    )


def four_objective_frontier(data: pd.DataFrame, lower: float, upper: float) -> pd.Series:
    deviation = size_deviation(data["size_mean"], lower, upper)
    objectives = np.column_stack(
        [
            -data["transfection_pctile"].to_numpy(),
            data["pdi_mean"].to_numpy(),
            -data["ee_mean"].to_numpy(),
            deviation,
        ]
    )
    return pd.Series(pareto_mask_minimize(objectives), index=data.index)


def in_band_three_objective_frontier(
    data: pd.DataFrame, lower: float, upper: float
) -> pd.Series:
    eligible = data["size_mean"].between(lower, upper)
    mask = pd.Series(False, index=data.index)
    subset = data.loc[eligible]
    objectives = np.column_stack(
        [
            -subset["transfection_pctile"].to_numpy(),
            subset["pdi_mean"].to_numpy(),
            -subset["ee_mean"].to_numpy(),
        ]
    )
    mask.loc[subset.index] = pareto_mask_minimize(objectives)
    return mask


if not INPUT.exists():
    raise SystemExit(f"STOP: {INPUT} was not found. Run Phase E core analysis first.")

df = pd.read_csv(INPUT)
missing = [column for column in REQUIRED if column not in df.columns]
if missing:
    raise SystemExit(f"STOP: required columns are missing: {missing}")

if df[REQUIRED].isna().any().any():
    missing_counts = df[REQUIRED].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    raise SystemExit(
        "STOP: complete-case file contains missing required values:\n"
        + missing_counts.to_string()
    )

# -------------------------------------------------------------------------
# 1. Failure-mode decomposition among potent formulations
# -------------------------------------------------------------------------
potent = df.loc[df["potent_primary"]].copy()
potent["fail_size"] = ~potent["size_pass_primary"]
potent["fail_pdi"] = ~potent["pdi_pass_primary"]
potent["fail_ee"] = ~potent["ee_pass_primary"]
potent["fail_any_physchem"] = ~potent["developable_primary"]


def failure_label(row: pd.Series) -> str:
    failures = []
    if row["fail_size"]:
        failures.append("Size")
    if row["fail_pdi"]:
        failures.append("PDI")
    if row["fail_ee"]:
        failures.append("EE")
    return " + ".join(failures) if failures else "None"


potent["failure_mode"] = potent.apply(failure_label, axis=1)
potent.to_csv(RESULTS / "phase_e_potent_failure_annotations.csv", index=False)

failure_order = [
    "EE",
    "Size",
    "PDI",
    "PDI + EE",
    "Size + EE",
    "Size + PDI",
    "Size + PDI + EE",
    "None",
]
failure_counts = (
    potent["failure_mode"]
    .value_counts()
    .reindex(failure_order, fill_value=0)
    .rename_axis("failure_mode")
    .reset_index(name="n_potent_formulations")
)
failure_counts["pct_of_potent_formulations"] = (
    100 * failure_counts["n_potent_formulations"] / len(potent)
)
failure_counts.to_csv(RESULTS / "phase_e_failure_mode_summary.csv", index=False)

largest_study = (
    df.groupby("paper_doi", dropna=False)
    .size()
    .sort_values(ascending=False)
    .index[0]
)
potent_without_largest = potent.loc[potent["paper_doi"] != largest_study]

# -------------------------------------------------------------------------
# 2. Sensitivity to the particle-size interval used in classification
# -------------------------------------------------------------------------
size_bands = [(50, 150), (60, 150), (60, 160), (60, 200), (50, 200)]
size_rows = []
for lower, upper in size_bands:
    developable = (
        df["size_mean"].between(lower, upper)
        & (df["pdi_mean"] < 0.30)
        & (df["ee_mean"] > 80)
    )
    potent_mask = df["transfection_pctile"] > 0.50
    n_potent = int(potent_mask.sum())
    n_joint = int((potent_mask & developable).sum())
    n_fail = int((potent_mask & ~developable).sum())
    size_rows.append(
        {
            "size_lower_nm": lower,
            "size_upper_nm": upper,
            "n_potent": n_potent,
            "n_potent_and_developable": n_joint,
            "n_potent_failing_physchem": n_fail,
            "pct_potent_failing_physchem": 100 * n_fail / n_potent,
        }
    )
size_sensitivity = pd.DataFrame(size_rows)
size_sensitivity.to_csv(
    RESULTS / "phase_e_size_interval_sensitivity.csv", index=False
)

# -------------------------------------------------------------------------
# 3. Pareto refinement and size-handling sensitivity
# -------------------------------------------------------------------------
scenario_specs = [
    ("4D_penalty_50_150", "4d", 50, 150),
    ("4D_penalty_60_150_primary", "4d", 60, 150),
    ("4D_penalty_60_200", "4d", 60, 200),
    ("3D_in_band_50_150", "3d", 50, 150),
    ("3D_in_band_60_150_primary", "3d", 60, 150),
    ("3D_in_band_60_200", "3d", 60, 200),
]

scenario_masks = {}
scenario_rows = []
for name, method, lower, upper in scenario_specs:
    if method == "4d":
        mask = four_objective_frontier(df, lower, upper)
    else:
        mask = in_band_three_objective_frontier(df, lower, upper)
    scenario_masks[name] = mask
    ids = df.loc[mask, "lnp_id"].astype(str).tolist()
    scenario_rows.append(
        {
            "scenario": name,
            "method": method,
            "size_lower_nm": lower,
            "size_upper_nm": upper,
            "n_candidates": int(mask.sum()),
            "candidate_lnp_ids": ";".join(ids),
        }
    )

pareto_sensitivity = pd.DataFrame(scenario_rows)
pareto_sensitivity.to_csv(
    RESULTS / "pareto_size_handling_sensitivity.csv", index=False
)

stability = df[
    [
        "lnp_id",
        "paper_doi",
        "paper_title",
        "transfection_pctile",
        "size_mean",
        "pdi_mean",
        "ee_mean",
        "category_primary",
    ]
].copy()
for name, mask in scenario_masks.items():
    stability[name] = mask.astype(bool)
scenario_columns = [name for name, _, _, _ in scenario_specs]
stability["n_scenarios_selected"] = stability[scenario_columns].sum(axis=1)
stability["selected_in_primary_4d"] = stability[
    "4D_penalty_60_150_primary"
]
stability["selected_in_primary_in_band_3d"] = stability[
    "3D_in_band_60_150_primary"
]
stability = stability.sort_values(
    ["n_scenarios_selected", "transfection_pctile", "pdi_mean", "ee_mean"],
    ascending=[False, False, True, False],
)
stability.to_csv(RESULTS / "pareto_candidate_stability.csv", index=False)

primary_4d_mask = scenario_masks["4D_penalty_60_150_primary"]
primary_3d_mask = scenario_masks["3D_in_band_60_150_primary"]
primary_4d = df.loc[primary_4d_mask].copy()
primary_3d = df.loc[primary_3d_mask].copy()
primary_3d.to_csv(RESULTS / "pareto_primary_in_band_3d_candidates.csv", index=False)

# -------------------------------------------------------------------------
# 4. Revised figures that highlight the same primary 4D candidate set
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
category_order = [
    "Potent + Developable",
    "Potent only",
    "Developable only",
    "Neither",
]
category_counts = df["category_primary"].value_counts().reindex(category_order)
bars = ax.bar(category_order, category_counts.values)
ax.set_ylabel("Number of formulations")
ax.set_title(f"Potency and physicochemical developability (n={len(df)})")
ax.tick_params(axis="x", rotation=20)
for bar, n in zip(bars, category_counts.values):
    pct = 100 * n / len(df)
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.6,
        f"{int(n)}\n({pct:.1f}%)",
        ha="center",
    )
fig.tight_layout()
fig.savefig(FIGURES / "phase_e_primary_classification_revised.png", dpi=300)
plt.close(fig)

failure_plot = failure_counts.loc[
    failure_counts["n_potent_formulations"] > 0
].copy()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(failure_plot["failure_mode"], failure_plot["n_potent_formulations"])
ax.set_ylabel("Number of potent formulations")
ax.set_title("Physicochemical failure modes among potent formulations")
ax.tick_params(axis="x", rotation=25)
for bar, n in zip(bars, failure_plot["n_potent_formulations"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        str(int(n)),
        ha="center",
    )
fig.tight_layout()
fig.savefig(FIGURES / "phase_e_potent_failure_modes.png", dpi=300)
plt.close(fig)


def save_primary_pareto_scatter(
    y_col: str,
    y_label: str,
    title: str,
    output_name: str,
    threshold: float | None = None,
    acceptable_band: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        df["transfection_pctile"],
        df[y_col],
        alpha=0.6,
        label="All complete formulations",
    )
    ax.scatter(
        df.loc[primary_4d_mask, "transfection_pctile"],
        df.loc[primary_4d_mask, y_col],
        marker="X",
        s=100,
        label="Primary 4-objective Pareto candidates",
    )
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1.2)
    if acceptable_band is not None:
        ax.axhspan(acceptable_band[0], acceptable_band[1], alpha=0.12)
    ax.set_xlabel("Within-study transfection percentile")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, dpi=300)
    plt.close(fig)


save_primary_pareto_scatter(
    "pdi_mean",
    "Polydispersity index (PDI)",
    "Primary multi-objective candidates: transfection versus PDI",
    "pareto_primary4d_transfection_vs_pdi.png",
    threshold=0.30,
)
save_primary_pareto_scatter(
    "ee_mean",
    "Encapsulation efficiency (%)",
    "Primary multi-objective candidates: transfection versus encapsulation",
    "pareto_primary4d_transfection_vs_ee.png",
    threshold=80,
)
save_primary_pareto_scatter(
    "size_mean",
    "Particle size (nm)",
    "Primary multi-objective candidates: transfection versus size",
    "pareto_primary4d_transfection_vs_size.png",
    acceptable_band=(60, 150),
)

# -------------------------------------------------------------------------
# 5. Report
# -------------------------------------------------------------------------
n_potent = len(potent)
n_fail_any = int(potent["fail_any_physchem"].sum())
n_fail_ee = int(potent["fail_ee"].sum())
n_fail_size = int(potent["fail_size"].sum())
n_fail_pdi = int(potent["fail_pdi"].sum())
n_ee_only = int((potent["failure_mode"] == "EE").sum())

without_n = len(potent_without_largest)
without_fail = int(potent_without_largest["fail_any_physchem"].sum())
without_ee = int(potent_without_largest["fail_ee"].sum())

primary_4d_ids = ", ".join(primary_4d["lnp_id"].astype(str))
primary_3d_ids = ", ".join(primary_3d["lnp_id"].astype(str))

stable = stability.loc[stability["n_scenarios_selected"] >= 4]
stable_ids = ", ".join(stable["lnp_id"].astype(str))

report = f"""PHASE E REFINEMENT ANALYSIS
===========================

Complete four-endpoint formulations: {len(df)}
Studies represented: {df['paper_doi'].nunique()}

Failure-mode decomposition among potent formulations:
  Potent formulations: {n_potent}
  Failing at least one physicochemical criterion: {n_fail_any} ({100*n_fail_any/n_potent:.1f}%)
  Fail encapsulation-efficiency criterion: {n_fail_ee}
  Fail particle-size criterion: {n_fail_size}
  Fail PDI criterion: {n_fail_pdi}
  Fail encapsulation efficiency only: {n_ee_only}

After excluding the largest contributing study ({largest_study}):
  Potent formulations: {without_n}
  Failing at least one criterion: {without_fail} ({100*without_fail/without_n:.1f}%)
  Fail encapsulation-efficiency criterion: {without_ee}

Size-interval sensitivity of the pooled failure percentage:
  Minimum across tested intervals: {size_sensitivity['pct_potent_failing_physchem'].min():.1f}%
  Maximum across tested intervals: {size_sensitivity['pct_potent_failing_physchem'].max():.1f}%

Pareto refinement:
  Primary four-objective distance-penalty frontier (60-150 nm): {int(primary_4d_mask.sum())} candidates
  Candidate IDs: {primary_4d_ids}
  Primary hard-constrained 60-150 nm three-objective frontier: {int(primary_3d_mask.sum())} candidates
  Candidate IDs: {primary_3d_ids}
  Candidates selected in at least 4 of 6 size-handling scenarios: {len(stable)}
  Candidate IDs: {stable_ids}

Interpretation:
  The pooled potency-developability gap is not driven by the exact particle-size interval;
  the failure percentage remained within the range reported above. Encapsulation efficiency
  is the dominant failed criterion. The exact Pareto candidate count is more sensitive to
  how particle size is encoded, so candidate-set sensitivity should accompany the primary
  Pareto result rather than presenting one frontier as universally definitive.
""".strip()

(RESULTS / "phase_e_refinement_report.txt").write_text(report + "\n")

print(report)
print("\nSaved refined outputs and revised figures successfully.")
