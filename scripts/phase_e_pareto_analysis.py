from pathlib import Path
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
    "category_primary",
]


def pareto_mask_minimize(values: np.ndarray) -> np.ndarray:
    """Return True for non-dominated rows when every column is minimized.

    Row A dominates row B when A is no worse than B in every objective and
    strictly better in at least one objective.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if not np.isfinite(values).all():
        raise ValueError("Pareto objective matrix contains non-finite values")

    n_rows = values.shape[0]
    keep = np.ones(n_rows, dtype=bool)

    for i in range(n_rows):
        dominates_i = (
            np.all(values <= values[i], axis=1)
            & np.any(values < values[i], axis=1)
        )
        dominates_i[i] = False
        if dominates_i.any():
            keep[i] = False

    return keep


def save_scatter(
    data: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    output_name: str,
    pareto_col: str,
    threshold: float | None = None,
    acceptable_band: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.scatter(
        data["transfection_pctile"],
        data[y_col],
        alpha=0.65,
        label="All complete formulations",
    )

    selected = data[pareto_col]
    ax.scatter(
        data.loc[selected, "transfection_pctile"],
        data.loc[selected, y_col],
        marker="X",
        s=100,
        label="Pairwise Pareto-optimal",
    )

    if threshold is not None:
        ax.axhline(threshold, linestyle="--", linewidth=1.2)

    if acceptable_band is not None:
        ax.axhspan(
            acceptable_band[0],
            acceptable_band[1],
            alpha=0.12,
        )

    ax.set_xlabel("Within-study transfection percentile")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / output_name, dpi=300)
    plt.close(fig)


if not INPUT.exists():
    raise SystemExit(
        f"STOP: {INPUT} was not found. Run Phase E core analysis first."
    )

df = pd.read_csv(INPUT)
missing = [column for column in REQUIRED if column not in df.columns]
if missing:
    raise SystemExit(f"STOP: required columns are missing: {missing}")

if df[REQUIRED].isna().any().any():
    missing_counts = df[REQUIRED].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    raise SystemExit(
        "STOP: the Phase E complete-case file contains missing values:\n"
        + missing_counts.to_string()
    )

# Particle size is not a simple maximize/minimize objective. The prespecified
# acceptable interval is 60-150 nm, so the Pareto objective is the distance
# outside that interval. Every size inside the interval receives deviation 0.
df["size_deviation_from_60_150_nm"] = np.where(
    df["size_mean"] < 60,
    60 - df["size_mean"],
    np.where(df["size_mean"] > 150, df["size_mean"] - 150, 0.0),
)

# Convert all objectives to minimization form for non-dominated sorting:
#   - maximize transfection -> minimize negative transfection
#   - minimize PDI          -> minimize PDI
#   - maximize EE           -> minimize negative EE
#   - target size interval  -> minimize distance outside 60-150 nm
four_objectives = np.column_stack(
    [
        -df["transfection_pctile"].to_numpy(),
        df["pdi_mean"].to_numpy(),
        -df["ee_mean"].to_numpy(),
        df["size_deviation_from_60_150_nm"].to_numpy(),
    ]
)

df["pareto_optimal_4d"] = pareto_mask_minimize(four_objectives)

# Pairwise frontiers are used for transparent 2D figures.
df["pareto_transfection_pdi"] = pareto_mask_minimize(
    np.column_stack(
        [
            -df["transfection_pctile"].to_numpy(),
            df["pdi_mean"].to_numpy(),
        ]
    )
)

df["pareto_transfection_ee"] = pareto_mask_minimize(
    np.column_stack(
        [
            -df["transfection_pctile"].to_numpy(),
            -df["ee_mean"].to_numpy(),
        ]
    )
)

df["pareto_transfection_size"] = pareto_mask_minimize(
    np.column_stack(
        [
            -df["transfection_pctile"].to_numpy(),
            df["size_deviation_from_60_150_nm"].to_numpy(),
        ]
    )
)

all_output = RESULTS / "pareto_all_complete_formulations.csv"
pareto_output = RESULTS / "pareto_optimal_4d_formulations.csv"
summary_output = RESULTS / "pareto_analysis_report.txt"
category_output = RESULTS / "pareto_4d_category_summary.csv"

pareto_df = df.loc[df["pareto_optimal_4d"]].copy()
pareto_df = pareto_df.sort_values(
    ["transfection_pctile", "pdi_mean", "ee_mean"],
    ascending=[False, True, False],
)

df.to_csv(all_output, index=False)
pareto_df.to_csv(pareto_output, index=False)

category_summary = (
    pareto_df["category_primary"]
    .value_counts()
    .rename_axis("category_primary")
    .reset_index(name="n_pareto_4d")
)
category_summary.to_csv(category_output, index=False)

save_scatter(
    df,
    y_col="pdi_mean",
    y_label="Polydispersity index (PDI)",
    title="Transfection versus particle dispersity",
    output_name="pareto_transfection_vs_pdi.png",
    pareto_col="pareto_transfection_pdi",
    threshold=0.30,
)

save_scatter(
    df,
    y_col="ee_mean",
    y_label="Encapsulation efficiency (%)",
    title="Transfection versus encapsulation efficiency",
    output_name="pareto_transfection_vs_ee.png",
    pareto_col="pareto_transfection_ee",
    threshold=80,
)

save_scatter(
    df,
    y_col="size_mean",
    y_label="Particle size (nm)",
    title="Transfection versus particle size",
    output_name="pareto_transfection_vs_size.png",
    pareto_col="pareto_transfection_size",
    acceptable_band=(60, 150),
)

n_total = len(df)
n_pareto = int(df["pareto_optimal_4d"].sum())
n_pareto_studies = int(pareto_df["paper_doi"].nunique())
n_potent_developable = int(
    (pareto_df["category_primary"] == "Potent + Developable").sum()
)

report = f"""PHASE E PARETO ANALYSIS
=======================
Complete four-endpoint formulations analyzed: {n_total}
Studies represented in complete dataset: {df['paper_doi'].nunique()}

Four-objective definition:
  maximize within-study transfection percentile
  minimize PDI
  maximize encapsulation efficiency
  minimize distance outside the 60-150 nm size interval

Four-dimensional Pareto-optimal formulations: {n_pareto}
Percentage of complete formulations: {100 * n_pareto / n_total:.1f}%
Studies represented among Pareto-optimal formulations: {n_pareto_studies}
Pareto-optimal formulations also classified Potent + Developable: {n_potent_developable}

Pairwise Pareto counts:
  transfection versus PDI: {int(df['pareto_transfection_pdi'].sum())}
  transfection versus encapsulation efficiency: {int(df['pareto_transfection_ee'].sum())}
  transfection versus size target deviation: {int(df['pareto_transfection_size'].sum())}

Important interpretation:
Pareto-optimal does not mean universally best. It means that, within this
complete-case dataset, no other formulation was at least as good across all
four objectives and strictly better in at least one objective.

Size handling:
Particle size was treated as a target interval rather than as a quantity to
maximize or minimize. Sizes within 60-150 nm received zero target deviation;
sizes outside that interval were penalized by their distance from the nearest
boundary.
""".strip()

summary_output.write_text(report + "\n")

print(report)
print("\nPARETO-OPTIMAL FORMULATIONS:")
print(
    pareto_df[
        [
            "lnp_id",
            "paper_doi",
            "transfection_pctile",
            "size_mean",
            "pdi_mean",
            "ee_mean",
            "category_primary",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
for path in [all_output, pareto_output, category_output, summary_output]:
    print(f"  {path}")
print("  figures/pareto_transfection_vs_pdi.png")
print("  figures/pareto_transfection_vs_ee.png")
print("  figures/pareto_transfection_vs_size.png")
