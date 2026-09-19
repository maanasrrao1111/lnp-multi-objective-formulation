#!/usr/bin/env python3
"""
Phase F1: Strict Leave-One-Publication-Out Validation for Particle Size
======================================================================

Input
-----
results/phase_c_feature_matrix.csv

Outputs
-------
results/phase_f_particle_size_lopo_predictions.csv
results/phase_f_particle_size_lopo_per_study.csv
results/phase_f_particle_size_lopo_summary.csv
results/phase_f_particle_size_lopo_bootstrap.csv
results/phase_f_particle_size_lopo_report.txt
figures/phase_f_particle_size_lopo_observed_vs_predicted.png
figures/phase_f_particle_size_lopo_study_improvement.png

Purpose
-------
Test whether particle-size prediction generalizes when each publication is held out
completely. This is stricter than the grouped 5-fold analysis from Phase D.

Models compared
---------------
1. Mean baseline
2. Random forest using composition features only
3. Random forest using molecular descriptors only
4. Random forest using combined composition + descriptors

Important
---------
- Entire publications are held out one at a time.
- No random row-level split is used.
- Missing outcome values are not imputed.
- Hyperparameters are fixed from Phase D; there is no tuning on held-out studies.
- Study-level and pooled metrics are both reported because publication sizes differ.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT = Path("results/phase_c_feature_matrix.csv")
RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

PRED_OUT = RESULTS / "phase_f_particle_size_lopo_predictions.csv"
STUDY_OUT = RESULTS / "phase_f_particle_size_lopo_per_study.csv"
SUMMARY_OUT = RESULTS / "phase_f_particle_size_lopo_summary.csv"
BOOT_OUT = RESULTS / "phase_f_particle_size_lopo_bootstrap.csv"
REPORT_OUT = RESULTS / "phase_f_particle_size_lopo_report.txt"

OOF_FIG = FIGURES / "phase_f_particle_size_lopo_observed_vs_predicted.png"
STUDY_FIG = FIGURES / "phase_f_particle_size_lopo_study_improvement.png"

RANDOM_STATE = 42
N_BOOTSTRAP = 5000

COMPOSITION_FEATURES = [
    "log_il_to_sterol",
    "log_helper_to_sterol",
    "log_peg_to_sterol",
]

DESCRIPTOR_FEATURES = [
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

FEATURE_SETS = {
    "baseline_mean": [],
    "rf_composition": COMPOSITION_FEATURES,
    "rf_descriptors": DESCRIPTOR_FEATURES,
    "rf_combined": COMPOSITION_FEATURES + DESCRIPTOR_FEATURES,
}

MODEL_LABELS = {
    "baseline_mean": "Mean baseline",
    "rf_composition": "RF: composition",
    "rf_descriptors": "RF: descriptors",
    "rf_combined": "RF: combined",
}


def build_rf(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_features=0.7,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return np.nan
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else np.nan


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman_rho": safe_spearman(y_true, y_pred),
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "lnp_id",
        "paper_doi",
        "size_mean",
        "descriptor_complete",
        *COMPOSITION_FEATURES,
        *DESCRIPTOR_FEATURES,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sub = df[
        df["size_mean"].notna()
        & df["descriptor_complete"].astype(bool)
        & df["paper_doi"].notna()
    ].copy()

    all_features = COMPOSITION_FEATURES + DESCRIPTOR_FEATURES
    finite = np.isfinite(sub[all_features].to_numpy(dtype=float)).all(axis=1)
    sub = sub.loc[finite].reset_index(drop=True)

    studies = sorted(sub["paper_doi"].astype(str).unique())
    prediction_rows = []

    print("PHASE F1 STRICT LEAVE-ONE-PUBLICATION-OUT VALIDATION")
    print("===================================================")
    print(f"Rows: {len(sub)}")
    print(f"Studies: {len(studies)}")

    for i, held_out in enumerate(studies, start=1):
        print(f"  held-out study {i}/{len(studies)}")

        train = sub[sub["paper_doi"].astype(str) != held_out]
        test = sub[sub["paper_doi"].astype(str) == held_out]

        y_train = train["size_mean"].astype(float)
        y_test = test["size_mean"].astype(float)

        for model_id, features in FEATURE_SETS.items():
            if model_id == "baseline_mean":
                model = DummyRegressor(strategy="mean")
                X_train = np.zeros((len(train), 1))
                X_test = np.zeros((len(test), 1))
            else:
                X_train = train[features]
                X_test = test[features]
                model = build_rf(seed=RANDOM_STATE)

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            for pos, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append(
                    {
                        "held_out_paper_doi": held_out,
                        "model_id": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "lnp_id": row["lnp_id"],
                        "observed": float(y_test.iloc[pos]),
                        "predicted": float(pred[pos]),
                        "absolute_error": float(abs(y_test.iloc[pos] - pred[pos])),
                    }
                )

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(PRED_OUT, index=False)

    study_rows = []
    for (study, model_id), g in pred_df.groupby(
        ["held_out_paper_doi", "model_id"], sort=False
    ):
        m = pooled_metrics(
            g["observed"].to_numpy(dtype=float),
            g["predicted"].to_numpy(dtype=float),
        )
        study_rows.append(
            {
                "held_out_paper_doi": study,
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "n_test": len(g),
                **m,
            }
        )

    study_df = pd.DataFrame(study_rows)

    baseline_study = (
        study_df[study_df["model_id"] == "baseline_mean"]
        [["held_out_paper_doi", "mae"]]
        .rename(columns={"mae": "baseline_mae"})
    )
    study_df = study_df.merge(baseline_study, on="held_out_paper_doi", how="left")
    study_df["mae_improvement_vs_baseline_pct"] = (
        100.0
        * (study_df["baseline_mae"] - study_df["mae"])
        / study_df["baseline_mae"]
    )
    study_df["beats_baseline"] = study_df["mae"] < study_df["baseline_mae"]
    study_df.to_csv(STUDY_OUT, index=False)

    summary_rows = []
    for model_id, g in pred_df.groupby("model_id", sort=False):
        m = pooled_metrics(
            g["observed"].to_numpy(dtype=float),
            g["predicted"].to_numpy(dtype=float),
        )
        s = study_df[study_df["model_id"] == model_id]

        summary_rows.append(
            {
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "n_formulations": len(g),
                "n_studies": g["held_out_paper_doi"].nunique(),
                "pooled_mae": m["mae"],
                "pooled_rmse": m["rmse"],
                "pooled_r2": m["r2"],
                "pooled_spearman_rho": m["spearman_rho"],
                "macro_mean_study_mae": float(s["mae"].mean()),
                "median_study_mae": float(s["mae"].median()),
                "studies_beating_baseline": int(s["beats_baseline"].sum()),
                "total_studies": len(s),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    baseline_pooled_mae = float(
        summary_df.loc[
            summary_df["model_id"] == "baseline_mean", "pooled_mae"
        ].iloc[0]
    )
    summary_df["pooled_mae_improvement_vs_baseline_pct"] = (
        100.0
        * (baseline_pooled_mae - summary_df["pooled_mae"])
        / baseline_pooled_mae
    )
    summary_df.to_csv(SUMMARY_OUT, index=False)

    # Study-level bootstrap for the primary combined model.
    combined_study = (
        study_df[study_df["model_id"] == "rf_combined"]
        .merge(
            baseline_study,
            on="held_out_paper_doi",
            how="left",
            suffixes=("", "_again"),
        )
    )

    rng = np.random.default_rng(RANDOM_STATE)
    study_ids = combined_study["held_out_paper_doi"].to_numpy()
    bootstrap_rows = []

    for b in range(N_BOOTSTRAP):
        sampled = rng.choice(study_ids, size=len(study_ids), replace=True)
        sampled_df = pd.concat(
            [
                combined_study[
                    combined_study["held_out_paper_doi"] == study
                ]
                for study in sampled
            ],
            ignore_index=True,
        )

        model_mae = float(sampled_df["mae"].mean())
        baseline_mae = float(sampled_df["baseline_mae"].mean())
        improvement = (
            100.0 * (baseline_mae - model_mae) / baseline_mae
            if baseline_mae > 0
            else np.nan
        )

        bootstrap_rows.append(
            {
                "bootstrap_iteration": b + 1,
                "macro_mae_improvement_vs_baseline_pct": improvement,
            }
        )

    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv(BOOT_OUT, index=False)

    ci_low, ci_high = np.nanpercentile(
        boot_df["macro_mae_improvement_vs_baseline_pct"],
        [2.5, 97.5],
    )

    combined_summary = summary_df[
        summary_df["model_id"] == "rf_combined"
    ].iloc[0]

    # Observed vs predicted for the combined model.
    combined_pred = pred_df[pred_df["model_id"] == "rf_combined"].copy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        combined_pred["observed"],
        combined_pred["predicted"],
        alpha=0.75,
    )
    lo = min(combined_pred["observed"].min(), combined_pred["predicted"].min())
    hi = max(combined_pred["observed"].max(), combined_pred["predicted"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed particle size (nm)")
    ax.set_ylabel("LOPO prediction (nm)")
    ax.set_title("Particle size: strict leave-one-publication-out validation")
    fig.tight_layout()
    fig.savefig(OOF_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Study-level improvement figure.
    study_plot = study_df[
        study_df["model_id"] == "rf_combined"
    ].sort_values("mae_improvement_vs_baseline_pct")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        study_plot["held_out_paper_doi"],
        study_plot["mae_improvement_vs_baseline_pct"],
    )
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("MAE improvement versus training-mean baseline (%)")
    ax.set_ylabel("")
    ax.set_title("Particle size: study-level LOPO performance")
    fig.tight_layout()
    fig.savefig(STUDY_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    best_nonbaseline = (
        summary_df[summary_df["model_id"] != "baseline_mean"]
        .sort_values("pooled_mae")
        .iloc[0]
    )

    report_lines = [
        "PHASE F1 STRICT LEAVE-ONE-PUBLICATION-OUT REPORT",
        "================================================",
        "",
        f"Formulations: {len(sub)}",
        f"Studies held out one at a time: {len(studies)}",
        "",
        "Primary combined random-forest model:",
        f"  Pooled LOPO MAE: {combined_summary['pooled_mae']:.3f} nm",
        f"  Pooled LOPO RMSE: {combined_summary['pooled_rmse']:.3f} nm",
        f"  Pooled LOPO R2: {combined_summary['pooled_r2']:.3f}",
        (
            "  Pooled LOPO Spearman rho: "
            f"{combined_summary['pooled_spearman_rho']:.3f}"
        ),
        (
            "  Pooled MAE improvement versus baseline: "
            f"{combined_summary['pooled_mae_improvement_vs_baseline_pct']:.1f}%"
        ),
        (
            "  Studies where combined model beat baseline: "
            f"{int(combined_summary['studies_beating_baseline'])}/"
            f"{int(combined_summary['total_studies'])}"
        ),
        (
            "  Macro mean study MAE: "
            f"{combined_summary['macro_mean_study_mae']:.3f} nm"
        ),
        (
            "  Median study MAE: "
            f"{combined_summary['median_study_mae']:.3f} nm"
        ),
        "",
        "Study-bootstrap uncertainty:",
        (
            "  95% CI for macro study-level MAE improvement versus baseline: "
            f"{ci_low:.1f}% to {ci_high:.1f}%"
        ),
        "",
        "Best non-baseline model by pooled LOPO MAE:",
        f"  {best_nonbaseline['model_label']}",
        f"  Pooled MAE: {best_nonbaseline['pooled_mae']:.3f} nm",
        (
            "  Improvement versus baseline: "
            f"{best_nonbaseline['pooled_mae_improvement_vs_baseline_pct']:.1f}%"
        ),
        "",
        "Interpretation:",
        "  LOPO is stricter than grouped 5-fold validation because every publication",
        "  is tested separately. Report pooled and study-level performance together.",
        "  A positive pooled result with wide or crossing-zero study-bootstrap",
        "  uncertainty indicates partial but heterogeneous cross-publication transfer.",
        "",
        "Saved:",
        f"  {PRED_OUT}",
        f"  {STUDY_OUT}",
        f"  {SUMMARY_OUT}",
        f"  {BOOT_OUT}",
        f"  {OOF_FIG}",
        f"  {STUDY_FIG}",
    ]

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")
    print("\n" + report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
