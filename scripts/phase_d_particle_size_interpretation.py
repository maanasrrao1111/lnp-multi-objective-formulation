#!/usr/bin/env python3
"""
Phase D2: Particle-Size Model Robustness and Interpretation
==========================================================

Inputs
------
results/phase_c_feature_matrix.csv
results/phase_d_model_summary.csv
results/phase_d_best_models.csv

Outputs
-------
results/phase_d_particle_size_fold_metrics.csv
results/phase_d_particle_size_grouped_permutation_importance.csv
results/phase_d_particle_size_shap_importance.csv
results/phase_d_particle_size_interpretation_report.txt
figures/phase_d_particle_size_grouped_permutation_importance.png
figures/phase_d_particle_size_shap_bar.png
figures/phase_d_particle_size_shap_beeswarm.png

Purpose
-------
Interpret only the particle-size model, because it was the only Phase D endpoint
that passed the pre-specified publication-grouped predictive-signal criterion.

The script:
  1. Recreates the publication-grouped 5-fold random-forest evaluation.
  2. Calculates permutation importance on held-out folds.
  3. Permutes each feature WITHIN held-out publications where possible, reducing
     the chance that importance is driven only by study-level shifts.
  4. Fits the same model on the full particle-size dataset and computes SHAP
     values as a secondary, in-sample explanation.

Important
---------
SHAP results are descriptive for the fitted full-data model. The held-out grouped
permutation importance is the more defensible measure of transferable importance.
The model's out-of-fold R2 is modest, so findings should be described as candidate
associations rather than universal design rules.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT = Path("results/phase_c_feature_matrix.csv")
MODEL_SUMMARY_IN = Path("results/phase_d_model_summary.csv")
BEST_MODELS_IN = Path("results/phase_d_best_models.csv")

RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FOLD_OUT = RESULTS / "phase_d_particle_size_fold_metrics.csv"
PERM_OUT = RESULTS / "phase_d_particle_size_grouped_permutation_importance.csv"
SHAP_OUT = RESULTS / "phase_d_particle_size_shap_importance.csv"
REPORT_OUT = RESULTS / "phase_d_particle_size_interpretation_report.txt"

PERM_FIG = FIGURES / "phase_d_particle_size_grouped_permutation_importance.png"
SHAP_BAR_FIG = FIGURES / "phase_d_particle_size_shap_bar.png"
SHAP_BEE_FIG = FIGURES / "phase_d_particle_size_shap_beeswarm.png"

RANDOM_STATE = 42
N_SPLITS = 5
N_PERMUTATIONS = 50
TOP_N = 12

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

FEATURES = COMPOSITION_FEATURES + DESCRIPTOR_FEATURES

DISPLAY_NAMES = {
    "log_il_to_sterol": "Ionizable/sterol log ratio",
    "log_helper_to_sterol": "Helper/sterol log ratio",
    "log_peg_to_sterol": "PEG/sterol log ratio",
    "mw": "Molecular weight",
    "logp": "LogP",
    "tpsa": "TPSA",
    "num_rotatable_bonds": "Rotatable bonds",
    "num_hbd": "H-bond donors",
    "num_hba": "H-bond acceptors",
    "num_ester_bonds": "Ester bonds",
    "num_tertiary_amines": "Tertiary amines",
    "tail_length_carbons": "Longest carbon-chain length",
    "num_unsaturated_bonds": "Unsaturated bonds",
    "num_branch_points": "Carbon branch points",
    "num_rings": "Rings",
    "fraction_csp3": "Fraction Csp3",
    "heavy_atom_count": "Heavy-atom count",
}


def build_model(seed: int = RANDOM_STATE) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_features=0.7,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    value = spearmanr(a, b, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else np.nan


def grouped_permutation(
    X: pd.DataFrame,
    groups: pd.Series,
    feature: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Permute feature values within each publication in the held-out fold.
    Publications containing a single row cannot be permuted and remain unchanged.
    """
    Xp = X.copy()
    group_values = groups.astype(str).to_numpy()

    for group in np.unique(group_values):
        idx = np.where(group_values == group)[0]
        if len(idx) > 1:
            vals = Xp.iloc[idx][feature].to_numpy(copy=True)
            Xp.iloc[idx, Xp.columns.get_loc(feature)] = rng.permutation(vals)

    return Xp


def main() -> None:
    for path in [INPUT, MODEL_SUMMARY_IN, BEST_MODELS_IN]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    df = pd.read_csv(INPUT)
    best = pd.read_csv(BEST_MODELS_IN)
    model_summary = pd.read_csv(MODEL_SUMMARY_IN)

    required = {
        "paper_doi",
        "size_mean",
        "descriptor_complete",
        *FEATURES,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    particle_best = best[best["endpoint"] == "particle_size"]
    if particle_best.empty:
        raise ValueError("Particle-size row missing from phase_d_best_models.csv")

    carried = particle_best.iloc[0]["best_model_id"]
    passed = bool(particle_best.iloc[0]["predictive_signal_passed"])

    if carried != "rf_combined" or not passed:
        raise ValueError(
            "This script is intended only when particle size carries forward "
            "the rf_combined model and passes the predictive-signal criterion."
        )

    sub = df[
        df["size_mean"].notna()
        & df["descriptor_complete"].astype(bool)
        & df["paper_doi"].notna()
    ].copy()

    # Phase C already generated complete descriptor rows; enforce finite features.
    finite_mask = np.isfinite(sub[FEATURES].to_numpy(dtype=float)).all(axis=1)
    sub = sub.loc[finite_mask].reset_index(drop=True)

    X = sub[FEATURES].copy()
    y = sub["size_mean"].astype(float)
    groups = sub["paper_doi"].astype(str)

    n_groups = groups.nunique()
    n_splits = min(N_SPLITS, n_groups)
    cv = GroupKFold(n_splits=n_splits)

    fold_rows = []
    perm_rows = []
    oof_pred = np.full(len(sub), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        g_test = groups.iloc[test_idx]

        model = build_model(seed=RANDOM_STATE + fold)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        oof_pred[test_idx] = pred

        baseline_mae = mean_absolute_error(y_test, pred)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_studies": g_test.nunique(),
                "mae": baseline_mae,
                "rmse": mean_squared_error(y_test, pred) ** 0.5,
                "r2": r2_score(y_test, pred),
                "spearman_rho": safe_spearman(
                    y_test.to_numpy(dtype=float),
                    np.asarray(pred, dtype=float),
                ),
            }
        )

        rng = np.random.default_rng(RANDOM_STATE + fold * 1000)

        for feature in FEATURES:
            increases = []

            for _ in range(N_PERMUTATIONS):
                X_perm = grouped_permutation(
                    X_test,
                    g_test,
                    feature,
                    rng,
                )
                perm_pred = model.predict(X_perm)
                perm_mae = mean_absolute_error(y_test, perm_pred)
                increases.append(perm_mae - baseline_mae)

            perm_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "display_name": DISPLAY_NAMES[feature],
                    "n_test": len(test_idx),
                    "n_test_studies": g_test.nunique(),
                    "baseline_mae": baseline_mae,
                    "mae_increase_mean": float(np.mean(increases)),
                    "mae_increase_sd": float(np.std(increases, ddof=1)),
                    "mae_increase_median": float(np.median(increases)),
                    "positive_importance_fraction": float(
                        np.mean(np.asarray(increases) > 0)
                    ),
                }
            )

    if np.isnan(oof_pred).any():
        raise RuntimeError("Missing out-of-fold predictions")

    fold_df = pd.DataFrame(fold_rows)
    perm_fold_df = pd.DataFrame(perm_rows)

    # Aggregate fold-level importance, weighted by held-out sample count.
    aggregate_rows = []

    for feature, group in perm_fold_df.groupby("feature"):
        weights = group["n_test"].to_numpy(dtype=float)
        values = group["mae_increase_mean"].to_numpy(dtype=float)

        weighted_mean = float(np.average(values, weights=weights))
        aggregate_rows.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "weighted_mean_mae_increase": weighted_mean,
                "median_fold_mae_increase": float(np.median(values)),
                "sd_across_folds": float(np.std(values, ddof=1)),
                "folds_positive": int((values > 0).sum()),
                "total_folds": len(values),
                "mean_positive_importance_fraction": float(
                    group["positive_importance_fraction"].mean()
                ),
            }
        )

    perm_df = pd.DataFrame(aggregate_rows).sort_values(
        "weighted_mean_mae_increase",
        ascending=False,
    )

    # Fit the same model on the full particle-size dataset for SHAP.
    full_model = build_model(seed=RANDOM_STATE)
    full_model.fit(X, y)

    explainer = shap.TreeExplainer(full_model)
    shap_values = explainer.shap_values(X)
    shap_values = np.asarray(shap_values)

    shap_rows = []

    for i, feature in enumerate(FEATURES):
        values = shap_values[:, i]
        shap_rows.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "mean_abs_shap": float(np.mean(np.abs(values))),
                "median_abs_shap": float(np.median(np.abs(values))),
                "feature_shap_spearman": safe_spearman(
                    X[feature].to_numpy(dtype=float),
                    values,
                ),
            }
        )

    shap_df = pd.DataFrame(shap_rows).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    fold_df.to_csv(FOLD_OUT, index=False)
    perm_df.to_csv(PERM_OUT, index=False)
    shap_df.to_csv(SHAP_OUT, index=False)

    # Grouped permutation-importance figure.
    top_perm = perm_df.head(TOP_N).sort_values(
        "weighted_mean_mae_increase",
        ascending=True,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top_perm["display_name"],
        top_perm["weighted_mean_mae_increase"],
    )
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Increase in held-out MAE after within-study permutation (nm)")
    ax.set_ylabel("")
    ax.set_title("Particle-size model: grouped permutation importance")
    fig.tight_layout()
    fig.savefig(PERM_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # SHAP mean-absolute bar figure.
    top_shap = shap_df.head(TOP_N).sort_values(
        "mean_abs_shap",
        ascending=True,
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_shap["display_name"], top_shap["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value (nm)")
    ax.set_ylabel("")
    ax.set_title("Particle-size model: full-data SHAP importance")
    fig.tight_layout()
    fig.savefig(SHAP_BAR_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # SHAP beeswarm with readable feature labels.
    X_display = X.rename(columns=DISPLAY_NAMES)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_display,
        show=False,
        max_display=TOP_N,
    )
    plt.title("Particle-size model: SHAP distribution")
    plt.tight_layout()
    plt.savefig(SHAP_BEE_FIG, dpi=300, bbox_inches="tight")
    plt.close()

    oof_mae = mean_absolute_error(y, oof_pred)
    oof_rmse = mean_squared_error(y, oof_pred) ** 0.5
    oof_r2 = r2_score(y, oof_pred)
    oof_rho = safe_spearman(y.to_numpy(dtype=float), oof_pred)

    summary_row = model_summary[
        (model_summary["endpoint"] == "particle_size")
        & (model_summary["model_id"] == "rf_combined")
    ].iloc[0]

    top_perm_lines = [
        (
            f"  {i+1}. {row.display_name}: "
            f"mean held-out MAE increase "
            f"{row.weighted_mean_mae_increase:.3f} nm; "
            f"positive in {int(row.folds_positive)}/{int(row.total_folds)} folds"
        )
        for i, row in enumerate(perm_df.head(10).itertuples())
    ]

    top_shap_lines = [
        (
            f"  {i+1}. {row.display_name}: "
            f"mean |SHAP| {row.mean_abs_shap:.3f} nm; "
            f"feature-SHAP rho {row.feature_shap_spearman:.3f}"
        )
        for i, row in enumerate(shap_df.head(10).itertuples())
    ]

    report_lines = [
        "PHASE D2 PARTICLE-SIZE INTERPRETATION REPORT",
        "============================================",
        "",
        f"Usable formulations: {len(sub)}",
        f"Studies represented: {n_groups}",
        f"Features: {len(FEATURES)}",
        "",
        "Publication-grouped out-of-fold performance:",
        f"  MAE: {oof_mae:.3f} nm",
        f"  RMSE: {oof_rmse:.3f} nm",
        f"  R2: {oof_r2:.3f}",
        f"  Spearman rho: {oof_rho:.3f}",
        (
            "  Phase D reported MAE improvement versus mean baseline: "
            f"{float(summary_row['mae']):.3f} nm model MAE"
        ),
        "",
        "Top held-out grouped permutation features:",
        *top_perm_lines,
        "",
        "Top full-data SHAP features:",
        *top_shap_lines,
        "",
        "Interpretation:",
        "  The grouped permutation results are the primary importance analysis.",
        "  A feature is more credible when held-out MAE increases after permutation",
        "  in several folds, not merely when its pooled mean importance is positive.",
        "  SHAP summarizes the fitted full-data random forest and is secondary.",
        "",
        "Caution:",
        "  OOF R2 is modest. These results identify candidate associations with",
        "  particle size, not universal causal design rules. Correlated descriptors",
        "  and composition features can share or redistribute importance.",
        "",
        "Saved:",
        f"  {FOLD_OUT}",
        f"  {PERM_OUT}",
        f"  {SHAP_OUT}",
        f"  {PERM_FIG}",
        f"  {SHAP_BAR_FIG}",
        f"  {SHAP_BEE_FIG}",
    ]

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")
    print(report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
