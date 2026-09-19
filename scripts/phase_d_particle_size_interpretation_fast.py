#!/usr/bin/env python3
"""
Phase D2 (fast/resumable): Particle-size robustness and interpretation
=====================================================================

Inputs
------
results/phase_c_feature_matrix.csv
results/phase_d_model_summary.csv
results/phase_d_best_models.csv

Primary outputs (always saved before SHAP)
------------------------------------------
results/phase_d_particle_size_fold_metrics.csv
results/phase_d_particle_size_grouped_permutation_importance.csv
results/phase_d_particle_size_oof_predictions.csv
results/phase_d_particle_size_interpretation_report.txt
figures/phase_d_particle_size_grouped_permutation_importance.png
figures/phase_d_particle_size_observed_vs_predicted.png

Optional SHAP outputs (only with --with-shap)
----------------------------------------------
results/phase_d_particle_size_shap_importance.csv
figures/phase_d_particle_size_shap_bar.png
figures/phase_d_particle_size_shap_beeswarm.png

Why this revision exists
------------------------
The earlier script saved outputs only after exact TreeSHAP completed. Exact SHAP for
a 500-tree random forest can be slow. This revision:
  1. saves all grouped-CV and held-out permutation results first;
  2. treats SHAP as optional and secondary;
  3. uses approximate TreeSHAP on at most 150 formulations;
  4. preserves completed primary results even if SHAP fails or is interrupted.

Usage
-----
Primary analysis only:
    python scripts/phase_d_particle_size_interpretation_fast.py

Primary analysis plus approximate SHAP:
    python scripts/phase_d_particle_size_interpretation_fast.py --with-shap
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
OOF_OUT = RESULTS / "phase_d_particle_size_oof_predictions.csv"
SHAP_OUT = RESULTS / "phase_d_particle_size_shap_importance.csv"
REPORT_OUT = RESULTS / "phase_d_particle_size_interpretation_report.txt"

PERM_FIG = FIGURES / "phase_d_particle_size_grouped_permutation_importance.png"
OOF_FIG = FIGURES / "phase_d_particle_size_observed_vs_predicted.png"
SHAP_BAR_FIG = FIGURES / "phase_d_particle_size_shap_bar.png"
SHAP_BEE_FIG = FIGURES / "phase_d_particle_size_shap_beeswarm.png"

RANDOM_STATE = 42
N_SPLITS = 5
N_PERMUTATIONS = 30
TOP_N = 12
MAX_SHAP_ROWS = 150

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
    """Permute one feature within each held-out publication."""
    Xp = X.copy()
    group_values = groups.astype(str).to_numpy()

    for group in np.unique(group_values):
        idx = np.where(group_values == group)[0]
        if len(idx) > 1:
            vals = Xp.iloc[idx][feature].to_numpy(copy=True)
            Xp.iloc[idx, Xp.columns.get_loc(feature)] = rng.permutation(vals)

    return Xp


def write_primary_report(
    n_rows: int,
    n_groups: int,
    oof_mae: float,
    oof_rmse: float,
    oof_r2: float,
    oof_rho: float,
    perm_df: pd.DataFrame,
    shap_status: str,
) -> str:
    top_perm_lines = [
        (
            f"  {i+1}. {row.display_name}: "
            f"held-out MAE increase {row.weighted_mean_mae_increase:.3f} nm; "
            f"positive in {int(row.folds_positive)}/{int(row.total_folds)} folds"
        )
        for i, row in enumerate(perm_df.head(10).itertuples())
    ]

    lines = [
        "PHASE D2 PARTICLE-SIZE INTERPRETATION REPORT",
        "============================================",
        "",
        f"Usable formulations: {n_rows}",
        f"Studies represented: {n_groups}",
        f"Features: {len(FEATURES)}",
        "",
        "Publication-grouped out-of-fold performance:",
        f"  MAE: {oof_mae:.3f} nm",
        f"  RMSE: {oof_rmse:.3f} nm",
        f"  R2: {oof_r2:.3f}",
        f"  Spearman rho: {oof_rho:.3f}",
        "",
        "Top held-out grouped permutation features:",
        *top_perm_lines,
        "",
        "Interpretation:",
        "  Grouped held-out permutation importance is the primary explanation.",
        "  A feature is more credible when permutation worsens held-out MAE in",
        "  several folds, not merely when its pooled mean importance is positive.",
        "",
        "Caution:",
        "  Out-of-fold R2 is modest. These are candidate associations with particle",
        "  size, not universal or causal design rules. Correlated features may share",
        "  or redistribute importance.",
        "",
        f"SHAP status: {shap_status}",
        "",
        "Saved primary outputs:",
        f"  {FOLD_OUT}",
        f"  {PERM_OUT}",
        f"  {OOF_OUT}",
        f"  {PERM_FIG}",
        f"  {OOF_FIG}",
    ]
    return "\n".join(lines)


def main(with_shap: bool) -> None:
    for path in [INPUT, MODEL_SUMMARY_IN, BEST_MODELS_IN]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    df = pd.read_csv(INPUT)
    best = pd.read_csv(BEST_MODELS_IN)

    particle_best = best[best["endpoint"] == "particle_size"]
    if particle_best.empty:
        raise ValueError("Particle-size row missing from phase_d_best_models.csv")

    row = particle_best.iloc[0]
    if row["best_model_id"] != "rf_combined" or not bool(row["predictive_signal_passed"]):
        raise ValueError(
            "Expected particle size to carry forward rf_combined and pass the "
            "predictive-signal criterion."
        )

    required = {"paper_doi", "size_mean", "descriptor_complete", *FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    sub = df[
        df["size_mean"].notna()
        & df["descriptor_complete"].astype(bool)
        & df["paper_doi"].notna()
    ].copy()

    finite = np.isfinite(sub[FEATURES].to_numpy(dtype=float)).all(axis=1)
    sub = sub.loc[finite].reset_index(drop=True)

    X = sub[FEATURES].copy()
    y = sub["size_mean"].astype(float)
    groups = sub["paper_doi"].astype(str)

    n_groups = groups.nunique()
    cv = GroupKFold(n_splits=min(N_SPLITS, n_groups))

    fold_rows = []
    perm_rows = []
    prediction_rows = []
    oof_pred = np.full(len(sub), np.nan, dtype=float)

    print("Running publication-grouped particle-size interpretation...")
    print(f"  rows={len(sub)}, studies={n_groups}, permutations={N_PERMUTATIONS}")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        print(f"  fold {fold}/{min(N_SPLITS, n_groups)}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        g_test = groups.iloc[test_idx]

        model = build_model(seed=RANDOM_STATE)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        oof_pred[test_idx] = pred

        base_mae = mean_absolute_error(y_test, pred)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_studies": g_test.nunique(),
                "mae": base_mae,
                "rmse": mean_squared_error(y_test, pred) ** 0.5,
                "r2": r2_score(y_test, pred),
                "spearman_rho": safe_spearman(
                    y_test.to_numpy(dtype=float),
                    np.asarray(pred, dtype=float),
                ),
            }
        )

        for pos, idx in enumerate(test_idx):
            prediction_rows.append(
                {
                    "fold": fold,
                    "lnp_id": sub.iloc[idx].get("lnp_id"),
                    "paper_doi": sub.iloc[idx]["paper_doi"],
                    "observed": float(y_test.iloc[pos]),
                    "predicted": float(pred[pos]),
                }
            )

        rng = np.random.default_rng(RANDOM_STATE + fold * 1000)

        for feature in FEATURES:
            increases = []
            for _ in range(N_PERMUTATIONS):
                Xp = grouped_permutation(X_test, g_test, feature, rng)
                p = model.predict(Xp)
                increases.append(mean_absolute_error(y_test, p) - base_mae)

            perm_rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "display_name": DISPLAY_NAMES[feature],
                    "n_test": len(test_idx),
                    "n_test_studies": g_test.nunique(),
                    "baseline_mae": base_mae,
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
    oof_df = pd.DataFrame(prediction_rows)

    aggregate = []
    for feature, g in perm_fold_df.groupby("feature"):
        weights = g["n_test"].to_numpy(dtype=float)
        values = g["mae_increase_mean"].to_numpy(dtype=float)
        aggregate.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAMES[feature],
                "weighted_mean_mae_increase": float(
                    np.average(values, weights=weights)
                ),
                "median_fold_mae_increase": float(np.median(values)),
                "sd_across_folds": float(np.std(values, ddof=1)),
                "folds_positive": int((values > 0).sum()),
                "total_folds": len(values),
                "mean_positive_importance_fraction": float(
                    g["positive_importance_fraction"].mean()
                ),
            }
        )

    perm_df = pd.DataFrame(aggregate).sort_values(
        "weighted_mean_mae_increase", ascending=False
    )

    # Save primary results BEFORE optional SHAP.
    fold_df.to_csv(FOLD_OUT, index=False)
    perm_df.to_csv(PERM_OUT, index=False)
    oof_df.to_csv(OOF_OUT, index=False)

    top = perm_df.head(TOP_N).sort_values(
        "weighted_mean_mae_increase", ascending=True
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["display_name"], top["weighted_mean_mae_increase"])
    ax.axvline(0, linestyle="--")
    ax.set_xlabel("Increase in held-out MAE after within-study permutation (nm)")
    ax.set_ylabel("")
    ax.set_title("Particle-size model: grouped permutation importance")
    fig.tight_layout()
    fig.savefig(PERM_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(oof_df["observed"], oof_df["predicted"], alpha=0.75)
    lo = min(oof_df["observed"].min(), oof_df["predicted"].min())
    hi = max(oof_df["observed"].max(), oof_df["predicted"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed particle size (nm)")
    ax.set_ylabel("Grouped out-of-fold prediction (nm)")
    ax.set_title("Particle-size model: observed vs predicted")
    fig.tight_layout()
    fig.savefig(OOF_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    oof_mae = mean_absolute_error(y, oof_pred)
    oof_rmse = mean_squared_error(y, oof_pred) ** 0.5
    oof_r2 = r2_score(y, oof_pred)
    oof_rho = safe_spearman(y.to_numpy(dtype=float), oof_pred)

    report = write_primary_report(
        len(sub),
        n_groups,
        oof_mae,
        oof_rmse,
        oof_r2,
        oof_rho,
        perm_df,
        "not requested",
    )
    REPORT_OUT.write_text(report + "\n")
    print("\nPrimary grouped interpretation complete and saved.")
    print(report)

    if not with_shap:
        print("\nSHAP skipped. Re-run with --with-shap only if desired.")
        return

    print("\nRunning approximate SHAP on a capped sample...")

    try:
        import shap

        full_model = build_model(seed=RANDOM_STATE)
        full_model.fit(X, y)

        if len(X) > MAX_SHAP_ROWS:
            sample_idx = (
                pd.Series(np.arange(len(X)))
                .sample(MAX_SHAP_ROWS, random_state=RANDOM_STATE)
                .sort_values()
                .to_numpy()
            )
            X_shap = X.iloc[sample_idx].copy()
        else:
            X_shap = X.copy()

        explainer = shap.TreeExplainer(full_model)
        shap_values = explainer.shap_values(
            X_shap,
            approximate=True,
            check_additivity=False,
        )
        shap_values = np.asarray(shap_values)

        rows = []
        for i, feature in enumerate(FEATURES):
            vals = shap_values[:, i]
            rows.append(
                {
                    "feature": feature,
                    "display_name": DISPLAY_NAMES[feature],
                    "mean_abs_shap": float(np.mean(np.abs(vals))),
                    "median_abs_shap": float(np.median(np.abs(vals))),
                    "feature_shap_spearman": safe_spearman(
                        X_shap[feature].to_numpy(dtype=float), vals
                    ),
                    "n_shap_rows": len(X_shap),
                    "approximate_shap": True,
                }
            )

        shap_df = pd.DataFrame(rows).sort_values(
            "mean_abs_shap", ascending=False
        )
        shap_df.to_csv(SHAP_OUT, index=False)

        top_shap = shap_df.head(TOP_N).sort_values(
            "mean_abs_shap", ascending=True
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top_shap["display_name"], top_shap["mean_abs_shap"])
        ax.set_xlabel("Mean absolute approximate SHAP value (nm)")
        ax.set_ylabel("")
        ax.set_title("Particle-size model: approximate full-data SHAP")
        fig.tight_layout()
        fig.savefig(SHAP_BAR_FIG, dpi=300, bbox_inches="tight")
        plt.close(fig)

        X_display = X_shap.rename(columns=DISPLAY_NAMES)
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_display,
            show=False,
            max_display=TOP_N,
        )
        plt.title("Particle-size model: approximate SHAP distribution")
        plt.tight_layout()
        plt.savefig(SHAP_BEE_FIG, dpi=300, bbox_inches="tight")
        plt.close()

        report = write_primary_report(
            len(sub),
            n_groups,
            oof_mae,
            oof_rmse,
            oof_r2,
            oof_rho,
            perm_df,
            f"completed approximately on {len(X_shap)} rows",
        )
        report += (
            "\n\nOptional SHAP outputs:\n"
            f"  {SHAP_OUT}\n"
            f"  {SHAP_BAR_FIG}\n"
            f"  {SHAP_BEE_FIG}\n"
        )
        REPORT_OUT.write_text(report + "\n")
        print("\nApproximate SHAP completed and saved.")

    except Exception as exc:
        report = write_primary_report(
            len(sub),
            n_groups,
            oof_mae,
            oof_rmse,
            oof_r2,
            oof_rho,
            perm_df,
            f"failed, but primary outputs remain valid: {type(exc).__name__}: {exc}",
        )
        REPORT_OUT.write_text(report + "\n")
        print("\nSHAP failed, but primary grouped results were already saved.")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-shap",
        action="store_true",
        help="Also run approximate SHAP on at most 150 formulations.",
    )
    args = parser.parse_args()
    main(with_shap=args.with_shap)
