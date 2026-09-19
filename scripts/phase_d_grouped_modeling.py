#!/usr/bin/env python3
"""
Phase D: Publication-Grouped Single-Objective Model Screening
=============================================================

Input
-----
results/phase_c_feature_matrix.csv

Outputs
-------
results/phase_d_fold_metrics.csv
results/phase_d_oof_predictions.csv
results/phase_d_model_summary.csv
results/phase_d_best_models.csv
results/phase_d_modeling_report.txt
figures/phase_d_<endpoint>_normalized_mae.png
figures/phase_d_<endpoint>_observed_vs_predicted.png

Design
------
For each endpoint, compare:
  1. Mean-only baseline
  2. Ridge regression using composition features
  3. Ridge regression using molecular descriptors
  4. Ridge regression using combined features
  5. Random forest using composition features
  6. Random forest using molecular descriptors
  7. Random forest using combined features

Validation
----------
Outer validation uses publication-grouped 5-fold cross-validation. Formulations
from the same paper are never split between training and test sets.

Ridge alpha is selected inside each outer training set using an inner grouped
cross-validation loop. Random forest hyperparameters are intentionally fixed to
avoid a large search on small datasets.

This script performs model screening only. It does not generate SHAP values or
make design-rule claims. Feature interpretation should be run only for endpoints
where a non-baseline model shows credible out-of-study predictive signal.
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
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

INPUT = Path("results/phase_c_feature_matrix.csv")
RESULTS = Path("results")
FIGURES = Path("figures")
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FOLD_METRICS_OUT = RESULTS / "phase_d_fold_metrics.csv"
OOF_OUT = RESULTS / "phase_d_oof_predictions.csv"
SUMMARY_OUT = RESULTS / "phase_d_model_summary.csv"
BEST_OUT = RESULTS / "phase_d_best_models.csv"
REPORT_OUT = RESULTS / "phase_d_modeling_report.txt"

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
MIN_IMPROVEMENT_PCT = 5.0

ENDPOINTS = {
    "transfection": "transfection_pctile",
    "particle_size": "size_mean",
    "pdi": "pdi_mean",
    "encapsulation_efficiency": "ee_mean",
}

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
    "composition": COMPOSITION_FEATURES,
    "descriptors": DESCRIPTOR_FEATURES,
    "combined": COMPOSITION_FEATURES + DESCRIPTOR_FEATURES,
}

MODEL_LABELS = {
    "baseline_mean": "Mean baseline",
    "ridge_composition": "Ridge: composition",
    "ridge_descriptors": "Ridge: descriptors",
    "ridge_combined": "Ridge: combined",
    "rf_composition": "RF: composition",
    "rf_descriptors": "RF: descriptors",
    "rf_combined": "RF: combined",
}


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return np.nan
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else np.nan


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    rho = safe_spearman(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "spearman_rho": rho,
    }


def build_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=alpha)),
        ]
    )


def build_random_forest() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features=0.7,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def tune_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> tuple[Pipeline, float]:
    unique_groups = groups_train.nunique()

    if unique_groups < 3:
        model = build_ridge(alpha=1.0)
        model.fit(X_train, y_train)
        return model, 1.0

    n_inner = min(INNER_FOLDS, unique_groups)
    inner_cv = GroupKFold(n_splits=n_inner)

    grid = GridSearchCV(
        estimator=build_ridge(),
        param_grid={
            "regressor__alpha": [
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
                3.0,
                10.0,
                30.0,
                100.0,
            ]
        },
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    best_alpha = float(grid.best_params_["regressor__alpha"])
    return grid.best_estimator_, best_alpha


def evaluate_endpoint(
    df: pd.DataFrame,
    endpoint_name: str,
    endpoint_col: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    sub = df[
        df[endpoint_col].notna()
        & df["descriptor_complete"].astype(bool)
        & df["paper_doi"].notna()
    ].copy()

    if len(sub) < 20:
        raise ValueError(
            f"{endpoint_name}: only {len(sub)} usable rows; too few for this workflow"
        )

    groups = sub["paper_doi"].astype(str)
    n_groups = groups.nunique()

    if n_groups < 3:
        raise ValueError(
            f"{endpoint_name}: only {n_groups} studies; grouped CV is not feasible"
        )

    n_outer = min(OUTER_FOLDS, n_groups)
    outer_cv = GroupKFold(n_splits=n_outer)
    split_indices = list(outer_cv.split(sub, sub[endpoint_col], groups))

    model_specs = [("baseline_mean", "baseline", None)]
    for feature_set_name in FEATURE_SETS:
        model_specs.append(
            (
                f"ridge_{feature_set_name}",
                "ridge",
                feature_set_name,
            )
        )
        model_specs.append(
            (
                f"rf_{feature_set_name}",
                "rf",
                feature_set_name,
            )
        )

    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    endpoint_summary_rows: list[dict] = []

    for model_id, model_type, feature_set_name in model_specs:
        oof_pred = np.full(len(sub), np.nan, dtype=float)
        selected_alphas = []

        for fold, (train_idx, test_idx) in enumerate(split_indices, start=1):
            train = sub.iloc[train_idx]
            test = sub.iloc[test_idx]

            y_train = train[endpoint_col]
            y_test = test[endpoint_col]
            groups_train = train["paper_doi"].astype(str)

            if model_type == "baseline":
                model = DummyRegressor(strategy="mean")
                X_train = np.zeros((len(train), 1))
                X_test = np.zeros((len(test), 1))
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                alpha = np.nan

            else:
                features = FEATURE_SETS[feature_set_name]
                X_train = train[features]
                X_test = test[features]

                if model_type == "ridge":
                    model, alpha = tune_ridge(
                        X_train,
                        y_train,
                        groups_train,
                    )
                    selected_alphas.append(alpha)
                elif model_type == "rf":
                    model = build_random_forest()
                    model.fit(X_train, y_train)
                    alpha = np.nan
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                pred = model.predict(X_test)

            oof_pred[test_idx] = pred
            fold_metric = metrics(
                y_test.to_numpy(dtype=float),
                np.asarray(pred, dtype=float),
            )

            fold_rows.append(
                {
                    "endpoint": endpoint_name,
                    "endpoint_column": endpoint_col,
                    "model_id": model_id,
                    "model_label": MODEL_LABELS[model_id],
                    "feature_set": feature_set_name or "none",
                    "fold": fold,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_train_studies": train["paper_doi"].nunique(),
                    "n_test_studies": test["paper_doi"].nunique(),
                    "selected_ridge_alpha": alpha,
                    **fold_metric,
                }
            )

            for local_pos, row_idx in enumerate(test_idx):
                row = sub.iloc[row_idx]
                prediction_rows.append(
                    {
                        "endpoint": endpoint_name,
                        "endpoint_column": endpoint_col,
                        "model_id": model_id,
                        "model_label": MODEL_LABELS[model_id],
                        "feature_set": feature_set_name or "none",
                        "fold": fold,
                        "lnp_id": row.get("lnp_id"),
                        "paper_doi": row["paper_doi"],
                        "observed": float(row[endpoint_col]),
                        "predicted": float(pred[local_pos]),
                    }
                )

        if np.isnan(oof_pred).any():
            raise RuntimeError(
                f"{endpoint_name}/{model_id}: missing OOF predictions"
            )

        y_all = sub[endpoint_col].to_numpy(dtype=float)
        summary_metric = metrics(y_all, oof_pred)

        q25, q75 = np.percentile(y_all, [25, 75])
        iqr = float(q75 - q25)
        normalized_mae = (
            summary_metric["mae"] / iqr
            if iqr > 0
            else np.nan
        )

        endpoint_summary_rows.append(
            {
                "endpoint": endpoint_name,
                "endpoint_column": endpoint_col,
                "model_id": model_id,
                "model_label": MODEL_LABELS[model_id],
                "feature_set": feature_set_name or "none",
                "n_formulations": len(sub),
                "n_studies": n_groups,
                "outer_folds": n_outer,
                "outcome_iqr": iqr,
                "normalized_mae": normalized_mae,
                "median_selected_ridge_alpha": (
                    float(np.median(selected_alphas))
                    if selected_alphas
                    else np.nan
                ),
                **summary_metric,
            }
        )

    return fold_rows, prediction_rows, endpoint_summary_rows


def make_figures(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    best: pd.DataFrame,
) -> None:
    for endpoint in ENDPOINTS:
        s = summary[summary["endpoint"] == endpoint].copy()
        s = s.sort_values("normalized_mae", ascending=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(s["model_label"], s["normalized_mae"])
        ax.set_xlabel("Out-of-fold MAE / outcome IQR")
        ax.set_ylabel("")
        ax.set_title(
            f"{endpoint.replace('_', ' ').title()}: publication-grouped model screening"
        )
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"phase_d_{endpoint}_normalized_mae.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        best_row = best[best["endpoint"] == endpoint].iloc[0]
        best_model = best_row["best_model_id"]
        p = predictions[
            (predictions["endpoint"] == endpoint)
            & (predictions["model_id"] == best_model)
        ].copy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(p["observed"], p["predicted"], alpha=0.75)
        lo = min(p["observed"].min(), p["predicted"].min())
        hi = max(p["observed"].max(), p["predicted"].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Publication-grouped out-of-fold prediction")
        ax.set_title(
            f"{endpoint.replace('_', ' ').title()}: "
            f"{MODEL_LABELS[best_model]}"
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"phase_d_{endpoint}_observed_vs_predicted.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "paper_doi",
        "descriptor_complete",
        *COMPOSITION_FEATURES,
        *DESCRIPTOR_FEATURES,
        *ENDPOINTS.values(),
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_fold_rows: list[dict] = []
    all_prediction_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    print("PHASE D PUBLICATION-GROUPED MODEL SCREENING")
    print("===========================================")

    for endpoint_name, endpoint_col in ENDPOINTS.items():
        print(f"\nRunning endpoint: {endpoint_name}")
        fold_rows, prediction_rows, summary_rows = evaluate_endpoint(
            df,
            endpoint_name,
            endpoint_col,
        )
        all_fold_rows.extend(fold_rows)
        all_prediction_rows.extend(prediction_rows)
        all_summary_rows.extend(summary_rows)
        print(
            f"  completed {len(summary_rows)} model comparisons "
            f"across grouped folds"
        )

    fold_df = pd.DataFrame(all_fold_rows)
    pred_df = pd.DataFrame(all_prediction_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    best_rows = []

    for endpoint in ENDPOINTS:
        s = summary_df[summary_df["endpoint"] == endpoint].copy()
        baseline = s[s["model_id"] == "baseline_mean"].iloc[0]
        candidates = s[s["model_id"] != "baseline_mean"].copy()
        winner = candidates.sort_values("mae").iloc[0]

        improvement = (
            100.0 * (baseline["mae"] - winner["mae"]) / baseline["mae"]
            if baseline["mae"] > 0
            else np.nan
        )

        predictive_signal = bool(
            np.isfinite(improvement)
            and improvement >= MIN_IMPROVEMENT_PCT
            and np.isfinite(winner["r2"])
            and winner["r2"] > 0
        )

        chosen_model_id = (
            winner["model_id"]
            if predictive_signal
            else "baseline_mean"
        )

        best_rows.append(
            {
                "endpoint": endpoint,
                "baseline_mae": baseline["mae"],
                "best_nonbaseline_model_id": winner["model_id"],
                "best_nonbaseline_model_label": winner["model_label"],
                "best_nonbaseline_mae": winner["mae"],
                "best_nonbaseline_r2": winner["r2"],
                "best_nonbaseline_spearman_rho": winner["spearman_rho"],
                "mae_improvement_vs_baseline_pct": improvement,
                "predictive_signal_passed": predictive_signal,
                "best_model_id": chosen_model_id,
                "best_model_label": MODEL_LABELS[chosen_model_id],
            }
        )

    best_df = pd.DataFrame(best_rows)

    fold_df.to_csv(FOLD_METRICS_OUT, index=False)
    pred_df.to_csv(OOF_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)
    best_df.to_csv(BEST_OUT, index=False)

    make_figures(summary_df, pred_df, best_df)

    report_lines = [
        "PHASE D PUBLICATION-GROUPED MODELING REPORT",
        "===========================================",
        "",
        "Validation design:",
        "  Outer validation: publication-grouped 5-fold CV",
        "  Inner Ridge tuning: publication-grouped CV within each training fold",
        "  Random forest hyperparameters fixed before evaluation",
        "  No random row-level train/test split was used",
        "",
        "Decision rule for claiming predictive signal:",
        (
            f"  At least {MIN_IMPROVEMENT_PCT:.1f}% lower OOF MAE than the "
            "mean baseline AND positive OOF R²"
        ),
        "",
        "Endpoint results:",
    ]

    for _, row in best_df.iterrows():
        report_lines.extend(
            [
                f"",
                f"  {row['endpoint']}:",
                f"    best non-baseline model: {row['best_nonbaseline_model_label']}",
                f"    MAE improvement vs baseline: {row['mae_improvement_vs_baseline_pct']:.1f}%",
                f"    OOF R²: {row['best_nonbaseline_r2']:.3f}",
                f"    OOF Spearman rho: {row['best_nonbaseline_spearman_rho']:.3f}",
                f"    predictive-signal criterion passed: {row['predictive_signal_passed']}",
                f"    model carried forward: {row['best_model_label']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "Interpretation rule:",
            "  SHAP or feature-importance analysis should be performed only for",
            "  endpoints whose non-baseline model passes the predictive-signal",
            "  criterion above. Otherwise, report that the current features did",
            "  not demonstrate out-of-study predictive value.",
            "",
            "Important limitation:",
            "  Only a small number of ionizable-lipid structures repeat across",
            "  independent studies. Grouped validation therefore tests substantial",
            "  chemical and experimental extrapolation, which is intentionally",
            "  harder and more realistic than a random split.",
            "",
            "Saved:",
            f"  {FOLD_METRICS_OUT}",
            f"  {OOF_OUT}",
            f"  {SUMMARY_OUT}",
            f"  {BEST_OUT}",
        ]
    )

    report = "\n".join(report_lines)
    REPORT_OUT.write_text(report + "\n")
    print("\n" + report)
    print(f"  {REPORT_OUT}")


if __name__ == "__main__":
    main()
