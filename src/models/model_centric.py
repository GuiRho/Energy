"""Model-centric regression analysis pipeline with SHAP and RFE."""

import logging
import time
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import shap

    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False

logger = logging.getLogger(__name__)


def split_and_select_numeric_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], pd.Index, pd.Index]:
    """Split data; keep numeric features; median impute NaNs."""
    logger.info("Preparing data (target='%s')", target_col)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError("Target must be numeric for regression.")

    df_c = df.dropna(subset=[target_col]).copy()
    if len(df_c) < 2:
        raise ValueError("Not enough data after cleaning.")
    if dropped := len(df) - len(df_c):
        logger.warning("Dropped %d rows with missing target.", dropped)

    X, y = df_c.drop(target_col, axis=1), df_c[target_col]
    X_num = X.select_dtypes(include=np.number)
    fnames = list(X_num.columns)
    if not fnames:
        raise ValueError("No numeric features found.")

    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=test_size, random_state=random_state)

    if X_tr.isnull().sum().sum() > 0 or X_te.isnull().sum().sum() > 0:
        logger.warning("Imputing NaNs with training median.")
        imp_vals = X_tr.median()
        X_tr, X_te = X_tr.fillna(imp_vals), X_te.fillna(imp_vals)

    return X_tr, X_te, y_tr, y_te, fnames, X_te.index, X_tr.index


def apply_preprocessing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str,
    pca_n_comp: Optional[int],
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], Any, Any]:
    """Apply scaling and PCA to features."""
    X_tr_s, X_te_s = X_train.copy(), X_test.copy()
    orig_names = list(X_train.columns)
    scaler: Any = None

    if scaler_type and scaler_type.lower() != "none":
        scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}
        s = scalers.get(scaler_type.lower())
        if s is None:
            raise ValueError(f"Unknown scaler: {scaler_type}")
        scaler = s
        X_tr_np = scaler.fit_transform(X_tr_s)
        X_te_np = scaler.transform(X_te_s)
    else:
        X_tr_np = X_tr_s.values if isinstance(X_tr_s, pd.DataFrame) else X_tr_s
        X_te_np = X_te_s.values if isinstance(X_te_s, pd.DataFrame) else X_te_s

    Xp_tr, Xp_te, fnames, pca = X_tr_np, X_te_np, orig_names, None

    if pca_n_comp is not None:
        max_feat = Xp_tr.shape[1]
        if max_feat == 0:
            raise ValueError("0 features for PCA.")
        valid = (isinstance(pca_n_comp, int) and 0 < pca_n_comp <= max_feat) or (
            isinstance(pca_n_comp, float) and 0 < pca_n_comp <= 1.0
        )
        if not valid:
            raise ValueError(f"Invalid pca_n_comp: {pca_n_comp}")
        pca = PCA(n_components=pca_n_comp, random_state=random_state)
        Xp_tr = pca.fit_transform(Xp_tr)
        Xp_te = pca.transform(Xp_te)
        fnames = [f"PC{i}" for i in range(Xp_tr.shape[1])]

    return Xp_tr, Xp_te, fnames, scaler, pca


def perform_rfe(
    X_train: np.ndarray,
    y_train: pd.Series,
    feature_names: list[str],
    target_feature_ratio: float = 0.6,
    random_state: int = 42,
    rfe_estimator_params: Optional[dict[str, Any]] = None,
) -> tuple[Any, list[str], np.ndarray, np.ndarray]:
    """Recursive Feature Elimination with RandomForest."""
    n_init = X_train.shape[1]
    should = target_feature_ratio is not None and 0 < target_feature_ratio < 1.0 and n_init > 0
    if not should:
        return None, feature_names, X_train, np.ones(n_init, dtype=bool)

    n_sel = max(1, int(n_init * target_feature_ratio))
    params = dict(rfe_estimator_params) if rfe_estimator_params else {"n_estimators": 150, "max_depth": 10, "n_jobs": -1}
    params.pop("random_state", None)
    rfe = RFE(estimator=RandomForestRegressor(random_state=random_state, **params), n_features_to_select=min(n_sel, n_init), step=1)
    try:
        rfe.fit(X_train, y_train)
    except Exception as exc:
        logger.error("RFE failed: %s", exc)
        return None, feature_names, X_train, np.ones(n_init, dtype=bool)

    mask = rfe.support_
    if mask.sum() == 0:
        return None, feature_names, X_train, np.ones(n_init, dtype=bool)

    sel_names = [n for n, s in zip(feature_names, mask) if s] if len(feature_names) == n_init else [f"S_{i}" for i in range(mask.sum())]
    return rfe, sel_names, rfe.transform(X_train), mask


def train_random_forest_regressor(
    X_train: np.ndarray,
    y_train: pd.Series,
    random_state: int,
    rf_params: Optional[dict[str, Any]] = None,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor."""
    if X_train.shape[1] == 0 or len(X_train) == 0:
        raise ValueError("Cannot train with 0 features or samples.")
    params = {"n_estimators": 150, "max_depth": 10, "n_jobs": -1}
    if rf_params:
        params.update(rf_params)
    params["random_state"] = random_state
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_regression_performance(
    model: RandomForestRegressor,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """Calculate train/test regression metrics."""
    if X_test.shape[0] == 0 or X_train.shape[1] == 0:
        return {"Train": {}, "Test": {}}
    yp_tr, yp_te = model.predict(X_train), model.predict(X_test)
    return {
        "Train": {"R²": r2_score(y_train, yp_tr), "MAE": mean_absolute_error(y_train, yp_tr), "RMSE": float(np.sqrt(mean_squared_error(y_train, yp_tr)))},
        "Test": {"R²": r2_score(y_test, yp_te), "MAE": mean_absolute_error(y_test, yp_te), "RMSE": float(np.sqrt(mean_squared_error(y_test, yp_te)))},
    }


def analyze_regression_residuals(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    y_test: pd.Series,
    df_original: pd.DataFrame,
    test_original_indices: pd.Index,
    n_worst: int = 20,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Analyze residuals and identify worst predictions."""
    if X_test.shape[0] == 0 or y_test.empty:
        return pd.Series(dtype=float), pd.DataFrame(), np.array([])

    y_pred = model.predict(X_test)
    residuals = pd.Series(y_test.values - y_pred, index=test_original_indices, name="Residual")

    worst = pd.DataFrame()
    if n_worst > 0 and not residuals.empty:
        abs_r = residuals.abs().sort_values(ascending=False)
        idx = abs_r.index[: min(n_worst, len(residuals))]
        yps = pd.Series(y_pred, index=test_original_indices, name="Predicted_Target")
        worst = pd.DataFrame({"Original_Index": idx, "Actual_Target": y_test.loc[idx].values, "Predicted_Target": yps.loc[idx].values, "Residual": residuals.loc[idx].values, "Absolute_Residual": abs_r.loc[idx].values})

    return residuals, worst, y_pred


def analyze_rf_feature_importance_mdi(
    model: RandomForestRegressor,
    feature_names: list[str],
) -> tuple[pd.DataFrame, Optional[float]]:
    """MDI feature importance with Lorenz curve Gini."""
    if not feature_names or not hasattr(model, "feature_importances_"):
        return pd.DataFrame(), None
    imp = model.feature_importances_
    feat = list(feature_names)
    if len(imp) > len(feat):
        feat.extend([f"U_{i}" for i in range(len(feat), len(imp))])
    elif len(imp) < len(feat):
        feat = feat[: len(imp)]

    imp_df = pd.DataFrame({"Feature": feat, "Importance_MDI": imp}).sort_values("Importance_MDI", ascending=False).reset_index(drop=True)
    vals = np.maximum(imp_df["Importance_MDI"].values, 0)
    total = np.sum(vals)
    gini: Optional[float] = None
    if not np.isclose(total, 0):
        cum = np.cumsum(vals) / total
        gini = float(np.clip(1 - 2 * np.trapz(np.insert(cum, 0, 0), np.linspace(0, 1, len(vals) + 1)), 0, 1))
        imp_df["Cumulative_MDI_Importance_Normalized"] = cum.tolist()
    return imp_df, gini


def analyze_shap_values(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    feature_names: list[str],
    test_original_indices: pd.Index,
) -> tuple[Any, Any, Any]:
    """Calculate SHAP values (TreeExplainer)."""
    if not SHAP_INSTALLED or X_test.shape[1] == 0 or X_test.shape[0] == 0:
        return None, None, None

    X_df = pd.DataFrame(X_test, columns=feature_names, index=test_original_indices)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_df)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = float(ev[0])

    sv_df = pd.DataFrame(sv, columns=feature_names, index=test_original_indices)
    ss_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": np.abs(sv).mean(axis=0)}).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    return sv_df, ss_df, ev


def save_results_to_excel(results: dict[str, Any], filename: str = "rf_regression_analysis_results.xlsx") -> None:
    """Save results dictionary to multi-sheet Excel file."""
    if not filename:
        return
    if not filename.endswith((".xlsx", ".xls")):
        filename += ".xlsx"

    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            ri = results.get("run_info")
            if isinstance(ri, dict):
                cleaned = {k: (str(v) if isinstance(v, (dict, list, tuple, set, BaseEstimator, np.ndarray)) else v) for k, v in ri.items()}
                pd.DataFrame(list(cleaned.items()), columns=["Parameter", "Value"]).to_excel(writer, sheet_name="Run Info", index=False)

            m = results.get("metrics")
            if isinstance(m, dict):
                pd.DataFrame(m).reset_index().rename(columns={"index": "Metric"}).melt(id_vars="Metric", var_name="Set", value_name="Value").to_excel(writer, sheet_name="Metrics", index=False)

            for key, sheet in [("mdi_importance_df", "MDI Importance"), ("shap_summary_df", "SHAP Importance Summary")]:
                v = results.get(key)
                if isinstance(v, pd.DataFrame) and not v.empty:
                    v.to_excel(writer, sheet_name=sheet, index=False)

            sv = results.get("shap_values_df")
            if isinstance(sv, pd.DataFrame) and not sv.empty:
                local = sv.copy()
                local.columns = ["SHAP_" + c for c in local.columns]
                col_order = []
                if "y_test" in results:
                    local["Actual_Target"], local = results["y_test"].align(local, join="right", axis=0)
                    col_order.append("Actual_Target")
                if isinstance(results.get("y_test_pred"), np.ndarray):
                    local["Predicted_Target"] = pd.Series(results["y_test_pred"], index=local.index)
                    col_order.append("Predicted_Target")
                if isinstance(results.get("residuals"), pd.Series):
                    local["Residual"], local = results["residuals"].align(local, join="right", axis=0)
                    col_order.append("Residual")
                shc = [c for c in local.columns if c.startswith("SHAP_")]
                local[col_order + shc].to_excel(writer, sheet_name="SHAP Values (Test Set)", index=True, index_label="Original_Index")

            wr = results.get("worst_residuals_df")
            if isinstance(wr, pd.DataFrame) and not wr.empty:
                wr.copy().to_excel(writer, sheet_name="Worst Residuals", index=False)
        logger.info("Results saved to %s", filename)
    except Exception as exc:
        logger.error("Error saving Excel: %s", exc)


def run_full_regression_pipeline(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "Standard",
    pca_n_comp: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    target_feature_ratio: float = 0.6,
    rfe_estimator_params: Optional[dict[str, Any]] = None,
    n_worst_residuals: int = 20,
    rf_params: Optional[dict[str, Any]] = None,
    output_filename: str = "rf_regression_analysis_results.xlsx",
    run_shap_analysis: bool = True,
) -> Optional[dict[str, Any]]:
    """Run the complete regression analysis pipeline with RFE, SHAP, and Excel export."""
    logger.info("Starting pipeline (target='%s')", target_col)
    start_ts = time.time()
    results: dict[str, Any] = {"run_info": {"target_column": target_col, "test_size": test_size, "random_state": random_state, "scaler_type": scaler_type, "pca_n_comp": pca_n_comp, "target_feature_ratio_rfe": target_feature_ratio, "rf_params": rf_params or "Defaults"}}

    try:
        X_tr, X_te, y_tr, y_te, fnames, tidx, _ = split_and_select_numeric_data(df, target_col, test_size, random_state)
        results.update({"y_train": y_tr, "y_test": y_te, "test_original_indices": tidx})

        Xp_tr, Xp_te, fp, _, _ = apply_preprocessing(X_tr, X_te, scaler_type, pca_n_comp, random_state)
        results.update({"X_train_processed": Xp_tr, "X_test_processed": Xp_te, "feature_names_after_processing": fp})

        rfe_obj, sn, Xf_tr, _ = perform_rfe(Xp_tr, y_tr, fp, target_feature_ratio, random_state, rfe_estimator_params)
        Xf_te, ff = Xp_te, fp
        if rfe_obj is not None and Xf_tr.shape[1] > 0 and Xf_tr.shape[1] < Xp_tr.shape[1]:
            Xf_te, ff = rfe_obj.transform(Xp_te), sn

        model = train_random_forest_regressor(Xf_tr, y_tr, random_state, rf_params)
        results.update({"model": model, "selected_feature_names_final": ff})
        results["metrics"] = evaluate_regression_performance(model, Xf_tr, y_tr, Xf_te, y_te)

        resid, wdf, yp = analyze_regression_residuals(model, Xf_te, y_te, df, tidx, n_worst_residuals)
        results.update({"residuals": resid, "worst_residuals_df": wdf, "y_test_pred": yp})

        imp_df, gini = analyze_rf_feature_importance_mdi(model, ff)
        results["mdi_importance_df"] = imp_df
        results["run_info"]["MDI_Feature_Importance_Gini"] = gini

        if run_shap_analysis and SHAP_INSTALLED:
            sv_df, ss_df, ev = analyze_shap_values(model, Xf_te, ff, tidx)
            results.update({"shap_values_df": sv_df, "shap_summary_df": ss_df, "shap_expected_value": ev})

        save_results_to_excel(results, output_filename)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        traceback.print_exc()
        return None

    elapsed = time.time() - start_ts
    results["run_info"]["total_duration_seconds"] = round(elapsed, 2)
    logger.info("Pipeline finished in %.2f seconds.", elapsed)
    return results
