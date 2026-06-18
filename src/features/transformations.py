"""Feature transformation functions for iterative correlation improvement.

Provides power, log, and scaling transformations with iterative
application to maximize feature-target correlation.
"""

import logging
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _calculate_correlation(
    transformed_series: pd.Series, target_series: pd.Series
) -> Optional[float]:
    """Calculate Pearson correlation between two series with safety guards."""
    if transformed_series.isnull().all() or target_series.isnull().all():
        return np.nan
    if transformed_series.nunique() <= 1:
        return np.nan
    try:
        cleaned = transformed_series.replace([np.inf, -np.inf], np.nan)
        corr = cleaned.corr(target_series, method="pearson")
        return float(corr) if pd.notna(corr) else np.nan
    except Exception:
        return np.nan


def _apply_power_transformation(series: pd.Series, power: float) -> pd.Series:
    """Apply a power transformation, handling non-positive and zero values."""
    if not isinstance(series, pd.Series):
        return pd.Series([np.nan] * len(series))
    result = series.copy()
    try:
        if power < 0:
            non_zero = result != 0
            if non_zero.any():
                result.where(~non_zero, result[non_zero] ** power, inplace=True)
                result.update(pd.Series(np.nan, index=result.index[~non_zero]))
            else:
                result[:] = np.nan
        elif 0 < power < 1:
            positive = result > 0
            if positive.any():
                result.where(~positive, result[positive] ** power, inplace=True)
                result.update(pd.Series(np.nan, index=result.index[~positive]))
            else:
                result[:] = np.nan
        else:
            with np.errstate(over="ignore"):
                result = result ** power
                result = result.replace([np.inf, -np.inf], np.nan)
    except (TypeError, ValueError):
        result = pd.Series(np.nan, index=series.index)
    return result


def _apply_log_transformation(series: pd.Series, log_func: Callable) -> pd.Series:
    """Apply a logarithmic transformation, handling non-positive values."""
    if not isinstance(series, pd.Series):
        return pd.Series([np.nan] * len(series))
    result = series.copy()
    try:
        positive = result > 0
        if positive.any():
            result.where(~positive, log_func(result[positive]), inplace=True)
            result.update(pd.Series(np.nan, index=result.index[~positive]))
        else:
            result[:] = np.nan
    except (TypeError, ValueError):
        result = pd.Series(np.nan, index=series.index)
    return result


def _apply_scaling_transformation(
    series: pd.Series,
    scaler_instance: Union[StandardScaler, RobustScaler, PowerTransformer],
) -> pd.Series:
    """Fit a scaler on the series and return the transformed values."""
    if not isinstance(series, pd.Series):
        return pd.Series([np.nan] * len(series))
    col_data = series.values.reshape(-1, 1)
    if pd.isna(col_data).all():
        return pd.Series(np.nan, index=series.index)
    if np.nanstd(col_data.astype(float)) == 0 and isinstance(
        scaler_instance, (StandardScaler, PowerTransformer)
    ):
        return pd.Series(np.nan, index=series.index)
    try:
        scaler_instance.fit(col_data)
        scaled = scaler_instance.transform(col_data)
        return pd.Series(scaled.flatten(), index=series.index)
    except (ValueError, TypeError):
        return pd.Series(np.nan, index=series.index)
    except Exception:
        return pd.Series(np.nan, index=series.index)


TRANSFORMATIONS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "power_0_25": lambda s: _apply_power_transformation(s, 0.25),
    "power_0_33": lambda s: _apply_power_transformation(s, 1 / 3),
    "power_0_50": lambda s: _apply_power_transformation(s, 0.5),
    "power_2_00": lambda s: _apply_power_transformation(s, 2),
    "power_3_00": lambda s: _apply_power_transformation(s, 5),
    "power_5_00": lambda s: _apply_power_transformation(s, 5),
    "power_8_00": lambda s: _apply_power_transformation(s, 8),
    "log2": lambda s: _apply_log_transformation(s, np.log2),
    "log10": lambda s: _apply_log_transformation(s, np.log10),
    "standard_scale": lambda s: _apply_scaling_transformation(
        s, StandardScaler()
    ),
    "robust_scale": lambda s: _apply_scaling_transformation(
        s, RobustScaler()
    ),
    "yeo_johnson": lambda s: _apply_scaling_transformation(
        s, PowerTransformer(method="yeo-johnson", standardize=True)
    ),
}


def apply_and_test_all_transformations(
    current_series: pd.Series,
    target_series: pd.Series,
    transformations: Dict[str, Callable[[pd.Series], pd.Series]],
) -> Dict[str, float]:
    """Apply all transformations and return a dict mapping name to correlation.

    Args:
        current_series: The feature series to transform.
        target_series: The target series for correlation.
        transformations: Dict of transformation name to callable.

    Returns:
        Dict mapping transformation name to correlation value (NaN on failure).
    """
    results: Dict[str, float] = {}
    for name, func in transformations.items():
        transformed = func(current_series)
        results[name] = _calculate_correlation(transformed, target_series)
    return results


def analyze_iterative_transformations(
    df: pd.DataFrame,
    target: str,
    max_turns: int = 5,
    min_improvement: float = 5e-4,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[None, None]]:
    """Iteratively apply the best transformation to maximise correlation with target.

    For each numeric feature, at each turn the transformation yielding the
    highest absolute correlation is applied.  Iteration stops when no feature
    improves by more than *min_improvement*.

    Args:
        df: Input DataFrame.
        target: Name of the target column.
        max_turns: Maximum transformation iterations per feature.
        min_improvement: Minimum absolute correlation increase to continue.

    Returns:
        Tuple of (summary_df, history_df).  *summary_df* has columns
        ``Initial Correlation``, ``Final Correlation``, ``Num Transformations``
        indexed by feature.  *history_df* records the sequence of transforms
        and correlations per feature.  Returns (None, None) on invalid input.
    """
    if not isinstance(df, pd.DataFrame):
        logger.error("Input 'df' must be a pandas DataFrame.")
        return None, None
    if target not in df.columns:
        logger.error("Target column '%s' not found in DataFrame.", target)
        return None, None
    if not pd.api.types.is_numeric_dtype(df[target]):
        logger.error("Target column '%s' must be numeric.", target)
        return None, None
    if df[target].isnull().all():
        logger.error("Target column '%s' contains only NaN values.", target)
        return None, None

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)
    if not numerical_cols:
        logger.error("No numerical features found (excluding target).")
        return None, None

    target_series = df[target].copy()
    valid_idx = target_series.dropna().index
    df_numeric = df[numerical_cols].copy().loc[valid_idx]
    target_series = target_series.loc[valid_idx]
    if df_numeric.empty:
        logger.error("No valid data after removing NaN targets.")
        return None, None

    logger.info(
        "Starting iterative transformation analysis for %d features...",
        len(numerical_cols),
    )

    feature_states: Dict[str, dict] = {}
    feature_history: Dict[str, list] = {}

    for col in numerical_cols:
        original = df_numeric[col]
        initial_corr = _calculate_correlation(original, target_series)
        if pd.isna(initial_corr):
            initial_corr = 0.0
        feature_states[col] = {
            "current_series": original,
            "best_corr_abs": abs(initial_corr),
            "best_corr_signed": initial_corr,
            "stopped": False,
        }
        feature_history[col] = [("original", initial_corr)]

    for turn in range(1, max_turns + 1):
        logger.info("--- Turn %d/%d ---", turn, max_turns)
        improved = 0

        for col in numerical_cols:
            state = feature_states[col]
            if state["stopped"]:
                continue

            current = state["current_series"]
            turn_corrs = apply_and_test_all_transformations(
                current, target_series, TRANSFORMATIONS
            )

            best_turn_name: Optional[str] = None
            best_turn_signed = np.nan
            best_turn_abs = -1.0

            for t_name, corr_val in turn_corrs.items():
                if pd.notna(corr_val):
                    ca = abs(corr_val)
                    if ca > best_turn_abs:
                        best_turn_abs = ca
                        best_turn_signed = corr_val
                        best_turn_name = t_name

            if (
                best_turn_name is not None
                and best_turn_abs > state["best_corr_abs"] + min_improvement
            ):
                improved += 1
                logger.info(
                    "  Feature '%s': corr %.4f -> %.4f via '%s'",
                    col,
                    state["best_corr_signed"],
                    best_turn_signed,
                    best_turn_name,
                )
                next_series = TRANSFORMATIONS[best_turn_name](current)
                state["current_series"] = next_series
                state["best_corr_abs"] = best_turn_abs
                state["best_corr_signed"] = best_turn_signed
                feature_history[col].append(
                    (best_turn_name, best_turn_signed)
                )
            else:
                state["stopped"] = True

        if improved == 0:
            logger.info(
                "No features improved in turn %d. Stopping early.", turn
            )
            break

    logger.info("Iterative analysis complete.")

    summary_data = []
    for col in numerical_cols:
        init_c = feature_history[col][0][1]
        final_c = feature_states[col]["best_corr_signed"]
        n_trans = len(feature_history[col]) - 1
        summary_data.append([col, init_c, final_c, n_trans])

    summary_df = pd.DataFrame(
        summary_data,
        columns=[
            "Feature",
            "Initial Correlation",
            "Final Correlation",
            "Num Transformations",
        ],
    ).set_index("Feature")

    history_cols: List[str] = []
    for i in range(1, max_turns + 1):
        history_cols.extend([f"Transform_{i}", f"Corr_{i}"])

    history_data: Dict[str, dict] = {col: {} for col in numerical_cols}
    for col in numerical_cols:
        hist = feature_history[col]
        for i in range(1, max_turns + 1):
            tc = f"Transform_{i}"
            cc = f"Corr_{i}"
            if i < len(hist):
                history_data[col][tc] = hist[i][0]
                history_data[col][cc] = hist[i][1]
            else:
                history_data[col][tc] = None
                history_data[col][cc] = np.nan

    history_df = pd.DataFrame.from_dict(history_data, orient="index")
    history_df = history_df.reindex(columns=history_cols)
    return summary_df, history_df


def _apply_named_transformation(
    series: pd.Series, transform_name: Optional[str]
) -> pd.Series:
    """Apply a single named transformation to a series.

    Args:
        series: Input series.
        transform_name: Name of the transformation (key in TRANSFORMATIONS),
            ``"original"``, or ``None``.

    Returns:
        Transformed series, or the original if the name is invalid.
    """
    if transform_name in (None, "original") or not isinstance(
        transform_name, str
    ):
        return series
    func = TRANSFORMATIONS.get(transform_name)
    if func is None:
        logger.warning("Transformation '%s' not found. Skipping.", transform_name)
        return series
    return func(series)


def create_transformed_dataframe(
    df: pd.DataFrame, history_results: pd.DataFrame
) -> pd.DataFrame:
    """Apply transformations recorded in *history_results* to a copy of *df*.

    Args:
        df: Original DataFrame.
        history_results: DataFrame output from
            :func:`analyze_iterative_transformations` (index = features,
            columns = ``Transform_1``, ``Corr_1``, …).

    Returns:
        DataFrame with transformed feature columns.  Returns an empty DataFrame
        on error.
    """
    df_transformed = df.copy()
    features = history_results.index.tolist()

    try:
        for feature in features:
            if feature not in df_transformed.columns:
                continue
            current = df_transformed[feature].copy()
            for i in range(1, 6):
                tc = f"Transform_{i}"
                if tc not in history_results.columns:
                    break
                t_name = history_results.loc[feature, tc]
                current = _apply_named_transformation(current, t_name)
            df_transformed[feature] = current
    except KeyError as e:
        logger.error("KeyError during transformation: %s", e)
        return pd.DataFrame()
    except Exception as e:
        logger.error("Unexpected error during transformation: %s", e)
        return pd.DataFrame()

    return df_transformed
