"""Outlier detection and removal utilities.

Primary export: outlier_stat
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def outlier_stat(
    df: pd.DataFrame,
    num_col: List[str],
    cat_col: List[str],
) -> pd.DataFrame:
    """Compute outlier statistics and mode information for columns.

    For numeric columns: calculates IQR bounds, Z-score bounds, and
    counts of outliers using both methods.
    For categorical columns: calculates mode and its occurrence count.

    Args:
        df: Input DataFrame.
        num_col: List of numeric column names.
        cat_col: List of categorical column names.

    Returns:
        DataFrame with outlier statistics indexed by feature name.
    """
    output_df = pd.DataFrame(index=df.columns)
    output_df.index.name = "feature"

    for feat in num_col:
        try:
            q1 = df[feat].quantile(0.25)
            q3 = df[feat].quantile(0.75)
            iqr = q3 - q1
            outlier_max_iqr = q3 + 1.5 * iqr
            outlier_min_iqr = max(q1 - 1.5 * iqr, 0)
            mean_val = df[feat].mean()
            std_val = df[feat].std()
            outlier_max_zscore = mean_val + 3 * std_val
            outlier_min_zscore = mean_val - 3 * std_val

            iqr_outliers = df[
                (df[feat] < outlier_min_iqr) | (df[feat] > outlier_max_iqr)
            ][feat].count()
            output_df.loc[feat, 'IQR_OUT_NB'] = iqr_outliers

            zscore_outliers = df[
                (df[feat] < outlier_min_zscore) | (df[feat] > outlier_max_zscore)
            ][feat].count()
            output_df.loc[feat, 'Z_OUT_NB'] = zscore_outliers

            output_df.loc[feat, 'outlier_max_iqr'] = outlier_max_iqr
            output_df.loc[feat, 'outlier_min_iqr'] = outlier_min_iqr
            output_df.loc[feat, 'outlier_max_zscore'] = outlier_max_zscore
            output_df.loc[feat, 'outlier_min_zscore'] = outlier_min_zscore

        except Exception as e:
            logger.error("Error calculating outlier statistics for %s: %s", feat, e)
            for col_name in [
                'outlier_max_iqr', 'outlier_min_iqr', 'outlier_max_zscore',
                'outlier_min_zscore', 'IQR_OUT_NB', 'Z_OUT_NB',
            ]:
                output_df.loc[feat, col_name] = "Error"

    for feat in cat_col:
        try:
            mode_series = df[feat].mode()
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
                mode_count = int(df[feat].value_counts().get(mode_value, 0))
                output_df.loc[feat, 'mode'] = mode_value
                output_df.loc[feat, 'mode_occurrence'] = mode_count
        except Exception as e:
            logger.error("Error calculating mode for %s: %s", feat, e)
            output_df.loc[feat, 'mode'] = "Error"
            output_df.loc[feat, 'mode_occurrence'] = "Error"

    numeric_index = [col for col in output_df.index if col not in cat_col]
    categorical_index = cat_col
    output_df = output_df.reindex(index=numeric_index + categorical_index)
    return output_df


def remove_z_outlier(df: pd.DataFrame, num_col: List[str]) -> pd.DataFrame:
    """Remove rows with Z-score outliers (|z| > 3) across numeric columns.

    Collects all outlier indices across all specified numeric columns,
    deduplicates them, and drops the corresponding rows.

    Args:
        df: Input DataFrame.
        num_col: List of numeric column names to check.

    Returns:
        DataFrame with outlier rows removed.
    """
    all_outlier_indices: List[int] = []

    for feat in num_col:
        mean_val = df[feat].mean()
        std_val = df[feat].std()
        outlier_max_zscore = mean_val + 3 * std_val
        outlier_min_zscore = mean_val - 3 * std_val

        outlier_mask = (
            (df[feat] < outlier_min_zscore) | (df[feat] > outlier_max_zscore)
        )
        outlier_indices = df[outlier_mask].index.tolist()
        all_outlier_indices.extend(outlier_indices)

    unique_outlier_indices = list(set(all_outlier_indices))
    logger.info("Number of total Z-score outliers: %d", len(unique_outlier_indices))

    result = df.drop(index=unique_outlier_indices)
    return result


def remove_1percent_outliers(df: pd.DataFrame, num_col: List[str]) -> pd.DataFrame:
    """Remove rows with values above the 99th percentile.

    For each numeric column, identifies rows exceeding the 99th
    percentile, collects all unique indices, and drops them.

    Args:
        df: Input DataFrame.
        num_col: List of numeric column names to check.

    Returns:
        DataFrame with top 1% outlier rows removed.
    """
    all_outlier_indices: List[int] = []

    for feat in num_col:
        top_99_val = df[feat].quantile(0.99)
        outlier_mask = df[feat] > top_99_val
        outlier_indices = df[outlier_mask].index.tolist()
        all_outlier_indices.extend(outlier_indices)

    unique_outlier_indices = list(set(all_outlier_indices))
    logger.info("Number of total 1%% outliers: %d", len(unique_outlier_indices))

    result = df.drop(index=unique_outlier_indices)
    return result
