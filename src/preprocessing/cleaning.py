"""Column type detection, string cleaning, deduplication, and NaN handling.

Primary export: type_definition
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def type_definition(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Categorize DataFrame columns into numeric and categorical.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (numeric_columns, categorical_columns).
    """
    num_col: List[str] = []
    cat_col: List[str] = []

    for feat in df.columns:
        if pd.api.types.is_numeric_dtype(df[feat]):
            num_col.append(feat)
        else:
            cat_col.append(feat)

    logger.info(
        "Numeric columns: %d | Categorical columns: %d", len(num_col), len(cat_col),
    )
    return num_col, cat_col


def get_modalities(
    df: pd.DataFrame,
) -> Tuple[Tuple[Tuple[str, Any], ...], List[Tuple[str, int]]]:
    """Find columns with low modality (few unique values).

    Separates columns into low-information (<2 unique values) and
    ranked columns by number of unique values.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of:
            - Low-information columns as ((name, unique_values), ...).
            - Ranked columns as [(name, nunique), ...] sorted ascending.
    """
    info_from_unique_column: List[Tuple[str, Any]] = []
    rank: List[Tuple[str, int]] = []

    for feat in df.columns:
        nunique = df[feat].nunique()
        if nunique < 2:
            info_from_unique_column.append((feat, df[feat].unique().tolist()))
        else:
            rank.append((feat, nunique))

    info_tuple = tuple(info_from_unique_column)
    rank_sorted = sorted(rank, key=lambda x: x[1])

    logger.info(
        "Columns with >=2 values: %d | Low-information columns: %d",
        len(rank_sorted), len(info_tuple),
    )
    return info_tuple, rank_sorted


def drop_low_modalities(
    df: pd.DataFrame,
    low_info_col: Tuple[Tuple[str, Any], ...],
) -> pd.DataFrame:
    """Drop low-information columns identified by get_modalities.

    Args:
        df: Input DataFrame.
        low_info_col: Tuple of (column_name, unique_values) from get_modalities.

    Returns:
        DataFrame with low-information columns removed.
    """
    cols_to_drop = [item[0] for item in low_info_col]
    result = df.drop(columns=cols_to_drop, errors='ignore')
    logger.info(
        "Shape: %s -> %s (dropped %d columns)",
        df.shape, result.shape, len(cols_to_drop),
    )
    return result


def clean_strings(df: pd.DataFrame, cat_col: List[str]) -> pd.DataFrame:
    """Clean string columns: strip whitespace, lowercase, remove parentheses.

    Applies the following to each column in cat_col:
        - Converts to string
        - Strips whitespace after splitting on commas
        - Lowercases
        - Removes parenthesized content

    Args:
        df: Input DataFrame.
        cat_col: List of categorical column names to clean.

    Returns:
        New DataFrame with cleaned string columns.
    """
    result = df.copy()
    for col in cat_col:
        if col not in result.columns:
            continue
        cleaned = (
            result[col]
            .astype(str)
            .apply(lambda x: ','.join(s.strip() for s in x.split(',')).lower())
            .str.replace(r'\(.*?\)', '', regex=True)
            .str.strip()
        )
        result[col] = cleaned
    return result


def find_error_col(
    df: pd.DataFrame,
    col: str,
    errors: List[Any],
) -> pd.DataFrame:
    """Drop rows where column values are in the error list.

    Args:
        df: Input DataFrame.
        col: Column name to check.
        errors: List of error values to remove.

    Returns:
        DataFrame with matching rows removed.
    """
    error_mask = df[col].isin(errors)
    result = df.drop(index=df[error_mask].index)
    logger.info(
        "Shape: %s -> %s (dropped %d error rows)",
        df.shape, result.shape, error_mask.sum(),
    )
    return result


def keep_value_col(
    df: pd.DataFrame,
    col: str,
    values: List[Any],
) -> pd.DataFrame:
    """Keep only rows where column value is in the specified list.

    Args:
        df: Input DataFrame.
        col: Column name to filter on.
        values: List of values to keep.

    Returns:
        Filtered DataFrame.
    """
    result = df[df[col].isin(values)].copy()
    logger.info("Shape: %s -> %s", df.shape, result.shape)
    return result


def keep_unique(
    df: pd.DataFrame,
    pkey: Union[str, List[str]],
    keep: str = 'first',
) -> pd.DataFrame:
    """Deduplicate DataFrame based on a primary key.

    Args:
        df: Input DataFrame.
        pkey: Column name or list of column names for the primary key.
        keep: Which duplicate to keep ('first', 'last', False).

    Returns:
        DataFrame with duplicates removed.
    """
    num_duplicates = df.duplicated(subset=pkey, keep=keep).sum()
    logger.info("Duplicate rows based on primary key %s: %d", pkey, num_duplicates)

    dup_mask = df.duplicated(subset=pkey, keep=keep)
    result = df[~dup_mask].copy()
    logger.info("Unique shape: %s", result.shape)
    return result


def get_duplicate(
    df: pd.DataFrame,
    pkey: Union[str, List[str]],
    keep: str = 'first',
) -> int:
    """Count duplicate rows based on a primary key.

    Args:
        df: Input DataFrame.
        pkey: Column name or list for the primary key.
        keep: Which duplicate to consider ('first', 'last', False).

    Returns:
        Number of duplicate rows.
    """
    num_duplicates = int(df.duplicated(subset=pkey, keep=keep).sum())
    logger.info("Duplicate rows based on primary key %s: %d", pkey, num_duplicates)
    return num_duplicates


def fill_na_values(
    df: pd.DataFrame,
    na_filling_rules: Dict[str, callable],
) -> pd.DataFrame:
    """Fill NaN values using specified rules per column.

    Args:
        df: Input DataFrame.
        na_filling_rules: Dict mapping column names to callables that
            accept the DataFrame and return fill values.

    Returns:
        DataFrame with NaN values filled.
    """
    result = df.copy()
    for feature, rule in na_filling_rules.items():
        if feature in result.columns:
            fill_values = rule(result)
            result[feature] = result[feature].fillna(fill_values)
    return result


def conditional_fill_na(
    df: pd.DataFrame,
    rules: Dict[str, List[Tuple[callable, Any]]],
) -> pd.DataFrame:
    """Conditionally fill NaN values based on dynamic rules.

    For each column, applies a list of (condition, fill_value) pairs.
    A condition is a callable that receives the DataFrame and returns
    a boolean Series.

    Args:
        df: Input DataFrame.
        rules: Dict mapping column names to lists of
            (condition_fn, fill_value) pairs.

    Returns:
        DataFrame with conditional NaN fills applied.
    """
    result = df.copy()
    for col, rule_list in rules.items():
        for condition_fn, fill_value in rule_list:
            na_mask = result[col].isna()
            cond_mask = condition_fn(result)
            mask = na_mask & cond_mask
            result[col] = np.where(mask, fill_value, result[col])
    return result
