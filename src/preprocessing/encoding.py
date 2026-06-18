"""Categorical encoding utilities including one-hot, GFA, and binning.

Primary export: onehot_encode_column
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def special_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Create GFA percentage encoding for property use type columns.

    For each of LargestPropertyUseType, SecondLargestPropertyUseType,
    and ThirdLargestPropertyUseType, creates new columns with the
    corresponding %_GFA value for each modality.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with added GFA percentage columns.
    """
    result = df.copy()
    property_cols = [
        "LargestPropertyUseType",
        "SecondLargestPropertyUseType",
        "ThirdLargestPropertyUseType",
    ]
    suffix = "%_GFA"

    for col in property_cols:
        modalities = result[col].dropna().unique()
        value_col = f"{suffix}_{col.replace('PropertyUseType', '_Use')}"

        for name in modalities:
            final_name = f"{col}_{name}_{suffix}"
            mask = result[col] == name
            result[final_name] = 0
            result.loc[mask, final_name] = result.loc[mask, value_col]

    return result


def sum_special_encoding_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GFA percentage columns by modality.

    Sums the columns created by special_encoding, grouping them by the
    modality name (middle part of the column name).

    Args:
        df: Input DataFrame with special encoding columns.

    Returns:
        DataFrame with additional aggregated total columns.
    """
    result = df.copy()

    special_cols = [
        col for col in result.columns
        if "%_GFA" in col and any(
            use_type in col
            for use_type in [
                "LargestPropertyUseType",
                "SecondLargestPropertyUseType",
                "ThirdLargestPropertyUseType",
            ]
        )
    ]

    modalities: set = set()
    for col in special_cols:
        parts = col.split("_")
        if len(parts) > 2:
            modalities.add(parts[1])

    for modality in modalities:
        cols_to_sum = [col for col in special_cols if f"_{modality}_" in col]
        if cols_to_sum:
            result[f"{modality}_Total_%_GFA"] = result[cols_to_sum].sum(axis=1)
        else:
            logger.warning("No columns found to sum for modality: %s", modality)

    return result


def onehot_encode_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """One-hot encode a single categorical column.

    Creates dummy variables for the specified column, appends them to
    the DataFrame, and drops the original column.

    Args:
        df: Input DataFrame.
        column_name: Name of the column to encode.

    Returns:
        DataFrame with the column one-hot encoded.
    """
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(df[[column_name]])
    transformed_data = encoder.transform(df[[column_name]])

    feature_names = encoder.get_feature_names_out([column_name])
    encoded_df = pd.DataFrame(
        transformed_data,
        columns=feature_names,
        index=df.index,
    )

    result = pd.concat([df, encoded_df], axis=1)
    result = result.drop(columns=[column_name])
    return result


def bin_categories(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    cutoff: float = 0.007,
    replace_with: str = 'other_grouped',
) -> pd.DataFrame:
    """Bin rare categories in categorical columns.

    Groups categories that appear less than `cutoff` fraction of rows
    into a single label.

    Args:
        df: Input DataFrame.
        features: List of categorical column names to bin. If None,
            defaults to all non-numeric columns.
        cutoff: Frequency threshold below which to bin (default 0.007).
        replace_with: Label for binned categories (default 'other_grouped').

    Returns:
        DataFrame with rare categories binned.
    """
    result = df.copy()
    if features is None:
        features = [
            col for col in result.columns
            if not pd.api.types.is_numeric_dtype(result[col])
        ]

    for feat in features:
        if feat not in result.columns:
            logger.warning("'%s' not found in DataFrame. No binning performed.", feat)
            continue
        if pd.api.types.is_numeric_dtype(result[feat]):
            continue
        value_counts = result[feat].value_counts()
        other_list = value_counts[value_counts / len(result) < cutoff].index
        result.loc[result[feat].isin(other_list), feat] = replace_with

    return result
