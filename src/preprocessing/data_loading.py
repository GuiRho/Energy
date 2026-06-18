"""Data loading and splitting utilities for regression modeling.

Provides functions for basic data validation and train/test splitting
with optional imputation and feature selection.

Primary export: split_and_select_numeric_data
"""

import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def check_data(df: pd.DataFrame, target_column: str) -> None:
    """Basic data checks before modeling.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.

    Raises:
        ValueError: If target_column is not found in df.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the dataframe.",
        )
    if df.isnull().sum().sum() > 0:
        logger.warning(
            "Data contains missing values. Consider imputation before running.",
        )


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Split DataFrame into train/test sets with numeric features only.

    Drops rows with missing target values and selects only numeric
    features for modeling. Does not impute feature NaNs.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column (must be numeric).
        test_size: Proportion of data for the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).

    Raises:
        ValueError: If target column is missing, non-numeric, or no
            data remains after cleaning.
    """
    logger.info(
        "Splitting data (test_size=%s, random_state=%s)", test_size, random_state,
    )

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric for regression.")

    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_column])
    if len(df_cleaned) < initial_rows:
        logger.warning("Dropped %d rows with missing target values.", initial_rows - len(df_cleaned))
    if df_cleaned.empty:
        raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
        raise ValueError("Not enough samples left to perform a split.")

    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]

    X_numeric = X.select_dtypes(include=np.number)
    original_feature_names = X_numeric.columns.tolist()

    logger.info("Features used for modeling (numeric only): %d", X_numeric.shape[1])
    if X_numeric.shape[1] == 0:
        raise ValueError(
            "No numeric features found after dropping target and selecting numeric types.",
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state,
    )
    logger.info("Split: X_train %s, X_test %s", X_train.shape, X_test.shape)

    return X_train, X_test, y_train, y_test, original_feature_names


def split_data_cat(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Split data with median imputation for CatBoost pipelines.

    Drops rows with missing target, selects numeric features, and
    imputes remaining feature NaNs using the training set median.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        test_size: Proportion for the test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).

    Raises:
        ValueError: If target is missing, non-numeric, or no data/features remain.
    """
    logger.info("Splitting data for CatBoost (target='%s')", target_column)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric for regression.")

    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_column])
    if len(df_cleaned) < initial_rows:
        logger.warning(
            "Dropped %d rows with missing target values.", initial_rows - len(df_cleaned),
        )
    if df_cleaned.empty:
        raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
        raise ValueError("Not enough samples left after handling missing target values.")

    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]

    X_numeric = X.select_dtypes(include=np.number)
    feature_names = X_numeric.columns.tolist()

    if not feature_names:
        raise ValueError("No numeric features found after dropping target.")

    logger.info("Features used for modeling (numeric only): %d", X_numeric.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state,
    )

    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        logger.warning("NaNs found in numeric features. Imputing with median from training data.")
        imputation_values = X_train.median()
        X_train = X_train.fillna(imputation_values)
        X_test = X_test.fillna(imputation_values)

    logger.info(
        "Split and imputed: X_train %s, X_test %s", X_train.shape, X_test.shape,
    )
    return X_train, X_test, y_train, y_test, feature_names


def split_and_select_numeric_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    List[str],
    pd.Index,
    pd.Index,
]:
    """Split data, select numeric features, and impute with median.

    Performs input validation, drops rows with missing target, selects
    only numeric features, splits into train/test, and imputes remaining
    NaNs using the training set median.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        test_size: Proportion for the test split (default 0.2).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Tuple containing:
            - X_train: Training features (numeric, imputed).
            - X_test: Testing features (numeric, imputed).
            - y_train: Training target.
            - y_test: Testing target.
            - initial_numeric_feature_names: Names of original numeric features.
            - test_original_indices: Original index of test samples.
            - train_original_indices: Original index of train samples.

    Raises:
        TypeError: If input is not a DataFrame.
        ValueError: If target is missing, non-numeric, or data is insufficient.
    """
    logger.info("Preparing and splitting data (target='%s')", target_col)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric for regression.")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_col])
    if len(df_cleaned) < initial_rows:
        logger.warning(
            "Dropped %d rows with missing target values.", initial_rows - len(df_cleaned),
        )
    if df_cleaned.empty:
        raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
        raise ValueError("Not enough samples left to perform a split.")

    X = df_cleaned.drop(target_col, axis=1)
    y = df_cleaned[target_col]

    X_numeric = X.select_dtypes(include=np.number)
    initial_numeric_feature_names = X_numeric.columns.tolist()

    if not initial_numeric_feature_names:
        raise ValueError("No numeric features found after dropping target.")

    logger.info(
        "Total features: %d, Using numeric: %d",
        X.shape[1], len(initial_numeric_feature_names),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state,
    )

    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        logger.warning("NaNs found in numeric features. Imputing with median from training data.")
        imputation_values = X_train.median()
        X_train = X_train.fillna(imputation_values)
        X_test = X_test.fillna(imputation_values)

    test_original_indices = X_test.index.copy()
    train_original_indices = X_train.index.copy()

    logger.info(
        "Train: X=%s, y=%s | Test: X=%s, y=%s",
        X_train.shape, y_train.shape, X_test.shape, y_test.shape,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        initial_numeric_feature_names,
        test_original_indices,
        train_original_indices,
    )
