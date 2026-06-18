import logging
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import scipy.stats as ss
from typing import Any

logger = logging.getLogger(__name__)


def run_ols_with_two_features(
    df: pd.DataFrame, target_col: str, feat_col: list[str]
) -> list[dict[str, Any]]:
    """
    Run OLS regression for every 2-feature combination.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        feat_col: List of candidate feature column names.

    Returns:
        List of dicts with keys 'feature1', 'feature2', 'r_squared'.
    """
    feature_names = [
        col
        for col in feat_col
        if col != target_col and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(feature_names) < 2:
        logger.error("Need at least two numerical features (excluding target) to run combinations.")
        return []

    results: list[dict[str, Any]] = []
    for feat1, feat2 in itertools.combinations(feature_names, 2):
        try:
            X = df[[feat1, feat2]]
            X = sm.add_constant(X)
            y = df[target_col]
            model = sm.OLS(y, X).fit()
            results.append(
                {
                    "feature1": feat1,
                    "feature2": feat2,
                    "r_squared": model.rsquared,
                }
            )
        except Exception as e:
            logger.error("Error during OLS regression for %s, %s: %s", feat1, feat2, e)

    return results


def run_ols_with_three_features(
    df: pd.DataFrame, target_col: str, feat_col: list[str]
) -> list[dict[str, Any]]:
    """
    Run OLS regression for every 3-feature combination.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.
        feat_col: List of candidate feature column names.

    Returns:
        List of dicts with keys 'feature1', 'feature2', 'feature3', 'r_squared'.
    """
    feature_names = [
        col
        for col in feat_col
        if col != target_col and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(feature_names) < 3:
        logger.error(
            "Need at least three numerical features (excluding target) to run combinations."
        )
        return []

    results: list[dict[str, Any]] = []
    for feat1, feat2, feat3 in itertools.combinations(feature_names, 3):
        try:
            X = df[[feat1, feat2, feat3]]
            X = sm.add_constant(X)
            y = df[target_col]
            model = sm.OLS(y, X).fit()
            results.append(
                {
                    "feature1": feat1,
                    "feature2": feat2,
                    "feature3": feat3,
                    "r_squared": model.rsquared,
                }
            )
        except Exception as e:
            logger.error(
                "Error during OLS regression for %s, %s, %s: %s",
                feat1,
                feat2,
                feat3,
                e,
            )

    return results


def compare_r_squared_two(results: list[dict[str, Any]]) -> None:
    """
    Display top 2-feature combinations sorted by R-squared.

    Args:
        results: List of dicts with keys 'feature1', 'feature2', 'r_squared'.
    """
    sorted_results = sorted(results, key=lambda x: x["r_squared"], reverse=True)
    for i, res in enumerate(sorted_results):
        logger.info(
            "Result %d: Features=(%s, %s), R-squared = %.4f",
            i,
            res["feature1"],
            res["feature2"],
            res["r_squared"],
        )


def compare_r_squared_three(results: list[dict[str, Any]]) -> None:
    """
    Display top 3-feature combinations sorted by R-squared.

    Args:
        results: List of dicts with keys 'feature1', 'feature2', 'feature3', 'r_squared'.
    """
    sorted_results = sorted(results, key=lambda x: x["r_squared"], reverse=True)
    for i, res in enumerate(sorted_results):
        logger.info(
            "Result %d: Features=(%s, %s, %s), R-squared = %.4f",
            i,
            res["feature1"],
            res["feature2"],
            res["feature3"],
            res["r_squared"],
        )


def cramers_v(contingency_matrix: pd.DataFrame) -> float:
    """
    Calculate Cramer's V statistic for association between two nominal variables.

    Args:
        contingency_matrix: Contingency table as a DataFrame.

    Returns:
        Cramer's V value.
    """
    chi2 = ss.chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))))


def analyze_categorical_pairs(
    df: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame | None:
    """
    Perform chi-squared analysis on all pairs of categorical columns.

    For each pair with a significant p-value, calculates Cramer's V.

    Args:
        df: Input DataFrame.
        alpha: Significance threshold for p-value.

    Returns:
        DataFrame with columns 'col1', 'col2', 'p_value', 'cramers_v',
        or None if no valid pairs found.
    """
    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    if not categorical_columns:
        logger.info("No categorical features found in the dataset.")
        return None

    results: list[dict[str, Any]] = []

    for i in range(len(categorical_columns)):
        for j in range(i + 1, len(categorical_columns)):
            col1 = categorical_columns[i]
            col2 = categorical_columns[j]

            if (
                df[col1].value_counts().iloc[0] < 15
                or df[col2].value_counts().iloc[0] < 15
            ):
                logger.info(
                    "Skipping %s vs. %s: one or both modes occur < 15 times.",
                    col1,
                    col2,
                )
                continue

            try:
                contingency_table = pd.crosstab(df[col1], df[col2])
                stat, p, dof, expected = ss.chi2_contingency(contingency_table)

                if p < alpha:
                    cramers_v_value = cramers_v(contingency_table)
                    results.append(
                        {
                            "col1": col1,
                            "col2": col2,
                            "p_value": p,
                            "cramers_v": cramers_v_value,
                        }
                    )
            except Exception as e:
                logger.error(
                    "Error during analysis of %s vs. %s: %s", col1, col2, e
                )

    if not results:
        logger.info("No significant associations found.")
        return None

    return pd.DataFrame(results)
