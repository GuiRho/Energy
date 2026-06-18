import logging
import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Any

from src.analysis.visualization import plot_histogram, plot_bar_chart

logger = logging.getLogger(__name__)


def type_definition(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Split DataFrame columns into numeric and categorical lists.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (numeric_column_names, categorical_column_names).
    """
    num_col: list[str] = []
    cat_col: list[str] = []

    for feat in df.columns:
        if pd.api.types.is_numeric_dtype(df[feat]):
            num_col.append(feat)
        else:
            cat_col.append(feat)

    logger.info("Numeric columns: %s", num_col)
    logger.info("Categorical columns: %s", cat_col)
    return num_col, cat_col


def get_modalities(df: pd.DataFrame) -> tuple[tuple[Any, ...], list[tuple[str, int]]]:
    """
    Find columns with few unique values and rank the rest by modality count.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (low_info_columns, ranked_columns).
        low_info_columns contains (col_name, unique_values) for cols with < 2 uniques.
        ranked_columns contains (col_name, nunique) sorted ascending.
    """
    low_info: list[tuple[str, Any]] = []
    ranked: list[tuple[str, int]] = []

    for feat in df.columns:
        unique_count = df[feat].nunique()
        if unique_count < 2:
            low_info.append((feat, df[feat].unique()))
        else:
            ranked.append((feat, unique_count))

    low_info_tuple = tuple(low_info)
    ranked_sorted = sorted(ranked, key=lambda x: x[1])

    logger.info(
        "Columns with at least 2 values: %d. Modality counts: %s",
        len(ranked),
        ranked_sorted,
    )
    logger.info(
        "Columns with < 2 unique values (%d): %s", len(low_info), low_info_tuple
    )
    return low_info_tuple, ranked_sorted


def get_duplicate(
    df: pd.DataFrame, pkey: str | list[str], keep: str = "first"
) -> int:
    """
    Count duplicate rows based on a primary key.

    Args:
        df: Input DataFrame.
        pkey: Column name or list of column names forming the primary key.
        keep: Which duplicate to keep ('first', 'last', False).

    Returns:
        Number of duplicate rows.
    """
    num_duplicates = int(df.duplicated(subset=pkey, keep=keep).sum())
    logger.info(
        "Number of duplicate rows based on primary key %s: %d", pkey, num_duplicates
    )
    return num_duplicates


def univariate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform comprehensive univariate analysis with statistics and plots.

    Computes count, missing, unique, mode, min, mean, max, std, skew,
    kurtosis, and quartiles for each column. Also plots histograms and
    bar charts.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with univariate statistics indexed by feature name.
    """
    output_df = pd.DataFrame(index=df.columns)
    output_df.index.name = "feature"
    output_df["type"] = df.dtypes
    output_df["count"] = df.count()
    output_df["missing"] = df.isna().sum()
    output_df["unique"] = df.nunique()

    try:
        output_df["mode"] = df.astype(str).mode().iloc[0]
    except Exception:
        logger.warning("Could not calculate mode for all columns.")
        output_df["mode"] = "N/A"

    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    if not numerical_cols.empty:
        numerical_stats = (
            df[numerical_cols]
            .agg(
                [
                    "min",
                    "mean",
                    "max",
                    "std",
                    "skew",
                    "kurt",
                    lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.5),
                    lambda x: x.quantile(0.75),
                ]
            )
            .T
        )
        numerical_stats.columns = [
            "min",
            "mean",
            "max",
            "std",
            "skew",
            "kurt",
            "q1",
            "median",
            "q3",
        ]
        output_df = output_df.combine_first(numerical_stats)

    plot_histogram(df)
    plot_bar_chart(df)

    return output_df


def bar_chart(
    df: pd.DataFrame,
    feature: str,
    target_variable: str,
    roundto: int = 4,
    p_threshold: float = 0.05,
    sig_ttest_only: bool = True,
    min_group_size: int = 2,
    max_t_tests: int = 5,
) -> None:
    """
    Plot a bar chart of a categorical feature against a numeric target with
    ANOVA and pairwise t-tests (Bonferroni-corrected).

    Args:
        df: Input DataFrame.
        feature: Column name (numeric or categorical).
        target_variable: Column name for the y-axis.
        roundto: Decimal places for rounding statistics.
        p_threshold: Significance threshold for ANOVA / t-tests.
        sig_ttest_only: Only display significant t-tests if True.
        min_group_size: Minimum group size to include.
        max_t_tests: Max number of t-tests to display.

    Note:
        Displays inline (intended for Jupyter notebooks).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))

    if pd.api.types.is_numeric_dtype(df[feature]):
        cat = target_variable
        num = feature
    else:
        cat = feature
        num = target_variable

    sns.barplot(x=cat, y=num, data=df, ci=None)

    groups = df[cat].unique()
    group_lists: list[pd.Series] = []
    valid_groups: list[Any] = []

    for g in groups:
        group_data = df[df[cat] == g][num]
        if len(group_data) >= min_group_size:
            group_lists.append(group_data)
            valid_groups.append(g)
        else:
            logger.info("Skipping group '%s' due to insufficient data (n < %d).", g, min_group_size)

    if len(group_lists) >= 2:
        try:
            f_stat, p_val = stats.f_oneway(*group_lists)
        except Exception as e:
            logger.error("Error during ANOVA: %s", e)
            f_stat, p_val = np.nan, np.nan
    else:
        logger.info("Not enough groups with sufficient data for ANOVA.")
        f_stat, p_val = np.nan, np.nan

    ttests: list[list[Any]] = []
    num_comparisons = 0
    for i1, g1 in enumerate(valid_groups):
        for i2 in range(i1 + 1, len(valid_groups)):
            g2 = valid_groups[i2]
            list1 = df[df[cat] == g1][num]
            list2 = df[df[cat] == g2][num]
            try:
                t_stat, tp_val = stats.ttest_ind(list1, list2)
                ttests.append([f"{g1} - {g2}", round(t_stat, roundto), round(tp_val, roundto)])
                num_comparisons += 1
            except Exception as e:
                logger.error("Error during t-test between '%s' and '%s': %s", g1, g2, e)
                ttests.append([f"{g1} - {g2}", np.nan, np.nan])

    bonferroni = p_threshold / num_comparisons if num_comparisons > 0 else np.nan

    ttests.sort(
        key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0,
        reverse=True,
    )
    top_ttests = ttests[:max_t_tests]

    textstr = f"ANOVA:\nF = {round(f_stat, roundto)}\np = {round(p_val, roundto)}\n"
    textstr += f"Bonferroni p threshold: {round(bonferroni, roundto)}\n"

    sig_ttest_count = 0
    for ttest in top_ttests:
        if len(ttest) == 3:
            if not np.isnan(bonferroni) and ttest[2] <= bonferroni:
                textstr += f"\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}"
                sig_ttest_count += 1
            elif not sig_ttest_only:
                textstr += f"\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}"

    if sig_ttest_only and sig_ttest_count == 0:
        textstr += "\nNo significant t-tests"

    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)

    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", edgecolor="black", facecolor="white", alpha=0.8
        ),
    )

    plt.title(f"Bar Chart: {target_variable} by {feature}")
    plt.xlabel(target_variable)
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


def print_top_correlations(
    dataset: pd.DataFrame, n: int = 5, threshold: float = 0.85
) -> tuple[list[tuple[str, str, float]], set[tuple[str, str]]]:
    """
    Print top N correlations for each numeric feature and return pairs
    exceeding a given threshold.

    Args:
        dataset: Input DataFrame.
        n: Number of top correlations to show per feature.
        threshold: Absolute correlation threshold for flagging high pairs.

    Returns:
        Tuple of (high_correlation_list, added_pairs_set).
        Each entry in the list is (feature, other_feature, correlation_value).
    """
    numerical_features = dataset.select_dtypes(include=np.number).columns.tolist()

    if not numerical_features:
        logger.info("No numerical features found in the dataset.")
        return [], set()

    correlation_matrix = dataset[numerical_features].corr()
    high_correlations: list[tuple[str, str, float]] = []
    added_pairs: set[tuple[str, str]] = set()

    for feature in numerical_features:
        correlations = correlation_matrix[feature].drop(feature, errors="ignore")
        abs_correlations = abs(correlations).sort_values(ascending=False)
        top_correlations = abs_correlations.head(n)

        logger.info("Top %d correlations for %s:", n, feature)
        for other_feature, _ in top_correlations.items():
            correlation = correlations[other_feature]
            logger.info("  %s: %.2f", other_feature, correlation)

            if abs(correlation) > threshold:
                pair = tuple(sorted((feature, other_feature)))
                if pair not in added_pairs:
                    high_correlations.append((feature, other_feature, correlation))
                    added_pairs.add(pair)

    return high_correlations, added_pairs
