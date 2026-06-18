"""Correlation analysis functions for numerical and categorical features.

Provides correlation matrix visualisation, top-correlation reporting,
Cramer's V for categorical pairs, and annotated bar charts.
"""

import logging
from collections import Counter
from typing import List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cramers_v(contingency_matrix: pd.DataFrame) -> float:
    """Calculate Cramer's V association statistic for two nominal variables.

    Args:
        contingency_matrix: Contingency table as a DataFrame.

    Returns:
        Cramer's V value in [0, 1].
    """
    chi2 = ss.chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1))))


def count_string_occurrences(data: List[Tuple]) -> Counter:
    """Count occurrences of the first two elements in each tuple.

    Args:
        data: List of tuples whose first two elements are counted.

    Returns:
        Counter mapping element to occurrence count.
    """
    counter: Counter = Counter()
    for item in data:
        for i in range(min(2, len(item))):
            counter[item[i]] += 1
    return counter


def print_top_correlations(
    dataset: pd.DataFrame, n: int = 5, threshold: float = 0.85
) -> Tuple[List[Tuple], Set[Tuple]]:
    """Print top *n* correlations per feature and return pairs above *threshold*.

    Args:
        dataset: Input DataFrame.
        n: Number of top correlations to display per feature.
        threshold: Absolute correlation threshold for inclusion in the returned list.

    Returns:
        Tuple of (high_correlations, added_pairs).  *high_correlations* is a list
        of ``(feature, other_feature, correlation)``.  *added_pairs* contains
        sorted tuples that have already been added (to avoid duplicates).
    """
    numerical = dataset.select_dtypes(include=np.number).columns.tolist()
    if not numerical:
        logger.warning("No numerical features found.")
        return [], set()

    corr_matrix = dataset[numerical].corr()
    high_corrs: List[Tuple] = []
    added: Set[Tuple] = set()

    for feature in numerical:
        correlations = corr_matrix[feature].drop(feature, errors="ignore")
        sorted_abs = correlations.abs().sort_values(ascending=False)
        top = sorted_abs.head(n)

        logger.info("Top %d correlations for %s:", n, feature)
        for other, _ in top.items():
            val = correlations[other]
            logger.info("  %s: %.2f", other, val)
            if abs(val) > threshold:
                pair = tuple(sorted((feature, other)))
                if pair not in added:
                    high_corrs.append((feature, other, val))
                    added.add(pair)
        logger.info("-" * 40)

    return high_corrs, added


def corr_matrix(df: pd.DataFrame) -> None:
    """Plot a heatmap of the correlation matrix.

    Args:
        df: DataFrame with numeric columns.
    """
    matrix = df.corr()
    _, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(
        matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()


def analyse_cat_cat(
    df: pd.DataFrame, alpha: float = 0.05
) -> Optional[pd.DataFrame]:
    """Run chi-squared + Cramer's V on every pair of categorical columns.

    Only pairs whose chi-squared *p*-value is below *alpha* are returned.

    Args:
        df: Input DataFrame.
        alpha: Significance threshold for the chi-squared test.

    Returns:
        DataFrame with columns ``col1``, ``col2``, ``p_value``, ``cramers_v``,
        or ``None`` if no significant associations are found.
    """
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()
    if not categorical:
        logger.warning("No categorical features found.")
        return None

    results: List[dict] = []

    for i in range(len(categorical)):
        for j in range(i + 1, len(categorical)):
            c1, c2 = categorical[i], categorical[j]
            vc1, vc2 = df[c1].value_counts(), df[c2].value_counts()
            if vc1.iloc[0] < 15 or vc2.iloc[0] < 15:
                logger.debug(
                    "Skipping %s vs. %s: mode < 15.", c1, c2
                )
                continue
            try:
                table = pd.crosstab(df[c1], df[c2])
                stat, p, dof, expected = ss.chi2_contingency(table)
                if p < alpha:
                    cv = cramers_v(table)
                    results.append(
                        {"col1": c1, "col2": c2, "p_value": p, "cramers_v": cv}
                    )
            except Exception as e:
                logger.error("Error analysing %s vs. %s: %s", c1, c2, e)

    if not results:
        logger.info("No significant associations found.")
        return None

    return pd.DataFrame(results)


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
    """Plot a bar chart of a categorical feature against a numeric target.

    An ANOVA *p*-value and top *t*-test results with Bonferroni correction are
    overlaid on the chart.

    Args:
        df: Input DataFrame.
        feature: One of the two variables (auto-detected as categorical).
        target_variable: The other variable (auto-detected as numeric).
        roundto: Decimal places for reported statistics.
        p_threshold: Significance threshold for *t*-tests.
        sig_ttest_only: If True, only annotate significant *t*-tests.
        min_group_size: Minimum group size to include in tests.
        max_t_tests: Maximum number of *t*-test results to annotate.
    """
    _, ax = plt.subplots(figsize=(10, 6))

    if pd.api.types.is_numeric_dtype(df[feature]):
        cat_col = target_variable
        num_col = feature
    else:
        cat_col = feature
        num_col = target_variable

    sns.barplot(x=cat_col, y=num_col, data=df, ci=None, ax=ax)

    groups = df[cat_col].unique()
    group_lists: List[pd.Series] = []
    valid_groups: List = []
    for g in groups:
        data = df[df[cat_col] == g][num_col]
        if len(data) >= min_group_size:
            group_lists.append(data)
            valid_groups.append(g)
        else:
            logger.info("Skipping group '%s': n < %d.", g, min_group_size)

    if len(group_lists) >= 2:
        try:
            f_stat, p_val = ss.f_oneway(*group_lists)
        except Exception as e:
            logger.error("ANOVA failed: %s", e)
            f_stat, p_val = np.nan, np.nan
    else:
        logger.info("Not enough groups with sufficient data for ANOVA.")
        f_stat, p_val = np.nan, np.nan

    ttests: List[list] = []
    n_comparisons = 0
    for i1, g1 in enumerate(valid_groups):
        for i2 in range(i1 + 1, len(valid_groups)):
            g2 = valid_groups[i2]
            list1 = df[df[cat_col] == g1][num_col]
            list2 = df[df[cat_col] == g2][num_col]
            try:
                t_stat, tp = ss.ttest_ind(list1, list2)
                ttests.append(
                    [f"{g1} - {g2}", round(t_stat, roundto), round(tp, roundto)]
                )
                n_comparisons += 1
            except Exception as e:
                logger.error("t-test error between '%s' and '%s': %s", g1, g2, e)
                ttests.append([f"{g1} - {g2}", np.nan, np.nan])

    bonferroni = p_threshold / n_comparisons if n_comparisons > 0 else np.nan

    ttests.sort(
        key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True
    )
    top_ttests = ttests[:max_t_tests]

    text = (
        f"ANOVA:\nF = {round(f_stat, roundto)}\np = {round(p_val, roundto)}\n"
    )
    text += f"Bonferroni p threshold: {round(bonferroni, roundto)}\n"

    sig_count = 0
    for ttest in top_ttests:
        if len(ttest) == 3:
            if not np.isnan(bonferroni) and ttest[2] <= bonferroni:
                text += f"\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}"
                sig_count += 1
            elif not sig_ttest_only:
                text += f"\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}"
    if sig_ttest_only and sig_count == 0:
        text += "\nNo significant t-tests"

    if df[feature].nunique() > 7:
        plt.setp(ax.get_xticklabels(), rotation=90)

    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", edgecolor="black", facecolor="white", alpha=0.8
        ),
    )
    ax.set_title(f"Bar Chart: {target_variable} by {feature}")
    ax.set_xlabel(target_variable)
    ax.set_ylabel(feature)
    plt.tight_layout()
    plt.show()
