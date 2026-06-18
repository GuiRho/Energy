import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_na_distribution(
    distrib_na_row: pd.Series | None = None,
    totalna_row: int | None = None,
) -> None:
    """
    Plot bar chart of NaN distribution per row.

    Displays how many rows contain a given number of NaN values.

    Args:
        distrib_na_row: Series with NaN count as index and row count as values.
        totalna_row: Total number of NaN values in the DataFrame (kept for API compat).

    Note:
        Displays inline (intended for Jupyter notebooks).
    """
    if distrib_na_row is None:
        logger.error(
            "distrib_na_row is not defined. Run check_na_values_rows() first."
        )
        return

    plt.figure(figsize=(14, 6))
    bars = plt.bar(distrib_na_row.index, distrib_na_row.values, color="blue")
    plt.xlabel("Number of NaN Values per Row")
    plt.ylabel("Number of Rows")
    plt.title("Distribution of NA Values per Row")
    plt.xticks(distrib_na_row.index)
    plt.ylim(0, distrib_na_row.values.max() * 1.1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            int(yval),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot a heatmap of the correlation matrix for numeric columns.

    Args:
        df: Input DataFrame.

    Note:
        Displays inline (intended for Jupyter notebooks).
    """
    numeric_df = df.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(18, 14))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.7},
    )
    plt.title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_histogram(dataset: pd.DataFrame) -> None:
    """
    Plot histograms with KDE for all numeric columns in the dataset.

    Args:
        dataset: Input DataFrame.

    Note:
        Displays inline (intended for Jupyter notebooks).
    """
    numerical_features = dataset.select_dtypes(include=np.number).columns.tolist()

    if not numerical_features:
        logger.info("No numerical features found in the dataset.")
        return

    for feature in numerical_features:
        plt.figure(figsize=(12, 6))
        ax = sns.histplot(
            dataset[feature], bins=30, kde=True, color="blue"
        )

        total = len(dataset[feature])
        max_height = 0

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                percentage = "{:.1f}%".format(100 * height / total)
                x = p.get_x() + p.get_width() / 2
                y = height
                ax.text(x, y + max_height * 0.01, percentage, ha="center", va="bottom")
            if height > max_height:
                max_height = height

        plt.title(f"Histogram of {feature}", fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Number of Rows", fontsize=14)
        ax.set_ylim(0, max_height * 1.05)
        plt.show()


def plot_bar_chart(dataset: pd.DataFrame) -> None:
    """
    Plot bar charts for categorical features showing top 10 value counts.

    Args:
        dataset: Input DataFrame.

    Note:
        Displays inline (intended for Jupyter notebooks).
    """
    categorical_features = dataset.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if not categorical_features:
        logger.info("No categorical features found in the dataset.")
        return

    for feature in categorical_features:
        value_counts = dataset[feature].value_counts().nlargest(10)
        total = len(dataset[feature])
        max_value = value_counts.max()

        if max_value < 20:
            logger.info(
                "Skipping feature '%s' because its max count (%d) < 20.",
                feature,
                max_value,
            )
            continue

        x = np.arange(len(value_counts))
        truncated_labels = [str(label)[:25] for label in value_counts.index]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(x, value_counts.values)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(f"Bar Graph of Top 10 Values for {feature}")
        plt.xticks(x, truncated_labels, rotation=45, ha="right")

        for bar in bars:
            yval = bar.get_height()
            percentage = "{:.1f}%".format(100 * yval / total)
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + max_value * 0.01,
                f"{int(yval)}\n({percentage})",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.show()
