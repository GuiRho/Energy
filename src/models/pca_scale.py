"""PCA, scaling, and RandomForest pipeline experiments."""

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_data(df: pd.DataFrame, target_column: str) -> None:
    """Validate that the target column exists in the DataFrame.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.

    Raises:
        ValueError: If target_column is not found.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the dataframe."
        )
    if df.isnull().sum().sum() > 0:
        logger.warning("Data contains missing values. Consider imputation.")


def split_data(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Split DataFrame into train/test sets using only numeric features.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        test_size: Proportion for test split.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, original_feature_names).
    """
    logger.info("Splitting data (test_size=%s, random_state=%s)", test_size, random_state)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric.")

    df_cleaned = df.dropna(subset=[target_column]).copy()
    dropped = len(df) - len(df_cleaned)
    if dropped:
        logger.warning("Dropped %d rows with missing target values.", dropped)
    if df_cleaned.empty:
        raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
        raise ValueError("Not enough samples after handling missing target values.")

    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]
    X_numeric = X.select_dtypes(include=np.number)
    original_feature_names = list(X_numeric.columns)

    logger.info("Features used (numeric only): %d", X_numeric.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Split: X_train=%s, X_test=%s", X_train.shape, X_test.shape
    )
    return X_train, X_test, y_train, y_test, original_feature_names


def run_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scalers: dict[str, Any],
    pca_options: list[Optional[int]],
    rf_params: dict[str, Any],
    n_top_results_to_show: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """Run scaling + PCA + RandomForest experiment combinations.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        scalers: Dictionary of scaler name to scaler instance (or None).
        pca_options: List of PCA n_components values (includes None).
        rf_params: Parameters for RandomForestRegressor.
        n_top_results_to_show: Number of top results to display.
        random_state: Random seed for reproducibility.

    Returns:
        List of result dictionaries per experiment.
    """
    results: list[dict[str, Any]] = []
    start_time = time.time()
    logger.info("Starting experiments...")

    for scaler_name, scaler in scalers.items():
        logger.info("Testing scaler: %s", scaler_name)
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        if scaler is not None:
            try:
                X_train_scaled = scaler.fit_transform(X_train_scaled)
                X_test_scaled = scaler.transform(X_test_scaled)
            except Exception as exc:
                logger.error("Scaling failed for %s: %s", scaler_name, exc)
                continue

        current_n_features = X_train_scaled.shape[1]

        for n_components in pca_options:
            experiment_name = (
                f"Scaler: {scaler_name}, "
                f"PCA: {n_components if n_components is not None else 'None'}"
            )

            if n_components is not None and n_components > current_n_features:
                continue

            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled

            if n_components is not None:
                pca_instance = PCA(n_components=n_components, random_state=random_state)
                try:
                    X_train_processed = pca_instance.fit_transform(X_train_processed)
                    X_test_processed = pca_instance.transform(X_test_processed)
                except Exception as exc:
                    logger.error("%s - PCA error: %s", experiment_name, exc)
                    results.append(
                        {
                            "Scaler": scaler_name,
                            "PCA_Components": n_components,
                            "R2_Score": np.nan,
                            "RMSE": np.nan,
                        }
                    )
                    continue

            try:
                model = RandomForestRegressor(**rf_params)
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results.append(
                    {
                        "Scaler": scaler_name,
                        "PCA_Components": "None"
                        if n_components is None
                        else n_components,
                        "R2_Score": r2,
                        "RMSE": rmse,
                    }
                )
                logger.info(
                    "%s - R2: %.4f, RMSE: %.4f", experiment_name, r2, rmse
                )
            except Exception as exc:
                logger.error("%s - Model error: %s", experiment_name, exc)
                results.append(
                    {
                        "Scaler": scaler_name,
                        "PCA_Components": "None"
                        if n_components is None
                        else n_components,
                        "R2_Score": np.nan,
                        "RMSE": np.nan,
                    }
                )

    elapsed = time.time() - start_time
    logger.info("Experiments finished in %.2f seconds.", elapsed)
    return results


def process_and_display_results(
    results: list[dict[str, Any]],
    n_top_results_to_show: int,
    output_file: str,
) -> None:
    """Sort, display, and save experiment results.

    Args:
        results: List of result dictionaries.
        n_top_results_to_show: Number of top results to display.
        output_file: Path to save results (xlsx or csv).
    """
    if not results:
        logger.warning("No results generated.")
        return

    results_df = pd.DataFrame(results)
    results_df_cleaned = results_df.dropna(subset=["R2_Score", "RMSE"])
    results_df_sorted = results_df_cleaned.sort_values(
        by="R2_Score", ascending=False
    ).reset_index(drop=True)

    logger.info(
        "Top %d results by R2:\n%s",
        n_top_results_to_show,
        results_df_sorted.head(n_top_results_to_show).to_string(),
    )

    try:
        results_df.to_excel(output_file, index=False, engine="openpyxl")
        logger.info("Results saved to '%s'", output_file)
    except Exception:
        logger.warning("Failed to save as xlsx, trying CSV.")
        try:
            csv_output = output_file.replace(".xlsx", ".csv")
            if csv_output == output_file:
                csv_output = output_file + ".csv"
            results_df.to_csv(csv_output, index=False)
            logger.info("Results saved to '%s'", csv_output)
        except Exception as e_csv:
            logger.error("Failed to save CSV: %s", e_csv)


def main_pca_scale(
    df: pd.DataFrame,
    acp_min: int,
    target: str,
    test_size: float,
    random_state: int,
    output_file: str,
    rf_params: dict[str, Any],
    n_top_results_to_show: int,
) -> None:
    """Orchestrate data splitting, scaling, PCA, and RandomForest experiments.

    Args:
        df: Input DataFrame.
        acp_min: Minimum number of PCA components to test.
        target: Name of the target column.
        test_size: Proportion for test split.
        random_state: Random seed.
        output_file: Path to save results.
        rf_params: Parameters for RandomForestRegressor.
        n_top_results_to_show: Number of top results to display.
    """
    logger.info("Starting main_pca_scale process for target '%s'...", target)

    try:
        check_data(df, target)
        X_train, X_test, y_train, y_test, _ = split_data(
            df, target, test_size, random_state
        )

        scalers: dict[str, Any] = {
            "None": None,
            "Standard": StandardScaler(),
            "Robust": RobustScaler(),
            "MinMax(1-100)": MinMaxScaler(feature_range=(1, 100)),
        }

        max_pca_components = X_train.shape[1]
        if max_pca_components == 0:
            raise ValueError("No numeric features available for PCA.")

        acp_min_adjusted = acp_min
        if acp_min > max_pca_components:
            logger.warning(
                "acp_min (%d) > features (%d), adjusting.", acp_min, max_pca_components
            )
            acp_min_adjusted = max_pca_components
        elif acp_min <= 0:
            logger.warning("acp_min <= 0, adjusting to 1.")
            acp_min_adjusted = 1

        if acp_min_adjusted > max_pca_components:
            pca_range: list[int] = []
        else:
            pca_range = list(range(acp_min_adjusted, max_pca_components + 1))

        pca_options: list[Optional[int]] = [None] + pca_range

        results = run_experiments(
            X_train,
            X_test,
            y_train,
            y_test,
            scalers,
            pca_options,
            rf_params,
            n_top_results_to_show,
            random_state,
        )

        process_and_display_results(results, n_top_results_to_show, output_file)

    except ValueError as ve:
        logger.error("Configuration error: %s", ve)
    except Exception as exc:
        logger.error("Unexpected error: %s", exc)

    logger.info("main_pca_scale process finished.")
