"""Overview of multiple regression models for energy efficiency prediction."""

import logging
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    import xgboost as xgb

    XGB_INSTALLED = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    XGB_INSTALLED = False

try:
    import lightgbm as lgb

    LGB_INSTALLED = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    LGB_INSTALLED = False

try:
    import catboost as cb

    CB_INSTALLED = True
except ImportError:
    cb = None  # type: ignore[assignment]
    CB_INSTALLED = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _build_model_dict() -> dict:
    """Build dictionary of regression models to evaluate.

    Returns:
        dict: Mapping of model names to sklearn regressor instances.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.1, max_iter=2000, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(
            alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_STATE
        ),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=3),
        "SVR_linear": SVR(kernel="linear", C=1.0, cache_size=500),
        "SVR_rbf": SVR(kernel="rbf", C=1.0, gamma="scale", cache_size=500),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=100, learning_rate=1.0, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE
        ),
    }
    if XGB_INSTALLED:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.15,
            max_depth=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    if LGB_INSTALLED:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.15,
            max_depth=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    if CB_INSTALLED:
        models["CatBoost"] = cb.CatBoostRegressor(
            iterations=100,
            learning_rate=0.15,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=0,
            thread_count=-1,
        )
    return models


def _evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict,
) -> pd.DataFrame:
    """Evaluate all models and return sorted results DataFrame.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        models: Dictionary of model name to regressor instance.

    Returns:
        DataFrame sorted by R2 descending.
    """
    results_list: list[dict] = []
    for model_name, model_instance in models.items():
        logger.info("Evaluating %s...", model_name)
        try:
            start = time.time()
            model_instance.fit(X_train, y_train)
            fit_time = time.time() - start

            start = time.time()
            y_pred = model_instance.predict(X_test)
            predict_time = time.time() - start

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results_list.append(
                {
                    "Model": model_name,
                    "R2 (Test)": r2,
                    "MAE (Test)": mae,
                    "RMSE (Test)": rmse,
                    "Fit Time (s)": fit_time,
                    "Predict Time (s)": predict_time,
                    "Notes": "",
                }
            )
        except Exception as exc:
            logger.error("ERROR evaluating %s: %s", model_name, exc)
            results_list.append(
                {
                    "Model": model_name,
                    "R2 (Test)": np.nan,
                    "MAE (Test)": np.nan,
                    "RMSE (Test)": np.nan,
                    "Fit Time (s)": np.nan,
                    "Predict Time (s)": np.nan,
                    "Notes": f"Error: {exc}",
                }
            )

    results_df = pd.DataFrame(results_list)
    results_df_sorted = results_df.sort_values(
        by="R2 (Test)", ascending=False
    ).reset_index(drop=True)
    return results_df_sorted


def _prepare_data(
    df: pd.DataFrame, target: str, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split and scale data for model evaluation.

    Args:
        df: Input DataFrame.
        target: Target column name.
        test_size: Proportion for test split.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df_clean = df.copy()
    y = df_clean[target]
    X = df_clean.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def overview_models_for_df(
    df: pd.DataFrame,
    target: str = "SiteEUIWN(kBtu/sf)",
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Evaluate 14+ regression models on a default target.

    Args:
        df: Input DataFrame with features and target columns.
        target: Name of the target column.
        test_size: Proportion of data to hold out for testing.

    Returns:
        DataFrame of model performance metrics sorted by R2.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    models = _build_model_dict()
    X_train, X_test, y_train, y_test = _prepare_data(df, target, test_size)
    results = _evaluate_models(X_train, X_test, y_train, y_test, models)

    logger.info("Model evaluation complete. Top model: %s", results.iloc[0]["Model"])
    return results


def overview_models_target(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
) -> pd.DataFrame:
    """Evaluate regression models on a specified target column.

    Args:
        df: Input DataFrame with features and target columns.
        target: Name of the target column to predict.
        test_size: Proportion of data to hold out for testing.

    Returns:
        DataFrame of model performance metrics sorted by R2.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    models = _build_model_dict()
    X_train, X_test, y_train, y_test = _prepare_data(df, target, test_size)
    results = _evaluate_models(X_train, X_test, y_train, y_test, models)

    logger.info("Model evaluation complete. Top model: %s", results.iloc[0]["Model"])
    return results
