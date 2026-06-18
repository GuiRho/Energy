import numpy as np
import pandas as pd
import pytest

from src.models.overview import overview_models_for_df, _build_model_dict


def test_overview_models_for_df_returns_dataframe(model_features):
    result = overview_models_for_df(model_features)
    assert isinstance(result, pd.DataFrame)
    assert "Model" in result.columns
    assert "R2 (Test)" in result.columns
    assert "MAE (Test)" in result.columns
    assert "RMSE (Test)" in result.columns
    assert len(result) > 0


def test_overview_models_for_df_sorted_by_r2(model_features):
    result = overview_models_for_df(model_features)
    r2_vals = result["R2 (Test)"].values
    for i in range(len(r2_vals) - 1):
        assert r2_vals[i] >= r2_vals[i + 1] or np.isnan(r2_vals[i + 1])


def test_overview_models_columns_are_correct(model_features):
    result = overview_models_for_df(model_features)
    expected = {"Model", "R2 (Test)", "MAE (Test)", "RMSE (Test)"}
    assert expected.issubset(set(result.columns))


def test_build_model_dict_has_core_models():
    models = _build_model_dict()
    core = {"LinearRegression", "Ridge", "Lasso", "DecisionTree", "RandomForest"}
    assert core.issubset(set(models.keys()))


def test_build_model_dict_all_instances():
    models = _build_model_dict()
    for name, instance in models.items():
        assert hasattr(instance, "fit")
        assert hasattr(instance, "predict")


def test_overview_models_with_constant_features():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "f1": [1.0] * 30,
        "f2": rng.normal(0, 1, 30),
        "SiteEUIWN(kBtu/sf)": rng.normal(50, 5, 30),
    })
    result = overview_models_for_df(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_overview_models_with_many_features():
    rng = np.random.default_rng(0)
    data = {f"f{i}": rng.normal(0, 1, 60) for i in range(10)}
    data["SiteEUIWN(kBtu/sf)"] = rng.normal(50, 5, 60)
    df = pd.DataFrame(data)
    result = overview_models_for_df(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) >= 11
