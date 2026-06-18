import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaning import type_definition, keep_unique, get_duplicate
from src.preprocessing.data_loading import split_data
from src.preprocessing.outliers import outlier_stat
from src.preprocessing.encoding import onehot_encode_column


def test_type_definition(sample_df):
    num_col, cat_col = type_definition(sample_df)
    assert "num_feat_1" in num_col
    assert "num_feat_2" in num_col
    assert "target" in num_col
    assert "cat_feat" in cat_col


def test_type_definition_empty():
    df = pd.DataFrame()
    num_col, cat_col = type_definition(df)
    assert num_col == []
    assert cat_col == []


def test_split_data(sample_df):
    X_train, X_test, y_train, y_test, feature_names = split_data(
        sample_df, "target", test_size=0.4, random_state=42
    )
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert set(feature_names) == {"num_feat_1", "num_feat_2"}
    assert y_train.name == "target"


def test_split_data_insufficient():
    df = pd.DataFrame({"a": [1], "b": [2], "target": [10]})
    with pytest.raises(ValueError, match="Not enough samples"):
        split_data(df, "target", test_size=0.5, random_state=42)


def test_split_data_missing_target_column(sample_df):
    with pytest.raises(ValueError, match="not found"):
        split_data(sample_df, "nonexistent", test_size=0.2, random_state=42)


def test_keep_unique(sample_df):
    dups = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    result = keep_unique(dups, pkey="num_feat_1")
    assert len(result) < len(dups)
    assert result.duplicated(subset="num_feat_1").sum() == 0


def test_keep_unique_all_duplicates():
    df = pd.DataFrame({"id": [1, 1, 1], "val": [10, 20, 30]})
    result = keep_unique(df, pkey="id", keep="first")
    assert len(result) == 1


def test_outlier_stat(sample_df):
    result = outlier_stat(
        sample_df,
        num_col=["num_feat_1", "num_feat_2"],
        cat_col=["cat_feat"],
    )
    assert "num_feat_1" in result.index
    assert "cat_feat" in result.index
    assert "IQR_OUT_NB" in result.columns
    assert "Z_OUT_NB" in result.columns
    assert "mode" in result.columns
    assert result.loc["cat_feat", "mode"] in ("A", "B", "C")


def test_outlier_stat_detects_outlier():
    df = pd.DataFrame({"a": [1.0, 2.0, 1.5, 1.8, 1000.0]})
    result = outlier_stat(df, num_col=["a"], cat_col=[])
    assert result.loc["a", "IQR_OUT_NB"] >= 1


def test_onehot_encode_column(sample_df):
    result = onehot_encode_column(sample_df, "cat_feat")
    assert "cat_feat" not in result.columns
    for val in ["A", "B", "C"]:
        col = f"cat_feat_{val}"
        assert col in result.columns
        assert result[col].dtype in (np.float64, np.int64, bool)


def test_onehot_encode_column_preserves_index(sample_df):
    result = onehot_encode_column(sample_df, "cat_feat")
    assert list(result.index) == list(sample_df.index)
    assert "num_feat_1" in result.columns


def test_get_duplicate(sample_df):
    dups = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    n = get_duplicate(dups, pkey="num_feat_1")
    assert n >= 1


def test_get_duplicate_no_dups(sample_df):
    n = get_duplicate(sample_df, pkey="num_feat_1")
    assert n == 0
