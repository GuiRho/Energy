import numpy as np
import pandas as pd
import pytest

from src.features.correlation import print_top_correlations, cramers_v


def test_print_top_correlations(sample_df):
    high_corrs, added = print_top_correlations(sample_df, n=3, threshold=0.0)
    assert len(high_corrs) > 0
    assert len(added) > 0
    for item in high_corrs:
        assert len(item) == 3


def test_print_top_correlations_no_numeric():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    high_corrs, added = print_top_correlations(df)
    assert high_corrs == []
    assert added == set()


def test_print_top_correlations_high_threshold(sample_df):
    high_corrs, added = print_top_correlations(sample_df, n=5, threshold=1.0)
    assert len(high_corrs) == 0


def test_cramers_v_independent(cat_correlated_df):
    ct = pd.crosstab(cat_correlated_df["cat3"], cat_correlated_df["cat1"])
    v = cramers_v(ct)
    assert 0.0 <= v <= 0.6


def test_cramers_v_identical():
    df = pd.DataFrame({
        "a": ["x", "x", "x", "y", "y", "y", "z", "z", "z"],
        "b": ["x", "x", "x", "y", "y", "y", "z", "z", "z"],
    })
    ct = pd.crosstab(df["a"], df["b"])
    v = cramers_v(ct)
    assert v == pytest.approx(1.0, abs=0.01)


def test_cramers_v_min_correlation():
    df = pd.DataFrame({
        "a": ["x", "x", "x", "y", "y", "y", "z", "z", "z"],
        "b": ["a", "a", "a", "b", "b", "c", "c", "c", "c"],
    })
    ct = pd.crosstab(df["a"], df["b"])
    v = cramers_v(ct)
    assert 0.0 <= v <= 1.0


def test_print_top_correlations_returns_pairs(sample_df):
    _, added = print_top_correlations(sample_df, n=2, threshold=0.0)
    for pair in added:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert pair[0] < pair[1]
