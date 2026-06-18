"""Preprocessing utilities for the energy analysis pipeline."""

from .cleaning import (
    clean_strings,
    conditional_fill_na,
    drop_low_modalities,
    fill_na_values,
    find_error_col,
    get_duplicate,
    get_modalities,
    keep_unique,
    keep_value_col,
    type_definition,
)
from .data_loading import (
    check_data,
    split_and_select_numeric_data,
    split_data,
    split_data_cat,
)
from .encoding import (
    bin_categories,
    onehot_encode_column,
    special_encoding,
    sum_special_encoding_columns,
)
from .outliers import (
    outlier_stat,
    remove_1percent_outliers,
    remove_z_outlier,
)

__all__ = [
    "check_data",
    "split_data",
    "split_data_cat",
    "split_and_select_numeric_data",
    "type_definition",
    "get_modalities",
    "drop_low_modalities",
    "clean_strings",
    "find_error_col",
    "keep_value_col",
    "keep_unique",
    "get_duplicate",
    "fill_na_values",
    "conditional_fill_na",
    "outlier_stat",
    "remove_z_outlier",
    "remove_1percent_outliers",
    "special_encoding",
    "sum_special_encoding_columns",
    "onehot_encode_column",
    "bin_categories",
]

__version__ = "0.1.0"
