# Architecture

## Module Dependency Diagram

```
preprocessing    features      models          analysis
┌────────────┐  ┌────────┐   ┌────────┐     ┌──────────┐
│ data_      │  │ trans- │   │ overview │   │descriptive│
│ loading    │  │ forma- │   │          │   │          │
│ cleaning   │──│ tions  │──▶│ pca_scale│──▶│ stats    │
│ encoding   │  │ corre- │   │ catboost │   │visualiza-│
│ outliers   │  │ lation │   │ _pca     │   │ tion     │
└────────────┘  └────────┘   │model_    │   └──────────┘
                             │ centric  │
                             └──────────┘
```

Dependencies flow left-to-right: raw CSV → preprocessing → features → models → analysis.

## Package Descriptions

### `src/preprocessing/` — Data Preparation

| Module | Key Functions |
|--------|--------------|
| `data_loading.py` | `check_data`, `split_data`, `split_data_cat`, `split_and_select_numeric_data` |
| `cleaning.py` | `type_definition`, `get_modalities`, `drop_low_modalities`, `clean_strings`, `find_error_col`, `keep_value_col`, `keep_unique`, `get_duplicate`, `fill_na_values`, `conditional_fill_na` |
| `encoding.py` | `onehot_encode_column`, `special_encoding`, `sum_special_encoding_columns`, `bin_categories` |
| `outliers.py` | `outlier_stat`, `remove_z_outlier`, `remove_1percent_outliers` |

### `src/features/` — Feature Engineering

| Module | Key Functions |
|--------|--------------|
| `transformations.py` | `analyze_iterative_transformations`, `apply_and_test_all_transformations`, `create_transformed_dataframe` |
| `correlation.py` | `print_top_correlations`, `cramers_v`, `corr_matrix`, `analyse_cat_cat`, `bar_chart` |

### `src/models/` — Model Training & Evaluation

| Module | Key Functions |
|--------|--------------|
| `overview.py` | `overview_models_for_df`, `overview_models_target` (14+ regression models) |
| `pca_scale.py` | `main_pca_scale`, `run_experiments` (PCA + scaling + RandomForest) |
| `catboost_pca.py` | `main_pca_scale_catboost`, `run_experiments_catboost` (PCA + scaling + CatBoost) |
| `model_centric.py` | `run_full_regression_pipeline` (RFE, SHAP, RandomForest, Excel export) |

### `src/analysis/` — Statistics & Visualization

| Module | Key Functions |
|--------|--------------|
| `descriptive.py` | `univariate_analysis`, `print_top_correlations`, `bar_chart` |
| `stats.py` | `run_ols_with_two_features`, `run_ols_with_three_features`, `cramers_v`, `analyze_categorical_pairs` |
| `visualization.py` | `plot_histogram`, `plot_bar_chart`, `plot_correlation_matrix`, `plot_na_distribution` |

## Data Flow

```
Raw CSV
  │
  ▼
preprocessing/data_loading.py  — load, validate, train/test split
preprocessing/cleaning.py      — type detection, dedup, NaN handling, string cleaning
preprocessing/encoding.py      — one-hot, binning, GFA encoding
preprocessing/outliers.py      — IQR/Z-score outlier detection & removal
  │
  ▼
features/transformations.py    — iterative power/log/scaling transforms
features/correlation.py        — numerical + categorical correlation analysis
  │
  ▼
models/overview.py             — 14+ model comparison, R²/MAE/RMSE
models/pca_scale.py            — PCA + scaling + RandomForest experiments
models/catboost_pca.py         — PCA + scaling + CatBoost experiments
models/model_centric.py        — RFE + SHAP + full pipeline → Excel export
  │
  ▼
analysis/descriptive.py        — univariate stats, distributions
analysis/stats.py              — OLS regression, categorical pair analysis
analysis/visualization.py      — histograms, bar charts, correlation heatmaps
```
