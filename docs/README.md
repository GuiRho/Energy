# Energy — Building Energy Use Intensity Prediction

Predict Energy Use Intensity (EUI) and GHG emissions for commercial buildings
using regression models (Random Forest, CatBoost, XGBoost, Linear, SVM, etc.).
The full pipeline covers preprocessing, feature engineering, model comparison,
and analysis.

## Directory Structure

```
energy/
├── src/
│   ├── preprocessing/   # Data loading, cleaning, encoding, outliers
│   ├── features/        # Transformations, correlation analysis
│   ├── models/          # Overview, PCA+scaling, CatBoost, model-centric pipeline
│   └── analysis/        # Descriptive stats, statistical tests, visualization
├── tests/               # Pytest test suite
├── docs/                # Documentation
├── config/              # Configuration files
├── v1/                  # Original monolithic code (archive)
├── requirements.txt
├── .gitignore
└── .env.example
```

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/
```

## Code Examples

**Basic model overview:**
```python
import pandas as pd
from src.models.overview import overview_models_for_df

df = pd.read_csv("data/building_data.csv")
results = overview_models_for_df(df, target="SiteEUIWN(kBtu/sf)")
print(results.head())
```

**Preprocessing pipeline:**
```python
from src.preprocessing.cleaning import type_definition, keep_unique

num_col, cat_col = type_definition(df)
df_clean = keep_unique(df, pkey="BuildingId")
```

**Feature correlation analysis:**
```python
from src.features.correlation import print_top_correlations

high_corrs, pairs = print_top_correlations(df, threshold=0.85)
for feat, other, val in high_corrs:
    print(f"{feat} vs {other}: {val:.2f}")
```
