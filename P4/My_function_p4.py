# TARGET_COLUMN = "SiteEUIWN(kBtu/sf)"

# log_GHGEmissionsIntensity


import pandas as pd
from sklearn.preprocessing import OneHotEncoder



"Librairies : "
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import itertools
from scipy import stats

import scipy.stats as stats # Needed again for residual Q-Q plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor # Changed model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # Regression metrics
from sklearn.impute import SimpleImputer
from time import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import time
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
import time
import warnings


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer # Better imputation practice
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import traceback
import time # To measure training time


import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import time

# Suppress warnings for cleaner output (optional)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def type_definition(df):
    
    features = df.columns
    df_colnumb = df.shape[1] 
    num_col = []
    cat_col = []

    for feat in features:
        if pd.api.types.is_numeric_dtype(df[feat]):
            num_col.append(feat)
        else:
            cat_col.append(feat)

    print(f"Colonnes numériques : {num_col}")
    print(f"Colonnes catégoriques : {cat_col}")
    return num_col, cat_col



## XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX NOTEBOOK PREDICTIONS T1 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
## XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX NOTEBOOK PREDICTIONS T1 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
## XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX NOTEBOOK PREDICTIONS T1 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
## XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX NOTEBOOK PREDICTIONS T1 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
## XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX NOTEBOOK PREDICTIONS T1 XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


def overview_models_for_df(df):
    import pandas as pd
    import numpy as np
    import time
    import warnings
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    # --- Model Imports ---
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    # Optional: Boosting libraries (ensure installed: pip install xgboost lightgbm catboost)
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    # --- Metrics ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # --- Suppress Warnings ---
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # --- 1. Define Models with Fixed, Sensible Parameters for Quick Overview ---

    RANDOM_STATE = 42 # for reproducibility

    # Using near-default or slightly adjusted parameters suitable for a quick check
    models_to_evaluate = {
        # Linear Models
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.1, max_iter=2000, random_state=RANDOM_STATE), # alpha=0.1 often better starting point than 1.0
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_STATE),

        # Neighbors
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3), # Default neighbors

        # SVM (can be slow, especially RBF)
        'SVR_linear': SVR(kernel='linear', C=1.0, cache_size=500),
        'SVR_rbf': SVR(kernel='rbf', C=1.0, gamma='scale', cache_size=500), # Default C and gamma

        # Tree-based
        'DecisionTree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE), # Limit depth for speed
        'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1), # Common defaults, limited depth

        # Boosting (often perform well)
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE), 
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1, objective='reg:squarederror'),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.15, max_depth=-1, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1), 
        'CatBoost': cb.CatBoostRegressor(iterations=100, learning_rate=0.15, depth=6, random_state=RANDOM_STATE, verbose=0, thread_count=-1) 
    }

    # --- 2. Provide Your Preprocessed Data Here ---
    df_clean = df.copy()
    T1, T2 = 'SiteEUIWN(kBtu/sf)', 'GHGEmissionsIntensity'

    X1 = df_clean.drop(columns= [T1,T2])
    y1 = df_clean[T1] # Using T1 as the target for this run


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42, shuffle=True)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # --- 3. Model Evaluation Loop ---

    results_list = []
    print("\n--- Starting Model Evaluation ---")

    # Check if data exists
    if 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
        raise NameError("Data variables (X_train, y_train, X_test, y_test) are not defined. Provide data in Section 2.")

    for model_name, model_instance in models_to_evaluate.items():
        print(f"Evaluating {model_name}...")

        try:
            # --- Training ---
            start_time = time.time()
            model_instance.fit(X_train, y_train)
            fit_time = time.time() - start_time

            # --- Prediction ---
            start_time = time.time()
            y_pred = model_instance.predict(X_test)
            predict_time = time.time() - start_time

            # --- Metrics Calculation ---
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            # --- Store Results ---
            results_list.append({
                'Model': model_name,
                'R2 (Test)': r2,
                'MAE (Test)': mae,
                'RMSE (Test)': rmse,
                'Fit Time (s)': fit_time,
                'Predict Time (s)': predict_time,
                'Notes': '' # Placeholder for any specific notes
            })

        except Exception as e:
            print(f"  ERROR evaluating {model_name}: {e}")
            results_list.append({
                'Model': model_name,
                'R2 (Test)': np.nan,
                'MAE (Test)': np.nan,
                'RMSE (Test)': np.nan,
                'Fit Time (s)': np.nan,
                'Predict Time (s)': np.nan,
                'Notes': f'Error: {e}'
            })

    print("--- Evaluation Complete ---")

    # --- 4. Display Results ---

    if not results_list:
        print("\nNo models were evaluated successfully.")
    else:
        results_df = pd.DataFrame(results_list)
        # Sort by R-squared (higher is better)
        results_df_sorted = results_df.sort_values(by='R2 (Test)', ascending=False).reset_index(drop=True)

        print("\n--- Model Performance Overview (Sorted by Test R2) ---")
        # Display relevant columns, format floats
        pd.set_option('display.float_format', lambda x: '%.4f' % x) # Format floats
        print(results_df_sorted[['Model', 'R2 (Test)', 'RMSE (Test)', 'MAE (Test)', 'Fit Time (s)', 'Notes']])
        pd.reset_option('display.float_format') # Reset float format

        # Identify top candidates based on R2 / RMSE
        if not results_df_sorted.empty and pd.notna(results_df_sorted.loc[0, 'R2 (Test)']):
            top_model_r2 = results_df_sorted.loc[0, 'Model']
            print(f"\n=> Top model based on R2: {top_model_r2}")

            # Find best by RMSE (lower is better)
            results_df_rmse = results_df.sort_values(by='RMSE (Test)', ascending=True).reset_index(drop=True)
            if not results_df_rmse.empty and pd.notna(results_df_rmse.loc[0, 'RMSE (Test)']):
                top_model_rmse = results_df_rmse.loc[0, 'Model']
                print(f"=> Top model based on RMSE: {top_model_rmse}")
                print("\nConsider these models for further analysis and hyperparameter tuning.")
            else:
                print("\nCould not reliably determine the best model by RMSE.")

        else:
            print("\nCould not reliably determine the best model.")
    return results_df_sorted






def overview_models_target(df, target):
    import pandas as pd
    import numpy as np
    import time
    import warnings
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    # --- Model Imports ---
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    # Optional: Boosting libraries (ensure installed: pip install xgboost lightgbm catboost)
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    # --- Metrics ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # --- Suppress Warnings ---
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # --- 1. Define Models with Fixed, Sensible Parameters for Quick Overview ---

    RANDOM_STATE = 42 # for reproducibility

    # Using near-default or slightly adjusted parameters suitable for a quick check
    models_to_evaluate = {
        # Linear Models
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.1, max_iter=2000, random_state=RANDOM_STATE), # alpha=0.1 often better starting point than 1.0
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000, random_state=RANDOM_STATE),

        # Neighbors
        'KNeighborsRegressor': KNeighborsRegressor(n_neighbors=3), # Default neighbors

        # SVM (can be slow, especially RBF)
        'SVR_linear': SVR(kernel='linear', C=1, cache_size=500),
        'SVR_rbf': SVR(kernel='rbf', C=1, gamma='scale', cache_size=500), # Default C and gamma
        'SVR_poly2' : SVR( kernel='poly', degree=2, gamma='scale', coef0=0.7, C=1.5, epsilon=0.001),
        'SVR_poly3' : SVR( kernel='poly', degree=3, gamma='scale', coef0=0.7, C=1.5, epsilon=0.001),
        'SVR_poly4' : SVR( kernel='poly', degree=4, gamma='scale', coef0=0.7, C=1.5, epsilon=0.001),
        'SVR_poly5' : SVR( kernel='poly', degree=5, gamma='scale', coef0=0.7, C=1.5, epsilon=0.001),

        # Tree-based
        'DecisionTree': DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE), # Limit depth for speed
        'RandomForest': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1), # Common defaults, limited depth

        # Boosting (often perform well)
        'AdaBoost': AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=RANDOM_STATE),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE), 
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1, objective='reg:squarederror'),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.15, max_depth=-1, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1), 
        'CatBoost': cb.CatBoostRegressor(iterations=100, learning_rate=0.15, depth=6, random_state=RANDOM_STATE, verbose=0, thread_count=-1) 
    }

    # --- 2. Provide Your Preprocessed Data Here ---
    df_clean = df.copy()
    X1 = df_clean.drop(columns= [target])
    y1 = df_clean[target] # Using T1 as the target for this run


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42, shuffle=True)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # --- 3. Model Evaluation Loop ---

    results_list = []
    print("\n--- Starting Model Evaluation ---")

    # Check if data exists
    if 'X_train' not in locals() or 'y_train' not in locals() or 'X_test' not in locals() or 'y_test' not in locals():
        raise NameError("Data variables (X_train, y_train, X_test, y_test) are not defined. Provide data in Section 2.")

    for model_name, model_instance in models_to_evaluate.items():
        print(f"Evaluating {model_name}...")

        try:
            # --- Training ---
            start_time = time.time()
            model_instance.fit(X_train, y_train)
            fit_time = time.time() - start_time

            # --- Prediction ---
            start_time = time.time()
            y_pred = model_instance.predict(X_test)
            predict_time = time.time() - start_time

            # --- Metrics Calculation ---
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            # --- Store Results ---
            results_list.append({
                'Model': model_name,
                'R2 (Test)': r2,
                'MAE (Test)': mae,
                'RMSE (Test)': rmse,
                'Fit Time (s)': fit_time,
                'Predict Time (s)': predict_time,
                'Notes': '' # Placeholder for any specific notes
            })

        except Exception as e:
            print(f"  ERROR evaluating {model_name}: {e}")
            results_list.append({
                'Model': model_name,
                'R2 (Test)': np.nan,
                'MAE (Test)': np.nan,
                'RMSE (Test)': np.nan,
                'Fit Time (s)': np.nan,
                'Predict Time (s)': np.nan,
                'Notes': f'Error: {e}'
            })

    print("--- Evaluation Complete ---")

    # --- 4. Display Results ---

    if not results_list:
        print("\nNo models were evaluated successfully.")
    else:
        results_df = pd.DataFrame(results_list)
        # Sort by R-squared (higher is better)
        results_df_sorted = results_df.sort_values(by='R2 (Test)', ascending=False).reset_index(drop=True)

        print("\n--- Model Performance Overview (Sorted by Test R2) ---")
        # Display relevant columns, format floats
        pd.set_option('display.float_format', lambda x: '%.4f' % x) # Format floats
        print(results_df_sorted[['Model', 'R2 (Test)', 'RMSE (Test)', 'MAE (Test)', 'Fit Time (s)', 'Notes']])
        pd.reset_option('display.float_format') # Reset float format

        # Identify top candidates based on R2 / RMSE
        if not results_df_sorted.empty and pd.notna(results_df_sorted.loc[0, 'R2 (Test)']):
            top_model_r2 = results_df_sorted.loc[0, 'Model']
            print(f"\n=> Top model based on R2: {top_model_r2}")

            # Find best by RMSE (lower is better)
            results_df_rmse = results_df.sort_values(by='RMSE (Test)', ascending=True).reset_index(drop=True)
            if not results_df_rmse.empty and pd.notna(results_df_rmse.loc[0, 'RMSE (Test)']):
                top_model_rmse = results_df_rmse.loc[0, 'Model']
                print(f"=> Top model based on RMSE: {top_model_rmse}")
                print("\nConsider these models for further analysis and hyperparameter tuning.")
            else:
                print("\nCould not reliably determine the best model by RMSE.")

        else:
            print("\nCould not reliably determine the best model.")
    return results_df_sorted




# 2 RF REGRESSOR ANALYSIS = Same as RFE at chapter 4



# 3 AMELIORATIONS DATA CENTRIC
# 3.1. ACP & SCALING




# Add necessary imports at the top of the file
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
# You also need openpyxl installed for to_excel, but no explicit import is strictly needed by pandas itself

# Function to perform basic data checks
def check_data(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")
    if df.isnull().sum().sum() > 0:
        print("Warning: Data contains missing values. Consider imputation before running.")
        # Simple mean imputation for demonstration if needed
        # df = df.fillna(df.mean())
# Function to split the data
def split_data(df, target_column, test_size, random_state):
    """
    Splits the DataFrame into training and testing sets, keeping only numeric features.
    Handles missing target values by dropping rows.
    Selects numeric features and returns their names.
    """
    print(f"--- Splitting Data (Test Size: {test_size}, Random State: {random_state}) ---")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataframe.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
         raise ValueError(f"Target column '{target_column}' must be numeric for regression.")

    # Drop rows where the target is NaN
    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_column]).copy()
    if len(df_cleaned) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df_cleaned)} rows with missing target values.")
    if df_cleaned.empty:
         raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
         raise ValueError("Not enough samples left after handling missing target values to perform a split.")

    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]

    # Ensure X contains only numeric data for scaling/PCA
    X_numeric = X.select_dtypes(include=np.number)
    original_feature_names = X_numeric.columns.tolist() # Capture the names here

    print(f"Features used for modeling (numeric only): {X_numeric.shape[1]}")
    if X_numeric.shape[1] == 0:
        raise ValueError("No numeric features found after dropping target and selecting numeric types.")

    # Note: This split function does NOT handle NaNs in features.
    # Imputation should happen BEFORE calling this function or as a pipeline step.
    # Your `run_experiments` function does handle imputation *after* splitting
    # but before scaling/PCA, which is also an option.

    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state
    )
    print(f"Data split into train/test sets: X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Return 5 values now
    return X_train, X_test, y_train, y_test, original_feature_names


# Function to run experiments
def run_experiments(X_train, X_test, y_train, y_test, scalers, pca_options, rf_params, n_top_results_to_show, random_state):
    results = []
    start_time = time.time()
    print("\n--- Starting Experiments ---")
    for scaler_name, scaler in scalers.items():
        print(f"\nTesting Scaler: {scaler_name}")
        # Apply scaling (if specified) - Fit on train only
        X_train_scaled = X_train.copy() # Start with a copy
        X_test_scaled = X_test.copy()   # Start with a copy
        if scaler:
            try:
                # Scaling requires numpy arrays or pandas DataFrames
                # If X_train/X_test were already numpy arrays from select_dtypes, this is fine.
                # If they were pandas DataFrames, fit_transform returns numpy arrays.
                X_train_scaled = scaler.fit_transform(X_train_scaled)
                X_test_scaled = scaler.transform(X_test_scaled)
            except Exception as e:
                print(f"Error during scaling with {scaler_name}: {e}")
                continue # Skip to next scaler if scaling fails

        current_n_features = X_train_scaled.shape[1]

        for n_components in pca_options:
            experiment_name = f"Scaler: {scaler_name}, PCA: {n_components if n_components is not None else 'None'}"
            # print(f"  Running: {experiment_name}") # Keep print for progress

            if n_components is not None and n_components > current_n_features:
                # print(f"    Skipping PCA {n_components} components (>{current_n_features} features available)")
                continue # Skip PCA value if too large

            X_train_processed = X_train_scaled # Start with scaled data
            X_test_processed = X_test_scaled   # Start with scaled data
            pca_instance = None

            if n_components is not None:
                pca_instance = PCA(n_components=n_components, random_state=random_state)
                try:
                    # PCA also returns numpy arrays
                    X_train_processed = pca_instance.fit_transform(X_train_processed)
                    X_test_processed = pca_instance.transform(X_test_processed)
                    print(f"  {experiment_name} - PCA applied. New feature shape: {X_train_processed.shape}")
                except Exception as e:
                    print(f"  {experiment_name} - Error during PCA: {e}")
                    results.append({
                        'Scaler': scaler_name,
                        'PCA_Components': n_components,
                        'R2_Score': np.nan,
                        'RMSE': np.nan
                    })
                    continue # Skip model training if PCA failed

            try:
                model = RandomForestRegressor(**rf_params)
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                results.append({
                    'Scaler': scaler_name,
                    'PCA_Components': 'None' if n_components is None else n_components,
                    'R2_Score': r2,
                    'RMSE': rmse
                })
                print(f"  {experiment_name} - Completed - R2: {r2:.4f}, RMSE: {rmse:.4f}")
            except Exception as e:
                print(f"  {experiment_name} - Error during model training/prediction: {e}")
                results.append({
                    'Scaler': scaler_name,
                    'PCA_Components': 'None' if n_components is None else n_components,
                    'R2_Score': np.nan,
                    'RMSE': np.nan
                })

    end_time = time.time()
    print(f"\n--- Experiments Finished ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    return results

# Function to process and display results
def process_and_display_results(results, n_top_results_to_show, output_file):
    if not results:
        print("\nNo results were generated. Please check for errors in the loops.")
    else:
        results_df = pd.DataFrame(results)
        # Remove rows where R2 or RMSE is NaN (due to errors during experiment) before sorting and displaying
        results_df_cleaned = results_df.dropna(subset=['R2_Score', 'RMSE'])
        results_df_sorted = results_df_cleaned.sort_values(by='R2_Score', ascending=False).reset_index(drop=True)
        print(f"\n--- Top {n_top_results_to_show} Results (Sorted by R2 Score) ---")
        # Use to_string() to ensure all rows/columns are displayed if within limit
        print(results_df_sorted.head(n_top_results_to_show).to_string())

        # Also save the full results, including errors marked by NaN, before cleaning
        try:
            # Ensure the directory exists if needed - not covered here, assuming standard path
            results_df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"\nAll results (including potential errors) saved successfully to '{output_file}'")
        except Exception as e:
            print(f"\nError saving results to Excel: {e}")
            print("Attempting to save as CSV instead.")
            try:
                csv_output_file = output_file.replace('.xlsx', '.csv')
                if csv_output_file == output_file: # Avoid infinite loop if filename doesn't have .xlsx
                     csv_output_file = output_file + '.csv'
                results_df.to_csv(csv_output_file, index=False)
                print(f"All results (including potential errors) saved successfully as CSV to '{csv_output_file}'")
            except Exception as e_csv:
                print(f"Error saving results to CSV: {e_csv}")


# Main function to orchestrate the entire process
def main_pca_scale(df, acp_min, target, test_size, random_state, output_file, rf_params, N_TOP_RESULTS_TO_SHOW):
    """
    Orchestrates the process of data splitting, scaling, PCA,
    Random Forest modeling, and evaluating different preprocessing pipelines.

    Args:
        df (pd.DataFrame): The input DataFrame.
        acp_min (int): Minimum number of PCA components to test.
        target (str): Name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        output_file (str): Path to save the results Excel/CSV file.
        rf_params (dict): Dictionary of parameters for RandomForestRegressor.
        N_TOP_RESULTS_TO_SHOW (int): Number of top results to display.
    """
    print("--- Starting main_pca_scale process ---")
    try:
        # Perform basic data checks
        print("Step 1: Checking data...")
        check_data(df, target)
        print("Data checks passed.")

        # Split the data
        print("\nStep 2: Splitting data...")
        X_train, X_test, y_train, y_test, original_feature_names = split_data(df, target, test_size, random_state)
        print("Data splitting complete.")

        # Define scalers and PCA options
        print("\nStep 3: Defining experiment parameters...")
        scalers = {
            'None': None,
            'Standard': StandardScaler(),
            'Robust': RobustScaler(),
            'MinMax(1-100)': MinMaxScaler(feature_range=(1, 100))
        }
        # Ensure max_pca_components is based on features *after* selecting numeric types
        max_pca_components = X_train.shape[1]
        if max_pca_components == 0:
             raise ValueError("No numeric features available for PCA after splitting.")

        # PCA options include None (no PCA) and components from acp_min up to max_pca_components
        # Ensure acp_min is not greater than max_pca_components, adjust if needed
        if acp_min > max_pca_components:
             print(f"Warning: acp_min ({acp_min}) is greater than the number of available numeric features ({max_pca_components}). Adjusting acp_min to {max_pca_components}.")
             acp_min_adjusted = max_pca_components
        elif acp_min <= 0:
              print(f"Warning: acp_min ({acp_min}) is less than or equal to 0. Adjusting acp_min to 1.")
              acp_min_adjusted = 1
        else:
             acp_min_adjusted = acp_min

        # Ensure the range for PCA is valid
        if acp_min_adjusted > max_pca_components: # Should not happen after previous checks, but safeguard
             pca_range = []
        else:
             pca_range = list(range(acp_min_adjusted, max_pca_components + 1))

        pca_options = [None] + pca_range

        print(f"Scalers to test: {list(scalers.keys())}")
        print(f"PCA components to test (including None): {pca_options}")
        print("Experiment parameters defined.")


        # Run experiments
        print("\nStep 4: Running experiments...")
        results = run_experiments(X_train, X_test, y_train, y_test, scalers, pca_options, rf_params, N_TOP_RESULTS_TO_SHOW, random_state)
        print("Experiments finished.")

        # Process and display results
        print("\nStep 5: Processing and displaying results...")
        process_and_display_results(results, N_TOP_RESULTS_TO_SHOW, output_file)
        print("Results processing complete.")

    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\n--- main_pca_scale process finished ---")




# Function to split the data (Mostly Unchanged, added imputation)
def split_data_cat(df, target_column, test_size, random_state):
    """
    Splits the DataFrame into features (numeric only) and target.
    Handles missing values in target by dropping rows.
    Handles missing values in numeric features by median imputation.
    """
    print(f"--- Splitting Data (Target: '{target_column}') ---")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
         raise ValueError(f"Target column '{target_column}' must be numeric for regression.")

    # Drop rows where the target is NaN
    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_column]).copy()
    if len(df_cleaned) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df_cleaned)} rows with missing target values.")
    if df_cleaned.empty:
         raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
         raise ValueError("Not enough samples left after handling missing target values to perform a split.")


    X = df_cleaned.drop(target_column, axis=1)
    y = df_cleaned[target_column]

    # Select only numeric features (for this pipeline version)
    X_numeric = X.select_dtypes(include=np.number)
    feature_names = X_numeric.columns.tolist()

    if not feature_names:
         raise ValueError("No numeric features found after dropping target.")

    print(f"Features used for modeling (numeric only): {X_numeric.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=test_size, random_state=random_state)

    # Impute NaNs in train and test sets using the TRAIN set's median
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("Warning: NaNs found in numeric features. Imputing with median from training data.")
        imputation_values = X_train.median()
        X_train = X_train.fillna(imputation_values)
        X_test = X_test.fillna(imputation_values) # Use train median for test set too

    print(f"Data split and imputed into train/test sets: X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names # Also return feature names

# Function to run experiments (Adapted for CatBoost)
def run_experiments_catboost(X_train, X_test, y_train, y_test, scalers, pca_options, catboost_params, n_top_results_to_show, random_state):
    """
    Runs experiments with different scaling and PCA combinations using CatBoostRegressor.

    Args:
        X_train (pd.DataFrame): Training features (numeric, imputed).
        X_test (pd.DataFrame): Testing features (numeric, imputed).
        y_train (pd.Series): Training target.
        y_test (pd.Series): Testing target.
        scalers (dict): Dictionary of scaler names and instances.
        pca_options (list): List of PCA components to test (including None).
        catboost_params (dict): Parameters for the CatBoostRegressor.
        n_top_results_to_show (int): Number of top results to display (used in display function, passed here).
        random_state (int): Random state for reproducibility.

    Returns:
        list: A list of dictionaries, each containing results for an experiment.
    """
    results = []
    start_time = time.time()
    print("\n--- Starting CatBoost Experiments ---")

    if CatBoostRegressor is None:
        print("CatBoostRegressor not available. Skipping experiments.")
        return results # Return empty results if CatBoost not installed

    # Ensure CatBoost params are set up correctly for the loop
    base_catboost_params = catboost_params.copy()
    base_catboost_params.setdefault('random_state', random_state)
    base_catboost_params.setdefault('verbose', 0) # Keep verbosity low during grid search
    base_catboost_params.setdefault('loss_function', 'RMSE') # Ensure regression loss

    # Note: Early stopping is typically used *outside* a grid search like this
    # because it depends on an eval_set. For a simple grid search comparing preprocess,
    # we train for a fixed number of iterations or use cross-validation.
    # If you want early stopping, you'd need to pass X_test/y_test as eval_set
    # to the model.fit call inside the loop, but this can bias results
    # towards the specific test set. A better approach for hyperparameter tuning
    # is often using cross-validation with early stopping on the validation fold.
    # For this grid search, we'll train for the fixed iterations in catboost_params.
    early_stopping_rounds = base_catboost_params.pop('early_stopping_rounds', None)
    if early_stopping_rounds is not None:
        print(f"Warning: early_stopping_rounds ({early_stopping_rounds}) found in catboost_params. Early stopping is generally not ideal inside this type of grid search loop as it uses the test set. Training for fixed iterations instead.")
        # You could add logic here to use eval_set=(X_test_processed, y_test) if you accept the bias.
        # For now, we train for the full iterations.


    for scaler_name, scaler in scalers.items():
        print(f"\nTesting Scaler: {scaler_name}")

        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        if scaler:
            try:
                # Ensure data is compatible with scaler (pandas DataFrame or numpy array)
                # split_data returns DataFrames, which is fine.
                X_train_scaled = scaler.fit_transform(X_train_scaled)
                X_test_scaled = scaler.transform(X_test_scaled)
                print(f"  Scaling with {scaler_name} applied.")
            except Exception as e:
                print(f"Error during scaling with {scaler_name}: {e}")
                # Record this failure
                for n_components in pca_options: # Record failure for all PCA options with this scaler
                     results.append({
                        'Scaler': scaler_name,
                        'PCA_Components': 'None' if n_components is None else n_components,
                        'R2_Score': np.nan,
                        'RMSE': np.nan,
                        'Error': f"Scaling failed: {e}"
                     })
                continue # Skip to next scaler if scaling fails

        current_n_features = X_train_scaled.shape[1]

        for n_components in pca_options:
            experiment_name = f"Scaler: {scaler_name}, PCA: {n_components if n_components is not None else 'None'}"
            print(f"  Running: {experiment_name}")

            if n_components is not None:
                 # Check if n_components is valid (int > 0 or float (0,1])
                 is_valid_pca_n = False
                 if isinstance(n_components, int) and n_components > 0 and n_components <= current_n_features:
                      is_valid_pca_n = True
                 elif isinstance(n_components, float) and 0 < n_components <= 1 and current_n_features > 0:
                      is_valid_pca_n = True
                 elif n_components > current_n_features:
                      print(f"    Skipping PCA {n_components} components (>{current_n_features} features available)")
                      continue # Skip if n_components exceeds available features

                 if not is_valid_pca_n and n_components is not None:
                      print(f"    Skipping PCA {n_components} components (invalid value or 0 features)")
                      continue

            X_train_processed = X_train_scaled # Start with scaled data (numpy array)
            X_test_processed = X_test_scaled   # Start with scaled data (numpy array)
            pca_instance = None

            if n_components is not None and current_n_features > 0:
                pca_instance = PCA(n_components=n_components, random_state=random_state)
                try:
                    # PCA requires numpy arrays
                    X_train_processed = pca_instance.fit_transform(X_train_processed)
                    X_test_processed = pca_instance.transform(X_test_processed)
                    # Update feature count after PCA
                    current_n_features_after_pca = X_train_processed.shape[1]
                    print(f"    PCA applied. New feature shape: {X_train_processed.shape}")
                    if current_n_features_after_pca == 0:
                         print("    Warning: PCA resulted in 0 features. Skipping model training.")
                         results.append({
                            'Scaler': scaler_name,
                            'PCA_Components': n_components,
                            'R2_Score': np.nan,
                            'RMSE': np.nan,
                            'Error': "PCA resulted in 0 features"
                         })
                         continue # Skip model training if PCA resulted in 0 features

                except Exception as e:
                    print(f"  {experiment_name} - Error during PCA: {e}")
                    results.append({
                        'Scaler': scaler_name,
                        'PCA_Components': n_components,
                        'R2_Score': np.nan,
                        'RMSE': np.nan,
                        'Error': f"PCA failed: {e}"
                    })
                    continue # Skip model training if PCA failed

            # --- Model Training and Evaluation ---
            # Use the base_catboost_params (without early stopping logic from this loop)
            model_params_this_exp = base_catboost_params.copy()

            try:
                # CatBoostRegressor can train on numpy arrays directly
                model = CatBoostRegressor(**model_params_this_exp)

                # Train the model
                print(f"    Training model with shape: {X_train_processed.shape}")
                model.fit(X_train_processed, y_train)
                print(f"    Training complete.")

                # Predict and evaluate
                y_pred = model.predict(X_test_processed)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    'Scaler': scaler_name,
                    'PCA_Components': 'None' if n_components is None else n_components,
                    'R2_Score': r2,
                    'RMSE': rmse,
                    'Error': None # Mark as successful
                })
                print(f"  {experiment_name} - Completed - R2: {r2:.4f}, RMSE: {rmse:.4f}")

            except Exception as e:
                print(f"  {experiment_name} - Error during model training/prediction: {e}")
                results.append({
                    'Scaler': scaler_name,
                    'PCA_Components': 'None' if n_components is None else n_components,
                    'R2_Score': np.nan,
                    'RMSE': np.nan,
                    'Error': f"Model training/prediction failed: {e}"
                })

    end_time = time.time()
    print(f"\n--- CatBoost Experiments Finished ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    return results


# Main function to orchestrate the entire process (Adapted for CatBoost experiments)
def main_pca_scale_catboost(df, acp_min, target, test_size, random_state, output_file, catboost_params, N_TOP_RESULTS_TO_SHOW):
    """
    Orchestrates the process of data splitting, scaling, PCA experiments,
    CatBoost modeling, and evaluating different preprocessing pipelines.

    Args:
        df (pd.DataFrame): The input DataFrame.
        acp_min (int): Minimum number of PCA components (integer) to test.
                        Must be > 0.
        target (str): Name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
        output_file (str): Path to save the results Excel/CSV file.
        catboost_params (dict): Dictionary of parameters for CatBoostRegressor.
        N_TOP_RESULTS_TO_SHOW (int): Number of top results to display.
    """
    print("--- Starting main_pca_scale_catboost process ---")
    try:
        # Perform basic data checks
        print("Step 1: Checking data...")
        check_data(df, target)
        print("Data checks passed.")

        # Split the data
        print("\nStep 2: Splitting data...")
        # split_data now returns feature_names as well
        X_train, X_test, y_train, y_test, initial_feature_names = split_data_cat(df, target, test_size, random_state)
        print("Data splitting complete.")

        # Define scalers and PCA options
        print("\nStep 3: Defining experiment parameters...")
        scalers = {
            'None': None,
            'Standard': StandardScaler(),
            'Robust': RobustScaler(),
            # Example of a different scaler range
            'MinMax(0-1)': MinMaxScaler(feature_range=(0, 1))
            # Add more scalers here if desired
        }

        # Determine the maximum number of PCA components possible after splitting
        max_pca_components = X_train.shape[1]
        if max_pca_components == 0:
             print("\nCRITICAL ERROR: 0 numeric features available after splitting. Cannot run experiments.")
             # No results to save or display in this case
             return

        # Adjust acp_min based on available features
        if not isinstance(acp_min, int) or acp_min <= 0:
             print(f"Warning: acp_min ({acp_min}) must be a positive integer. Setting to 1.")
             acp_min_adjusted = 1
        elif acp_min > max_pca_components:
             print(f"Warning: acp_min ({acp_min}) is greater than the number of available numeric features ({max_pca_components}). Adjusting acp_min to {max_pca_components}.")
             acp_min_adjusted = max_pca_components
        else:
             acp_min_adjusted = acp_min


        # PCA options list: None + integers from acp_min up to max features.
        # You could also add floats (e.g., variance retention) if desired, but
        # your current loop structure is for integer components or None.
        pca_range = list(range(acp_min_adjusted, max_pca_components + 1))
        pca_options = [None] + pca_range

        print(f"Scalers to test: {list(scalers.keys())}")
        print(f"PCA components to test (including None): {pca_options}")
        print("Experiment parameters defined.")


        # Run experiments
        print("\nStep 4: Running experiments...")
        # Pass the CatBoost parameters dictionary
        results = run_experiments_catboost(
            X_train, X_test, y_train, y_test,
            scalers, pca_options, catboost_params, # Pass catboost_params here
            N_TOP_RESULTS_TO_SHOW, random_state
        )
        print("Experiments finished.")

        # Process and display results
        print("\nStep 5: Processing and displaying results...")
        process_and_display_results(results, N_TOP_RESULTS_TO_SHOW, output_file)
        print("Results processing complete.")

    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the main process: {e}")
        traceback.print_exc() # Print traceback for unexpected errors

    print("\n--- main_pca_scale_catboost process finished ---")



# 4 Model - Centric XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 4 Model - Centric XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# 4 Model - Centric XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import pandas as pd
import numpy as np
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator # Needed for type checking


# --- NEW: Import SHAP ---
try:
    import shap
    SHAP_INSTALLED = True
    # Initialize JS visualization code for plots in environments like Jupyter
    shap.initjs()
except ImportError:
    print("Warning: 'shap' library not installed. SHAP analysis will be skipped.")
    print("Install it using: pip install shap")
    SHAP_INSTALLED = False


# --- 1. Data Preparation & Splitting ---
# (No changes needed in this function)
def split_and_select_numeric_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into features (numeric only) and target.
    Handles missing values in target by dropping rows.
    Handles missing values in numeric features by median imputation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): Name of the target column.
        test_size (float): Proportion for the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Contains:
            - X_train (pd.DataFrame): Training features (numeric, imputed).
            - X_test (pd.DataFrame): Testing features (numeric, imputed).
            - y_train (pd.Series): Training target.
            - y_test (pd.Series): Testing target.
            - initial_numeric_feature_names (list): Names of original numeric features.
            - test_original_indices (pd.Index): Original index of test samples.
            - train_original_indices (pd.Index): Original index of train samples. # Added for consistency if needed later
    Raises:
        ValueError, TypeError
    """
    print(f"--- 1. Preparing & Splitting Data (Target: '{target_col}') ---")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
         raise ValueError(f"Target column '{target_col}' must be numeric for regression.")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Drop rows where the target is NaN
    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_col]).copy()
    if len(df_cleaned) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df_cleaned)} rows with missing target values.")

    if df_cleaned.empty:
         raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
         raise ValueError("Not enough samples left after handling missing target values to perform a split.")
    min_train_samples = int(np.ceil(len(df_cleaned)*(1-test_size)))
    min_test_samples = int(np.ceil(len(df_cleaned)*test_size))
    if min_train_samples < 1 or min_test_samples < 1:
         print(f"Warning: Test size {test_size} results in very small train/test sets ({min_train_samples} train, {min_test_samples} test). Proceeding cautiously.")

    X = df_cleaned.drop(target_col, axis=1)
    y = df_cleaned[target_col]

    # Select only numeric features
    X_numeric = X.select_dtypes(include=np.number)
    initial_numeric_feature_names = X_numeric.columns.tolist()

    if not initial_numeric_feature_names:
         raise ValueError("No numeric features found after dropping target.")

    print(f"Initial total features: {X.shape[1]}, Using {len(initial_numeric_feature_names)} numeric features.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state
    )

    # Impute NaNs in train and test sets using the TRAIN set's median
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("Warning: NaNs found in numeric features. Imputing with median from training data.")
        imputation_values = X_train.median()
        X_train = X_train.fillna(imputation_values)
        X_test = X_test.fillna(imputation_values) # Use train median for test set too

    test_original_indices = X_test.index # Keep track of original indices for residual analysis
    train_original_indices = X_train.index # Keep track of original indices for train analysis if needed

    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    return (X_train, X_test, y_train, y_test,
            initial_numeric_feature_names, test_original_indices, train_original_indices)


# --- 2. Apply Preprocessing (Scaling and PCA) ---
# (No changes needed in this function)
def apply_preprocessing(X_train, X_test, scaler_type, pca_n_comp, random_state=42):
    """
    Applies specified scaling and PCA to the data.

    Args:
        X_train (pd.DataFrame): Training features (numeric, imputed).
        X_test (pd.DataFrame): Testing features (numeric, imputed).
        scaler_type (str): Type of scaler ('Standard', 'Robust', 'MinMax', 'None').
        pca_n_comp (int or float or None): Number of PCA components or variance ratio.
                                            If None, no PCA is applied.
        random_state (int): Random seed for PCA.

    Returns:
        tuple: Contains:
            - X_train_processed (np.ndarray): Processed training features.
            - X_test_processed (np.ndarray): Processed testing features.
            - feature_names_after_processing (list): Names corresponding to processed features.
            - scaler_instance: Fitted scaler object (or None).
            - pca_instance: Fitted PCA object (or None).
    Raises:
        ValueError, RuntimeError
    """
    print("\n--- 2. Applying Preprocessing (Scaling & PCA) ---")

    # 2.1 Apply Scaling
    scaler_instance = None
    X_train_scaled = X_train.copy() # Start with copies
    X_test_scaled = X_test.copy()
    original_feature_names = X_train.columns.tolist() # Store original names before scaling/PCA

    if scaler_type and scaler_type.lower() != 'none':
        print(f"  Applying Scaler: {scaler_type}")
        if scaler_type.lower() == 'standard':
            scaler_instance = StandardScaler()
        elif scaler_type.lower() == 'robust':
            scaler_instance = RobustScaler()
        elif scaler_type.lower() == 'minmax':
             # Use default range (0, 1) for simplicity unless specified otherwise
             scaler_instance = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}. Choose from 'Standard', 'Robust', 'MinMax', 'None'.")

        if scaler_instance:
            try:
                # Fit on train, transform train and test. Returns numpy arrays.
                X_train_scaled_np = scaler_instance.fit_transform(X_train_scaled)
                X_test_scaled_np = scaler_instance.transform(X_test_scaled)
            except Exception as e:
                print(f"  Error during scaling with {scaler_type}: {e}")
                raise RuntimeError(f"Scaling failed with {scaler_type}: {e}")
    else:
        print("  No scaling applied.")
        # Convert to numpy arrays even if no scaling for consistency with PCA output
        if isinstance(X_train_scaled, pd.DataFrame):
            X_train_scaled_np = X_train_scaled.values
        else: # Already numpy?
            X_train_scaled_np = X_train_scaled
        if isinstance(X_test_scaled, pd.DataFrame):
            X_test_scaled_np = X_test_scaled.values
        else: # Already numpy?
            X_test_scaled_np = X_test_scaled


    # Store scaled data (numpy arrays) before potential PCA
    X_train_processed = X_train_scaled_np
    X_test_processed = X_test_scaled_np
    feature_names_after_processing = original_feature_names # Names before potential PCA
    pca_instance = None

    # 2.2 Apply PCA
    if pca_n_comp is not None:
        max_features = X_train_processed.shape[1]
        if max_features == 0:
             print("  0 features available after scaling, skipping PCA.")
             feature_names_after_processing = []
        else:
            valid_pca_comp = False
            if isinstance(pca_n_comp, int) and 0 < pca_n_comp <= max_features:
                print(f"  Applying PCA with n_components={pca_n_comp}")
                valid_pca_comp = True
            elif isinstance(pca_n_comp, float) and 0 < pca_n_comp <= 1.0:
                print(f"  Applying PCA to retain {pca_n_comp:.1%} variance")
                valid_pca_comp = True
            else:
                 raise ValueError(f"Invalid pca_n_comp type or value: {pca_n_comp}. Must be int (0, {max_features}], float (0,1], or None.")

            if valid_pca_comp:
                try:
                    pca_instance = PCA(n_components=pca_n_comp, random_state=random_state)
                    # Fit PCA on the scaled training data
                    X_train_processed = pca_instance.fit_transform(X_train_processed)
                    # Transform the scaled test data
                    X_test_processed = pca_instance.transform(X_test_processed)

                    # Feature names become 'PC0', 'PC1', etc.
                    feature_names_after_processing = [f'PC{i}' for i in range(X_train_processed.shape[1])]
                    print(f"  PCA applied. New feature shape: {X_train_processed.shape}")
                    print(f"  Explained variance ratio by component: {pca_instance.explained_variance_ratio_}")
                    print(f"  Total explained variance: {np.sum(pca_instance.explained_variance_ratio_):.4f}")


                except Exception as e:
                     print(f"  Error during PCA with n_components={pca_n_comp}: {e}")
                     raise RuntimeError(f"PCA failed with n_components={pca_n_comp}: {e}")

    else:
        print("  No PCA applied.")
        # Feature names remain the original numeric names if no PCA
        # X_train_processed and X_test_processed are already numpy arrays


    print(f"Preprocessing complete. Final feature shape before RFE: {X_train_processed.shape}")
    # Ensure data is numpy array for RFE and model training (already done above, but double check)
    if not isinstance(X_train_processed, np.ndarray):
         raise TypeError(f"X_train_processed is not a numpy array after preprocessing: {type(X_train_processed)}")
    if not isinstance(X_test_processed, np.ndarray):
         raise TypeError(f"X_test_processed is not a numpy array after preprocessing: {type(X_test_processed)}")


    return X_train_processed, X_test_processed, feature_names_after_processing, scaler_instance, pca_instance


# --- 3. Recursive Feature Elimination ---
# (No changes needed in this function)
def perform_rfe(X_train_processed, y_train, feature_names_after_processing, target_feature_ratio=0.6, random_state=42, rfe_estimator_params=None):
    """
    Performs Recursive Feature Elimination using RandomForestRegressor.

    Args:
        X_train_processed (np.ndarray): Processed training features (scaled/PCA'd).
        y_train (pd.Series): Training target variable.
        feature_names_after_processing (list): List of feature names corresponding
                                               to X_train_processed (original or PC names).
        target_feature_ratio (float): The desired ratio of features to keep (e.g., 0.6 for 60%).
                                       If None or >= 1.0, RFE selection is skipped.
        random_state (int): Random seed for the RFE estimator.
        rfe_estimator_params (dict or None): Parameters for the RFE's internal RF estimator.
                                             Should NOT contain 'random_state'.

    Returns:
        tuple: Contains:
            - rfe (RFE): Fitted RFE object (or None if skipped/failed).
            - selected_feature_names (list): List of feature names selected by RFE
                                             (these are names from feature_names_after_processing).
            - X_train_rfe (np.ndarray): Training data with only selected features.
            - selected_mask (np.ndarray): Boolean mask indicating selected features
                                          relative to X_train_processed.
    Raises:
        ValueError, RuntimeError
    """
    print("\n--- 3. Performing Recursive Feature Elimination (RFE) ---")
    n_initial_features = X_train_processed.shape[1]
    print(f"Initial number of features for RFE: {n_initial_features}")

    # --- Skip RFE Logic ---
    # Skip if ratio is None, >= 1.0, <= 0, or if no features to begin with.
    # Use a specific check here that the orchestrator also uses.
    rfe_should_run = (target_feature_ratio is not None and 0 < target_feature_ratio < 1.0 and n_initial_features > 0)

    if not rfe_should_run:
        if n_initial_features == 0:
            print("Cannot perform RFE on data with 0 features. Skipping RFE.")
        else:
             print(f"target_feature_ratio ({target_feature_ratio}) is not in (0, 1). Skipping RFE selection.")
        # Return values indicating RFE was skipped
        # Mask selects all (original) features
        return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    # --- Proceed with RFE ---
    n_features_to_select = max(1, int(n_initial_features * target_feature_ratio))
    print(f"Target number of features to select: {n_features_to_select} ({target_feature_ratio*100:.0f}%)")

    # RFE Estimator Setup
    rfe_est_params_clean = rfe_estimator_params.copy() if rfe_estimator_params is not None else {'n_estimators': 150, 'max_depth': 10, 'n_jobs': -1}
    rfe_est_params_clean.pop('random_state', None) # Ensure random_state is NOT in rfe_estimator_params dict

    print(f"  Using RandomForestRegressor with params {rfe_est_params_clean} as RFE estimator (random_state={random_state}).")
    estimator = RandomForestRegressor(random_state=random_state, **rfe_est_params_clean)

    # Ensure n_features_to_select is valid
    n_features_to_select = min(n_features_to_select, n_initial_features)
    if n_features_to_select <= 0: # Should only happen if n_initial_features was 0, handled above
         print("Internal Error: n_features_to_select became <= 0 unexpectedly.")
         return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)

    try:
        # Check if X_train_processed has enough samples
        min_samples_needed = max(2, n_features_to_select + 1) # Heuristic: need more samples than features
        if X_train_processed.shape[0] < min_samples_needed:
             print(f"Warning: Not enough samples ({X_train_processed.shape[0]}) relative to features ({n_features_to_select}) for stable RFE. Skipping RFE.")
             return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

        print(f"Fitting RFE to select {n_features_to_select} features...")
        rfe.fit(X_train_processed, y_train)
        print("RFE fitting complete.")

    except Exception as e:
        print(f"Error fitting RFE: {e}")
        traceback.print_exc()
        print("RFE fitting failed. Proceeding without feature selection (using all processed features).")
        return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    # Post-RFE processing
    selected_mask = rfe.support_
    n_selected = selected_mask.sum()

    if n_selected == 0:
         print("Warning: RFE selected 0 features. This might indicate issues with the data or RFE estimator. Proceeding without feature selection.")
         return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    # Get the names of the selected features *from the processed data*
    if len(feature_names_after_processing) != n_initial_features:
         print(f"Error: Mismatch in feature names list length ({len(feature_names_after_processing)}) and processed data columns ({n_initial_features}). Cannot map selected features to names.")
         # Fallback: generic names
         selected_feature_names = [f'Selected_Feature_{i}' for i in range(n_selected)]
    else:
        selected_feature_names = [name for name, selected in zip(feature_names_after_processing, selected_mask) if selected]

    # Transform the training data immediately
    X_train_rfe = rfe.transform(X_train_processed)

    # Sanity check shapes
    if X_train_rfe.shape[1] != len(selected_feature_names):
         print(f"Error: Shape mismatch after RFE transform ({X_train_rfe.shape[1]}) vs selected names ({len(selected_feature_names)}).")
         # Fallback to generic names if mismatch
         selected_feature_names = [f'Selected_Feature_{i}' for i in range(X_train_rfe.shape[1])]


    print(f"RFE complete. Selected {len(selected_feature_names)} features.")

    return rfe, selected_feature_names, X_train_rfe, selected_mask


# --- 4. Model Training ---
# (No changes needed in this function)
def train_random_forest_regressor(X_train_final, y_train, random_state, rf_params=None):
    """
    Trains a RandomForestRegressor model on the final selected features.

    Args:
        X_train_final (np.ndarray): Training features after preprocessing & RFE.
        y_train (pd.Series): Training target variable.
        random_state (int): Random seed for the model.
        rf_params (dict or None): Parameters for the RandomForestRegressor.
                                  Defaults provided if None. 'random_state' will be overwritten.

    Returns:
        RandomForestRegressor: Fitted model object.
    Raises:
        RuntimeError, ValueError
    """
    print("\n--- 4. Training Final RandomForestRegressor Model ---")
    if X_train_final.shape[1] == 0:
        raise ValueError("Cannot train model with 0 features.")
    if len(X_train_final) == 0:
         raise ValueError("Cannot train model with 0 training samples.")

    # Define defaults and override with user params, ensuring random_state
    model_params = {
        'n_estimators': 150,
        'max_depth': 10,
        'n_jobs': -1,
        # Add other desired defaults here
    }
    if rf_params:
        model_params.update(rf_params)
    model_params['random_state'] = random_state # Ensure correct random state is used

    print(f"  Using RandomForestRegressor with params: {model_params}")

    try:
        model = RandomForestRegressor(**model_params)
        print(f"Training with {X_train_final.shape[0]} samples and {X_train_final.shape[1]} features.")
        model.fit(X_train_final, y_train)
        print("Model training complete.")
        return model
    except Exception as e:
        print(f"Error during model training: {e}")
        traceback.print_exc()
        raise RuntimeError(f"RandomForestRegressor training failed: {e}")


# --- 5. Performance Evaluation ---
# (Modified to accept final train/test data directly)
def evaluate_regression_performance(model, X_train_final, y_train, X_test_final, y_test):
    """
    Calculates regression metrics for train and test sets using final data.

    Args:
        model: Trained regression model.
        X_train_final (np.ndarray): Final training features (after preproc/RFE).
        y_train (pd.Series): Training target.
        X_test_final (np.ndarray): Final testing features (after preproc/RFE).
        y_test (pd.Series): Testing target.

    Returns:
        dict: Dictionary of performance metrics.
    Raises:
        RuntimeError
    """
    print("\n--- 5. Evaluating Regressor Performance ---")
    if X_test_final.shape[0] == 0:
        print("Error: Final Test set is empty. Cannot evaluate performance.")
        return {"Train": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan},
                "Test": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}}
    if X_train_final.shape[1] == 0 or X_test_final.shape[1] == 0:
        print("Error: Final data has 0 features. Cannot evaluate performance.")
        return {"Train": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan},
                "Test": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}}

    print(f"  Evaluating on {X_test_final.shape[1]} final features.")

    try:
        y_train_pred = model.predict(X_train_final)
        y_test_pred = model.predict(X_test_final)

        metrics = {
            "Train": {
                "R²": r2_score(y_train, y_train_pred),
                "MAE": mean_absolute_error(y_train, y_train_pred),
                "MSE": mean_squared_error(y_train, y_train_pred),
                "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            },
            "Test": {
                "R²": r2_score(y_test, y_test_pred),
                "MAE": mean_absolute_error(y_test, y_test_pred),
                "MSE": mean_squared_error(y_test, y_test_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            },
        }

        print("Scores on train set:")
        for metric, value in metrics["Train"].items():
            print(f"  {metric:<5}: {value:.4f}")

        print("\nScores on test set:")
        for metric, value in metrics["Test"].items():
            print(f"  {metric:<5}: {value:.4f}")

        # --- Plot Predicted vs Actual ---
        plt.figure(figsize=(12, 5))

        # Test Set
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test Data')
        all_values = np.concatenate([y_test.values, y_test_pred])
        min_val, max_val = all_values.min(), all_values.max()
        plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2, label='Ideal')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs. Actual (Test Set)")
        plt.legend()
        plt.grid(True)

        # Train Set
        plt.subplot(1, 2, 2)
        plt.scatter(y_train, y_train_pred, alpha=0.6, label='Train Data')
        all_values_train = np.concatenate([y_train.values, y_train_pred])
        min_val_train, max_val_train = all_values_train.min(), all_values_train.max()
        plt.plot([min_val_train, max_val_train], [min_val_train, max_val_train], '--', color='red', lw=2, label='Ideal')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs. Actual (Train Set)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return metrics
    except Exception as e:
        print(f"Error during performance evaluation: {e}")
        traceback.print_exc()
        return {"Train": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan},
                "Test": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}}


# --- 6. Residual Analysis ---
# (Modified to accept final test data directly)
def analyze_regression_residuals(model, X_test_final, y_test, df_original, test_original_indices, n_worst=20):
    """
    Performs residual analysis for regression using final test data.

    Args:
        model: Trained regression model.
        X_test_final (np.ndarray): Final testing features (after preproc/RFE).
        y_test (pd.Series): Testing target.
        df_original (pd.DataFrame): The original input DataFrame.
        test_original_indices (pd.Index): Original index of test samples.
        n_worst (int): Number of worst residuals to detail.

    Returns:
        tuple: Contains:
            - residuals (pd.Series): Residuals for the test set, indexed by original index.
            - worst_residuals_df (pd.DataFrame): Details of worst predictions.
            - y_test_pred (np.ndarray): Predictions on the test set. # Added for SHAP sheet later
    Raises:
        RuntimeError
    """
    print(f"\n--- 6. Analyzing Regression Residuals (Test Set) for top {n_worst} worst predictions ---")
    if X_test_final.shape[0] == 0 or y_test.empty:
        print("Error: Final Test set is empty. Cannot analyze residuals.")
        return pd.Series(dtype=float), pd.DataFrame(), np.array([]) # Return empty

    if X_test_final.shape[1] == 0:
        print("Error: Final data has 0 features. Cannot analyze residuals.")
        return pd.Series(dtype=float), pd.DataFrame(), np.array([]) # Return empty

    print(f"  Analyzing residuals based on {X_test_final.shape[1]} final features.")

    try:
        y_test_pred = model.predict(X_test_final)
        residuals_array = y_test.values - y_test_pred # Actual - Predicted

        # Create a pandas Series for residuals, using the original test indices
        residuals = pd.Series(residuals_array, index=test_original_indices, name='Residual')

        print(f"Residuals Summary: Min={residuals.min():.4f}, Mean={residuals.mean():.4f}, Max={residuals.max():.4f}, Std={residuals.std():.4f}")

        # --- Residual Plots ---
        plt.figure(figsize=(15, 5))
        # 1. Residuals vs. Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(y_test_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals (Actual - Predicted)")
        plt.title("Residuals vs. Predicted Values")
        plt.grid(True)
        # 2. Histogram of Residuals
        plt.subplot(1, 3, 2)
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals")
        plt.title("Histogram of Residuals")
        plt.grid(True)
        # 3. Q-Q Plot
        plt.subplot(1, 3, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title("Q-Q Plot of Residuals")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Identify Worst Residuals ---
        worst_residuals_df = pd.DataFrame()
        if n_worst > 0 and not residuals.empty:
            abs_residuals = residuals.abs()
            n_worst_actual = min(n_worst, len(residuals))
            if n_worst_actual > 0:
                # Sort by absolute residual to get indices
                worst_indices_sorted = abs_residuals.sort_values(ascending=False).index
                worst_original_indices = worst_indices_sorted[:n_worst_actual]

                print(f"\nDetailing Top {len(worst_original_indices)} Worst Predictions (by Absolute Residual):")

                # Get the original rows from the *original* dataframe
                original_data_worst = df_original.loc[worst_original_indices].copy()

                # Get the corresponding predictions and residuals using the original indices
                # Need to map original indices back to the order of y_test_pred if necessary
                # Easier: Re-index y_test_pred and residuals to match original_data_worst
                y_test_pred_series = pd.Series(y_test_pred, index=test_original_indices, name='Predicted_Target')

                worst_residuals_info = pd.DataFrame({
                    'Original_Index': worst_original_indices,
                    'Actual_Target': y_test.loc[worst_original_indices].values,
                    'Predicted_Target': y_test_pred_series.loc[worst_original_indices].values,
                    'Residual': residuals.loc[worst_original_indices].values,
                    'Absolute_Residual': abs_residuals.loc[worst_original_indices].values
                }).set_index('Original_Index') # Use original index temporarily for joining

                # Combine prediction info with original feature values using the index
                worst_residuals_df = worst_residuals_info.join(original_data_worst)
                worst_residuals_df = worst_residuals_df.sort_values('Absolute_Residual', ascending=False).reset_index() # Sort and reset index

                # Drop duplicate columns if any (e.g., Original_Index might be added twice)
                worst_residuals_df = worst_residuals_df.loc[:, ~worst_residuals_df.columns.duplicated()]

                print("Showing head of worst residuals dataframe (Prediction Info & Original Features):")
                cols_to_show = ['Original_Index', 'Actual_Target', 'Predicted_Target', 'Residual', 'Absolute_Residual']
                if y_test.name in worst_residuals_df.columns:
                     cols_to_show.append(y_test.name) # Add target column if name is valid
                existing_cols_to_show = [col for col in cols_to_show if col in worst_residuals_df.columns]
                print(worst_residuals_df[existing_cols_to_show].head().to_string())
            else:
                 print("Cannot identify worst residuals (n_worst > 0 but residuals are empty or test set too small).")

        else:
            print("Skipping worst residuals analysis (n_worst is 0 or test set empty).")

        return residuals, worst_residuals_df, y_test_pred # Return predictions too

    except Exception as e:
        print(f"Error during residual analysis: {e}")
        traceback.print_exc()
        return pd.Series(dtype=float), pd.DataFrame(), np.array([])


# --- 7. Feature Importance (MDI/Gini) ---
# (Renamed step number, no functional changes needed)
def analyze_rf_feature_importance_mdi(model, feature_names_after_rfe, plot=True, figsize=(16,6)):
    """
    Analyzes and plots feature importance (MDI) for Random Forest,
    including Lorenz curve and Gini coefficient.

    Args:
        model: Trained RandomForestRegressor instance (on final features).
        feature_names_after_rfe (list): List of names of features *used by the model*.
        plot (bool): Whether to plot the results.
        figsize (tuple): Figure size for plots.

    Returns:
        tuple: Contains:
            - importance_df (pd.DataFrame): Sorted feature importances with Lorenz data.
            - gini_coefficient (float): Calculated Gini coefficient (0 to 1).
            Returns (pd.DataFrame(), None) if analysis fails or no features/importance.
    Raises:
        RuntimeError
    """
    print("\n--- 7. Analyzing MDI Feature Importance (Mean Impurity Decrease) ---")
    if not feature_names_after_rfe:
        print("No final features available. Cannot analyze MDI feature importance.")
        return pd.DataFrame(), None

    print(f"Analyzing MDI importance for {len(feature_names_after_rfe)} features used by the model.")
    print("Note: MDI Importance is for the features *after* preprocessing and RFE.")

    if not hasattr(model, 'feature_importances_'):
        print("Error: Model does not have 'feature_importances_'. Cannot analyze MDI.")
        return pd.DataFrame(), None

    importances = model.feature_importances_

    if len(importances) != len(feature_names_after_rfe):
         print(f"Error: Mismatch in number of MDI importances ({len(importances)}) and provided feature names ({len(feature_names_after_rfe)}). Cannot proceed.")
         # Try to pad names if possible, otherwise return empty
         if len(importances) > len(feature_names_after_rfe):
             feature_names_after_rfe.extend([f'Unknown_Feat_{i}' for i in range(len(feature_names_after_rfe), len(importances))])
         elif len(importances) < len(feature_names_after_rfe):
             feature_names_after_rfe = feature_names_after_rfe[:len(importances)]
         else: # Should not happen if length mismatch is the issue
             return pd.DataFrame(), None

    # Create DataFrame and sort by importance (descending)
    importance_df = pd.DataFrame({
        'Feature': feature_names_after_rfe,
        'Importance_MDI': importances # Renamed column
    }).sort_values('Importance_MDI', ascending=False).reset_index(drop=True)

    print("\nMDI Feature Importances (Top 10 or all):")
    print(importance_df.head(10).to_string(index=False))

    # --- Lorenz Curve and Gini Coefficient Calculation ---
    imp_sorted = importance_df['Importance_MDI'].values
    imp_sorted[imp_sorted < 0] = 0 # Ensure non-negative

    total_importance = np.sum(imp_sorted)
    if np.isclose(total_importance, 0):
        print("Warning: Total MDI feature importance is zero or close to zero. Cannot calculate Gini/Lorenz.")
        return importance_df, None

    cum_imp = np.cumsum(imp_sorted)
    cum_imp_norm = cum_imp / total_importance
    num_features = len(importance_df)
    proportion_features = np.linspace(0, 1, num_features + 1)
    cum_imp_norm_with_origin = np.insert(cum_imp_norm, 0, 0)
    area_under_lorenz = np.trapz(cum_imp_norm_with_origin, proportion_features)
    gini_coefficient = 1 - 2 * area_under_lorenz
    gini_coefficient = np.clip(gini_coefficient, 0, 1)

    print(f"\nGini Coefficient of MDI Feature Importance: {gini_coefficient:.4f}")

    if not importance_df.empty:
         importance_df['Cumulative_MDI_Importance_Normalized'] = cum_imp_norm.tolist()

    if plot:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            # Feature Importance Plot
            y_label_fontsize = 'small' if len(feature_names_after_rfe) > 20 else None
            # Plot only top N features if too many
            plot_df = importance_df.head(30) if len(importance_df) > 30 else importance_df
            sns.barplot(x='Importance_MDI', y='Feature', data=plot_df, palette='viridis', ax=ax1)
            title = 'Top MDI Feature Importances' if len(importance_df) > 30 else 'MDI Feature Importances'
            ax1.set_title(f'{title} (Final Features)')
            ax1.set_xlabel('Mean Decrease in Impurity (MDI)')
            ax1.set_ylabel('Feature')
            ax1.tick_params(axis='y', labelsize=y_label_fontsize)
            ax1.grid(axis='x')

            # Lorenz Curve Plot
            ax2.plot(proportion_features, cum_imp_norm_with_origin, marker='.', label='Lorenz Curve (MDI)')
            ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Line of Equality')
            ax2.set_title(f'Lorenz Curve (MDI Gini = {gini_coefficient:.3f})')
            ax2.set_xlabel('Cumulative Proportion of Features')
            ax2.set_ylabel('Cumulative Proportion of MDI Importance')
            ax2.legend()
            ax2.grid(True)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)

            fig.tight_layout()
            plt.show()
        except Exception as e:
             print(f"Error during MDI importance plotting: {e}")
             traceback.print_exc()

    return importance_df, gini_coefficient


# --- 8. NEW: SHAP Value Analysis ---

def analyze_shap_values(model, X_test_final, feature_names_final, test_original_indices, X_test_original=None, plot=True, figsize=(12, 10)):
    """
    Calculates SHAP values for the test set, generates summary plots,
    and returns SHAP values and importance summary.

    Args:
        model: Trained RandomForestRegressor model.
        X_test_final (np.ndarray): Final testing features (after preproc/RFE).
        feature_names_final (list): List of names corresponding to X_test_final columns.
        test_original_indices (pd.Index): Original indices of the test samples.
        X_test_original (pd.DataFrame, optional): Test features *before* scaling/PCA,
                                                  used for potentially more interpretable
                                                  dependence plots if provided. Must have
                                                  same indices as X_test_final.
        plot (bool): Whether to generate SHAP summary plots.
        figsize (tuple): Figure size for SHAP plots.

    Returns:
        tuple: Contains:
            - shap_values_df (pd.DataFrame): SHAP values for each test instance and feature,
                                             indexed by original test index. (None if SHAP fails)
            - shap_summary_df (pd.DataFrame): Mean absolute SHAP value per feature, sorted.
                                              (None if SHAP fails)
            - shap_expected_value (float): The base value (mean prediction over background data).
                                           (None if SHAP fails)
    """
    print("\n--- 8. Analyzing SHAP Values (Test Set) ---")

    if not SHAP_INSTALLED:
        print("SHAP library not found. Skipping SHAP analysis.")
        return None, None, None
    if X_test_final.shape[1] == 0:
        print("No final features available. Cannot calculate SHAP values.")
        return None, None, None
    if X_test_final.shape[0] == 0:
        print("Test set is empty. Cannot calculate SHAP values.")
        return None, None, None
    if len(feature_names_final) != X_test_final.shape[1]:
        print(f"Error: Mismatch between number of feature names ({len(feature_names_final)}) and data columns ({X_test_final.shape[1]}). Cannot proceed with SHAP.")
        return None, None, None

    # SHAP works best with DataFrames with column names
    X_test_final_df = pd.DataFrame(X_test_final, columns=feature_names_final, index=test_original_indices)

    print(f"Calculating SHAP values for {X_test_final_df.shape[0]} test samples and {X_test_final_df.shape[1]} features...")

    try:
        # Use TreeExplainer for RandomForest
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for the test set
        # For TreeExplainer, shap_values output is [N_samples, N_features] for regression
        shap_values_test = explainer.shap_values(X_test_final_df)

        # Get the expected value (base value for the explainer)
        # For TreeExplainer, this is often the mean prediction on the background data (implicitly training data)
        shap_expected_value = explainer.expected_value
        if isinstance(shap_expected_value, (list, np.ndarray)): # Handle multi-output case if it arises, though unlikely for RF regressor
            shap_expected_value = shap_expected_value[0]
        print(f"  SHAP Expected Value (Base): {shap_expected_value:.4f}")


        # --- Create DataFrames for output ---
        # 1. DataFrame of SHAP values per instance
        shap_values_df = pd.DataFrame(shap_values_test, columns=feature_names_final, index=test_original_indices)

        # 2. DataFrame for SHAP Summary (Global Importance)
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
        shap_summary_df = pd.DataFrame({
            'Feature': feature_names_final,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

        print("\nSHAP Feature Importance Summary (Top 10 or all):")
        print(shap_summary_df.head(10).to_string(index=False))

        # --- Generate Plots ---
        if plot:
            print("\nGenerating SHAP summary plots...")
            plt.figure(figsize=figsize) # Adjust figure size as needed

            # Plot 1: SHAP Summary Plot (Bar - Global Importance)
            plt.subplot(2, 1, 1) # Arrange plots vertically
            shap.summary_plot(shap_values_test, X_test_final_df, plot_type="bar", show=False)
            # Find the current Axes object created by shap and set title
            ax1 = plt.gca()
            ax1.set_title("SHAP Feature Importance (Mean Absolute SHAP)")
            # Manually adjust layout if needed after plotting
            # plt.tight_layout(pad=2.0) # Adjust padding


            # Plot 2: SHAP Summary Plot (Beeswarm - Value Distribution & Impact)
            plt.subplot(2, 1, 2)
            shap.summary_plot(shap_values_test, X_test_final_df, plot_type="dot", show=False) # Use 'dot' or default
            ax2 = plt.gca()
            ax2.set_title("SHAP Value Distribution and Impact on Prediction")
            plt.tight_layout(pad=2.0) # Adjust padding between subplots
            plt.show()


            # Optional: Dependence Plots for top N features
            # Use X_test_original if available for more interpretable x-axis
            data_for_dependence = X_test_original if X_test_original is not None else X_test_final_df
            if data_for_dependence is not None and isinstance(data_for_dependence, pd.DataFrame):
                 # Ensure indices align if using X_test_original
                 if not data_for_dependence.index.equals(X_test_final_df.index):
                     print("Warning: Index mismatch between X_test_original and X_test_final_df. Skipping dependence plots with original features.")
                     data_for_dependence = X_test_final_df # Fallback to processed data

                 n_dependence_plots = min(5, len(feature_names_final)) # Plot top 5
                 top_features = shap_summary_df['Feature'].head(n_dependence_plots).tolist()
                 print(f"\nGenerating SHAP dependence plots for top {n_dependence_plots} features: {top_features}")
                 for feature in top_features:
                     if feature in data_for_dependence.columns:
                         try:
                             plt.figure() # Create a new figure for each dependence plot
                             shap.dependence_plot(
                                 feature,
                                 shap_values_test,
                                 data_for_dependence, # Use original or processed data
                                 interaction_index="auto", # Let SHAP choose interaction feature
                                 show=False
                             )
                             ax_dep = plt.gca()
                             xlabel = f"{feature}"
                             if X_test_original is not None and feature in X_test_original.columns:
                                 xlabel += " (Original Scale)"
                             elif feature.startswith("PC"):
                                 xlabel += " (Principal Component)"

                             ax_dep.set_xlabel(xlabel)
                             ax_dep.set_title(f"SHAP Dependence Plot for {feature}")
                             plt.tight_layout()
                             plt.show()
                         except Exception as dep_e:
                             print(f"  Could not generate dependence plot for '{feature}': {dep_e}")
                     else:
                          print(f"  Feature '{feature}' not found in data_for_dependence for dependence plot.")

            else:
                print("Skipping SHAP dependence plots (missing suitable data).")


        return shap_values_df, shap_summary_df, shap_expected_value

    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        traceback.print_exc()
        return None, None, None


# --- 9. Save Results ---
# (Modified to include SHAP sheets)

def save_results_to_excel(results, filename="rf_regression_analysis_results.xlsx"):
    """
    Saves the key results of the regression analysis to an Excel file,
    including SHAP information if available.

    Args:
        results (dict): Dictionary containing analysis results.
        filename (str): Path to the output Excel file.
    """
    print(f"\n--- 9. Exporting Results to Excel: {filename} ---")
    if not filename:
         print("No output filename provided. Skipping results export.")
         return
    if not filename.endswith(('.xlsx', '.xls')):
        filename += '.xlsx'
        print(f"Warning: Filename did not end with .xlsx or .xls. Appending .xlsx: {filename}")


    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # --- Sheet 1: Run Info ---
            if 'run_info' in results and isinstance(results['run_info'], dict):
                print("  - Preparing 'Run Info' sheet.")
                run_info_export = {}
                for k, v in results['run_info'].items():
                    # Handle common non-serializable types more robustly
                    if isinstance(v, (dict, list, tuple, set)):
                         run_info_export[k] = str(v)
                    elif isinstance(v, (BaseEstimator, StandardScaler, RobustScaler, MinMaxScaler, PCA, RFE)):
                         # Get class name and maybe key params if simple
                         try:
                             params_str = str(v.get_params(deep=False)) # Get only top-level params
                             run_info_export[k] = f"{type(v).__name__}({params_str})"
                         except:
                             run_info_export[k] = type(v).__name__
                    elif pd.isna(v):
                         run_info_export[k] = 'NaN'
                    elif isinstance(v, (np.ndarray, pd.Series, pd.Index)):
                        run_info_export[k] = f"[Object of type {type(v).__name__}, Length: {len(v)}]" # Don't save large arrays
                    else:
                         run_info_export[k] = v

                # Add Gini coefficient directly to run info if available
                if 'mdi_gini_coefficient' in results and results['mdi_gini_coefficient'] is not None:
                     run_info_export['MDI_Feature_Importance_Gini'] = results['mdi_gini_coefficient']

                # Add SHAP Expected Value if available
                if 'shap_expected_value' in results and results['shap_expected_value'] is not None:
                    run_info_export['SHAP_Expected_Value'] = results['shap_expected_value']

                run_info_df = pd.DataFrame(list(run_info_export.items()), columns=['Parameter', 'Value'])
                run_info_df.to_excel(writer, sheet_name='Run Info', index=False)
                print("  - Saved 'Run Info' sheet.")
            else:
                print("  - 'run_info' not found or not a dict in results, skipping sheet.")

            # --- Sheet 2: Performance Metrics ---
            if 'metrics' in results and isinstance(results['metrics'], dict) and results['metrics'].get('Test'):
                 print("  - Preparing 'Metrics' sheet.")
                 metrics_df = pd.DataFrame(results['metrics']).reset_index().rename(columns={'index': 'Metric'})
                 metrics_df = metrics_df.melt(id_vars='Metric', var_name='Set', value_name='Value')
                 metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                 print("  - Saved 'Metrics' sheet.")
            elif 'metrics' in results:
                 status = 'Evaluation Failed or Incomplete' if results['metrics'] is None or not results['metrics'].get('Test') else 'Metrics key exists but invalid format'
                 pd.DataFrame([{'Status': status}]).to_excel(writer, sheet_name='Metrics', index=False)
                 print(f"  - Saved 'Metrics' sheet ({status}).")
            else:
                print("  - 'metrics' not found in results, skipping sheet.")

            # --- Sheet 3: MDI Feature Importance ---
            # Renamed from 'Feature Importance' for clarity
            if 'mdi_importance_df' in results and isinstance(results['mdi_importance_df'], pd.DataFrame) and not results['mdi_importance_df'].empty:
                print("  - Preparing 'MDI Importance' sheet.")
                results['mdi_importance_df'].to_excel(writer, sheet_name='MDI Importance', index=False)
                print("  - Saved 'MDI Importance' sheet.")
            else:
                 print("  - 'mdi_importance_df' not found, empty, or not a DataFrame in results, skipping sheet.")

            # --- Sheet 4: SHAP Importance Summary --- (NEW)
            if 'shap_summary_df' in results and isinstance(results['shap_summary_df'], pd.DataFrame) and not results['shap_summary_df'].empty:
                print("  - Preparing 'SHAP Importance Summary' sheet.")
                results['shap_summary_df'].to_excel(writer, sheet_name='SHAP Importance Summary', index=False)
                print("  - Saved 'SHAP Importance Summary' sheet.")
            else:
                 status = "SHAP analysis skipped or failed" if 'shap_summary_df' in results and results['shap_summary_df'] is None else "'shap_summary_df' not found, empty, or not a DataFrame"
                 print(f"  - {status}, skipping 'SHAP Importance Summary' sheet.")
                 if 'shap_summary_df' in results and results['shap_summary_df'] is None: # Add a sheet indicating failure if SHAP was attempted
                      pd.DataFrame([{'Status': 'SHAP analysis skipped or failed'}]).to_excel(writer, sheet_name='SHAP Importance Summary', index=False)


            # --- Sheet 5: SHAP Values (Test Set) --- (NEW - Local Explanations)
            if 'shap_values_df' in results and isinstance(results['shap_values_df'], pd.DataFrame) and not results['shap_values_df'].empty:
                print("  - Preparing 'SHAP Values (Test Set)' sheet.")
                # Combine SHAP values with actual, predicted, and residuals for context
                shap_local_df = results['shap_values_df'].copy()

                # Add prefix to SHAP value columns to avoid name clashes
                shap_local_df.columns = ['SHAP_' + col for col in shap_local_df.columns]

                # Add Actual Target (y_test)
                if 'y_test' in results and isinstance(results['y_test'], pd.Series):
                    # Ensure index alignment
                    y_test_aligned, shap_local_df = results['y_test'].align(shap_local_df, join='right', axis=0)
                    shap_local_df['Actual_Target'] = y_test_aligned
                else:
                    print("  - Warning: Could not find 'y_test' in results to add to SHAP values sheet.")

                # Add Predicted Target (y_test_pred) - Ensure it's available in results
                if 'y_test_pred' in results and isinstance(results['y_test_pred'], np.ndarray) and len(results['y_test_pred']) == len(shap_local_df):
                     # Need to align predictions with the original index
                     pred_series = pd.Series(results['y_test_pred'], index=shap_local_df.index, name='Predicted_Target')
                     shap_local_df['Predicted_Target'] = pred_series
                elif 'y_test_pred' in results:
                     print(f"  - Warning: 'y_test_pred' found but type/length mismatch ({type(results['y_test_pred'])}, len={len(results.get('y_test_pred',''))}). Cannot add to SHAP values sheet.")
                else:
                    print("  - Warning: Could not find 'y_test_pred' in results to add to SHAP values sheet.")

                # Add Residuals
                if 'residuals' in results and isinstance(results['residuals'], pd.Series):
                    # Ensure index alignment
                    residuals_aligned, shap_local_df = results['residuals'].align(shap_local_df, join='right', axis=0)
                    shap_local_df['Residual'] = residuals_aligned
                else:
                    print("  - Warning: Could not find 'residuals' in results to add to SHAP values sheet.")

                # Add SHAP Expected Value (Base Value) as a column for reference
                if 'shap_expected_value' in results and results['shap_expected_value'] is not None:
                     shap_local_df['SHAP_Base_Value'] = results['shap_expected_value']

                # Reorder columns for clarity (identifiers first, then SHAP values)
                id_cols = ['Actual_Target', 'Predicted_Target', 'Residual', 'SHAP_Base_Value']
                shap_cols = [col for col in shap_local_df.columns if col.startswith('SHAP_') and col != 'SHAP_Base_Value']
                final_cols = [col for col in id_cols if col in shap_local_df.columns] + shap_cols
                shap_local_df = shap_local_df[final_cols]


                shap_local_df.to_excel(writer, sheet_name='SHAP Values (Test Set)', index=True, index_label='Original_Index')
                print("  - Saved 'SHAP Values (Test Set)' sheet.")
            else:
                 status = "SHAP analysis skipped or failed" if 'shap_values_df' in results and results['shap_values_df'] is None else "'shap_values_df' not found, empty, or not a DataFrame"
                 print(f"  - {status}, skipping 'SHAP Values (Test Set)' sheet.")
                 if 'shap_values_df' in results and results['shap_values_df'] is None: # Add a sheet indicating failure if SHAP was attempted
                     pd.DataFrame([{'Status': 'SHAP analysis skipped or failed'}]).to_excel(writer, sheet_name='SHAP Values (Test Set)', index=False)

            # --- Sheet 6: Worst Residuals ---
            # Renumbered sheet
            if 'worst_residuals_df' in results and isinstance(results['worst_residuals_df'], pd.DataFrame) and not results['worst_residuals_df'].empty:
                print("  - Preparing 'Worst Residuals' sheet.")
                worst_residuals_df_cleaned_cols = results['worst_residuals_df'].copy()
                worst_residuals_df_cleaned_cols.columns = worst_residuals_df_cleaned_cols.columns.astype(str)
                # Limit columns saved for large original dataframes? Maybe not necessary.
                worst_residuals_df_cleaned_cols.to_excel(writer, sheet_name='Worst Residuals', index=False)
                print("  - Saved 'Worst Residuals' sheet.")
            else:
                print("  - 'worst_residuals_df' not found, empty, or not a DataFrame in results, skipping sheet.")

        print(f"Successfully saved results to {filename}")

    except ImportError:
        print("\nError: 'openpyxl' library is required to write Excel files.")
        print("Please install it using: pip install openpyxl")
    except Exception as e:
        print(f"\nError writing results to Excel file '{filename}': {e}")
        traceback.print_exc()




# --- 10. Main Orchestrator ---
# (MODIFIED to store X_train_processed, X_test_processed, feature_names_after_processing, and mask name)

def run_full_regression_pipeline(
    df,
    target_col,
    scaler_type='Standard',
    pca_n_comp=None,
    test_size=0.2,
    random_state=42,
    target_feature_ratio=0.6, # Ratio for RFE
    rfe_estimator_params=None, # Params for RF used *inside* RFE
    n_worst_residuals=20,
    rf_params={'n_estimators': 150, 'max_depth': 10, 'n_jobs': -1}, # Params for the FINAL RandomForest model (if not tuning afterwards)
    output_filename="rf_regression_analysis_results.xlsx",
    run_shap_analysis=True, # Option to skip SHAP if needed
    show_plots=True # Control plotting globally
    ):
    """
    Runs the complete regression analysis pipeline including SHAP analysis.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target variable column.
        scaler_type (str): Scaler type ('Standard', 'Robust', 'MinMax', 'None').
        pca_n_comp (int/float/None): PCA components or variance.
        test_size (float): Test split proportion.
        random_state (int): Global random seed.
        target_feature_ratio (float/None): RFE feature ratio to keep (0, 1). Skip if None/ >=1 / <=0.
        rfe_estimator_params (dict/None): Params for RF *inside* RFE.
        n_worst_residuals (int): Number of worst residuals to detail.
        rf_params (dict): Parameters for the FINAL RandomForestRegressor (if not tuning afterwards).
        output_filename (str): Path to save results Excel file.
        run_shap_analysis (bool): If True and SHAP is installed, run SHAP analysis.
        show_plots (bool): If True, display generated matplotlib/seaborn/SHAP plots.

    Returns:
        dict: A dictionary containing key results of the analysis.
              Returns None if a critical initial step fails.
    """
    print("\n--- Starting Full Regression Pipeline ---")
    print(f"Random State used throughout pipeline: {random_state}")
    start_time = time.time()

    # Turn off interactive plotting if show_plots is False
    current_backend = plt.get_backend()
    if not show_plots and current_backend != 'agg':
        plt.ioff() # Turn interactive mode off
        print("Plot display disabled (show_plots=False). Plots might still be generated but not shown.")
        plotting_disabled = True
    elif current_backend == 'agg':
         print("Detected non-interactive backend ('agg'). Plots will be generated but not displayed.")
         plotting_disabled = True
    else:
        plt.ion() # Ensure interactive mode is on (default)
        plotting_disabled = False


    results = {'run_info': {
        'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': 'RandomForestRegressor',
        'target_column': target_col,
        'test_size': test_size,
        'random_state': random_state,
        'scaler_type': scaler_type,
        'pca_n_comp': pca_n_comp,
        'target_feature_ratio_rfe': target_feature_ratio,
        'rfe_used': False, # Will be updated later
        'rfe_estimator_params': rfe_estimator_params if rfe_estimator_params else 'Defaults',
        'n_worst_residuals': n_worst_residuals,
        'rf_params_final_model': rf_params,
        'run_shap_analysis_requested': run_shap_analysis,
        'shap_analysis_performed': False, # Will be updated later
        'show_plots': show_plots,
        'output_filename': output_filename
    }}

    # --- 1. Data Preparation & Splitting ---
    try:
        (X_train, X_test, y_train, y_test,
         initial_numeric_feature_names, test_original_indices, train_original_indices) = split_and_select_numeric_data(
             df, target_col, test_size, random_state
         )
        results['y_train'] = y_train # Store Series
        results['y_test'] = y_test   # Store Series
        results['initial_numeric_feature_names'] = initial_numeric_feature_names
        results['test_original_indices'] = test_original_indices
        results['train_original_indices'] = train_original_indices
        # Store X_test (numeric, imputed) for potential use in SHAP dependence plots
        results['X_test_original_numeric'] = X_test.copy() # Store DataFrame copy

        print("Step 1: Data preparation and splitting complete.")
        if X_train.shape[1] == 0:
             print("\nCRITICAL ERROR: No numeric features found after splitting. Cannot proceed.")
             results['run_info']['error_step'] = 1
             results['run_info']['error_message'] = "No numeric features after splitting"
             save_results_to_excel(results, output_filename)
             if plotting_disabled: plt.close('all')
             return None
    except Exception as e:
        print(f"\nCRITICAL ERROR during Step 1 (Data Prep & Split): {e}")
        traceback.print_exc()
        results['run_info']['error_step'] = 1
        results['run_info']['error_message'] = str(e)
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 2. Apply Preprocessing (Scaling and PCA) ---
    try:
        # Pass X_train, X_test as DataFrames as expected by apply_preprocessing
        (X_train_processed, X_test_processed, feature_names_after_processing,
         scaler_instance, pca_instance) = apply_preprocessing(
             X_train, X_test, scaler_type, pca_n_comp, random_state
         )
        # --- Store the processed data and names in results dictionary ---
        results['X_train_processed'] = X_train_processed # Store numpy array
        results['X_test_processed'] = X_test_processed   # Store numpy array
        results['feature_names_after_processing'] = feature_names_after_processing # Store list
        # ---------------------------------------------------------------

        results['scaler_instance'] = scaler_instance # Store object
        results['pca_instance'] = pca_instance       # Store object
        results['run_info']['n_features_after_preprocessing'] = X_train_processed.shape[1]

        print("Step 2: Preprocessing complete.")
        if X_train_processed.shape[1] == 0:
             print("\nCRITICAL ERROR: 0 features remaining after preprocessing. Cannot proceed.")
             results['run_info']['error_step'] = 2
             results['run_info']['error_message'] = "0 features after preprocessing"
             save_results_to_excel(results, output_filename)
             if plotting_disabled: plt.close('all')
             return None
    except Exception as e:
        print(f"\nCRITICAL ERROR during Step 2 (Preprocessing): {e}")
        traceback.print_exc()
        results['run_info']['error_step'] = 2
        results['run_info']['error_message'] = str(e)
        results['scaler_instance'] = None
        results['pca_instance'] = None
        results['feature_names_after_processing'] = []
        results['X_train_processed'] = np.array([]) # Store empty arrays on failure
        results['X_test_processed'] = np.array([])
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 3. Perform RFE OR Skip ---
    rfe_object = None
    X_train_final = X_train_processed # Default if RFE fails/skipped
    X_test_final = X_test_processed   # Default if RFE fails/skipped
    selected_feature_names_final = feature_names_after_processing # Default if RFE fails/skipped
    selected_mask_final = np.ones(X_train_processed.shape[1], dtype=bool) # Default mask (select all)

    # Determine if RFE should run based on ratio and features available
    rfe_should_run = (target_feature_ratio is not None and 0 < target_feature_ratio < 1.0 and X_train_processed.shape[1] > 1) # Need > 1 to eliminate
    results['run_info']['rfe_used'] = rfe_should_run # Record if RFE was attempted

    if rfe_should_run:
        try:
            # RFE estimator params (simple default, can be customized)
            rfe_est_params_clean = rfe_estimator_params.copy() if rfe_estimator_params is not None else {'n_estimators': 150, 'max_depth': 10, 'n_jobs': rf_params.get('n_jobs', -1)}
            rfe_object, selected_feature_names_rfe, X_train_rfe, selected_mask_rfe = perform_rfe(
                X_train_processed, y_train, feature_names_after_processing,
                target_feature_ratio, random_state, rfe_est_params_clean # Use cleaned params
            )
            # Update final variables ONLY if RFE ran successfully and selected features > 0
            if rfe_object is not None and X_train_rfe is not None and X_train_rfe.shape[1] > 0:
                 selected_feature_names_final = selected_feature_names_rfe
                 X_train_final = X_train_rfe
                 # IMPORTANT: Transform the TEST set using the fitted RFE object
                 # Use the X_test_processed from Step 2
                 X_test_final = rfe_object.transform(X_test_processed)
                 selected_mask_final = selected_mask_rfe
                 print("Step 3: RFE selection applied successfully.")
                 print(f"  Final features selected: {len(selected_feature_names_final)}")
                 print(f"  Final data shapes: X_train={X_train_final.shape}, X_test={X_test_final.shape}")
            else:
                 # RFE was attempted but failed or selected 0 features, use pre-RFE data
                 print("Step 3: RFE was run but failed or selected 0 features. Using pre-RFE data.")
                 rfe_object = None # Ensure rfe_object is None if RFE didn't effectively run
                 results['run_info']['rfe_used'] = False # Mark RFE as not effectively used
                 # X_train_final etc. retain their default (pre-RFE) values
                 selected_feature_names_final = feature_names_after_processing
                 selected_mask_final = np.ones(X_train_processed.shape[1], dtype=bool)
                 print(f"  Using pre-RFE data. Final train shape: {X_train_final.shape}")


        except Exception as e:
            print(f"\nCRITICAL ERROR during Step 3 (RFE): {e}")
            traceback.print_exc()
            results['run_info']['error_step'] = 3
            results['run_info']['error_message'] = str(e)
            # Use pre-RFE data in case of critical failure
            rfe_object = None
            results['run_info']['rfe_used'] = False # Mark RFE as failed
            # X_train_final etc. retain their pre-RFE values (X_train_processed etc.)
            selected_feature_names_final = feature_names_after_processing
            selected_mask_final = np.ones(X_train_processed.shape[1], dtype=bool)
            print("  Proceeding with pre-RFE features due to error.")
    else:
        print("Step 3: RFE skipped (target_feature_ratio not in (0, 1) or <= 1 feature).")
        results['run_info']['rfe_used'] = False
        # Data and names remain as initialized (X_train_processed, etc.)


    results['rfe_object'] = rfe_object # Store the fitted RFE object (or None)
    # --- Store the RFE mask using the name expected by the tuning orchestrator ---
    results['selected_mask_after_rfe'] = selected_mask_final # Store boolean mask
    # ---------------------------------------------------------------------------
    results['selected_feature_names_final'] = selected_feature_names_final # Store list of final names
    results['run_info']['n_features_final'] = X_train_final.shape[1]

    if X_train_final.shape[1] == 0:
        print("\nCRITICAL ERROR: 0 features remaining after RFE/selection process. Cannot proceed.")
        results['run_info']['error_step'] = 3
        results['run_info']['error_message'] = "0 final features after RFE/selection"
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 4. Train Final Model ---
    # This model is only trained if we are NOT doing tuning immediately afterwards.
    # If tuning is done, the tuned model will be the "final" model.
    # For this pipeline, we assume it's the end of the line unless tuning is called separately.
    # The current structure is fine, we train a model based on the determined features.
    try:
        final_model = train_random_forest_regressor(
            X_train_final, y_train, random_state=random_state, rf_params=rf_params
        )
        results['model'] = final_model # Store the fitted model
        print("Step 4: Final model training complete.")
    except Exception as e:
        print(f"\nCRITICAL ERROR during Step 4 (Model Training): {e}")
        traceback.print_exc()
        results['run_info']['error_step'] = 4
        results['run_info']['error_message'] = str(e)
        results['model'] = None
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 5. Evaluate Performance ---
    try:
        # Pass the data that was actually used for training the model
        metrics = evaluate_regression_performance(
            results['model'], X_train_final, y_train, X_test_final, y_test
        )
        results['metrics'] = metrics
        print("Step 5: Performance evaluation complete.")
    except Exception as e:
        print(f"\nError during Step 5 (Performance Evaluation): {e}")
        traceback.print_exc()
        results['run_info']['warning_step_5'] = str(e)
        results['metrics'] = {"Train": {}, "Test": {}} # Mark evaluation as failed

    # --- 6. Analyze Residuals ---
    try:
        # Pass the data that was actually used for predicting
        residuals, worst_residuals_df, y_test_pred = analyze_regression_residuals(
            results['model'], X_test_final, y_test, df, test_original_indices, n_worst=n_worst_residuals
        )
        results['residuals'] = residuals # pd.Series, indexed by original index
        results['worst_residuals_df'] = worst_residuals_df # pd.DataFrame
        results['y_test_pred'] = y_test_pred # np.ndarray, corresponds to X_test_final order

        print("Step 6: Residual analysis complete.")
    except Exception as e:
        print(f"\nError during Step 6 (Residual Analysis): {e}")
        traceback.print_exc()
        results['run_info']['warning_step_6'] = str(e)
        results['residuals'] = pd.Series(dtype=float)
        results['worst_residuals_df'] = pd.DataFrame()
        results['y_test_pred'] = np.array([]) # Mark as failed

    # --- 7. Analyze MDI Feature Importance ---
    try:
        # Pass the model and the list of FINAL selected names
        mdi_importance_df, mdi_gini_coefficient = analyze_rf_feature_importance_mdi(
            results['model'], selected_feature_names_final, plot=show_plots
        )
        results['mdi_importance_df'] = mdi_importance_df
        results['mdi_gini_coefficient'] = mdi_gini_coefficient
        # Add Gini to run_info for easy access
        results['run_info']['MDI_Feature_Importance_Gini'] = mdi_gini_coefficient
        print("Step 7: MDI Feature importance analysis complete.")
    except Exception as e:
        print(f"\nError during Step 7 (MDI Importance Analysis): {e}")
        traceback.print_exc()
        results['run_info']['warning_step_7'] = str(e)
        results['mdi_importance_df'] = pd.DataFrame()
        results['mdi_gini_coefficient'] = None

    # --- 8. Analyze SHAP Values ---
    shap_values_df, shap_summary_df, shap_expected_value = None, None, None # Initialize
    if run_shap_analysis and SHAP_INSTALLED:
        results['run_info']['shap_analysis_performed'] = True
        try:
            # Pass original numeric X_test (DataFrame) for dependence plots if available
            X_test_orig_for_shap = results.get('X_test_original_numeric')
            # Pass X_test_final (numpy array), final feature names, and original test indices
            shap_values_df, shap_summary_df, shap_expected_value = analyze_shap_values(
                results['model'], # Pass the fitted model
                X_test_final, # Final features (numpy array)
                selected_feature_names_final, # Final feature names (list)
                test_original_indices, # Original indices (Index)
                X_test_original=X_test_orig_for_shap, # Original numeric X_test (DataFrame or None)
                plot=show_plots # Control plotting
            )
            results['shap_values_df'] = shap_values_df
            results['shap_summary_df'] = shap_summary_df
            results['shap_expected_value'] = shap_expected_value
            results['run_info']['SHAP_Expected_Value'] = shap_expected_value
            print("Step 8: SHAP analysis complete.")
        except Exception as e:
            print(f"\nError during Step 8 (SHAP Analysis): {e}")
            traceback.print_exc()
            results['run_info']['warning_step_8'] = str(e)
            results['shap_values_df'] = None # Mark as failed
            results['shap_summary_df'] = None
            results['shap_expected_value'] = None
            results['run_info']['SHAP_Expected_Value'] = 'Analysis Failed'
    elif not SHAP_INSTALLED:
         print("\nStep 8: SHAP Analysis skipped (SHAP library not installed).")
         results['run_info']['shap_analysis_performed'] = False
    else: # SHAP installed but run_shap_analysis is False
         print("\nStep 8: SHAP Analysis skipped (run_shap_analysis=False).")
         results['run_info']['shap_analysis_performed'] = False


    # --- 9. Save Results ---
    save_results_to_excel(results, output_filename)
    print("Step 9: Results export attempt complete.")

    end_time = time.time()
    total_duration = end_time - start_time
    results['run_info']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['run_info']['total_duration_seconds'] = round(total_duration, 2)
    print(f"\n--- Full Regression Pipeline Finished in {total_duration:.2f} seconds ---")

    # Final summary printout
    if 'metrics' in results and results['metrics'] and 'Test' in results['metrics']:
        print("\n--- Final Test Set Performance ---")
        test_metrics = results['metrics']['Test']
        if test_metrics:
            for metric, value in test_metrics.items():
                 print(f"  {metric:<5}: {value:.4f}")
        else:
            print("  Test metrics calculation failed or resulted empty.")
    else:
        print("\n--- Final Test Set Performance: Not Available or Evaluation Failed ---")

    # Close all plots if plotting was disabled
    if plotting_disabled:
        plt.close('all')

    # Re-enable interactive plotting if it was disabled by this function
    if not show_plots and current_backend != 'agg':
         plt.ion()


    return results

# TUNING HYPERPARAMETERS
# My_functions_p4.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
import sys
import warnings

# --- Data Splitting (Unchanged) ---
def split_data(df: pd.DataFrame,
               target_name: str,
               test_size: float = 0.2,
               random_state: int = 42):
    """Splits the dataframe into training and testing sets."""
    print(f"\n--- Splitting Data (Test Size: {test_size}, Random State: {random_state}) ---")
    try:
        X = df.drop(columns=[target_name])
        y = df[target_name]
    except KeyError:
        raise KeyError(f"Target column '{target_name}' not found in DataFrame columns: {df.columns.tolist()}")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError(f"Target column '{target_name}' must be numeric for regression.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test


# --- Tuning Function (Accepts Prefixed Params) ---
def tune_random_forest(X_train: pd.DataFrame,
                       y_train: pd.Series,
                       pipeline_param_distributions: dict, # Expects prefixed keys now
                       n_iter: int,
                       cv: int,
                       scoring: str,
                       random_state: int,
                       apply_scaling: bool = False,
                       apply_pca: bool = False,
                       pca_n_components=None):
    """
    Performs hyperparameter tuning for RandomForestRegressor within a Pipeline.
    Expects parameter keys in pipeline_param_distributions to be prefixed
    (e.g., 'rf__n_estimators').
    """
    print(f"\n--- Starting Randomized Search (Pipeline: Scale={apply_scaling}, PCA={apply_pca}) ---")
    print(f"   (n_iter={n_iter}, cv={cv}, scoring='{scoring}')")
    search_start_time = time.time()

    # Basic checks
    if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
        print("\nWarning: Training data contains NaN values. Ensure missing values are handled.")
    if apply_pca and pca_n_components is None:
         warnings.warn("apply_pca is True, but pca_n_components is None. PCA will use default n_components.", UserWarning)

    # 1. Define Pipeline Steps
    steps = []
    if apply_scaling:
        steps.append(('scaler', StandardScaler()))
    if apply_pca:
        if isinstance(pca_n_components, int) and pca_n_components > X_train.shape[1]:
             pca_n_components = X_train.shape[1] # Cap components at num features
             warnings.warn(f"pca_n_components adjusted to {pca_n_components} (max features).", UserWarning)
        steps.append(('pca', PCA(n_components=pca_n_components, random_state=random_state)))

    steps.append(('rf', RandomForestRegressor(random_state=random_state, n_jobs=-1)))
    pipeline = Pipeline(steps)
    print(f"   Pipeline steps: {[s[0] for s in pipeline.steps]}")

    # 2. Verify Parameter Keys (Optional check, as orchestrator should handle prefixing)
    print("   Hyperparameter search space:")
    for param, dist in pipeline_param_distributions.items():
         step_name = param.split('__')[0]
         if step_name not in [s[0] for s in pipeline.steps]:
             # This check is less critical if the orchestrator ensures correct prefixes
             warnings.warn(f"Parameter '{param}' targets step '{step_name}', which might not be in the current pipeline configuration: {[s[0] for s in pipeline.steps]}. Ensure configuration matches.", UserWarning)
         if hasattr(dist, 'dist'):
             print(f"     {param}: {dist.dist.name}({dist.args}, {dist.kwds})")
         else:
             print(f"     {param}: {dist}")

    # 3. Setup and Run Randomized Search
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=pipeline_param_distributions, # Use the prefixed dict directly
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_state,
        return_train_score=True,
        verbose=1
    )
    random_search.fit(X_train, y_train)
    search_end_time = time.time()
    search_time = search_end_time - search_start_time
    print(f"Search finished in {search_time:.2f} seconds.")

    best_estimator = random_search.best_estimator_
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values(by=f'rank_test_score', ascending=True)
    results_df.columns = results_df.columns.str.replace(":", "_", regex=False)

    print("\n--- Best Pipeline Results ---")
    print(f"Best Score ({scoring}): {random_search.best_score_:.4f}")
    print("Best Parameters Found:")
    best_pipeline_params = {k: v for k, v in random_search.best_params_.items()
                             if k.split('__')[0] in [s[0] for s in best_estimator.steps]}
    for param, value in best_pipeline_params.items():
        print(f"  {param}: {value}")

    return best_estimator, results_df, search_time


# --- Saving Results (Unchanged) ---
def save_tuning_results(results_df: pd.DataFrame, best_params: dict, best_score: float,
                        scoring: str, cv: int, n_iter: int, search_time: float,
                        random_state: int, results_dir: str):
    """Saves the detailed tuning results (CSV) and best parameters (TXT)."""
    print("\n--- Saving Tuning Results ---")
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, 'random_forest_tuning_results.csv')
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"Detailed tuning results saved to: {results_filename}")
    except Exception as e: print(f"Error saving results CSV: {e}")
    best_params_filename = os.path.join(results_dir, 'best_params.txt')
    try:
        with open(best_params_filename, 'w') as f:
            f.write(f"--- Random Forest Pipeline Tuning Summary ---\n")
            f.write(f"Scoring Metric: {scoring}\n")
            f.write(f"Best CV Score ({scoring}): {best_score:.4f}\n")
            f.write("\nBest Pipeline Parameters:\n")
            for param, value in best_params.items():
                if f'param_{param}' in results_df.columns or param in best_params:
                    f.write(f"  {param}: {value}\n")
            f.write("\n--- Search Configuration ---\n")
            f.write(f"CV Folds: {cv}\n")
            f.write(f"Search Iterations (n_iter): {n_iter}\n")
            f.write(f"Random State Seed: {random_state}\n")
            f.write(f"Tuning Time: {search_time:.2f} seconds\n")
        print(f"Best parameters summary saved to: {best_params_filename}")
    except Exception as e: print(f"Error saving best parameters text file: {e}")


# --- Plotting Results (Accepts Prefixed Params) ---
def plot_tuning_results(results_df: pd.DataFrame,
                        pipeline_param_distributions: dict, # Expects prefixed keys
                        scoring: str,
                        results_dir: str):
    """Generates and saves boxplots using prefixed parameter keys."""
    print("\n--- Generating Plots ---")
    try:
        # Identify varied params using prefixed keys directly from the input dict
        varied_params_keys = list(pipeline_param_distributions.keys())
        plotted_params = [] # Store the actual column names found in results_df

        for prefixed_key in varied_params_keys:
            param_col_name = f'param_{prefixed_key}'
            if param_col_name in results_df.columns and results_df[param_col_name].nunique() > 1:
                plotted_params.append(prefixed_key) # Store the key used in the search

        if not plotted_params:
            print("No parameters varied significantly. Skipping boxplots.")
            return

        n_params = len(plotted_params)
        n_cols = 3; n_rows = (n_params + n_cols - 1) // n_cols
        score_col = 'mean_test_score'
        plt.figure(figsize=(n_cols * 6, n_rows * 5))

        for i, param_key in enumerate(plotted_params): # Use the prefixed keys
            ax = plt.subplot(n_rows, n_cols, i + 1)
            param_col = f'param_{param_key}'
            plot_data = results_df[[param_col, score_col]].copy()
            if plot_data[param_col].isnull().any():
                plot_data[param_col] = plot_data[param_col].fillna('None')
            unique_vals = plot_data[param_col].unique()
            plot_order = None
            try: plot_order = sorted(unique_vals)
            except TypeError:
                try: plot_order = sorted([str(uv) for uv in unique_vals])
                except Exception: plot_order = None

            sns.boxplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                        palette="viridis", order=plot_order)
            ax.set_title(f"Score vs {param_key}") # Show prefixed name in title
            ax.set_xlabel(param_key)
            ax.set_ylabel(f'Mean CV Score ({scoring})')
            if len(unique_vals) > 7 or any(len(str(v)) > 12 for v in unique_vals):
                ax.tick_params(axis='x', rotation=60)
            else: ax.tick_params(axis='x', rotation=0)

        plt.suptitle(f'Hyperparameter Tuning Results (Score: {scoring})', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_filename = os.path.join(results_dir, 'hyperparameter_tuning_boxplots.png')
        plt.savefig(plot_filename, bbox_inches='tight'); print(f"Boxplots saved to: {plot_filename}")
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules: plt.show()
        else: plt.close()
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback; traceback.print_exc()


# --- Saving Model (Unchanged) ---
def save_best_model(estimator, filepath: str):
    """Saves the trained model/pipeline to a file using joblib."""
    print(f"\n--- Saving Best Model/Pipeline ---")
    try:
        joblib.dump(estimator, filepath)
        print(f"Best model/pipeline saved to: {filepath}")
    except Exception as e: print(f"Error saving best model/pipeline to {filepath}: {e}")


# --- Evaluating Model (Unchanged) ---
def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the final model/pipeline on the hold-out test set."""
    print("\n--- Evaluating Best Model/Pipeline on Test Set ---")
    y_pred = estimator.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test Set R2 Score: {r2:.4f}")
    print(f"Test Set RMSE: {rmse:.4f}")
    print(f"Test Set MAE: {mae:.4f}")
    return {'r2': r2, 'rmse': rmse, 'mae': mae}


# --- Feature Importance (Unchanged) ---
def get_feature_importances(estimator, feature_names: list = None, top_n: int = None):
    """Extracts feature importances. Adapts for pipelines with/without PCA."""
    final_estimator = None; transformer_steps = []
    if isinstance(estimator, Pipeline):
        final_estimator = estimator.steps[-1][1]
        transformer_steps = estimator.steps[:-1]
    else: final_estimator = estimator

    if not hasattr(final_estimator, 'feature_importances_'):
        print("Warning: Final estimator lacks 'feature_importances_'.")
        return None
    importances = final_estimator.feature_importances_
    pca_step = next((step for name, step in transformer_steps if isinstance(step, PCA)), None)

    if pca_step:
        print("Note: PCA applied. Importances relate to Principal Components.")
        n_components = len(importances)
        component_names = [f'PC_{i+1}' for i in range(n_components)]
        importance_df = pd.DataFrame({'Feature/Component': component_names, 'Importance': importances})
    elif feature_names:
        if len(importances) != len(feature_names):
             print(f"Warning: Mismatch between importances ({len(importances)}) and feature names ({len(feature_names)}). Using generic names.")
             feature_col = [f'feature_{i}' for i in range(len(importances))]
        else: feature_col = feature_names
        importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})
    else:
         print("Warning: Feature names not provided. Using generic names.")
         feature_col = [f'feature_{i}' for i in range(len(importances))]
         importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})

    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return importance_df.head(top_n) if top_n else importance_df


# --- Orchestrator Function ---
def orchestrate_rf_tuning(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    # Non-prefixed param distributions expected here
    param_distributions_config: dict,
    n_iter: int,
    cv_folds: int,
    scoring_metric: str,
    random_state: int,
    results_dir: str,
    best_model_filename: str,
    # Preprocessing args
    apply_scaling: bool = False,
    apply_pca: bool = False,
    pca_n_components = None,
    save_importance: bool = True,
    top_n_features: int = None
):
    """
    Runs the entire Random Forest tuning pipeline.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target variable.
        test_size (float): Proportion for the test set split.
        param_distributions_config (dict): Hyperparameter distributions WITHOUT pipeline prefixes.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Metric for tuning evaluation.
        random_state (int): Seed for reproducibility.
        results_dir (str): Directory to save outputs.
        best_model_filename (str): Filename for the saved best model/pipeline.
        apply_scaling (bool): Whether to apply StandardScaler.
        apply_pca (bool): Whether to apply PCA.
        pca_n_components: n_components for PCA (if apply_pca=True).
        save_importance (bool): Whether to calculate and save feature importances.
        top_n_features (int, optional): Number of top features to save/print.

    Returns:
        tuple: (best_pipeline, test_scores)
               best_pipeline: The best fitted pipeline object.
               test_scores: Dictionary of scores from evaluating on the test set.
    """
    print("===== Starting Random Forest Tuning Pipeline =====")
    print(f"Input data shape: {df.shape}")
    pipeline_start_time = time.time()

    # 1. Split Data
    X_train, X_test, y_train, y_test = split_data(
        df, target_column, test_size=test_size, random_state=random_state
    )
    original_feature_names = list(X_train.columns)

    # 2. Prepare Parameter Distributions for Pipeline
    # Add 'rf__' prefix to parameters intended for the RandomForestRegressor step
    pipeline_param_distributions = {}
    for key, value in param_distributions_config.items():
        # Basic assumption: if no prefix, it's for the RF model ('rf')
        if '__' not in key:
            pipeline_param_distributions[f'rf__{key}'] = value
        else:
            # If user provided prefixed keys already, use them directly
            # Add checks here if you plan to tune other steps like 'pca__'
            pipeline_param_distributions[key] = value

    # 3. Tune Hyperparameters (with optional scaling/PCA via pipeline)
    best_pipeline, tuning_results_df, search_time = tune_random_forest(
        X_train=X_train,
        y_train=y_train,
        # Pass the explicitly prefixed dictionary
        pipeline_param_distributions=pipeline_param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring=scoring_metric,
        random_state=random_state,
        apply_scaling=apply_scaling,
        apply_pca=apply_pca,
        pca_n_components=pca_n_components
    )

    # 4. Save Tuning Results
    # Determine best score from results_df based on scoring metric
    score_col = 'mean_test_score'
    best_cv_score = tuning_results_df[score_col].iloc[tuning_results_df[f'rank_test_score'].idxmin()]

    save_tuning_results(
        results_df=tuning_results_df,
        best_params=best_pipeline.get_params(), # Get all params from the best pipeline
        best_score=best_cv_score,
        scoring=scoring_metric,
        cv=cv_folds,
        n_iter=n_iter,
        search_time=search_time,
        random_state=random_state,
        results_dir=results_dir
    )

    # 5. Plot Tuning Results
    plot_tuning_results(
        results_df=tuning_results_df,
        # Pass the prefixed dict, plot function expects/uses these keys
        pipeline_param_distributions=pipeline_param_distributions,
        scoring=scoring_metric,
        results_dir=results_dir
    )

    # 6. Save the Best Pipeline
    model_filepath = os.path.join(results_dir, best_model_filename)
    save_best_model(best_pipeline, model_filepath)

    # 7. Evaluate the Final Pipeline on the Test Set
    test_scores = evaluate_model(best_pipeline, X_test, y_test)

    # 8. Get and Save Feature/Component Importances (Optional)
    if save_importance:
        print("\n--- Extracting Feature/Component Importances ---")
        importances_df = get_feature_importances(
            best_pipeline,
            feature_names=original_feature_names,
            top_n=top_n_features # Use top_n parameter
        )
        if importances_df is not None:
            fi_filename = os.path.join(results_dir, 'feature_component_importances.csv')
            try:
                # Save potentially filtered dataframe (if top_n is used)
                full_importances_df = get_feature_importances(best_pipeline, original_feature_names)
                if full_importances_df is not None:
                     full_importances_df.to_csv(fi_filename, index=False)
                     print(f"Full importances saved to: {fi_filename}")
                else: print("Could not generate full importances df for saving.")

                print(f"Top {top_n_features if top_n_features else 'All'} Features/Components:")
                print(importances_df) # Print top N or all if top_n is None

            except Exception as e:
                print(f"Error saving/printing importances: {e}")
        else:
             print("Could not extract feature importances.")

    pipeline_end_time = time.time()
    print("\n===== Pipeline Finished =====")
    print(f"Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds")

    return best_pipeline, test_scores



# on a RFE NOW below

# rf_grid_evaluator.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

def split_data_for_grid(df: pd.DataFrame,
                        target_name: str,
                        test_size: float = 0.2,
                        random_state: int = 42):
    print(f"\n--- Splitting Data (Test Size: {test_size}, Random State: {random_state}) ---")
    try:
        X = df.drop(columns=[target_name])
        y = df[target_name]
    except KeyError:
        raise KeyError(f"Target column '{target_name}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Target column must be numeric.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate_rf_grid_on_train(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              param_grid: dict,
                              cv: int,
                              scoring: str,
                              random_state: int):
    print(f"--- Starting RandomForest Grid Evaluation on Training Data ---")
    print(f"   Grid Size: {np.prod([len(v) for v in param_grid.values()])} combinations")
    print(f"   CV Folds: {cv}, Scoring: '{scoring}'")
    start_time = time.time()

    if X_train.isnull().sum().sum() > 0 or y_train.isnull().sum() > 0:
         print("Warning: Training data contains NaN values.")

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    end_time = time.time()
    print(f"Grid search finished in {end_time - start_time:.2f} seconds.")

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by='rank_test_score', ascending=True)
    results_df.columns = results_df.columns.str.replace(":", "_", regex=False)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_estimator = grid_search.best_estimator_ # The model trained on full train set with best params

    print("\n--- Grid Evaluation Best CV Results (on Train data) ---")
    print(f"Best CV Score ({scoring}): {best_score:.4f}")
    print("Best Parameters in Grid:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return results_df, best_params, best_score, best_estimator


def save_grid_results(results_df: pd.DataFrame,
                      best_params: dict,
                      best_score: float,
                      param_grid: dict,
                      scoring: str,
                      cv: int,
                      results_dir: str):
    print("\n--- Saving Grid Evaluation Results ---")
    os.makedirs(results_dir, exist_ok=True)

    results_filename = os.path.join(results_dir, 'rf_grid_evaluation_results.csv')
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"Detailed grid results saved to: {results_filename}")
    except Exception as e:
        print(f"Error saving results CSV: {e}")

    summary_filename = os.path.join(results_dir, 'rf_grid_evaluation_summary.txt')
    try:
        with open(summary_filename, 'w') as f:
            f.write("--- Random Forest Grid Evaluation Summary ---\n")
            f.write(f"Scoring Metric (CV on Train): {scoring}\n")
            f.write(f"CV Folds: {cv}\n")
            f.write(f"\nBest CV Score Found: {best_score:.4f}\n")
            f.write("Best Parameters:\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n--- Parameter Grid Evaluated ---\n")
            for param, values in param_grid.items():
                 f.write(f"  {param}: {values}\n")
        print(f"Summary saved to: {summary_filename}")
    except Exception as e:
        print(f"Error saving summary TXT: {e}")


def plot_grid_results(results_df: pd.DataFrame,
                      param_grid: dict,
                      scoring: str,
                      results_dir: str):
    print("\n--- Generating Grid Evaluation Plots ---")
    try:
        varied_params = [param for param, values in param_grid.items() if len(values) > 1]

        if not varied_params:
            print("No parameters varied in the grid. Skipping plots.")
            return

        n_params = len(varied_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        score_col = 'mean_test_score'

        plt.figure(figsize=(n_cols * 5, n_rows * 4.5))

        for i, param in enumerate(varied_params):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            param_col = f'param_{param}'

            if param_col not in results_df.columns:
                print(f"Warning: Column '{param_col}' not found. Skipping plot for '{param}'.")
                continue

            plot_data = results_df[[param_col, score_col]].copy()

            if plot_data[param_col].isnull().any():
                plot_data[param_col] = plot_data[param_col].fillna('None')
            if not pd.api.types.is_numeric_dtype(plot_data[param_col]):
                 plot_data[param_col] = plot_data[param_col].astype(str)

            unique_vals = plot_data[param_col].unique()
            plot_order = None
            try:
                numeric_vals = pd.to_numeric(unique_vals)
                plot_order = sorted(numeric_vals)
            except (ValueError, TypeError):
                try: plot_order = sorted([str(uv) for uv in unique_vals])
                except Exception: plot_order = None

            sns.boxplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                        palette="coolwarm", order=plot_order)
            sns.pointplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                          join=False, color='black', markers='d', scale=0.75, errorbar=None, order=plot_order)

            ax.set_title(f"{param}")
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel(f'Mean CV Score ({scoring})')
            if len(unique_vals) > 6 or any(len(str(v)) > 8 for v in unique_vals):
                ax.tick_params(axis='x', rotation=45, labelsize=9)
            else: ax.tick_params(axis='x', rotation=0, labelsize=10)
            ax.tick_params(axis='y', labelsize=9)

        plt.suptitle(f'Grid Search CV Results (Score: {scoring})', fontsize=14, y=1.03)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_filename = os.path.join(results_dir, 'grid_evaluation_boxplots.png')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
        print(f"Boxplots saved to: {plot_filename}")
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules: plt.show()
        else: plt.close()
    except Exception as e:
        print(f"\nError generating plots: {e}")
        import traceback; traceback.print_exc()


def evaluate_on_test(estimator, X_test: pd.DataFrame, y_test: pd.Series):
    print("\n--- Evaluating Best Estimator on Test Set ---")
    y_pred = estimator.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test Set R2 Score: {r2:.4f}")
    print(f"Test Set RMSE: {rmse:.4f}")
    return {'r2': r2, 'rmse': rmse}

# --- Orchestrator ---
def run_grid_evaluation_pipeline(
    df: pd.DataFrame,
    target_column: str,
    param_grid: dict,
    test_size: float = 0.2,
    cv_folds: int = 5,
    scoring_metric: str = 'r2',
    random_state: int = 42,
    results_dir: str = "rf_grid_eval_output"
):
    print("===== Running RF Grid Evaluation Pipeline =====")
    pipeline_start_time = time.time()

    # 1. Split Data
    X_train, X_test, y_train, y_test = split_data_for_grid(
        df=df,
        target_name=target_column,
        test_size=test_size,
        random_state=random_state
    )

    # 2. Run Grid Search on Training Data
    results_data, best_grid_params, best_grid_cv_score, best_estimator = evaluate_rf_grid_on_train(
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        cv=cv_folds,
        scoring=scoring_metric,
        random_state=random_state
    )

    # 3. Save Results (from CV on train data)
    if results_data is not None:
        save_grid_results(
            results_df=results_data,
            best_params=best_grid_params,
            best_score=best_grid_cv_score,
            param_grid=param_grid,
            scoring=scoring_metric,
            cv=cv_folds,
            results_dir=results_dir
        )

        # 4. Plot Results (from CV on train data)
        plot_grid_results(
            results_df=results_data,
            param_grid=param_grid,
            scoring=scoring_metric,
            results_dir=results_dir
        )

    # 5. Evaluate the best estimator found on the Test Set
    test_set_scores = evaluate_on_test(
        estimator=best_estimator,
        X_test=X_test,
        y_test=y_test
    )

    # Optional: Save test set scores
    test_score_filename = os.path.join(results_dir, 'test_set_scores.txt')
    try:
        with open(test_score_filename, 'w') as f:
            f.write("--- Test Set Evaluation Scores ---\n")
            f.write(f"Best Parameters Used (from CV):\n")
            for param, value in best_grid_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\nScores on Hold-Out Test Set:\n")
            for metric, score in test_set_scores.items():
                f.write(f"  {metric.upper()}: {score:.4f}\n")
        print(f"\nTest set scores saved to: {test_score_filename}")
    except Exception as e:
        print(f"Error saving test set scores: {e}")


    pipeline_end_time = time.time()
    print(f"\n===== Pipeline Finished in {pipeline_end_time - pipeline_start_time:.2f} seconds =====")

    return best_estimator, test_set_scores



############
##########
###################
###############
#####################


# --- Adapted Plot Function ---
def plot_tuning_results_adapted(results_df: pd.DataFrame,
                        param_distributions: dict,
                        scoring: str,
                        results_dir: str):
    """
    Generates and saves boxplots for tuning results.
    Adapts to work with results_df where param columns are 'param_<param_name>'
    and the input param_distributions uses non-prefixed names.
    """
    # Requires matplotlib.pyplot as plt and seaborn as sns imports
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n--- Generating Plots (Adapted) ---")
    try:
        # Identify varied params using the NON-prefixed keys from the input dict
        # Check against the 'param_' prefixed column names in the results_df
        varied_params_keys = [
            key for key in param_distributions.keys()
            if f'param_{key}' in results_df.columns and results_df[f'param_{key}'].nunique() > 1
        ]

        if not varied_params_keys:
            print("No parameters varied significantly or matched results columns. Skipping boxplots.")
            return

        n_params = len(varied_params_keys)
        n_cols = 3; n_rows = (n_params + n_cols - 1) // n_cols
        score_col = 'mean_test_score' # Assuming scoring is 'mean_test_score' after prefixing
        plt.figure(figsize=(n_cols * 6, n_rows * 5))

        for i, param_key in enumerate(varied_params_keys):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            param_col = f'param_{param_key}'

            plot_data = results_df[[param_col, score_col]].copy()
            if plot_data[param_col].isnull().any():
                plot_data[param_col] = plot_data[param_col].fillna('None')
            unique_vals = plot_data[param_col].unique()
            plot_order = None
            try: plot_order = sorted(unique_vals)
            except TypeError:
                try: plot_order = sorted([str(uv) for uv in unique_vals])
                except Exception: plot_order = None

            sns.boxplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                        palette="viridis", order=plot_order)
            sns.pointplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                          join=False, color='black', markers='d', scale=0.75, errorbar=None, order=plot_order)

            ax.set_title(f"Score vs {param_key}")
            ax.set_xlabel(param_key.replace("_", " ").title())
            ax.set_ylabel(f'Mean CV Score ({scoring})') # Use original scoring name
            if len(unique_vals) > 6 or any(len(str(v)) > 8 for v in unique_vals):
                ax.tick_params(axis='x', rotation=45, labelsize=9)
            else: ax.tick_params(axis='x', rotation=0, labelsize=10)
            ax.tick_params(axis='y', labelsize=9)

        plt.suptitle(f'Grid Search CV Results (Score: {scoring})', fontsize=14, y=1.03)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_filename = os.path.join(results_dir, 'grid_evaluation_boxplots.png') # Renamed filename slightly
        plt.savefig(plot_filename, bbox_inches='tight', dpi=150)
        print(f"Boxplots saved to: {plot_filename}")
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules: plt.show()
        else: plt.close()
    except Exception as e:
        print(f"\nError generating plots (adapted): {e}")
        traceback.print_exc()


# --- Modified Tuning Function ---
def tune_random_forest_on_preprocessed(
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    param_distributions: dict,
    n_iter: int,
    cv: int,
    scoring: str,
    random_state: int,
    results_dir: str,
    best_model_filename: str,
    feature_names_after_processing: list = None,
    save_importance: bool = True,
    top_n_features: int = None
):
    """
    Performs hyperparameter tuning for RandomForestRegressor directly
    on already processed (scaled, PCA'd, RFE'd) data.
    Does NOT build a preprocessing pipeline internally.
    """
    print("\n--- Starting Randomized Search (Directly on Processed Data) ---")
    print(f"   Input data shape: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")
    print(f"   (n_iter={n_iter}, cv={cv}, scoring='{scoring}', random_state={random_state})")
    search_start_time = time.time()

    if X_train_processed.shape[1] == 0 or len(X_train_processed) == 0:
         raise ValueError("Processed training data is empty or has 0 features. Cannot tune.")

    # Ensure CV folds is not more than samples
    if len(X_train_processed) < cv:
        warnings.warn(f"Number of samples ({len(X_train_processed)}) is less than CV folds ({cv}). Adjusting CV to {len(X_train_processed)}.", UserWarning)
        cv = len(X_train_processed) if len(X_train_processed) > 0 else 1
        if cv == 0: raise ValueError("0 training samples available.")


    # Define the Estimator (just RF, no pipeline)
    # Set fixed parameters like random_state and n_jobs outside the distributions
    fixed_params = {'random_state': random_state, 'n_jobs': -1}

    estimator = RandomForestRegressor(**fixed_params)

    # Check if any parameters from the distributions are NOT valid for the estimator
    valid_rf_params = estimator.get_params().keys()
    for param_name in param_distributions.keys():
         if param_name not in valid_rf_params:
              warnings.warn(f"Parameter '{param_name}' from distributions is not a valid parameter for RandomForestRegressor. It will be ignored by RandomizedSearchCV.", UserWarning)

    # Setup and Run Randomized Search
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    try:
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1,
            random_state=random_state,
            return_train_score=True,
            verbose=1
        )

        random_search.fit(X_train_processed, y_train)

    except Exception as e:
         print(f"\nCRITICAL ERROR during RandomizedSearchCV fit: {e}")
         traceback.print_exc()
         # Minimal error handling for return
         dummy_df = pd.DataFrame({'params': ['Error'], f'mean_test_{scoring}': [np.nan], 'rank_test_score': [1]})
         return None, dummy_df, 0.0, {'r2':np.nan,'rmse':np.nan,'mae':np.nan}, {}


    search_end_time = time.time()
    search_time = search_end_time - search_start_time
    print(f"Search finished in {search_time:.2f} seconds.")

    best_estimator = random_search.best_estimator_
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values(by=f'rank_test_score', ascending=True)
    results_df.columns = results_df.columns.str.replace(":", "_", regex=False)

    print("\n--- Best Tuning Results (CV on Train Data) ---")
    best_score = random_search.best_score_
    print(f"Best Score ({scoring}): {best_score:.4f}")
    best_params = random_search.best_params_
    print("Best Parameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Evaluate on the held-out test set *after* tuning is complete
    test_scores = evaluate_model(best_estimator, X_test_processed, y_test)

    # Save Tuning Results
    save_tuning_results(
        results_df=results_df,
        best_params=best_params,
        best_score=best_score,
        scoring=scoring,
        cv=cv,
        n_iter=n_iter,
        search_time=search_time,
        random_state=random_state,
        results_dir=results_dir
    )

    # Plot Tuning Results (using the adapted function)
    plot_tuning_results_adapted(
        results_df=results_df,
        param_distributions=param_distributions, # Pass the non-prefixed dict
        scoring=scoring,
        results_dir=results_dir
    )

    # Save the Best Model
    model_filepath = os.path.join(results_dir, best_model_filename)
    save_best_model(best_estimator, model_filepath)

    # Get and Save Feature Importance (for the best tuned model)
    if save_importance:
        print("\n--- Extracting Feature Importances (Tuned Model) ---")
        importances_df = get_feature_importances(
            best_estimator,
            feature_names=feature_names_after_processing, # Pass the final feature names
            top_n=top_n_features
        )
        if importances_df is not None:
            fi_filename = os.path.join(results_dir, 'tuned_feature_importances.csv')
            try:
                importances_df.to_csv(fi_filename, index=False) # Save the potentially filtered df
                print(f"Feature importances saved to: {fi_filename}")
                # Print top N or all if top_n is None
                print(f"Top {top_n_features if top_n_features else 'All'} Features:")
                print(importances_df)
            except Exception as e:
                print(f"Error saving/printing importances: {e}")
        else:
             print("Could not extract feature importances for tuned model.")


    return best_estimator, results_df, search_time, test_scores, best_params


# --- Orchestrator Function for Tuning Preprocessed Data ---
# (No major functional changes, just ensures it correctly extracts keys
#  based on the modified run_full_regression_pipeline output)
def orchestrate_tuned_rf_on_pipeline_output(
    results_from_run_full_regression_pipeline: dict,
    param_distributions_config: dict, # Hyperparameter distributions for RF (NO prefixes)
    n_iter: int,
    cv_folds: int,
    scoring_metric: str,
    random_state: int,
    results_dir: str = "tuned_rf_on_pipeline_output",
    best_model_filename: str = "tuned_random_forest_model.joblib",
    save_importance: bool = True,
    top_n_features: int = None,
    show_plots: bool = True # Add plotting control here too
):
    """
    Takes results from run_full_regression_pipeline, extracts the processed data
    (after scaling, PCA, and RFE), and performs hyperparameter tuning on
    a RandomForestRegressor trained directly on this preprocessed data.

    Args:
        results_from_run_full_regression_pipeline (dict): Results dictionary from a previous call
                                                          to run_full_regression_pipeline.
        param_distributions_config (dict): Hyperparameter distributions for RandomizedSearchCV.
                                           Keys should be model parameter names (e.g., 'n_estimators').
        n_iter (int): Number of iterations for RandomizedSearchCV.
        cv_folds (int): Number of CV folds.
        scoring_metric (str): Scoring metric for tuning.
        random_state (int): Random seed.
        results_dir (str): Directory to save tuning results and best model.
        best_model_filename (str): Filename for the saved best model.
        save_importance (bool): Whether to save feature importance of the best model.
        top_n_features (int or None): Number of top features to include in saved importance.
        show_plots (bool): Whether to display plots generated during tuning evaluation/importance.


    Returns:
        tuple: Contains:
            - best_model: The fitted best model found during tuning.
            - tuned_test_scores (dict): Test set performance metrics for the best model.
            Returns (None, None) if a critical step fails.
    """
    print("\n===== Starting Tuned RF Pipeline on Preprocessed Data =====")
    print(f"Random State used throughout tuning: {random_state}")
    pipeline_start_time = time.time()

    # Turn off interactive plotting if show_plots is False
    current_backend = plt.get_backend()
    if not show_plots and current_backend != 'agg':
        plt.ioff()
        print("Plot display disabled (show_plots=False) for tuning evaluation/importance.")
        plotting_disabled = True
    elif current_backend == 'agg':
         print("Detected non-interactive backend ('agg'). Plots for tuning evaluation/importance will be generated but not displayed.")
         plotting_disabled = True
    else:
        plt.ion()
        plotting_disabled = False


    # 1. Extract Data and Transformers from Previous Run Results
    print("Step 1: Extracting data and transformations from previous pipeline results...")
    try:
        # --- Extract the data *after* scaling/PCA (as stored in the modified pipeline) ---
        X_train_processed_initial = results_from_run_full_regression_pipeline.get('X_train_processed')
        X_test_processed_initial = results_from_run_full_regression_pipeline.get('X_test_processed')
        # ---------------------------------------------------------------------------------

        # Extract y_train and y_test
        y_train = results_from_run_full_regression_pipeline.get('y_train')
        y_test = results_from_run_full_regression_pipeline.get('y_test')

        # Extract the fitted RFE object and the mask it produced
        rfe_object = results_from_run_full_regression_pipeline.get('rfe_object')
        # --- Extract the mask using the name it's now stored with ---
        selected_mask_after_rfe = results_from_run_full_regression_pipeline.get('selected_mask_after_rfe')
        # -------------------------------------------------------------

        # Extract feature names corresponding to X_train_processed_initial
        feature_names_after_preprocessing = results_from_run_full_regression_pipeline.get('feature_names_after_processing', [])

        # Basic checks for critical data
        if X_train_processed_initial is None or X_test_processed_initial is None or y_train is None or y_test is None:
             raise ValueError("Critical data (X_train_processed, X_test_processed, y_train, or y_test) is missing in the results dictionary.")
        if not isinstance(X_train_processed_initial, np.ndarray) or not isinstance(X_test_processed_initial, np.ndarray):
             raise TypeError("Processed data is not in expected numpy array format.")
        if not isinstance(y_train, pd.Series): y_train = pd.Series(y_train) # Ensure Series
        if not isinstance(y_test, pd.Series): y_test = pd.Series(y_test)     # Ensure Series

        print(f"  Extracted initial processed data shapes: X_train={X_train_processed_initial.shape}, X_test={X_test_processed_initial.shape}")
        print(f"  RFE object found: {'Yes' if rfe_object is not None else 'No'}")
        print(f"  RFE mask found: {'Yes' if selected_mask_after_rfe is not None else 'No'}")


    except KeyError as e:
        print(f"\nCRITICAL ERROR: Required key '{e}' not found in the results dictionary. Ensure you ran run_full_regression_pipeline and captured its output correctly and it completed successfully.")
        # Also indicate which keys were found vs missing if possible for better debugging
        print(f"  Keys found in results dictionary: {results_from_run_full_regression_pipeline.keys()}")
        return None, None
    except Exception as e:
        print(f"\nCRITICAL ERROR during data extraction: {e}")
        traceback.print_exc()
        return None, None

    # 2. Apply RFE Transformation to the Data *Before* Tuning
    print("\nStep 2: Applying RFE transformation to the data...")
    try:
        # Determine if RFE was *intended and successful* in the previous run
        # Check if the rfe_object exists AND if the mask was applied (i.e., ratio was < 1.0)
        # We need the mask to be a numpy array for the shape check
        mask_is_valid = (selected_mask_after_rfe is not None and isinstance(selected_mask_after_rfe, np.ndarray))

        rfe_was_applied_successfully = (
             rfe_object is not None and
             mask_is_valid and
             np.sum(selected_mask_after_rfe) > 0 and # Ensure RFE selected at least one feature
             np.sum(selected_mask_after_rfe) < X_train_processed_initial.shape[1] # Ensure RFE actually reduced features
        )

        if rfe_was_applied_successfully:
            n_features_after_rfe = np.sum(selected_mask_after_rfe)
            print(f"  Applying RFE transformation (selected {n_features_after_rfe} features) to data...")
            # Use the stored RFE object to transform the processed data
            X_train_final = rfe_object.transform(X_train_processed_initial)
            X_test_final = rfe_object.transform(X_test_processed_initial)

            # Feature names are the names of the features *selected by RFE*
            if feature_names_after_preprocessing and len(feature_names_after_preprocessing) == X_train_processed_initial.shape[1]:
                 feature_names_final = [name for name, selected in zip(feature_names_after_preprocessing, selected_mask_after_rfe) if selected]
            else:
                 # Fallback if names mapping fails
                 print("Warning: Cannot reliably map RFE mask to original processed feature names. Using generic names for final features.")
                 feature_names_final = [f'Selected_Feature_{i}' for i in range(X_train_final.shape[1])]


            print(f"  Data shape after RFE transformation: X_train={X_train_final.shape}, X_test={X_test_final.shape}")

        else: # If RFE was skipped or failed/selected all features in the previous pipeline
            print("  RFE was skipped, failed, or selected all features in the previous pipeline. Using full processed data for tuning.")
            X_train_final = X_train_processed_initial
            X_test_final = X_test_processed_initial
            feature_names_final = feature_names_after_preprocessing # Use names after scaling/PCA
            print(f"  Using data shape: X_train={X_train_final.shape}, X_test={X_test_final.shape}")

        # Final check before tuning
        if X_train_final.shape[1] == 0:
             print("\nCRITICAL ERROR: 0 features available after preprocessing/RFE decision. Cannot tune.")
             # Close plots if disabled before exiting
             if plotting_disabled: plt.close('all')
             return None, None


    except Exception as e:
         print(f"\nCRITICAL ERROR during applying RFE transformation or checking RFE status: {e}")
         traceback.print_exc()
         # Close plots if disabled before exiting
         if plotting_disabled: plt.close('all')
         return None, None


    # 3. Tune Hyperparameters (directly on the final, preprocessed data)
    print("\nStep 3: Starting Hyperparameter Tuning...")
    try:
        # Pass the final data (after RFE if applied) and the corresponding names
        best_model, tuning_results_df, search_time, test_scores, best_params_found = tune_random_forest_on_preprocessed(
            X_train_processed=X_train_final, # This is X_train AFTER RFE if used
            X_test_processed=X_test_final,   # This is X_test AFTER RFE if used
            y_train=y_train,
            y_test=y_test,
            param_distributions=param_distributions_config, # Use the config dict
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring_metric,
            random_state=random_state,
            results_dir=results_dir,
            best_model_filename=best_model_filename,
            feature_names_after_processing=feature_names_final, # Pass the correct final names
            save_importance=save_importance,
            top_n_features=top_n_features
            # show_plots is handled internally by evaluate_regression_performance called by tuning func
        )

        if best_model is None:
             print("\nCRITICAL ERROR: Hyperparameter tuning failed internally.")
             # Close plots if disabled before exiting
             if plotting_disabled: plt.close('all')
             return None, None

    except Exception as e:
        print(f"\nCRITICAL ERROR during hyperparameter tuning process: {e}")
        traceback.print_exc()
        # Close plots if disabled before exiting
        if plotting_disabled: plt.close('all')
        return None, None

    # Save and Plotting are handled within tune_random_forest_on_preprocessed now.
    # Test Evaluation is also handled within tune_random_forest_on_preprocessed.
    # Feature Importance is also handled within tune_random_forest_on_preprocessed.

    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    print(f"\n===== Tuned RF Pipeline on Preprocessed Data Finished in {total_duration:.2f} seconds =====")

    # Add final test scores printout (already done in evaluate_model called by tuning func, but repeat here for final summary)
    if test_scores:
         print("\n--- Final Tuned Model Test Set Performance Summary ---")
         test_scores_display = test_scores.get('Test', {}) # Get test scores if available
         if test_scores_display:
              for metric, value in test_scores_display.items():
                   print(f"  {metric.upper():<5}: {value:.4f}")
         else:
              print("  Test metrics calculation failed or resulted empty.")
    else:
         print("\n--- Final Tuned Model Test Set Performance Summary: Not Available ---")


    # Close all plots if plotting was disabled by this function
    if plotting_disabled:
         plt.close('all')

    # Re-enable interactive plotting if it was disabled by this function
    if not show_plots and current_backend != 'agg':
         plt.ion()


    # Return the best tuned model and its test scores
    return best_model, test_scores


# --- New Grid Search Tuning Function ---
def tune_random_forest_grid_on_preprocessed(
    X_train_processed: np.ndarray,
    X_test_processed: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    param_grid: dict,           # Use param_grid for GridSearchCV
    cv: int,
    scoring: str,
    random_state: int,
    results_dir: str,
    best_model_filename: str,
    feature_names_after_processing: list = None,
    save_importance: bool = True,
    top_n_features: int = None
):
    """
    Performs hyperparameter tuning using GridSearchCV for RandomForestRegressor
    directly on already processed (scaled, PCA'd, RFE'd) data.
    Does NOT build a preprocessing pipeline internally.
    """
    print("\n--- Starting Grid Search (Directly on Processed Data) ---")
    print(f"   Input data shape: X_train={X_train_processed.shape}, X_test={X_test_processed.shape}")
    # Calculate grid size
    grid_size = 1
    for values in param_grid.values():
        grid_size *= len(values)
    print(f"   Grid Size: {grid_size} combinations")
    print(f"   (cv={cv}, scoring='{scoring}', random_state={random_state})")
    search_start_time = time.time()

    if X_train_processed.shape[1] == 0 or len(X_train_processed) == 0:
         raise ValueError("Processed training data is empty or has 0 features. Cannot tune.")

    # Ensure CV folds is not more than samples
    if len(X_train_processed) < cv:
        warnings.warn(f"Number of samples ({len(X_train_processed)}) is less than CV folds ({cv}). Adjusting CV to {len(X_train_processed)}.", UserWarning)
        cv = len(X_train_processed) if len(X_train_processed) > 0 else 1
        if cv == 0: raise ValueError("0 training samples available.")

    # Define the Estimator (just RF, no pipeline)
    fixed_params = {'random_state': random_state, 'n_jobs': -1}
    estimator = RandomForestRegressor(**fixed_params)

    # Check if any parameters from the grid are NOT valid for the estimator
    valid_rf_params = estimator.get_params().keys()
    for param_name in param_grid.keys():
         if param_name not in valid_rf_params:
              warnings.warn(f"Parameter '{param_name}' from grid is not a valid parameter for RandomForestRegressor. It will be ignored by GridSearchCV.", UserWarning)


    # Setup and Run Grid Search
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    try:
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1, # Use n_jobs=-1 for the search itself
            return_train_score=True,
            verbose=1
        )

        # Fit the search *directly* on the processed training data
        grid_search.fit(X_train_processed, y_train)

    except Exception as e:
         print(f"\nCRITICAL ERROR during GridSearchCV fit: {e}")
         traceback.print_exc()
         # Minimal error handling for return
         dummy_df = pd.DataFrame({'params': ['Error'], f'mean_test_{scoring}': [np.nan], 'rank_test_score': [1]})
         return None, dummy_df, 0.0, {'r2':np.nan,'rmse':np.nan,'mae':np.nan}, {}


    search_end_time = time.time()
    search_time = search_end_time - search_start_time
    print(f"Search finished in {search_time:.2f} seconds.")

    best_estimator = grid_search.best_estimator_
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by=f'rank_test_score', ascending=True)
    results_df.columns = results_df.columns.str.replace(":", "_", regex=False) # Clean column names


    print("\n--- Best Tuning Results (CV on Train Data) ---")
    best_score = grid_search.best_score_
    print(f"Best Score ({scoring}): {best_score:.4f}")
    best_params = grid_search.best_params_
    print("Best Parameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Evaluate on the held-out test set *after* tuning is complete
    test_scores = evaluate_model(best_estimator, X_test_processed, y_test)

    # Save Tuning Results
    # Reuse save_tuning_results, it works for grid search results structure
    save_tuning_results(
        results_df=results_df,
        best_params=best_params,
        best_score=best_score,
        scoring=scoring,
        cv=cv,
        n_iter=grid_size, # Report grid size as n_iter for clarity
        search_time=search_time,
        random_state=random_state,
        results_dir=results_dir
    )

    # Plot Tuning Results (using the adapted function)
    plot_tuning_results_adapted(
        results_df=results_df,
        param_distributions=param_grid, # Pass the param_grid
        scoring=scoring,
        results_dir=results_dir
    )

    # Save the Best Model
    model_filepath = os.path.join(results_dir, best_model_filename)
    save_best_model(best_estimator, model_filepath)

    # Get and Save Feature Importance (for the best tuned model)
    if save_importance:
        print("\n--- Extracting Feature Importances (Tuned Model) ---")
        importances_df = get_feature_importances(
            best_estimator,
            feature_names=feature_names_after_processing,
            top_n=top_n_features
        )
        if importances_df is not None:
            fi_filename = os.path.join(results_dir, 'tuned_feature_importances.csv')
            try:
                importances_df.to_csv(fi_filename, index=False)
                print(f"Feature importances saved to: {fi_filename}")
                print(f"Top {top_n_features if top_n_features else 'All'} Features:")
                print(importances_df)
            except Exception as e:
                print(f"Error saving/printing importances: {e}")
        else:
             print("Could not extract feature importances for tuned model.")

    return best_estimator, results_df, search_time, test_scores, best_params


# --- New Orchestrator Function for Grid Tuning Preprocessed Data ---
def orchestrate_grid_tuned_rf_on_pipeline_output(
    results_from_run_full_regression_pipeline: dict,
    param_grid_config: dict, # Hyperparameter grid for RF (NO prefixes)
    cv_folds: int,
    scoring_metric: str,
    random_state: int,
    results_dir: str = "grid_tuned_rf_on_pipeline_output",
    best_model_filename: str = "grid_tuned_random_forest_model.joblib",
    save_importance: bool = True,
    top_n_features: int = None
):
    """
    Takes results from run_full_regression_pipeline, extracts the processed data
    (after scaling, PCA, and RFE), and performs hyperparameter tuning using
    GridSearchCV on a RandomForestRegressor trained directly on this preprocessed data.
    """
    print("\n===== Starting Grid Tuned RF Pipeline on Preprocessed Data =====")
    print(f"Random State used throughout tuning: {random_state}")
    pipeline_start_time = time.time()

    # 1. Extract Data and Transformers from Previous Run Results
    print("Step 1: Extracting data and transformations from previous pipeline results...")
    try:
        X_train_processed_initial = results_from_run_full_regression_pipeline.get('X_train_processed')
        X_test_processed_initial = results_from_run_full_regression_pipeline.get('X_test_processed')
        y_train = results_from_run_full_regression_pipeline.get('y_train')
        y_test = results_from_run_full_regression_pipeline.get('y_test')
        rfe_object = results_from_run_full_regression_pipeline.get('rfe_object')
        selected_mask_after_rfe = results_from_run_full_regression_pipeline.get('selected_mask_after_rfe')
        feature_names_after_preprocessing = results_from_run_full_regression_pipeline.get('feature_names_after_processing', [])

        if X_train_processed_initial is None or X_test_processed_initial is None or y_train is None or y_test is None:
             raise ValueError("Critical data is missing in the results dictionary.")
        if not isinstance(X_train_processed_initial, np.ndarray) or not isinstance(X_test_processed_initial, np.ndarray):
             raise TypeError("Processed data is not in expected numpy array format.")
        if not isinstance(y_train, pd.Series): y_train = pd.Series(y_train)
        if not isinstance(y_test, pd.Series): y_test = pd.Series(y_test)

        print(f"  Extracted initial processed data shapes: X_train={X_train_processed_initial.shape}, X_test={X_test_processed_initial.shape}")
        print(f"  RFE object found: {'Yes' if rfe_object else 'No'}")

    except Exception as e:
        print(f"\nCRITICAL ERROR during data extraction: {e}")
        traceback.print_exc()
        return None, None

    # 2. Apply RFE Transformation to the Data *Before* Tuning
    print("\nStep 2: Applying RFE transformation to the data...")
    try:
        rfe_was_applied_successfully = (rfe_object is not None and selected_mask_after_rfe is not None and np.sum(selected_mask_after_rfe) < X_train_processed_initial.shape[1])

        if rfe_was_applied_successfully:
            n_features_after_rfe = np.sum(selected_mask_after_rfe)
            if n_features_after_rfe == 0:
                 print("\nCRITICAL ERROR: RFE was applied but resulted in 0 features. Cannot tune.")
                 return None, None

            print(f"  Applying RFE transformation (selected {n_features_after_rfe} features) to data...")
            X_train_final = rfe_object.transform(X_train_processed_initial)
            X_test_final = rfe_object.transform(X_test_processed_initial)
            feature_names_final = [name for name, selected in zip(feature_names_after_preprocessing, selected_mask_after_rfe) if selected]
            print(f"  Data shape after RFE transformation: X_train={X_train_final.shape}, X_test={X_test_final.shape}")

        else:
            print("  RFE was skipped or did not reduce features in the previous pipeline. Using full processed data.")
            X_train_final = X_train_processed_initial
            X_test_final = X_test_processed_initial
            feature_names_final = feature_names_after_preprocessing
            print(f"  Using data shape: X_train={X_train_final.shape}, X_test={X_test_final.shape}")
            if X_train_final.shape[1] == 0:
                 print("\nCRITICAL ERROR: 0 features available after preprocessing/skipping RFE. Cannot tune.")
                 return None, None


    except Exception as e:
         print(f"\nCRITICAL ERROR during applying RFE transformation or checking RFE status: {e}")
         traceback.print_exc()
         return None, None


    # 3. Tune Hyperparameters (directly on the final, preprocessed data)
    print("\nStep 3: Starting Hyperparameter Tuning (GridSearchCV)...")
    try:
        best_model, tuning_results_df, search_time, test_scores, best_params_found = tune_random_forest_grid_on_preprocessed(
            X_train_processed=X_train_final,
            X_test_processed=X_test_final,
            y_train=y_train,
            y_test=y_test,
            param_grid=param_grid_config, # Use param_grid_config
            cv=cv_folds,
            scoring=scoring_metric,
            random_state=random_state,
            results_dir=results_dir,
            best_model_filename=best_model_filename,
            feature_names_after_processing=feature_names_final,
            save_importance=save_importance,
            top_n_features=top_n_features
        )

        if best_model is None:
             print("\nCRITICAL ERROR: Hyperparameter tuning failed internally.")
             return None, None

    except Exception as e:
        print(f"\nCRITICAL ERROR during hyperparameter tuning process: {e}")
        traceback.print_exc()
        return None, None

    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    print(f"\n===== Grid Tuned RF Pipeline on Preprocessed Data Finished in {total_duration:.2f} seconds =====")

    if test_scores:
         print("\n--- Final Tuned Model Test Set Performance Summary ---")
         for metric, value in test_scores.items():
              print(f"  {metric.upper():<5}: {value:.4f}")
    else:
         print("\n--- Final Tuned Model Test Set Performance Summary: Not Available ---")

    return best_model, test_scores





######################################

#####################################

##########################################

###################################################
#######################################

#####################################

##########################################

###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time
import traceback
import sys # To check Python version for plotting

# Sci-kit Learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor # Used as RFE estimator in RF pipeline, not needed in CatBoost RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator # For type checking in save_results

# CatBoost Imports (handle missing library)
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_INSTALLED = True
except ImportError:
    print("Warning: CatBoost library not found. CatBoost functions will not work.")
    CATBOOST_INSTALLED = False
    # Define a dummy class to prevent NameError if CatBoost is not installed
    class CatBoostRegressor:
        def __init__(self, **kwargs):
            raise ImportError("CatBoost not installed")
        def fit(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): return np.array([])
        def get_feature_importance(self, *args, **kwargs): return np.array([])
        def save_model(self, *args, **kwargs): pass
        def load_model(self, *args, **kwargs): pass

# SHAP Imports (handle missing library)
try:
    import shap
    SHAP_INSTALLED = True
except ImportError:
    print("Warning: SHAP library not found. SHAP analysis will be skipped.")
    SHAP_INSTALLED = False
    # Define dummy functions/objects to prevent NameError if SHAP is not installed
    class DummyExplainer:
        def __init__(self, model): pass
        def shap_values(self, X): return np.array([])
        @property
        def expected_value(self): return 0.0
    shap = type('shap', (object,), {'TreeExplainer': DummyExplainer, 'summary_plot': lambda *args, **kwargs: None, 'dependence_plot': lambda *args, **kwargs: None, 'Explanation': None})


# Configure plotting backend for non-interactive environments if needed
# Check if running in a non-interactive environment (e.g., script vs notebook)
# This is a basic check, might need refinement depending on the exact environment
IS_INTERACTIVE = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
if not IS_INTERACTIVE:
     plt.switch_backend('agg') # Use non-interactive backend


# --- 1. Data Preparation & Splitting ---
# (No changes needed from PART 1)
def split_and_select_numeric_data(df, target_col, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into features (numeric only) and target.
    Handles missing values in target by dropping rows.
    Handles missing values in numeric features by median imputation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): Name of the target column.
        test_size (float): Proportion for the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: Contains:
            - X_train (pd.DataFrame): Training features (numeric, imputed).
            - X_test (pd.DataFrame): Testing features (numeric, imputed).
            - y_train (pd.Series): Training target.
            - y_test (pd.Series): Testing target.
            - initial_numeric_feature_names (list): Names of original numeric features.
            - test_original_indices (pd.Index): Original index of test samples.
            - train_original_indices (pd.Index): Original index of train samples. # Added for consistency if needed later
    Raises:
        ValueError, TypeError
    """
    print(f"--- 1. Preparing & Splitting Data (Target: '{target_col}') ---")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_col]):
         raise ValueError(f"Target column '{target_col}' must be numeric for regression.")
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Drop rows where the target is NaN
    initial_rows = len(df)
    df_cleaned = df.dropna(subset=[target_col]).copy()
    if len(df_cleaned) < initial_rows:
        print(f"Warning: Dropped {initial_rows - len(df_cleaned)} rows with missing target values.")

    if df_cleaned.empty:
         raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
         raise ValueError("Not enough samples left after handling missing target values to perform a split.")
    min_train_samples = int(np.ceil(len(df_cleaned)*(1-test_size)))
    min_test_samples = int(np.ceil(len(df_cleaned)*test_size))
    if min_train_samples < 1 or min_test_samples < 1:
         print(f"Warning: Test size {test_size} results in very small train/test sets ({min_train_samples} train, {min_test_samples} test). Proceeding cautiously.")

    X = df_cleaned.drop(target_col, axis=1)
    y = df_cleaned[target_col]

    # Select only numeric features
    X_numeric = X.select_dtypes(include=np.number)
    initial_numeric_feature_names = X_numeric.columns.tolist()

    if not initial_numeric_feature_names:
         # Allow pipeline to proceed with 0 features for now, but subsequent steps will fail
         print("Warning: No numeric features found after dropping target.")


    print(f"Initial total features: {X.shape[1]}, Using {len(initial_numeric_feature_names)} numeric features.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=test_size, random_state=random_state
    )

    # Impute NaNs in train and test sets using the TRAIN set's median
    if X_train.isnull().sum().sum() > 0 or X_test.isnull().sum().sum() > 0:
        print("Warning: NaNs found in numeric features. Imputing with median from training data.")
        imputation_values = X_train.median()
        X_train = X_train.fillna(imputation_values)
        X_test = X_test.fillna(imputation_values) # Use train median for test set too

    test_original_indices = X_test.index # Keep track of original indices for residual analysis
    train_original_indices = X_train.index # Keep track of original indices for train analysis if needed

    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")

    return (X_train, X_test, y_train, y_test,
            initial_numeric_feature_names, test_original_indices, train_original_indices)


# --- 2. Apply Preprocessing (Scaling and PCA) ---
# (No changes needed from PART 2)
def apply_preprocessing_cat(X_train, X_test, scaler_type, pca_n_comp, random_state=42):
    """
    Applies specified scaling and PCA to the data.

    Args:
        X_train (pd.DataFrame): Training features (numeric, imputed).
        X_test (pd.DataFrame): Testing features (numeric, imputed).
        scaler_type (str): Type of scaler ('Standard', 'Robust', 'MinMax', 'None').
        pca_n_comp (int or float or None): Number of PCA components or variance ratio.
                                            If None, no PCA is applied.
        random_state (int): Random seed for PCA.

    Returns:
        tuple: Contains:
            - X_train_processed (np.ndarray): Processed training features.
            - X_test_processed (np.ndarray): Processed testing features.
            - feature_names_after_processing (list): Names corresponding to processed features.
            - scaler_instance: Fitted scaler object (or None).
            - pca_instance: Fitted PCA object (or None).
    Raises:
        ValueError, RuntimeError, TypeError
    """
    print("\n--- 2. Applying Preprocessing (Scaling & PCA) ---")
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
         raise TypeError("X_train and X_test must be pandas DataFrames for this function.")

    # Store original feature names before scaling/PCA
    original_feature_names = X_train.columns.tolist()

    # 2.1 Apply Scaling
    scaler_instance = None
    X_train_scaled = X_train.copy() # Start with copies
    X_test_scaled = X_test.copy()

    if scaler_type and scaler_type.lower() != 'none':
        print(f"  Applying Scaler: {scaler_type}")
        if scaler_type.lower() == 'standard':
            scaler_instance = StandardScaler()
        elif scaler_type.lower() == 'robust':
            scaler_instance = RobustScaler()
        elif scaler_type.lower() == 'minmax':
             scaler_instance = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}. Choose from 'Standard', 'Robust', 'MinMax', 'None'.")

        if scaler_instance:
            try:
                # Fit on train, transform train and test. Returns numpy arrays.
                X_train_scaled_np = scaler_instance.fit_transform(X_train_scaled)
                X_test_scaled_np = scaler_instance.transform(X_test_scaled)
                print(f"  Scaling applied. Scaled train data shape: {X_train_scaled_np.shape}")
            except Exception as e:
                print(f"  Error during scaling with {scaler_type}: {e}")
                raise RuntimeError(f"Scaling failed with {scaler_type}: {e}")
    else:
        print("  No scaling applied.")
        # Convert to numpy arrays even if no scaling for consistency with PCA output
        X_train_scaled_np = X_train_scaled.values if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled
        X_test_scaled_np = X_test_scaled.values if isinstance(X_test_scaled, pd.DataFrame) else X_test_scaled
        print(f"  Data converted to NumPy array. Shape: {X_train_scaled_np.shape}")

    # Store scaled data (numpy arrays) before potential PCA
    X_train_processed = X_train_scaled_np
    X_test_processed = X_test_scaled_np
    feature_names_after_processing = original_feature_names # Names before potential PCA
    pca_instance = None

    # 2.2 Apply PCA
    if pca_n_comp is not None:
        max_features = X_train_processed.shape[1]
        if max_features == 0:
             print("  0 features available after scaling, skipping PCA.")
             feature_names_after_processing = []
        else:
            valid_pca_comp = False
            if isinstance(pca_n_comp, int) and 0 < pca_n_comp <= max_features:
                print(f"  Applying PCA with n_components={pca_n_comp}")
                valid_pca_comp = True
            elif isinstance(pca_n_comp, float) and 0 < pca_n_comp <= 1.0:
                print(f"  Applying PCA to retain {pca_n_comp:.1%} variance")
                valid_pca_comp = True
            else:
                 raise ValueError(f"Invalid pca_n_comp type or value: {pca_n_comp}. Must be int (0, {max_features}], float (0,1], or None.")

            if valid_pca_comp:
                try:
                    pca_instance = PCA(n_components=pca_n_comp, random_state=random_state)
                    # Fit PCA on the scaled training data
                    X_train_processed = pca_instance.fit_transform(X_train_processed)
                    # Transform the scaled test data
                    X_test_processed = pca_instance.transform(X_test_processed)

                    # Feature names become 'PC0', 'PC1', etc.
                    feature_names_after_processing = [f'PC{i}' for i in range(X_train_processed.shape[1])]
                    print(f"  PCA applied. New feature shape: {X_train_processed.shape}")
                    print(f"  Explained variance ratio by component: {pca_instance.explained_variance_ratio_}")
                    print(f"  Total explained variance: {np.sum(pca_instance.explained_variance_ratio_):.4f}")


                except Exception as e:
                     print(f"  Error during PCA with n_components={pca_n_comp}: {e}")
                     raise RuntimeError(f"PCA failed with n_components={pca_n_comp}: {e}")

    else:
        print("  No PCA applied.")
        # Feature names remain the original numeric names if no PCA
        # X_train_processed and X_test_processed are already numpy arrays


    print(f"Preprocessing complete. Final feature shape before RFE: {X_train_processed.shape}")
    # Ensure data is numpy array for RFE and model training (already done above, but double check)
    if not isinstance(X_train_processed, np.ndarray):
         raise TypeError(f"X_train_processed is not a numpy array after preprocessing: {type(X_train_processed)}")
    if not isinstance(X_test_processed, np.ndarray):
         raise TypeError(f"X_test_processed is not a numpy array after preprocessing: {type(X_test_processed)}")


    return X_train_processed, X_test_processed, feature_names_after_processing, scaler_instance, pca_instance


# --- 3. Recursive Feature Elimination (Using CatBoost as Estimator) ---
# (No changes needed from PART 2)
def perform_rfe_with_catboost(X_train_processed, y_train, feature_names_after_processing, target_feature_ratio=0.6, random_state=42, rfe_estimator_params=None):
    """
    Performs Recursive Feature Elimination using CatBoostRegressor.

    Args:
        X_train_processed (np.ndarray): Processed training features (scaled/PCA'd).
        y_train (pd.Series): Training target variable.
        feature_names_after_processing (list): List of feature names corresponding
                                               to X_train_processed (original or PC names).
        target_feature_ratio (float): The desired ratio of features to keep (e.g., 0.6 for 60%).
                                       If None or >= 1.0, RFE selection is skipped.
        random_state (int): Random seed for the RFE estimator.
        rfe_estimator_params (dict or None): Parameters for the RFE's internal CatBoost estimator.
                                             Should NOT contain 'random_state' or 'verbose'.

    Returns:
        tuple: Contains:
            - rfe (RFE): Fitted RFE object (or None if skipped/failed).
            - selected_feature_names (list): List of feature names selected by RFE
                                             (these are names from feature_names_after_processing).
            - X_train_rfe (np.ndarray): Training data with only selected features.
            - selected_mask (np.ndarray): Boolean mask indicating selected features
                                          relative to X_train_processed.
    Raises:
        ValueError, RuntimeError
    """
    print("\n--- 3. Performing Recursive Feature Elimination (RFE) with CatBoost ---")
    if not CATBOOST_INSTALLED:
        print("CatBoost library is not installed. Cannot perform RFE with CatBoost. Skipping RFE.")
        # Return values indicating RFE was skipped, using all original features
        n_initial_features = X_train_processed.shape[1] if isinstance(X_train_processed, np.ndarray) else 0
        return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)


    n_initial_features = X_train_processed.shape[1]
    print(f"Initial number of features for RFE: {n_initial_features}")

    # --- Skip RFE Logic ---
    # Skip if ratio is None, >= 1.0, <= 0, or if no features to begin with, or only 1 feature.
    rfe_should_run = (target_feature_ratio is not None and 0 < target_feature_ratio < 1.0 and n_initial_features > 1)

    if not rfe_should_run:
        if n_initial_features == 0:
            print("Cannot perform RFE on data with 0 features. Skipping RFE.")
        elif n_initial_features == 1:
             print("Only 1 feature available. RFE requires more than one feature to eliminate. Skipping RFE.")
        else:
             print(f"target_feature_ratio ({target_feature_ratio}) is not in (0, 1). Skipping RFE selection.")
        # Return values indicating RFE was skipped
        # Mask selects all (original) features
        return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    # --- Proceed with RFE ---
    n_features_to_select = max(1, int(n_initial_features * target_feature_ratio))
    # Ensure n_features_to_select is not more than available features
    n_features_to_select = min(n_features_to_select, n_initial_features)
    print(f"Target number of features to select: {n_features_to_select} ({target_feature_ratio*100:.0f}%)")

    # RFE Estimator Setup
    # Use a basic CatBoost for RFE estimator - not the final model params
    rfe_est_params_clean = rfe_estimator_params.copy() if rfe_estimator_params is not None else {}
    rfe_est_params_clean.pop('random_state', None)
    rfe_est_params_clean.pop('verbose', None)
    rfe_est_params_clean.setdefault('iterations', 100) # Fewer iterations for RFE speed
    rfe_est_params_clean.setdefault('learning_rate', 0.1)
    rfe_est_params_clean.setdefault('depth', 5)
    rfe_est_params_clean.setdefault('loss_function', 'RMSE')

    print(f"  Using CatBoostRegressor with params {rfe_est_params_clean} as RFE estimator (random_state={random_state}, verbose=0).")
    # Note: CatBoost needs feature names if not already a DataFrame for RFE
    # The RFE wrapper in scikit-learn works on numpy arrays, so feature names need careful handling.
    # We pass feature_names_after_processing *to this function*, but RFE itself doesn't use them during fit/transform.
    # We use them *after* RFE to get the selected names.
    estimator = CatBoostRegressor(random_state=random_state, verbose=0, **rfe_est_params_clean)


    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1, verbose=0) # RFE verbosity

    try:
        # Check if X_train_processed has enough samples for CatBoost (usually > 1)
        if X_train_processed.shape[0] < 2:
             print(f"Warning: Not enough samples ({X_train_processed.shape[0]}) for RFE fitting. Skipping RFE.")
             return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)


        print(f"  Fitting RFE to select {n_features_to_select} features...")
        rfe.fit(X_train_processed, y_train)
        print("  RFE fitting complete.")

        # Check if RFE actually selected features (it might fail gracefully)
        if not hasattr(rfe, 'support_') or rfe.support_.sum() == 0:
             print("Warning: RFE fit completed but selected 0 features, or 'support_' attribute not found. RFE may have failed internally or found no useful features.")
             return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    except Exception as e:
        print(f"Error fitting RFE with CatBoost: {e}")
        traceback.print_exc()
        print("RFE fitting failed. Proceeding without feature selection.")
        return None, feature_names_after_processing, X_train_processed, np.ones(n_initial_features, dtype=bool)

    selected_mask = rfe.support_
    n_selected = selected_mask.sum()

    # Get the names of the selected features *from the processed data*
    if len(feature_names_after_processing) != n_initial_features:
         print(f"Error: Mismatch in feature names list length ({len(feature_names_after_processing)}) and processed data columns ({n_initial_features}). Cannot map selected features to names.")
         # Fallback: generic names
         selected_feature_names = [f'Selected_Feature_{i}' for i in range(n_selected)]
    else:
        selected_feature_names = [name for name, selected in zip(feature_names_after_processing, selected_mask) if selected]


    # Transform the training data immediately
    X_train_rfe = rfe.transform(X_train_processed)

    # Sanity check shapes
    if X_train_rfe.shape[1] != len(selected_feature_names):
         print(f"Error: Shape mismatch after RFE transform ({X_train_rfe.shape[1]}) vs selected names ({len(selected_feature_names)}).")
         # Fallback to generic names if mismatch
         selected_feature_names = [f'Selected_Feature_{i}' for i in range(X_train_rfe.shape[1])]

    print(f"RFE complete. Selected {len(selected_feature_names)} features.")

    return rfe, selected_feature_names, X_train_rfe, selected_mask


# --- 4. Model Training (CatBoost) ---
# (No changes needed from PART 2)
def train_catboost_regressor(X_train_final, y_train, X_test_final, y_test,
                             catboost_params=None, early_stopping_rounds=50, random_state=42):
    """
    Trains a CatBoostRegressor model on the final selected features,
    using early stopping with the test set as evaluation data.

    Args:
        X_train_final (np.ndarray): Training features (after preprocessing & RFE).
        y_train (pd.Series): Training target variable.
        X_test_final (np.ndarray): Testing features (after preprocessing & RFE).
                                    Used for early stopping eval_set.
        y_test (pd.Series): Testing target variable. Used for early stopping.
        catboost_params (dict or None): Parameters for the CatBoostRegressor.
                                        Defaults to reasonable settings if None.
                                        Should NOT include 'random_state', 'eval_metric',
                                        'loss_function' unless overriding defaults.
        early_stopping_rounds (int): Activates CatBoost early stopping.
                                     Set to None to disable.
        random_state (int): Random seed for the CatBoost model.

    Returns:
        CatBoostRegressor: Fitted model object.
    Raises:
        RuntimeError, ValueError
    """
    print("\n--- 4. Training Final CatBoostRegressor Model ---")
    if not CATBOOST_INSTALLED:
        raise RuntimeError("CatBoost library is not installed. Cannot train model.")
    if X_train_final.shape[1] == 0:
        raise ValueError("Cannot train model with 0 features.")
    if len(X_train_final) == 0:
         raise ValueError("Cannot train model with 0 training samples.")
    if early_stopping_rounds is not None and (X_test_final is None or y_test is None or len(X_test_final) == 0):
        print("Warning: Early stopping requires test data (X_test_final, y_test), but it's missing or empty. Disabling early stopping.")
        early_stopping_rounds = None

    model_params = catboost_params.copy() if catboost_params is not None else {}

    # --- Set CatBoost Defaults and Overrides ---
    model_params.setdefault('iterations', 1000)
    model_params.setdefault('learning_rate', 0.05) # Often good starting point
    model_params.setdefault('depth', 6)
    model_params.setdefault('l2_leaf_reg', 3) # L2 regularization
    model_params.setdefault('loss_function', 'RMSE') # Common for regression
    model_params.setdefault('eval_metric', 'RMSE') # Metric for evaluation and early stopping
    # model_params.setdefault('nan_mode', 'Min') # Example: How CatBoost handles NaNs if not pre-imputed

    # --- Ensure random_state is included ---
    model_params['random_state'] = random_state
    # --- Control Verbosity ---
    # Make verbose less frequent unless specified
    verbose_level = model_params.pop('verbose', 100 if early_stopping_rounds else False) # Show progress every 100 iters if ES active, else silent

    print(f"  Using CatBoostRegressor with effective params: {model_params}")
    print(f"  random_state={random_state}, early_stopping_rounds={early_stopping_rounds}, verbose={verbose_level}")

    try:
        model = CatBoostRegressor(**model_params)

        fit_params = {
            'X': X_train_final,
            'y': y_train,
            'verbose': verbose_level
        }

        if early_stopping_rounds is not None and X_test_final is not None and y_test is not None:
            # CatBoost needs Pool object or tuple of (X, y) for eval_set
            # Using tuple of numpy arrays is fine here as we've already preprocessed
            eval_set = [(X_test_final, y_test)]
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            print(f"Training with early stopping using test set (stops if '{model_params['eval_metric']}' doesn't improve for {early_stopping_rounds} rounds).")
        else:
            print("Training without early stopping.")

        print(f"Training on {X_train_final.shape[0]} samples and {X_train_final.shape[1]} features.")
        model.fit(**fit_params)

        if early_stopping_rounds is not None and hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
            print(f"Model training complete. Best iteration found: {model.best_iteration_ + 1} (out of {model_params['iterations']})")
        else:
            print("Model training complete.")

        return model
    except Exception as e:
        print(f"Error during CatBoost model training: {e}")
        traceback.print_exc()
        raise RuntimeError(f"CatBoostRegressor training failed: {e}")


# --- 5. Performance Evaluation (Generic - Handles RFE object) ---
# (No changes needed from PART 2)
def evaluate_model_performance(model, X_train_processed, y_train, X_test_processed, y_test, rfe_object=None):
    """
    Calculates regression metrics for train and test sets using the trained model.
    Transforms data using the fitted RFE object if provided.

    Args:
        model: Trained regression model.
        X_train_processed (np.ndarray): Training features *before* final RFE selection.
                                        Used if RFE object is provided for transformation.
        y_train (pd.Series): Training target.
        X_test_processed (np.ndarray): Testing features *before* final RFE selection.
                                       Used for transformation.
        y_test (pd.Series): Testing target.
        rfe_object (RFE or None): Fitted RFE object. If None, processed data is used directly.

    Returns:
        dict: Dictionary of performance metrics.
    Raises:
        RuntimeError
    """
    print("\n--- 5. Evaluating Model Performance ---")
    if X_test_processed is None or y_test is None or X_test_processed.shape[0] == 0:
        print("Error: Test set (X_test_processed/y_test) is missing or empty. Cannot evaluate performance.")
        return {"Train": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan},
                "Test": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}}
    if X_train_processed is None or y_train is None or X_train_processed.shape[0] == 0:
        print("Warning: Train set (X_train_processed/y_train) is missing or empty. Cannot evaluate train performance.")

    try:
        # Apply RFE transformation if RFE was performed AND the rfe_object is valid
        if rfe_object is not None and hasattr(rfe_object, 'transform'):
             # Use n_features_to_select from the RFE object itself if available
             n_selected = rfe_object.n_features_to_select_ if hasattr(rfe_object, 'n_features_to_select_') else 'N/A'
             print(f"  Applying RFE transform (selecting features) for evaluation. Expected {n_selected} features.")
             if X_train_processed is not None and X_train_processed.shape[1] > 0:
                 X_train_eval = rfe_object.transform(X_train_processed)
             else:
                 X_train_eval = None # Cannot transform if input is empty/None
             X_test_eval = rfe_object.transform(X_test_processed)
             print(f"  Evaluating on {X_test_eval.shape[1]} features (after RFE).")
        else:
             # If no valid RFE object, use the processed data directly
             X_train_eval = X_train_processed
             X_test_eval = X_test_processed
             n_features = X_test_eval.shape[1] if X_test_eval is not None else 'N/A'
             print(f"  Evaluating on {n_features} features (RFE not applied or invalid).")

        # Check for zero features after potential RFE
        if X_test_eval is None or X_test_eval.shape[1] == 0:
             print("  0 features available for test evaluation after RFE/preprocessing. Cannot predict.")
             # Return structure indicating failure
             return {"Train": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan},
                     "Test": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}}


        # --- Calculate Metrics ---
        metrics = {"Train": {}, "Test": {}}

        # Train Metrics (if possible)
        if X_train_eval is not None and y_train is not None and X_train_eval.shape[1] > 0:
            y_train_pred = model.predict(X_train_eval)
            metrics["Train"] = {
                "R²": r2_score(y_train, y_train_pred),
                "MAE": mean_absolute_error(y_train, y_train_pred),
                "MSE": mean_squared_error(y_train, y_train_pred),
                "RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            }
            print("Scores on train set:")
            for metric, value in metrics["Train"].items():
                print(f"  {metric:<5}: {value:.4f}")
        else:
             print("Scores on train set: Skipped (missing data or 0 features).")
             metrics["Train"] = {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}

        # Test Metrics
        y_test_pred = model.predict(X_test_eval)
        metrics["Test"] = {
            "R²": r2_score(y_test, y_test_pred),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        }
        print("\nScores on test set:")
        for metric, value in metrics["Test"].items():
            print(f"  {metric:<5}: {value:.4f}")

        # --- Plot Predicted vs Actual ---
        # Only plot if show_plots is True (controlled by orchestrator)
        current_backend = plt.get_backend()
        if current_backend != 'agg': # Don't try to show plots with agg backend
            plt.figure(figsize=(12, 5))

            # Test Set Plot
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test Data')
            # Ensure plotting range handles predictions outside the range of actuals
            all_values_test = np.concatenate([y_test.values, y_test_pred])
            min_val_test, max_val_test = np.min(all_values_test), np.max(all_values_test)
            plt.plot([min_val_test, max_val_test], [min_val_test, max_val_test], '--', color='red', lw=2, label='Ideal')
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs. Actual (Test Set)")
            plt.legend()
            plt.grid(True)

            # Train Set Plot (if possible)
            if X_train_eval is not None and y_train is not None and 'y_train_pred' in locals() and len(y_train_pred) == len(y_train):
                plt.subplot(1, 2, 2)
                plt.scatter(y_train, y_train_pred, alpha=0.6, label='Train Data')
                all_values_train = np.concatenate([y_train.values, y_train_pred])
                min_val_train, max_val_train = np.min(all_values_train), np.max(all_values_train)
                plt.plot([min_val_train, max_val_train], [min_val_train, max_val_train], '--', color='red', lw=2, label='Ideal')
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Predicted vs. Actual (Train Set)")
                plt.legend()
                plt.grid(True)
            else:
                 # Add an empty subplot or text if train plot is skipped
                 ax = plt.subplot(1, 2, 2)
                 ax.text(0.5, 0.5, 'Train Plot Skipped\n(Missing Data)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                 ax.set_title("Predicted vs. Actual (Train Set)")
                 ax.set_xticks([])
                 ax.set_yticks([])


            plt.tight_layout()
            plt.show()
        else:
            print("Plotting skipped (show_plots=False or non-interactive backend).")


        return metrics

    except Exception as e:
        print(f"Error during performance evaluation: {e}")
        traceback.print_exc()
        # Return metrics with NaN values in case of calculation error
        return {"Train": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan},
                "Test": {"R²": np.nan, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan}}


# --- 6. Residual Analysis (Generic - Handles RFE object & returns y_test_pred) ---
# (Modified to return y_test_pred)
def analyze_residuals(model, X_test_processed, y_test, df_original, test_original_indices, rfe_object=None, n_worst=20):
    """
    Performs residual analysis for regression, plots diagnostics, identifies worst predictions.
    Transforms test data using the fitted RFE object if provided.

    Args:
        model: Trained regression model.
        X_test_processed (np.ndarray): Testing features *before* final RFE selection.
                                       Used for transformation.
        y_test (pd.Series): Testing target.
        df_original (pd.DataFrame): The original input DataFrame.
        test_original_indices (pd.Index): Original index of test samples.
        rfe_object (RFE or None): Fitted RFE object. If None, processed data is used directly.
        n_worst (int): Number of worst residuals to detail.

    Returns:
        tuple: Contains:
            - residuals (pd.Series): Residuals for the test set, indexed by original index.
            - worst_residuals_df (pd.DataFrame): Details of worst predictions.
            - y_test_pred (np.ndarray): Predictions on the test set. # Added this return value
    Raises:
        RuntimeError
    """
    print(f"\n--- 6. Analyzing Residuals (Test Set) for top {n_worst} worst predictions ---")
    if X_test_processed is None or y_test is None or X_test_processed.shape[0] == 0 or y_test.empty:
        print("Error: Test set (X_test_processed/y_test) is missing or empty. Cannot analyze residuals.")
        return pd.Series(dtype=float), pd.DataFrame(), np.array([]) # Return empty residuals, df, and preds

    try:
        # Apply RFE transformation if RFE was performed AND the rfe_object is valid
        if rfe_object is not None and hasattr(rfe_object, 'transform'):
             # Use n_features_to_select from the RFE object itself if available
             n_selected = rfe_object.n_features_to_select_ if hasattr(rfe_object, 'n_features_to_select_') else 'N/A'
             print(f"  Applying RFE transform (selecting features) for residual analysis. Expected {n_selected} features.")
             X_test_eval = rfe_object.transform(X_test_processed)
             print(f"  Analyzing residuals based on {X_test_eval.shape[1]} features (after RFE).")
        else:
             # If no valid RFE object, use the processed data directly
             X_test_eval = X_test_processed
             n_features = X_test_eval.shape[1] if X_test_eval is not None else 'N/A'
             print(f"  Analyzing residuals based on {n_features} features (RFE not applied or invalid).")


        if X_test_eval is None or X_test_eval.shape[1] == 0:
             print("  0 features available for residual analysis after RFE/preprocessing. Cannot predict.")
             return pd.Series(dtype=float), pd.DataFrame(), np.array([])

        y_test_pred = model.predict(X_test_eval)
        residuals_array = y_test.values - y_test_pred # Actual - Predicted

        # Create a pandas Series for residuals, using the original test indices
        residuals = pd.Series(residuals_array, index=test_original_indices, name='Residual')

        print(f"Residuals Summary: Min={residuals.min():.4f}, Mean={residuals.mean():.4f}, Max={residuals.max():.4f}, Std={residuals.std():.4f}")

        # --- Residual Plots ---
        # Only plot if show_plots is True (controlled by orchestrator)
        current_backend = plt.get_backend()
        if current_backend != 'agg':
            plt.figure(figsize=(15, 5))

            # 1. Residuals vs. Predicted
            plt.subplot(1, 3, 1)
            plt.scatter(y_test_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--', lw=2)
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals (Actual - Predicted)")
            plt.title("Residuals vs. Predicted Values")
            plt.grid(True)

            # 2. Histogram of Residuals
            plt.subplot(1, 3, 2)
            sns.histplot(residuals, kde=True)
            plt.xlabel("Residuals")
            plt.title("Histogram of Residuals")
            plt.grid(True)

            # 3. Q-Q Plot
            plt.subplot(1, 3, 3)
            # Handle potential NaNs/Infs in residuals before plotting
            residuals_finite = residuals.replace([np.inf, -np.inf], np.nan).dropna()
            if len(residuals_finite) > 0:
                stats.probplot(residuals_finite, dist="norm", plot=plt)
            else:
                 plt.text(0.5, 0.5, 'Cannot plot Q-Q:\nNo finite residuals', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
            plt.title("Q-Q Plot of Residuals")
            plt.grid(True)

            plt.tight_layout()
            plt.show()
        else:
             print("Residual plots skipped (show_plots=False or non-interactive backend).")


        # --- Identify Worst Residuals ---
        worst_residuals_df = pd.DataFrame() # Initialize as empty
        if n_worst > 0 and not residuals.empty:
            abs_residuals = residuals.abs().sort_values(ascending=False)
            # Ensure n_worst doesn't exceed the number of test samples
            n_worst_actual = min(n_worst, len(abs_residuals))

            if n_worst_actual > 0:
                worst_original_indices = abs_residuals.head(n_worst_actual).index
                print(f"\nDetailing Top {len(worst_original_indices)} Worst Predictions (by Absolute Residual):")

                # Need to align predictions back to original indices.
                # We predicted on X_test_eval, which corresponds row-wise to y_test.
                # So, we need a mapping from the row index of X_test_eval/y_test back to original index.
                # The 'residuals' Series already has the original index.
                # Re-index y_test_pred to match the original index
                y_test_pred_series = pd.Series(y_test_pred, index=y_test.index)
                worst_preds_aligned = y_test_pred_series.loc[worst_original_indices]


                # Get the original rows from the *original* dataframe
                original_data_worst = df_original.loc[worst_original_indices].copy()

                worst_residuals_info = pd.DataFrame({
                    'Original_Index': worst_original_indices,
                    'Actual_Target': y_test.loc[worst_original_indices].values, # Using .loc with original indices
                    'Predicted_Target': worst_preds_aligned.values,
                    'Residual': residuals.loc[worst_original_indices].values,
                    'Absolute_Residual': abs_residuals.loc[worst_original_indices].values
                }).sort_values('Absolute_Residual', ascending=False).reset_index(drop=True)

                # Combine prediction info with original feature values
                # Use Original_Index as a key for merging
                worst_residuals_df = pd.merge(
                     worst_residuals_info,
                     original_data_worst.reset_index().rename(columns={'index': 'Original_Index'}),
                     on='Original_Index',
                     how='left'
                )


                # Drop duplicate columns if any
                worst_residuals_df = worst_residuals_df.loc[:, ~worst_residuals_df.columns.duplicated()]

                print("Showing head of worst residuals dataframe (Prediction Info & Original Features):")
                # Display key columns for overview
                cols_to_show = ['Original_Index', 'Actual_Target', 'Predicted_Target', 'Residual', 'Absolute_Residual']
                # Add target column name if it exists in df_original and not already included
                if y_test.name in df_original.columns and y_test.name not in cols_to_show:
                    cols_to_show.append(y_test.name)
                # Display only columns that actually exist in the worst_residuals_df
                existing_cols_to_show = [col for col in cols_to_show if col in worst_residuals_df.columns]
                try:
                    print(worst_residuals_df[existing_cols_to_show].head().to_string())
                except KeyError as e:
                    print(f"Warning: Could not display all requested columns in worst residuals head: {e}")
                    print(worst_residuals_df.head().to_string()) # Print whatever is available
            else:
                 print("Not enough samples to identify worst residuals after filtering.")

        else:
            print("Skipping worst residuals analysis (n_worst is 0 or test set empty).")


        return residuals, worst_residuals_df, y_test_pred # Return predictions too

    except Exception as e:
        print(f"Error during residual analysis: {e}")
        traceback.print_exc()
        # Return empty results on failure
        return pd.Series(dtype=float), pd.DataFrame(), np.array([])


# --- 7. Feature Importance (Lorenz/Gini for CatBoost) ---
# (No changes needed from PART 2)
def analyze_catboost_feature_importance_lorenz(model, feature_names_after_rfe, plot=True, figsize=(16,6)):
    """
    Analyzes and plots CatBoost feature importance, including Lorenz curve and Gini coefficient.

    Note: CatBoost importance type depends on training parameters (default is often
    related to feature impact on prediction changes).

    Args:
        model: Trained CatBoostRegressor instance (on final selected features).
        feature_names_after_rfe (list): List of names of features *used by the model*.
        plot (bool): Whether to plot the results.
        figsize (tuple): Figure size for plots.

    Returns:
        tuple: Contains:
            - importance_df (pd.DataFrame): Sorted feature importances with Lorenz data.
            - gini_coefficient (float): Calculated Gini coefficient (0 to 1).
            Returns (pd.DataFrame(), None) if analysis fails or no features/importance.
    Raises:
        RuntimeError, AttributeError
    """
    print("\n--- 7. Analyzing CatBoost Feature Importance ---")
    if not CATBOOST_INSTALLED:
         print("CatBoost library not installed. Skipping importance analysis.")
         return pd.DataFrame(), None

    if not isinstance(model, CatBoostRegressor):
        print(f"Error: Expected a CatBoostRegressor model, but got {type(model)}. Cannot analyze importance.")
        return pd.DataFrame(), None
    if not feature_names_after_rfe:
        print("No feature names provided for the model. Cannot analyze feature importance.")
        return pd.DataFrame(), None

    print(f"Analyzing importance for {len(feature_names_after_rfe)} features used by the model.")
    print("Note: Importance is for the features *after* preprocessing and potential RFE.")
    print("  If PCA was used, importance is for the selected PCA components.")

    if not hasattr(model, 'get_feature_importance'):
        print("Error: Model does not have 'get_feature_importance' method. Cannot analyze.")
        return pd.DataFrame(), None

    try:
        importances = model.get_feature_importance()
        if len(importances) == 0:
             print("Model returned 0 importances.")
             return pd.DataFrame(), None
    except Exception as e:
        print(f"Error getting feature importance from CatBoost model: {e}")
        traceback.print_exc()
        return pd.DataFrame(), None


    if len(importances) != len(feature_names_after_rfe):
         print(f"Error: Mismatch in number of importances ({len(importances)}) and provided feature names ({len(feature_names_after_rfe)}). Cannot proceed reliably.")
         # Attempt to create df anyway, but warn
         feature_names_for_df = [f'Feature_{i}' for i in range(len(importances))]
         print("Warning: Using generic feature names due to length mismatch.")
    else:
        feature_names_for_df = feature_names_after_rfe


    # Create DataFrame and sort by importance (descending)
    importance_df = pd.DataFrame({
        'Feature': feature_names_for_df,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print("\nFeature Importances (CatBoost - Top 10 or all):")
    print(importance_df.head(10).to_string(index=False))

    # --- Lorenz Curve and Gini Coefficient Calculation ---
    imp_sorted = importance_df['Importance'].values
    imp_sorted[imp_sorted < 0] = 0 # Ensure non-negative

    total_importance = np.sum(imp_sorted)
    if np.isclose(total_importance, 0):
        print("Warning: Total feature importance is zero or close to zero. Cannot calculate Gini/Lorenz.")
        return importance_df, None # Return df with 0 importance, Gini=None

    cum_imp = np.cumsum(imp_sorted)
    cum_imp_norm = cum_imp / total_importance

    num_features = len(importance_df)
    # Linspace needs at least 2 points for trapz to work correctly
    if num_features > 0:
        proportion_features = np.linspace(0, 1, num_features + 1)
        cum_imp_norm_with_origin = np.insert(cum_imp_norm, 0, 0)

        area_under_lorenz = np.trapz(cum_imp_norm_with_origin, proportion_features)
        gini_coefficient = 1 - 2 * area_under_lorenz
        gini_coefficient = np.clip(gini_coefficient, 0, 1) # Clamp between 0 and 1
    else:
         print("Warning: 0 features after importance calculation. Cannot compute Gini.")
         gini_coefficient = None


    print(f"\nGini Coefficient of Feature Importance: {gini_coefficient:.4f}" if gini_coefficient is not None else "\nGini Coefficient: Not calculated.")

    if not importance_df.empty and gini_coefficient is not None:
         try:
             importance_df['Cumulative_Importance_Normalized'] = cum_imp_norm.tolist()
         except ValueError:
             print("Warning: Could not add cumulative importance column to DataFrame (length mismatch possible).")


    # Only plot if show_plots is True (controlled by orchestrator)
    current_backend = plt.get_backend()
    if plot and current_backend != 'agg':
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

            # Feature Importance Plot
            y_label_fontsize = 'small' if len(feature_names_for_df) > 20 else None
            # Use top N features if too many for readability
            n_plot = min(len(importance_df), 30)
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(n_plot), palette='viridis', ax=ax1)
            ax1.set_title(f'Top {n_plot} Feature Importances (CatBoost)')
            ax1.set_xlabel('Importance')
            ax1.set_ylabel('Feature')
            ax1.tick_params(axis='y', labelsize=y_label_fontsize)
            ax1.grid(axis='x')

            # Lorenz Curve Plot
            if gini_coefficient is not None:
                ax2.plot(proportion_features, cum_imp_norm_with_origin, marker='.', label='Lorenz Curve')
                ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Line of Equality')
                ax2.set_title(f'Lorenz Curve (Gini = {gini_coefficient:.3f})')
                ax2.set_xlabel('Cumulative Proportion of Features')
                ax2.set_ylabel('Cumulative Proportion of Importance')
                ax2.legend()
                ax2.grid(True)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
            else:
                 ax2.text(0.5, 0.5, 'Cannot plot Lorenz Curve:\nGini not calculated', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
                 ax2.set_title('Lorenz Curve (Not Available)')
                 ax2.set_xticks([])
                 ax2.set_yticks([])

            fig.tight_layout()
            plt.show()
        except Exception as e:
             print(f"Error during feature importance plotting: {e}")
             traceback.print_exc()
    else:
         print("Importance plots skipped (show_plots=False or non-interactive backend).")


    return importance_df, gini_coefficient



# --- 8. NEW: SHAP Value Analysis (for CatBoost) ---

def analyze_shap_values_catboost(model, X_test_final, feature_names_final, test_original_indices, X_test_original=None, plot=True, figsize=(12, 10)):
    """
    Calculates SHAP values for the test set using a CatBoostRegressor,
    generates summary plots, and returns SHAP values and importance summary.

    Args:
        model: Trained CatBoostRegressor model.
        X_test_final (np.ndarray): Final testing features (after preproc/RFE).
        feature_names_final (list): List of names corresponding to X_test_final columns.
        test_original_indices (pd.Index): Original indices of the test samples.
        X_test_original (pd.DataFrame, optional): Test features *before* scaling/PCA,
                                                  used for potentially more interpretable
                                                  dependence plots if provided AND PCA was NOT used.
                                                  Must have same indices as X_test_final.
        plot (bool): Whether to generate SHAP summary plots.
        figsize (tuple): Figure size for SHAP plots.

    Returns:
        tuple: Contains:
            - shap_values_df (pd.DataFrame): SHAP values for each test instance and feature,
                                             indexed by original test index. (None if SHAP fails)
            - shap_summary_df (pd.DataFrame): Mean absolute SHAP value per feature, sorted.
                                              (None if SHAP fails)
            - shap_expected_value (float): The base value (mean prediction over background data).
                                           (None if SHAP fails)
    """
    print("\n--- 8. Analyzing SHAP Values (Test Set) ---")

    if not SHAP_INSTALLED:
        print("SHAP library not found. Skipping SHAP analysis.")
        return None, None, None
    if not CATBOOST_INSTALLED:
         print("CatBoost library not found. Cannot perform SHAP analysis for CatBoost. Skipping SHAP analysis.")
         return None, None, None
    if X_test_final is None or X_test_final.shape[1] == 0:
        print("No final features available. Cannot calculate SHAP values.")
        return None, None, None
    if X_test_final.shape[0] == 0:
        print("Test set is empty. Cannot calculate SHAP values.")
        return None, None, None
    if feature_names_final is None or len(feature_names_final) != X_test_final.shape[1]:
        print(f"Error: Mismatch between number of feature names ({len(feature_names_final if feature_names_final is not None else 'None')}) and data columns ({X_test_final.shape[1]}). Cannot proceed with SHAP.")
        return None, None, None
    if not isinstance(model, CatBoostRegressor):
         print(f"Error: Expected CatBoostRegressor model for SHAP, but got {type(model)}. Skipping SHAP analysis.")
         return None, None, None

    # SHAP works best with DataFrames with column names for TreeExplainer
    # Use the final feature names and original test indices
    # This DataFrame will have the final feature names (PC names or original names subset)
    X_test_final_df = pd.DataFrame(X_test_final, columns=feature_names_final, index=test_original_indices)

    print(f"Calculating SHAP values for {X_test_final_df.shape[0]} test samples and {X_test_final_df.shape[1]} features...")

    try:
        # Use TreeExplainer for CatBoost
        # For CatBoost, it's recommended to use a background dataset (e.g., training data) for the explainer,
        # but for tree models trained on the full dataset, passing just the model is common and often sufficient.
        # If performance is an issue with large datasets, consider using a sample of training data.
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for the test set
        # For TreeExplainer, shap_values output is [N_samples, N_features] for regression
        # Pass the DataFrame with correct final names
        shap_values_test = explainer.shap_values(X_test_final_df)

        # Get the expected value (base value for the explainer)
        shap_expected_value = explainer.expected_value
        # Ensure it's a single value for regression
        if isinstance(shap_expected_value, (list, np.ndarray)):
             if len(shap_expected_value) == 1:
                 shap_expected_value = shap_expected_value[0]
             else:
                 print(f"Warning: SHAP returned multiple expected values ({len(shap_expected_value)}). Using the first one.")
                 shap_expected_value = shap_expected_value[0]
        print(f"  SHAP Expected Value (Base): {shap_expected_value:.4f}")


        # --- Create DataFrames for output ---
        # 1. DataFrame of SHAP values per instance
        shap_values_df = pd.DataFrame(shap_values_test, columns=feature_names_final, index=test_original_indices)

        # 2. DataFrame for SHAP Summary (Global Importance)
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
        shap_summary_df = pd.DataFrame({
            'Feature': feature_names_final,
            'Mean_Abs_SHAP': mean_abs_shap
        }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

        print("\nSHAP Feature Importance Summary (Top 10 or all):")
        print(shap_summary_df.head(10).to_string(index=False))

        # --- Generate Plots ---
        # Only plot if show_plots is True (controlled by orchestrator)
        current_backend = plt.get_backend()
        if plot and current_backend != 'agg':
            print("\nGenerating SHAP summary plots...")
            # Use try/except around plotting calls as they can sometimes fail
            try:
                plt.figure(figsize=figsize) # Adjust figure size as needed

                # Plot 1: SHAP Summary Plot (Bar - Global Importance)
                plt.subplot(2, 1, 1) # Arrange plots vertically
                # The shap.summary_plot function directly creates and modifies matplotlib axes
                # Pass the numpy SHAP values and the DataFrame with correct names for plotting
                shap.summary_plot(shap_values_test, X_test_final_df, plot_type="bar", show=False)
                plt.title("SHAP Feature Importance (Mean Absolute SHAP)") # Set title afterwards

                # Plot 2: SHAP Summary Plot (Beeswarm - Value Distribution & Impact)
                plt.subplot(2, 1, 2)
                shap.summary_plot(shap_values_test, X_test_final_df, plot_type="dot", show=False) # Use 'dot' or default
                plt.title("SHAP Value Distribution and Impact on Prediction")
                plt.tight_layout(pad=2.0) # Adjust padding between subplots
                try:
                    plt.show()
                except Exception as show_e:
                     print(f"Error showing SHAP summary plots: {show_e}")
                     traceback.print_exc()
                     if current_backend != 'agg': plt.close('all') # Close figure on error


            except Exception as plot_e:
                 print(f"Error during SHAP summary plotting: {plot_e}")
                 traceback.print_exc()
                 if current_backend != 'agg': plt.close('all') # Close figure on error


            # Optional: Dependence Plots for top N features
            n_dependence_plots = min(5, len(feature_names_final)) # Plot top 5
            top_features = shap_summary_df['Feature'].head(n_dependence_plots).tolist()

            if n_dependence_plots > 0 and top_features:
                 print(f"\nGenerating SHAP dependence plots for top {n_dependence_plots} features: {top_features}")

                 # --- Determine which data source to use for dependence plots ---
                 # Use original numeric data if PCA was NOT applied
                 # Use the final processed data (with PC names) if PCA WAS applied
                 is_pca_applied = feature_names_final and str(feature_names_final[0]).startswith('PC')

                 if is_pca_applied:
                     print("  Using PCA component values for dependence plots.")
                     data_for_dependence_plot = X_test_final_df # This already has PC names and values
                 else:
                     print("  Using original numeric feature values for dependence plots.")
                     data_for_dependence_plot = X_test_original # This has original names and values

                 if data_for_dependence_plot is not None and isinstance(data_for_dependence_plot, pd.DataFrame) and not data_for_dependence_plot.empty:
                      # Ensure indices align between SHAP values and data for plotting
                      if not data_for_dependence_plot.index.equals(X_test_final_df.index):
                          print("Warning: Index mismatch between data_for_dependence_plot and X_test_final_df. Falling back to X_test_final_df for dependence plots.")
                          data_for_dependence_plot = X_test_final_df # Fallback


                      for feature in top_features:
                          # Check if the specific feature exists in the chosen data source's columns
                          if feature in data_for_dependence_plot.columns:
                              try:
                                  plt.figure() # Create a new figure for each dependence plot
                                  shap.dependence_plot(
                                      feature,
                                      shap_values_test, # SHAP values (numpy array)
                                      data_for_dependence_plot, # Data for x-axis and interaction color
                                      interaction_index="auto", # Let SHAP choose interaction feature
                                      show=False # Prevent immediate show
                                  )
                                  ax_dep = plt.gca() # Get the current axes created by shap
                                  xlabel = f"{feature}"
                                  if is_pca_applied:
                                      xlabel += " (Principal Component Score)"
                                  elif X_test_original is not None and feature in X_test_original.columns:
                                       xlabel += " (Original Scale)"

                                  ax_dep.set_xlabel(xlabel)
                                  ax_dep.set_title(f"SHAP Dependence Plot for {feature}")
                                  plt.tight_layout() # Adjust layout
                                  try:
                                      plt.show() # Show this specific plot
                                  except Exception as show_e:
                                      print(f"Error showing dependence plot for '{feature}': {show_e}")
                                      traceback.print_exc()
                                      if current_backend != 'agg': plt.close('all') # Close figure on error

                              except Exception as dep_e:
                                  print(f"  Could not generate dependence plot for '{feature}': {dep_e}")
                                  traceback.print_exc()
                                  if current_backend != 'agg': plt.close('all') # Close figure on error
                          else:
                               print(f"  Feature '{feature}' not found in the selected data for dependence plot.")


                 else:
                     print("Skipping SHAP dependence plots (selected data source is missing, empty, or not a DataFrame).")

            else:
                 print("Skipping SHAP dependence plots (no top features or dependence plots requested).")

        else:
             print("SHAP plots skipped (show_plots=False or non-interactive backend).")


        return shap_values_df, shap_summary_df, shap_expected_value

    except Exception as e:
        print(f"Error during SHAP analysis: {e}")
        traceback.print_exc()
        return None, None, None



# --- 9. Save Results (CatBoost Version - Modified to include SHAP) ---

def save_results_to_excel(results, filename="catboost_regression_analysis_results.xlsx"):
    """
    Saves the key results of the regression analysis to an Excel file,
    including SHAP information if available.

    Args:
        results (dict): Dictionary containing analysis results.
                        Expected keys: 'run_info', 'metrics', 'importance_df',
                        'worst_residuals_df', 'gini_coefficient',
                        'shap_values_df', 'shap_summary_df', 'shap_expected_value',
                        'y_test', 'y_test_pred', 'residuals', etc.
        filename (str): Path to the output Excel file.
    """
    print(f"\n--- 9. Exporting Results to Excel: {filename} ---")
    if not filename:
         print("No output filename provided. Skipping results export.")
         return
    if not filename.endswith(('.xlsx', '.xls')):
        filename += '.xlsx'
        print(f"Warning: Filename did not end with .xlsx or .xls. Appending .xlsx: {filename}")

    try:
        import openpyxl # Check if openpyxl is available

        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # --- Sheet 1: Run Info ---
            if 'run_info' in results and isinstance(results['run_info'], dict):
                print("  - Preparing 'Run Info' sheet.")
                run_info_export = {}
                for k, v in results['run_info'].items():
                    # Handle common non-serializable types more robustly
                    if isinstance(v, (dict, list, tuple, set)):
                         run_info_export[k] = str(v)
                    elif isinstance(v, (BaseEstimator, StandardScaler, RobustScaler, MinMaxScaler, PCA, RFE)) or (CATBOOST_INSTALLED and isinstance(v, CatBoostRegressor)):
                         # Get class name and maybe key params if simple
                         try:
                             # Attempt to get params, but handle errors
                             if hasattr(v, 'get_params'):
                                 params_str = str(v.get_params(deep=False))
                                 run_info_export[k] = f"{type(v).__name__}({params_str})"
                             else:
                                 run_info_export[k] = type(v).__name__
                         except:
                              run_info_export[k] = type(v).__name__
                    elif pd.isna(v):
                         run_info_export[k] = 'NaN'
                    elif isinstance(v, (np.ndarray, pd.Series, pd.Index)):
                        run_info_export[k] = f"[Object of type {type(v).__name__}, Length: {len(v)}]" # Don't save large arrays
                    else:
                         run_info_export[k] = v

                # Add Gini coefficient directly to run info if available
                # Use the generic 'gini_coefficient' key from CatBoost pipeline
                if 'gini_coefficient' in results and results['gini_coefficient'] is not None:
                     run_info_export['Feature_Importance_Gini'] = results['gini_coefficient']

                # Add SHAP Expected Value if available
                if 'shap_expected_value' in results and results['shap_expected_value'] is not None:
                    run_info_export['SHAP_Expected_Value'] = results['shap_expected_value']
                elif 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True:
                    # Indicate SHAP was attempted but base value wasn't captured
                    run_info_export['SHAP_Expected_Value'] = 'Not Captured (Error/None)'


                run_info_df = pd.DataFrame(list(run_info_export.items()), columns=['Parameter', 'Value'])
                run_info_df.to_excel(writer, sheet_name='Run Info', index=False)
                print("  - Saved 'Run Info' sheet.")
            else:
                print("  - 'run_info' not found or not a dict in results, skipping sheet.")

            # --- Sheet 2: Performance Metrics ---
            if 'metrics' in results and isinstance(results['metrics'], dict) and results['metrics'].get('Test'):
                 print("  - Preparing 'Metrics' sheet.")
                 metrics_df = pd.DataFrame(results['metrics']).reset_index().rename(columns={'index': 'Metric'})
                 metrics_df = metrics_df.melt(id_vars='Metric', var_name='Set', value_name='Value')
                 metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                 print("  - Saved 'Metrics' sheet.")
            elif 'metrics' in results:
                 status = 'Evaluation Failed or Incomplete' if results['metrics'] is None or not results['metrics'].get('Test') else 'Metrics key exists but invalid format'
                 pd.DataFrame([{'Status': status}]).to_excel(writer, sheet_name='Metrics', index=False)
                 print(f"  - Saved 'Metrics' sheet ({status}).")
            else:
                print("  - 'metrics' not found in results, skipping sheet.")

            # --- Sheet 3: Feature Importance ---
            # Using the generic 'importance_df' key from the CatBoost pipeline
            if 'importance_df' in results and isinstance(results['importance_df'], pd.DataFrame) and not results['importance_df'].empty:
                print("  - Preparing 'Feature Importance (CatBoost)' sheet.")
                results['importance_df'].to_excel(writer, sheet_name='Feature Importance (CatBoost)', index=False)
                print("  - Saved 'Feature Importance (CatBoost)' sheet.")
            else:
                 print("  - 'importance_df' not found, empty, or not a DataFrame in results, skipping sheet.")

            # --- Sheet 4: SHAP Importance Summary --- (NEW)
            # Check if SHAP analysis was attempted and succeeded in generating summary
            if 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True and \
               'shap_summary_df' in results and isinstance(results['shap_summary_df'], pd.DataFrame) and not results['shap_summary_df'].empty:
                print("  - Preparing 'SHAP Importance Summary' sheet.")
                results['shap_summary_df'].to_excel(writer, sheet_name='SHAP Importance Summary', index=False)
                print("  - Saved 'SHAP Importance Summary' sheet.")
            else:
                 status = "SHAP analysis skipped or failed to produce summary" if 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True else "SHAP analysis not requested or libraries missing"
                 print(f"  - {status}, skipping 'SHAP Importance Summary' sheet.")
                 # Add a sheet indicating failure if SHAP was attempted but failed
                 if 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True:
                     pd.DataFrame([{'Status': status}]).to_excel(writer, sheet_name='SHAP Importance Summary', index=False)


            # --- Sheet 5: SHAP Values (Test Set) --- (NEW - Local Explanations)
            # Check if SHAP analysis was attempted and succeeded in generating values df
            if 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True and \
               'shap_values_df' in results and isinstance(results['shap_values_df'], pd.DataFrame) and not results['shap_values_df'].empty:
                print("  - Preparing 'SHAP Values (Test Set)' sheet.")
                # Combine SHAP values with actual, predicted, and residuals for context
                shap_local_df = results['shap_values_df'].copy()

                # Add prefix to SHAP value columns to avoid name clashes
                shap_local_df.columns = ['SHAP_' + str(col) for col in shap_local_df.columns] # Ensure column names are strings

                # Add Actual Target (y_test)
                if 'y_test' in results and isinstance(results['y_test'], pd.Series):
                    # Ensure index alignment (join keeps matching indices, fills others with NaN)
                    # Using align and then fillna(0) for SHAP columns or dropna() might be options
                    # A right join on shap_local_df's index seems most appropriate here
                    try:
                        y_test_aligned = results['y_test'].reindex(shap_local_df.index)
                        shap_local_df['Actual_Target'] = y_test_aligned
                    except Exception as e:
                        print(f"  - Warning: Could not align/add 'y_test' to SHAP values sheet: {e}")
                else:
                    print("  - Warning: Could not find 'y_test' in results to add to SHAP values sheet.")

                # Add Predicted Target (y_test_pred)
                if 'y_test_pred' in results and isinstance(results['y_test_pred'], np.ndarray) and len(results['y_test_pred']) == len(shap_local_df):
                     # Need to align predictions with the original index.
                     # y_test_pred is a numpy array corresponding to the order of X_test_final
                     # shap_local_df already has the original index from X_test_final_df
                     try:
                         pred_series = pd.Series(results['y_test_pred'], index=shap_local_df.index, name='Predicted_Target')
                         shap_local_df['Predicted_Target'] = pred_series
                     except Exception as e:
                          print(f"  - Warning: Could not align/add 'y_test_pred' to SHAP values sheet: {e}")
                elif 'y_test_pred' in results:
                     print(f"  - Warning: 'y_test_pred' found but type/length mismatch ({type(results['y_test_pred'])}, len={len(results.get('y_test_pred',[]))}). Cannot add to SHAP values sheet.")
                else:
                    print("  - Warning: Could not find 'y_test_pred' in results to add to SHAP values sheet.")


                # Add Residuals
                if 'residuals' in results and isinstance(results['residuals'], pd.Series):
                    # Ensure index alignment (residuals series already has the original index)
                    try:
                        residuals_aligned = results['residuals'].reindex(shap_local_df.index)
                        shap_local_df['Residual'] = residuals_aligned
                    except Exception as e:
                        print(f"  - Warning: Could not align/add 'residuals' to SHAP values sheet: {e}")
                else:
                    print("  - Warning: Could not find 'residuals' in results to add to SHAP values sheet.")


                # Add SHAP Expected Value (Base Value) as a column for reference
                if 'shap_expected_value' in results and results['shap_expected_value'] is not None:
                     shap_local_df['SHAP_Base_Value'] = results['shap_expected_value']

                # Reorder columns for clarity (identifiers first, then SHAP values)
                id_cols = ['Actual_Target', 'Predicted_Target', 'Residual', 'SHAP_Base_Value']
                shap_cols = [col for col in shap_local_df.columns if col.startswith('SHAP_') and col != 'SHAP_Base_Value']
                final_cols = [col for col in id_cols if col in shap_local_df.columns] + shap_cols

                # Ensure all columns exist before reordering
                final_cols = [col for col in final_cols if col in shap_local_df.columns]
                shap_local_df = shap_local_df[final_cols]


                shap_local_df.to_excel(writer, sheet_name='SHAP Values (Test Set)', index=True, index_label='Original_Index')
                print("  - Saved 'SHAP Values (Test Set)' sheet.")
            else:
                 status = "SHAP analysis skipped or failed to produce values" if 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True else "SHAP analysis not requested or libraries missing"
                 print(f"  - {status}, skipping 'SHAP Values (Test Set)' sheet.")
                 # Add a sheet indicating failure if SHAP was attempted but failed
                 if 'run_info' in results and results['run_info'].get('shap_analysis_performed') is True:
                      pd.DataFrame([{'Status': status}]).to_excel(writer, sheet_name='SHAP Values (Test Set)', index=False)


            # --- Sheet 6: Worst Residuals ---
            # Renumbered sheet
            if 'worst_residuals_df' in results and isinstance(results['worst_residuals_df'], pd.DataFrame) and not results['worst_residuals_df'].empty:
                print("  - Preparing 'Worst Residuals' sheet.")
                worst_residuals_df_cleaned_cols = results['worst_residuals_df'].copy()
                # Ensure column names are strings for Excel compatibility
                worst_residuals_df_cleaned_cols.columns = worst_residuals_df_cleaned_cols.columns.astype(str)
                # Limit columns saved for large original dataframes? Maybe not necessary.
                worst_residuals_df_cleaned_cols.to_excel(writer, sheet_name='Worst Residuals', index=False)
                print("  - Saved 'Worst Residuals' sheet.")
            else:
                print("  - 'worst_residuals_df' not found, empty, or not a DataFrame in results, skipping sheet.")

        print(f"Successfully saved results to {filename}")

    except ImportError:
        print("\nError: 'openpyxl' library is required to write Excel files.")
        print("Please install it using: pip install openpyxl")
    except Exception as e:
        print(f"\nError writing results to Excel file '{filename}': {e}")
        traceback.print_exc()


# --- 10. Main Orchestrator (CatBoost Version - Modified for SHAP) ---
# (Modified to calculate final data once, call SHAP, and store SHAP results)

def run_full_catboost_pipeline(
    df,
    target_col,
    scaler_type='Standard',
    pca_n_comp=None,
    test_size=0.2,
    random_state=42,
    target_feature_ratio=0.6, # Ratio for RFE
    rfe_catboost_params=None, # Params for CatBoost used *inside* RFE
    n_worst_residuals=20,
    catboost_params={'iterations': 100, 'learning_rate': 0.1, 'depth': 10, 'l2_leaf_reg': 3}, # Params for the FINAL CatBoost model
    early_stopping_rounds=50,
    output_filename="catboost_regression_analysis_results.xlsx",
    run_shap_analysis=True, # Option to skip SHAP if needed
    show_plots=True # Control plotting globally
    ):
    """
    Runs the complete regression analysis pipeline using CatBoost, including SHAP analysis.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Name of the target variable column.
        scaler_type (str): Scaler type ('Standard', 'Robust', 'MinMax', 'None').
        pca_n_comp (int/float/None): PCA components or variance.
        test_size (float): Test split proportion.
        random_state (int): Global random seed.
        target_feature_ratio (float/None): RFE feature ratio to keep (0, 1). Skip if None/ >=1 / <=0.
        rfe_catboost_params (dict/None): Params for CatBoost *inside* RFE.
        n_worst_residuals (int): Number of worst residuals to detail.
        catboost_params (dict): Parameters for the FINAL CatBoostRegressor.
        early_stopping_rounds (int/None): Early stopping rounds for final CatBoost training.
        output_filename (str): Path to save results Excel file.
        run_shap_analysis (bool): If True and SHAP is installed, run SHAP analysis.
        show_plots (bool): If True, display generated matplotlib/seaborn/SHAP plots.

    Returns:
        dict: A dictionary containing key results of the analysis.
              Returns None if a critical initial step fails.
    """
    print("\n--- Starting Full CatBoost Regression Pipeline ---")
    print(f"Random State used throughout pipeline: {random_state}")
    start_time = time.time()

    if not CATBOOST_INSTALLED:
        print("CRITICAL ERROR: CatBoost library is not installed or failed to import. Cannot run pipeline.")
        return None

    # Turn off interactive plotting if show_plots is False
    current_backend = plt.get_backend()
    if not show_plots and current_backend != 'agg':
        plt.ioff() # Turn interactive mode off
        print("Plot display disabled (show_plots=False). Plots might still be generated but not shown.")
        plotting_disabled = True
    elif current_backend == 'agg':
         print("Detected non-interactive backend ('agg'). Plots will be generated but not displayed.")
         plotting_disabled = True
    else:
        plt.ion() # Ensure interactive mode is on (default)
        plotting_disabled = False


    results = {'run_info': {
        'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': 'CatBoostRegressor',
        'target_column': target_col,
        'test_size': test_size,
        'random_state': random_state,
        'scaler_type': scaler_type,
        'pca_n_comp': pca_n_comp,
        'target_feature_ratio_rfe': target_feature_ratio,
        'rfe_estimator_params': rfe_catboost_params if rfe_catboost_params else 'Defaults',
        'n_worst_residuals': n_worst_residuals,
        'catboost_params_final_model': catboost_params,
        'early_stopping_rounds': early_stopping_rounds,
        'run_shap_analysis_requested': run_shap_analysis,
        'shap_analysis_performed': False, # Will be updated later
        'show_plots': show_plots,
        'output_filename': output_filename
    }}

    # --- 1. Data Preparation & Splitting ---
    try:
        (X_train, X_test, y_train, y_test,
         initial_numeric_feature_names, test_original_indices, train_original_indices) = split_and_select_numeric_data(
             df, target_col, test_size, random_state
         )
        results['y_train'] = y_train
        results['y_test'] = y_test
        results['initial_numeric_feature_names'] = initial_numeric_feature_names
        results['test_original_indices'] = test_original_indices
        results['train_original_indices'] = train_original_indices
        # Store X_test (numeric, imputed) for potential use in SHAP dependence plots
        results['X_test_original_numeric'] = X_test.copy() # Ensure it's a DataFrame copy

        print("Step 1: Data preparation and splitting complete.")
        if X_train.shape[1] == 0:
             print("\nCRITICAL ERROR: No numeric features found after splitting. Cannot proceed.")
             results['run_info']['error_step'] = 1
             results['run_info']['error_message'] = "No numeric features after splitting"
             save_results_to_excel(results, output_filename)
             if plotting_disabled: plt.close('all')
             return None
    except Exception as e:
        print(f"\nCRITICAL ERROR during Step 1 (Data Prep & Split): {e}")
        traceback.print_exc()
        results['run_info']['error_step'] = 1
        results['run_info']['error_message'] = str(e)
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 2. Apply Preprocessing (Scaling and PCA) ---
    try:
        # Pass X_train, X_test as DataFrames as expected by apply_preprocessing
        (X_train_processed, X_test_processed, feature_names_after_processing,
         scaler_instance, pca_instance) = apply_preprocessing_cat(
             X_train, X_test, scaler_type, pca_n_comp, random_state
         )
        results['X_train_processed_shape'] = X_train_processed.shape
        results['X_test_processed_shape'] = X_test_processed.shape
        results['scaler_instance'] = scaler_instance
        results['pca_instance'] = pca_instance
        results['feature_names_after_processing'] = feature_names_after_processing
        results['run_info']['n_features_after_preprocessing'] = X_train_processed.shape[1]
        print("Step 2: Preprocessing complete.")
        if X_train_processed.shape[1] == 0:
             print("\nCRITICAL ERROR: 0 features remaining after preprocessing. Cannot proceed.")
             results['run_info']['error_step'] = 2
             results['run_info']['error_message'] = "0 features after preprocessing"
             save_results_to_excel(results, output_filename)
             if plotting_disabled: plt.close('all')
             return None
    except Exception as e:
        print(f"\nCRITICAL ERROR during Step 2 (Preprocessing): {e}")
        traceback.print_exc()
        results['run_info']['error_step'] = 2
        results['run_info']['error_message'] = str(e)
        results['scaler_instance'] = None
        results['pca_instance'] = None
        results['feature_names_after_processing'] = []
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 3. Perform RFE (with CatBoost) OR Skip ---
    rfe_object = None
    X_train_final = X_train_processed # Default if RFE fails/skipped
    X_test_final = X_test_processed   # Default if RFE fails/skipped
    selected_feature_names_final = feature_names_after_processing # Default if RFE fails/skipped
    selected_mask_final = np.ones(X_train_processed.shape[1], dtype=bool) # Default

    # Determine if RFE should run based on ratio and features available
    rfe_should_run = (target_feature_ratio is not None and 0 < target_feature_ratio < 1.0 and X_train_processed.shape[1] > 1)
    results['run_info']['rfe_used'] = rfe_should_run # Record if RFE was attempted

    if rfe_should_run and CATBOOST_INSTALLED: # Also check if CatBoost is installed for RFE
        try:
            print("\nStep 3: Performing RFE selection...")
            rfe_object, selected_feature_names_rfe, X_train_rfe, selected_mask_rfe = perform_rfe_with_catboost(
                X_train_processed, y_train, feature_names_after_processing,
                target_feature_ratio, random_state, rfe_catboost_params
            )
            # Update final variables ONLY if RFE ran successfully and selected features
            if rfe_object is not None and X_train_rfe is not None and X_train_rfe.shape[1] > 0:
                 selected_feature_names_final = selected_feature_names_rfe
                 X_train_final = X_train_rfe
                 # IMPORTANT: Transform the TEST set using the fitted RFE object
                 X_test_final = rfe_object.transform(X_test_processed) # Use pre-RFE test data
                 selected_mask_final = selected_mask_rfe
                 print("Step 3: RFE selection applied successfully.")
                 print(f"  Final features selected: {len(selected_feature_names_final)}")
                 print(f"  Final data shapes: X_train={X_train_final.shape}, X_test={X_test_final.shape}")
            else:
                 # RFE was attempted but failed or selected 0 features, use pre-RFE data
                 print("Step 3: RFE was run but failed or selected 0 features. Using pre-RFE data.")
                 rfe_object = None # Ensure rfe_object is None if RFE didn't effectively run
                 results['run_info']['rfe_used'] = False # Mark RFE as not effectively used
                 # X_train_final etc. retain their default (pre-RFE) values
                 print(f"  Using pre-RFE data. Final train shape: {X_train_final.shape}")


        except Exception as e:
            print(f"\nCRITICAL ERROR during Step 3 (RFE): {e}")
            traceback.print_exc()
            results['run_info']['error_step'] = 3
            results['run_info']['error_message'] = str(e)
            # Use pre-RFE data in case of critical failure
            rfe_object = None
            results['run_info']['rfe_used'] = False # Mark RFE as failed
            # X_train_final etc. retain their pre-RFE values (X_train_processed etc.)
            print("  Proceeding with pre-RFE features due to error.")
    elif not CATBOOST_INSTALLED:
        print("\nStep 3: RFE skipped (CatBoost not installed).")
        results['run_info']['rfe_used'] = False
    else:
        print("\nStep 3: RFE skipped (target_feature_ratio not in (0, 1) or <= 1 feature).")
        results['run_info']['rfe_used'] = False
        # X_train_final etc. retain their pre-RFE values (X_train_processed etc.)


    results['rfe_object'] = rfe_object
    results['selected_feature_names_final'] = selected_feature_names_final
    results['selected_mask_final'] = selected_mask_final # Mask relative to X_..._processed
    results['run_info']['n_features_final'] = X_train_final.shape[1]

    if X_train_final.shape[1] == 0:
        print("\nCRITICAL ERROR: 0 features remaining after RFE/selection process. Cannot proceed.")
        results['run_info']['error_step'] = 3
        results['run_info']['error_message'] = "0 final features after RFE/selection"
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 4. Train Final Model ---
    try:
        final_model = train_catboost_regressor(
            X_train_final, y_train, X_test_final, y_test, # Pass final train/test data
            catboost_params=catboost_params,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state
        )
        results['model'] = final_model
        print("Step 4: Final model training complete.")
    except Exception as e:
        print(f"\nCRITICAL ERROR during Step 4 (Model Training): {e}")
        traceback.print_exc()
        results['run_info']['error_step'] = 4
        results['run_info']['error_message'] = str(e)
        results['model'] = None
        save_results_to_excel(results, output_filename)
        if plotting_disabled: plt.close('all')
        return None

    # --- 5. Evaluate Performance ---
    try:
        # Pass the original PROCESSED data, evaluation func will handle RFE transform using rfe_object
        # Pass y_train and y_test as Series
        metrics = evaluate_model_performance(
            final_model, X_train_processed, y_train, X_test_processed, y_test, rfe_object=rfe_object
        )
        results['metrics'] = metrics
        print("Step 5: Performance evaluation complete.")
    except Exception as e:
        print(f"\nError during Step 5 (Performance Evaluation): {e}")
        traceback.print_exc()
        results['run_info']['warning_step_5'] = str(e)
        results['metrics'] = {"Train": {}, "Test": {}} # Mark as failed

    # --- 6. Analyze Residuals ---
    try:
        # Pass the original PROCESSED data, analysis func will handle RFE transform using rfe_object
        # Pass y_test as Series. Pass original df and indices.
        # analyze_residuals now returns y_test_pred
        residuals, worst_residuals_df, y_test_pred = analyze_residuals(
            final_model, X_test_processed, y_test, df, test_original_indices, rfe_object=rfe_object, n_worst=n_worst_residuals
        )
        results['residuals'] = residuals # pd.Series, indexed by original index
        results['worst_residuals_df'] = worst_residuals_df # pd.DataFrame
        results['y_test_pred'] = y_test_pred # np.ndarray, corresponds to X_test_eval order

        print("Step 6: Residual analysis complete.")
    except Exception as e:
        print(f"\nError during Step 6 (Residual Analysis): {e}")
        traceback.print_exc()
        results['run_info']['warning_step_6'] = str(e)
        results['residuals'] = pd.Series(dtype=float)
        results['worst_residuals_df'] = pd.DataFrame()
        results['y_test_pred'] = np.array([]) # Mark as failed

    # --- 7. Analyze Feature Importance (CatBoost) ---
    try:
        # Pass the model and the list of FINAL selected names
        importance_df, gini_coefficient = analyze_catboost_feature_importance_lorenz(
            final_model, results.get('selected_feature_names_final', []), plot=show_plots
        )
        results['importance_df'] = importance_df
        results['gini_coefficient'] = gini_coefficient
        # Add Gini to run_info for easy access
        results['run_info']['Feature_Importance_Gini'] = gini_coefficient
        print("Step 7: Feature importance analysis complete.")
    except Exception as e:
        print(f"\nError during Step 7 (Feature Importance Analysis): {e}")
        traceback.print_exc()
        results['run_info']['warning_step_7'] = str(e)
        results['importance_df'] = pd.DataFrame()
        results['gini_coefficient'] = None
        results['run_info']['Feature_Importance_Gini'] = 'Error'


    # --- 8. Analyze SHAP Values ---
    shap_values_df, shap_summary_df, shap_expected_value = None, None, None # Initialize
    if run_shap_analysis and SHAP_INSTALLED and CATBOOST_INSTALLED:
        results['run_info']['shap_analysis_performed'] = True
        try:
            # Pass original numeric X_test if available for dependence plots
            X_test_orig_for_shap = results.get('X_test_original_numeric') # This is a DataFrame
            shap_values_df, shap_summary_df, shap_expected_value = analyze_shap_values_catboost(
                final_model,
                X_test_final, # This is the final numpy array after preproc/RFE
                results.get('selected_feature_names_final', []), # Final feature names
                test_original_indices, # Original indices for DataFrame conversion
                X_test_original=X_test_orig_for_shap, # Original numeric data for dependence plots
                plot=show_plots # Control plotting
            )
            results['shap_values_df'] = shap_values_df
            results['shap_summary_df'] = shap_summary_df
            results['shap_expected_value'] = shap_expected_value
            results['run_info']['SHAP_Expected_Value'] = shap_expected_value
            print("Step 8: SHAP analysis complete.")
        except Exception as e:
            print(f"\nError during Step 8 (SHAP Analysis): {e}")
            traceback.print_exc()
            results['run_info']['warning_step_8'] = str(e)
            results['shap_values_df'] = None # Mark as failed
            results['shap_summary_df'] = None
            results['shap_expected_value'] = None
            results['run_info']['SHAP_Expected_Value'] = 'Analysis Failed'
    elif not SHAP_INSTALLED:
         print("\nStep 8: SHAP Analysis skipped (SHAP library not installed).")
         results['run_info']['shap_analysis_performed'] = False
    elif not CATBOOST_INSTALLED: # Should be caught at the start, but double check
         print("\nStep 8: SHAP Analysis skipped (CatBoost library not installed).")
         results['run_info']['shap_analysis_performed'] = False
    else: # SHAP and CatBoost installed but run_shap_analysis is False
         print("\nStep 8: SHAP Analysis skipped (run_shap_analysis=False).")
         results['run_info']['shap_analysis_performed'] = False


    # --- 9. Save Results ---
    # (Uses the modified save function for CatBoost)
    save_results_to_excel(results, output_filename)
    print("Step 9: Results export attempt complete.")

    end_time = time.time()
    total_duration = end_time - start_time
    results['run_info']['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['run_info']['total_duration_seconds'] = round(total_duration, 2)
    print(f"\n--- Full CatBoost Regression Pipeline Finished in {total_duration:.2f} seconds ---")

    # Final summary printout
    if 'metrics' in results and results['metrics'] and 'Test' in results['metrics']:
        print("\n--- Final Test Set Performance ---")
        test_metrics = results['metrics']['Test']
        if test_metrics:
            for metric, value in test_metrics.items():
                 print(f"  {metric:<5}: {value:.4f}")
        else:
            print("  Test metrics calculation failed or resulted empty.")
    else:
        print("\n--- Final Test Set Performance: Not Available or Evaluation Failed ---")

    # Close all plots if plotting was disabled
    if plotting_disabled:
        plt.close('all')

    # Re-enable interactive plotting if it was disabled by this function
    if not show_plots and current_backend != 'agg':
         plt.ion()


    return results



## CATBOOST - CATBOOST - CATBOOST CATBOOST - CATBOOST - CATBOOST - ####################
## CATBOOST - CATBOOST - CATBOOST CATBOOST - CATBOOST - CATBOOST - ####################
## CATBOOST - CATBOOST - CATBOOST CATBOOST - CATBOOST - CATBOOST - ####################
## CATBOOST - CATBOOST - CATBOOST CATBOOST - CATBOOST - CATBOOST - ####################
## CATBOOST - CATBOOST - CATBOOST CATBOOST - CATBOOST - CATBOOST - ####################



# --- Data Splitting (Minor adjustments for potential CatBoost use) ---
def split_data(df: pd.DataFrame,
               target_name: str,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Splits the dataframe into training and testing sets.
    Returns X as DataFrame to preserve column names/types before potential pipeline steps.
    """
    print(f"\n--- Splitting Data (Test Size: {test_size}, Random State: {random_state}) ---")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_name not in df.columns:
        raise KeyError(f"Target column '{target_name}' not found in DataFrame columns: {df.columns.tolist()}")
    if not pd.api.types.is_numeric_dtype(df[target_name]):
        raise ValueError(f"Target column '{target_name}' must be numeric for regression.")
    if df[target_name].isnull().any():
         print(f"Warning: Target column '{target_name}' contains missing values. Dropping rows with NaN target.")
         df_cleaned = df.dropna(subset=[target_name]).copy()
    else:
         df_cleaned = df.copy()

    if df_cleaned.empty:
         raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
         raise ValueError("Not enough samples left after handling missing target values to perform a split.")

    X = df_cleaned.drop(columns=[target_name])
    y = df_cleaned[target_name]

    if X.isnull().sum().sum() > 0:
        print("Warning: Feature data contains missing values (NaNs).")
        print("  Pipeline steps like Scaling/PCA cannot handle NaNs.")
        print("  Consider imputing NaNs *before* calling split_data or adding an Imputer step to the pipeline.")

        numeric_cols_with_nan = X.select_dtypes(include=np.number).columns[X.select_dtypes(include=np.number).isnull().any()]
        if not numeric_cols_with_nan.empty:
             print(f"  Imputing {len(numeric_cols_with_nan)} numeric columns with median (train/test leakage risk).")
             imputation_values = X[numeric_cols_with_nan].median()
             X[numeric_cols_with_nan] = X[numeric_cols_with_nan].fillna(imputation_values)
             # Re-check for NaNs
             if X.isnull().sum().sum() > 0:
                  print("Warning: NaNs still exist after numeric imputation. Check non-numeric features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Store original feature names AFTER imputation but BEFORE pipeline steps
    # This is crucial for feature importance if PCA is NOT used.
    original_feature_names = list(X_train.columns)

    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    # Return X as DataFrame so pipeline steps can inspect columns if needed (e.g. for cat features)
    return X_train, X_test, y_train, y_test, original_feature_names


# --- Tuning Function (Adapted for CatBoost) ---
def tune_catboost(X_train: pd.DataFrame, # Keep as DataFrame for potential cat feature handling later
                  y_train: pd.Series,
                  pipeline_param_distributions: dict, # Expects prefixed keys
                  n_iter: int,
                  cv: int,
                  scoring: str,
                  random_state: int,
                  apply_scaling: bool = False,
                  apply_pca: bool = False,
                  pca_n_components=None,
                  # CatBoost specific args for the estimator itself (not hyperparameters to tune in this dict)
                  catboost_base_params: dict = None,
                  # If you decide to pass categorical features directly to CatBoost:
                  # categorical_features_indices='auto' # Or a list of indices
                  ):
    
    print(f"\n--- Starting Randomized Search (Pipeline: Scale={apply_scaling}, PCA={apply_pca}) ---")
    print(f"   (n_iter={n_iter}, cv={cv}, scoring='{scoring}')")
    search_start_time = time.time()

    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost library is not installed. Cannot perform tuning.")

    # 1. Define Pipeline Steps
    steps = []

    if apply_scaling:
        # Choose scaler based on need, StandardScaler is common default
        # You could make scaler_type a parameter here if you want to tune scaler types
        steps.append(('scaler', StandardScaler()))
        print("   Adding StandardScaler step.")

    if apply_pca:
        if isinstance(pca_n_components, int):
             if pca_n_components <= 0:
                 raise ValueError(f"pca_n_components must be > 0 for int input, got {pca_n_components}.")
             # Cap components at num features *after* any potential scaling
             max_features_after_scaling = X_train.shape[1] # Assuming scaling doesn't change num features
             if apply_scaling: # Need to fit scaler first to get shape if dynamic
                  # This check is tricky in a pipeline. Better to check/warn before the tuning function.
                  # Let's rely on PCA itself to handle n_components > n_features
                  pass # PCA will handle n_components > n_features gracefully by setting it to n_features
             if pca_n_components > max_features_after_scaling:
                  # PCA handles this by setting n_components_ to n_features
                  print(f"Warning: pca_n_components ({pca_n_components}) > features after scaling ({max_features_after_scaling}). PCA will adjust.")

        elif isinstance(pca_n_components, float) and not (0 < pca_n_components <= 1):
             raise ValueError(f"pca_n_components must be between 0 and 1 for float input, got {pca_n_components}.")
        elif pca_n_components is None:
             # PCA default is min(n_samples, n_features) - 1, often too many components
             warnings.warn("apply_pca is True, but pca_n_components is None. PCA will use default n_components, which might be high.", UserWarning)


        steps.append(('pca', PCA(n_components=pca_n_components, random_state=random_state)))
        print(f"   Adding PCA step (n_components={pca_n_components}).")

    # Define base CatBoost parameters (not the ones being tuned)
    catboost_base_params_final = catboost_base_params.copy() if catboost_base_params is not None else {}
    # Ensure random_state is set for reproducibility outside the search space
    catboost_base_params_final.setdefault('random_state', random_state)
    # Ensure verbosity is low during CV fitting
    catboost_base_params_final['verbose'] = 0
    # Ensure loss function is set for regression
    catboost_base_params_final.setdefault('loss_function', 'RMSE')
    # eval_metric is good for monitoring, but scoring determines the best model in search
    catboost_base_params_final.setdefault('eval_metric', 'RMSE')

    # Add the CatBoostRegressor estimator to the pipeline
    steps.append(('catboost', CatBoostRegressor(**catboost_base_params_final)))
    pipeline = Pipeline(steps)
    print(f"   Pipeline steps: {[s[0] for s in pipeline.steps]}")

    # 2. Verify Parameter Keys (Ensure prefixes match pipeline steps)
    print("   Hyperparameter search space:")
    valid_step_names = [s[0] for s in pipeline.steps]
    prefixed_search_space = {}
    for param_key, dist in pipeline_param_distributions.items():
         if '__' not in param_key:
              print(f"Warning: Parameter key '{param_key}' lacks a pipeline prefix. Assuming it's for the 'catboost' step.")
              prefixed_search_space[f'catboost__{param_key}'] = dist
              param_key_to_print = f'catboost__{param_key}' # Use prefixed name for print
         else:
              step_name = param_key.split('__')[0]
              if step_name not in valid_step_names:
                  warnings.warn(f"Parameter '{param_key}' targets step '{step_name}', which is NOT in the current pipeline configuration: {valid_step_names}. This parameter will be ignored by RandomizedSearchCV.", UserWarning)
                  continue # Skip parameter if step is not in pipeline
              prefixed_search_space[param_key] = dist
              param_key_to_print = param_key # Use original prefixed name for print


         # Pretty print the distribution
         if hasattr(dist, 'dist'): # For scipy.stats distributions
             try:
                 print(f"     {param_key_to_print}: {dist.dist.name}(args={dist.args}, kwds={dist.kwds})")
             except Exception:
                 print(f"     {param_key_to_print}: {dist} (scipy.stats distribution)")
         elif isinstance(dist, (list, np.ndarray)): # For lists/arrays of values
              print(f"     {param_key_to_print}: {dist}")
         else:
             print(f"     {param_key_to_print}: {dist} (single value or unknown type)")


    if not prefixed_search_space:
         raise ValueError("Hyperparameter search space is empty after processing prefixes and pipeline steps. Cannot perform tuning.")

    # 3. Setup and Run Randomized Search
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # CatBoostRegressor doesn't need n_jobs explicitly in the estimator if using Pipeline.
    # n_jobs in RandomizedSearchCV controls parallel search over parameter combinations.
    # For CatBoost, it's often recommended to let CatBoost manage threads internally
    # unless you are tuning many small models. For large models, use CatBoost's
    # thread_count parameter. n_jobs in RandomizedSearchCV is usually fine.

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=prefixed_search_space, # Use the validated/prefixed dict
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1, # Use all available cores for the search iterations
        random_state=random_state,
        return_train_score=True,
        verbose=1 # Set verbosity level for RandomizedSearchCV itself
    )

    print(f"   Fitting RandomizedSearchCV...")
    try:
        # X_train should be DataFrame here if you want potential CatBoost native features
        random_search.fit(X_train, y_train)
        print("   RandomizedSearchCV fitting complete.")
    except Exception as e:
        print(f"\nError during RandomizedSearchCV fit: {e}")
        traceback.print_exc()
        raise RuntimeError(f"RandomizedSearchCV fit failed: {e}")

    search_end_time = time.time()
    search_time = search_end_time - search_start_time
    print(f"Search finished in {search_time:.2f} seconds.")

    best_estimator = random_search.best_estimator_
    results_df = pd.DataFrame(random_search.cv_results_)
    # Clean up column names that might have special characters from prefixes
    results_df.columns = results_df.columns.str.replace(":", "_", regex=False)
    results_df = results_df.sort_values(by=f'rank_test_score', ascending=True)


    print("\n--- Best Pipeline Results ---")
    # Use best_score_ and best_params_ from the fitted search object
    best_score = random_search.best_score_
    best_params = random_search.best_params_

    print(f"Best CV Score ({scoring}): {best_score:.4f}")
    print("Best Parameters Found:")
    # Filter best_params to show only those that were actually searched/set
    # and match steps in the best pipeline.
    # The best_params_ attribute already gives you the final parameters set on the best_estimator.
    # We can just print those.
    params_to_print = {k: v for k, v in best_params.items() if k.split('__')[0] in valid_step_names}

    for param, value in params_to_print.items():
        print(f"  {param}: {value}")


    # Add best test score from the search results itself for the top model
    try:
        best_result_row = results_df[results_df['rank_test_score'] == 1].iloc[0]
        mean_train_score = best_result_row.get('mean_train_score')
        std_test_score = best_result_row.get('std_test_score')
        std_train_score = best_result_row.get('std_train_score')

        print(f"\nMean Train Score ({scoring}): {mean_train_score:.4f} (+/- {std_train_score:.4f})")
        print(f"Mean Test Score ({scoring}): {best_score:.4f} (+/- {std_test_score:.4f})") # Same as best_score_ but includes std

    except Exception as e:
         print(f"Warning: Could not extract detailed scores for best model from cv_results_: {e}")


    return best_estimator, results_df, search_time


# --- Saving Results (Adapted filename) ---
def save_tuning_results_catboost(results_df: pd.DataFrame, best_params: dict, best_score: float,
                        scoring: str, cv: int, n_iter: int, search_time: float,
                        random_state: int, results_dir: str):
    """Saves the detailed tuning results (CSV) and best parameters (TXT)."""
    print("\n--- Saving Tuning Results ---")
    os.makedirs(results_dir, exist_ok=True)
    # Use a distinct filename for CatBoost results
    results_filename = os.path.join(results_dir, 'catboost_tuning_results.csv')
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"Detailed tuning results saved to: {results_filename}")
    except Exception as e: print(f"Error saving results CSV: {e}")

    # Use a distinct filename for best params summary
    best_params_filename = os.path.join(results_dir, 'catboost_best_params.txt')
    try:
        with open(best_params_filename, 'w') as f:
            f.write(f"--- CatBoost Pipeline Tuning Summary ---\n")
            f.write(f"Scoring Metric: {scoring}\n")
            f.write(f"Best CV Score ({scoring}): {best_score:.4f}\n")
            f.write("\nBest Pipeline Parameters:\n")
            # Iterate through the best_params dictionary directly
            for param, value in best_params.items():
                 f.write(f"  {param}: {value}\n")

            f.write("\n--- Search Configuration ---\n")
            f.write(f"CV Folds: {cv}\n")
            f.write(f"Search Iterations (n_iter): {n_iter}\n")
            f.write(f"Random State Seed: {random_state}\n")
            f.write(f"Tuning Time: {search_time:.2f} seconds\n")
        print(f"Best parameters summary saved to: {best_params_filename}")
    except Exception as e: print(f"Error saving best parameters text file: {e}")




# --- Feature Importance (Adapted for CatBoost) ---
def get_feature_importances(estimator, feature_names: list = None, top_n: int = None):
    """Extracts feature importances from the final estimator. Adapts for pipelines."""
    print("\n--- Extracting Feature Importances ---")
    final_estimator = None; transformer_steps = []
    if isinstance(estimator, Pipeline):
        final_estimator = estimator.steps[-1][1]
        transformer_steps = estimator.steps[:-1]
    else: final_estimator = estimator

    # Check if the final estimator is a CatBoostRegressor
    if not isinstance(final_estimator, CatBoostRegressor):
        print(f"Warning: Final estimator is not a CatBoostRegressor ({type(final_estimator).__name__}). Cannot use CatBoost's feature importance method.")
        # Fallback: Check for generic feature_importances_ attribute if it exists (less likely for non-tree models)
        if hasattr(final_estimator, 'feature_importances_'):
            print("  Using generic 'feature_importances_' attribute.")
            importances = final_estimator.feature_importances_
            # Need feature names corresponding to data *input* to this estimator step
            # If PCA is before, names are PC names. If no PCA, original names (or scaled original names).
            # This part needs careful handling based on pipeline structure.
            # For simplicity, if PCA is present, use PC names. If not, use provided feature_names.
            pca_step = next((step for name, step in transformer_steps if isinstance(step, PCA)), None)
            if pca_step:
                 n_components = len(importances)
                 current_feature_names = [f'PC_{i+1}' for i in range(n_components)]
            else:
                 current_feature_names = feature_names # Use original names if no PCA

            if current_feature_names is None or len(importances) != len(current_feature_names):
                 print(f"Warning: Mismatch between generic importances ({len(importances)}) and available feature names ({len(current_feature_names) if current_feature_names else 0}). Using generic names.")
                 feature_col = [f'feature_{i}' for i in range(len(importances))]
            else:
                 feature_col = current_feature_names

            importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            return importance_df.head(top_n) if top_n else importance_df
        else:
            print("  Final estimator lacks feature importance attribute/method.")
            return None

    # --- CatBoost Specific Importance Extraction ---
    try:

        importances = final_estimator.get_feature_importance()

        pca_step = next((step for name, step in transformer_steps if isinstance(step, PCA)), None)

        if pca_step:
            print("Note: PCA applied before CatBoost. Importances relate to Principal Components.")
            n_components = len(importances)
            component_names = [f'PC_{i+1}' for i in range(n_components)]
            importance_df = pd.DataFrame({'Feature/Component': component_names, 'Importance': importances})
        elif feature_names:

             if len(importances) != len(feature_names):

                 print(f"Warning: Mismatch between CatBoost importances ({len(importances)}) and provided original feature names ({len(feature_names)}). Using generic names.")
                 feature_col = [f'feature_{i}' for i in range(len(importances))]
             else:
                 feature_col = feature_names # Use provided original names

             importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})
        else:
             print("Warning: Feature names not provided and no PCA. Cannot map importances to original names. Using generic names.")
             feature_col = [f'feature_{i}' for i in range(len(importances))]
             importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})

        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importance_df.head(top_n) if top_n else importance_df

    except Exception as e:
        print(f"Error extracting CatBoost feature importance: {e}")
        traceback.print_exc()
        return None


# --- Orchestrator Function (Unchanged from previous version) ---
def orchestrate_catboost_tuning(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    param_distributions_config: dict,
    n_iter: int,
    cv_folds: int,
    scoring_metric: str,
    random_state: int,
    results_dir: str,
    best_model_filename: str = 'best_catboost_pipeline.joblib',
    apply_scaling: bool = False,
    apply_pca: bool = False,
    pca_n_components = None,
    catboost_base_params: dict = None,
    save_importance: bool = True,
    top_n_features: int = None,
):
    """
    Runs the entire CatBoost tuning pipeline.
    ... (docstring truncated) ...
    """
    print("===== Starting CatBoost Tuning Pipeline =====")
    print(f"Input data shape: {df.shape}")
    pipeline_start_time = time.time()

    if CatBoostRegressor is None:
        print("CatBoostRegressor not available. Aborting pipeline.")
        return None, None

    os.makedirs(results_dir, exist_ok=True)

    # 1. Split Data
    try:
        X_train, X_test, y_train, y_test, original_feature_names = split_data(
            df, target_column, test_size=test_size, random_state=random_state
        )
        print("Step 1: Data splitting complete.")

        if X_train.shape[1] == 0:
             print("\nCRITICAL ERROR: 0 features available after splitting/imputation. Cannot proceed.")
             return None, None

        if apply_pca and pca_n_components is not None:
             max_features_available = X_train.shape[1]
             if isinstance(pca_n_components, int):
                  if pca_n_components <= 0:
                       raise ValueError(f"PCA n_components must be > 0 for integer input, got {pca_n_components}.")
                  if pca_n_components > max_features_available:
                       warnings.warn(f"PCA n_components ({pca_n_components}) > number of available features ({max_features_available}). PCA will automatically adjust to {max_features_available}.", UserWarning)
             elif isinstance(pca_n_components, float) and not (0 < pca_n_components <= 1):
                  raise ValueError(f"PCA n_components must be between 0 and 1 for float input, got {pca_n_components}.")
             elif pca_n_components is None:
                 warnings.warn("apply_pca is True, but pca_n_components is None. PCA will use default n_components, which might be high.", UserWarning)


    except Exception as e:
        print(f"\nCRITICAL ERROR during data splitting: {e}")
        traceback.print_exc()
        return None, None

    # 2. Prepare Parameter Distributions for Pipeline
    pipeline_param_distributions = {}
    for key, value in param_distributions_config.items():
        if '__' not in key:
            pipeline_param_distributions[f'catboost__{key}'] = value
        else:
            pipeline_param_distributions[key] = value

    # 3. Tune Hyperparameters
    best_pipeline = None # Initialize in case tuning fails completely
    tuning_results_df = pd.DataFrame() # Initialize empty dataframe
    search_time = 0 # Initialize search time

    try:
        best_pipeline, tuning_results_df, search_time = tune_catboost(
            X_train=X_train,
            y_train=y_train,
            pipeline_param_distributions=pipeline_param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring=scoring_metric,
            random_state=random_state,
            apply_scaling=apply_scaling,
            apply_pca=apply_pca,
            pca_n_components=pca_n_components,
            catboost_base_params=catboost_base_params
        )
        print("Step 3: Hyperparameter tuning complete.")

        # Report number of failed fits
        n_failed_fits = tuning_results_df['mean_test_score'].isnull().sum()
        if n_failed_fits > 0:
            print(f"\nNote: {n_failed_fits} out of {len(tuning_results_df)} parameter combinations resulted in failed CV fits.")


    except Exception as e:
        print(f"\nCRITICAL ERROR during hyperparameter tuning: {e}")
        traceback.print_exc()
        # Save partial results if tuning failed after starting
        if not tuning_results_df.empty:
             # Need best_params and best_score for save_tuning_results, which might not exist if fit failed
             # Let's skip saving best_params and set best_score to nan in case of critical failure here
             print("Attempting to save partial tuning results...")
             try:
                 save_tuning_results(
                     results_df=tuning_results_df, # Save what was collected
                     best_params={}, # Indicate no best params found
                     best_score=np.nan, # Indicate failure
                     scoring=scoring_metric,
                     cv=cv_folds,
                     n_iter=n_iter,
                     search_time=search_time,
                     random_state=random_state,
                     results_dir=results_dir
                 )
                 print("Partial tuning results saved.")
             except Exception as save_e:
                 print(f"Error saving partial tuning results: {save_e}")
        return None, None # Critical failure, stop pipeline


    # 4. Save Tuning Results
    # Check if tuning_results_df is not empty and has valid scores
    if not tuning_results_df.empty and 'rank_test_score' in tuning_results_df.columns and tuning_results_df['rank_test_score'].min() == 1:
        best_cv_score = tuning_results_df['mean_test_score'].iloc[tuning_results_df['rank_test_score'].idxmin()]
        best_params_found = best_pipeline.get_params() if best_pipeline else {} # Use best_pipeline if available
    else:
        print("Warning: No successful fits found during tuning. Cannot determine best score or parameters.")
        best_cv_score = np.nan
        best_params_found = {}

    save_tuning_results(
        results_df=tuning_results_df,
        best_params=best_params_found,
        best_score=best_cv_score,
        scoring=scoring_metric,
        cv=cv_folds,
        n_iter=n_iter,
        search_time=search_time,
        random_state=random_state,
        results_dir=results_dir
    )
    print("Step 4: Tuning results saved.")


    # 5. Plot Tuning Results
    try:
        plot_tuning_results_random_cat(
            results_df=tuning_results_df,
            pipeline_param_distributions=pipeline_param_distributions,
            scoring=scoring_metric,
            results_dir=results_dir,
            max_unique_for_boxplot=15 # Use a slightly higher threshold, adjust as needed
        )
        print("Step 5: Tuning plots generated.")
    except Exception as e:
         print(f"\nError during plotting results: {e}")
         traceback.print_exc()


    # 6. Save the Best Pipeline
    if best_pipeline: # Only save if a best pipeline was found
        model_filepath = os.path.join(results_dir, best_model_filename)
        save_best_model(best_pipeline, model_filepath)
        print("Step 6: Best pipeline saved.")
    else:
        print("Step 6: No best pipeline found (tuning failed). Skipping model saving.")


    # 7. Evaluate the Final Pipeline on the Test Set
    test_scores = {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan} # Initialize scores
    if best_pipeline and not X_test.empty: # Only evaluate if best pipeline exists and test set is not empty
        test_scores = evaluate_model(best_pipeline, X_test, y_test)
        print("Step 7: Test set evaluation complete.")
    else:
        print("Step 7: Skipping test set evaluation (No best pipeline or empty test set).")


    # 8. Get and Save Feature/Component Importances (Optional)
    if save_importance and best_pipeline: # Only extract if saving importance and best pipeline exists
        try:
            print("\nStep 8: Extracting Feature/Component Importances...")
            importances_df = get_feature_importances(
                best_pipeline,
                feature_names=original_feature_names,
                top_n=top_n_features
            )
            if importances_df is not None and not importances_df.empty:
                fi_filename = os.path.join(results_dir, 'catboost_feature_component_importances.csv')
                try:
                    full_importances_df = get_feature_importances(best_pipeline, original_feature_names)
                    if full_importances_df is not None:
                         full_importances_df.to_csv(fi_filename, index=False)
                         print(f"Full importances saved to: {fi_filename}")

                    print(f"Top {top_n_features if top_n_features else 'All'} Features/Components:")
                    print(importances_df.to_string())

                except Exception as e:
                    print(f"Error saving/printing importances: {e}")
                    traceback.print_exc()
            else:
                 print("Could not extract or found no valid feature importances.")

        except Exception as e:
             print(f"\nError during feature importance step: {e}")
             traceback.print_exc()
    elif save_importance: # If save_importance is True but best_pipeline is None
        print("Step 8: Skipping feature importance extraction (No best pipeline found).")


    pipeline_end_time = time.time()
    print("\n===== CatBoost Tuning Pipeline Finished =====")
    print(f"Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds")

    return best_pipeline, test_scores



# --- Plotting Results (Adapted for scatterplot option) ---
def plot_tuning_results_random_cat(results_df: pd.DataFrame,
                        pipeline_param_distributions: dict,
                        scoring: str,
                        results_dir: str,
                        max_unique_for_boxplot: int = 6): # Threshold for switching to scatter
    """
    Generates and saves plots (boxplots or scatterplots) for tuning results.
    Chooses plot type based on the number of unique parameter values.
    """
    print("\n--- Generating Plots ---")
    try:
        varied_params_keys = [k for k, v in pipeline_param_distributions.items()
                              if isinstance(v, (list, np.ndarray)) or hasattr(v, 'rvs')]

        plotted_params = []
        for prefixed_key in varied_params_keys:
            param_col_name = f'param_{prefixed_key}'
            if param_col_name in results_df.columns and results_df[param_col_name].nunique() > 1:
                plotted_params.append(prefixed_key)

        if not plotted_params:
            print("No parameters varied significantly in the search results. Skipping plots.")
            return

        n_params = len(plotted_params)
        n_cols = 3; n_rows = (n_params + n_cols - 1) // n_cols
        score_col = 'mean_test_score'
        plt.figure(figsize=(n_cols * 6, n_rows * 5))

        for i, param_key in enumerate(plotted_params):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            param_col = f'param_{param_key}'
            plot_data = results_df[[param_col, score_col]].copy()

            # Convert parameter values to appropriate type for plotting
            # Handle potential NaNs by dropping rows for plotting, or imputing 'None' for boxplots
            # For scatter, dropping NaNs is usually fine.
            plot_data = plot_data.dropna(subset=[param_col, score_col])


            unique_vals = plot_data[param_col].nunique()
            param_values = plot_data[param_col].values

            # Determine plot type
            use_boxplot = (unique_vals <= max_unique_for_boxplot) and (isinstance(param_values[0], (int, float, str, bool)) or (param_values.dtype.kind in 'biufc'))
            # Check if data type is suitable for scatter plot (numeric)
            is_numeric_param = pd.api.types.is_numeric_dtype(param_values)


            if use_boxplot:
                # Convert to string for boxplot if not numeric, to handle mixed types or objects
                plot_data[param_col] = plot_data[param_col].astype(str)
                unique_str_vals = plot_data[param_col].unique()
                plot_order = sorted(unique_str_vals) if all(isinstance(v, str) for v in unique_str_vals) else np.unique(plot_data[param_col]).tolist()

                sns.boxplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                            palette="viridis", order=plot_order)
                ax.set_title(f"CV Score vs {param_key} (Boxplot)")
                ax.set_xlabel(param_key)
                ax.set_ylabel(f'Mean CV Score ({scoring})')
                if len(unique_str_vals) > 7 or any(len(str(v)) > 12 for v in unique_str_vals):
                    ax.tick_params(axis='x', rotation=60)
                else: ax.tick_params(axis='x', rotation=0)

            elif is_numeric_param:
                # Use scatter plot for numeric parameters with many unique values
                ax.scatter(plot_data[param_col], plot_data[score_col], alpha=0.6)
                ax.set_title(f"CV Score vs {param_key} (Scatter)")
                ax.set_xlabel(param_key)
                ax.set_ylabel(f'Mean CV Score ({scoring})')
                # Optionally add a smoothed line
                sns.regplot(x=param_col, y=score_col, data=plot_data, scatter=False, color='red', ax=ax, ci=None)
                ax.grid(True) # Add grid for scatter plots

            else:
                # Fallback if not suitable for boxplot or scatter (e.g., complex objects, strings with many unique values)
                print(f"Warning: Parameter '{param_key}' has {unique_vals} unique non-numeric values. Skipping plot.")
                ax.text(0.5, 0.5, 'Plot Skipped\n(Too many unique non-numeric values)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"CV Score vs {param_key}")
                ax.set_xticks([])
                ax.set_yticks([])


        plt.suptitle(f'Hyperparameter Tuning Results (Score: {scoring})', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_filename = os.path.join(results_dir, 'catboost_tuning_plots.png') # Generic name now
        plt.savefig(plot_filename, bbox_inches='tight'); print(f"Plots saved to: {plot_filename}")
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules: plt.show()
        else: plt.close()
    except Exception as e:
        print(f"\nError generating plots: {e}")
        traceback.print_exc()



# --- Standard Imports (ensure all are present) ---
import pandas as pd
import numpy as np
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import warnings
import joblib
from scipy.stats import uniform, randint # Example distributions (used for Randomized Search, but kept for completeness)

# --- Scikit-learn Imports (ensure all are present) ---
from sklearn.model_selection import train_test_split, GridSearchCV, KFold # Use GridSearchCV now
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, make_scorer
from sklearn.exceptions import FitFailedWarning

# --- CatBoost Import (ensure it's present) ---
try:
    import catboost
    from catboost import CatBoostRegressor
    from catboost import Pool
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! CatBoost is not installed. Please install it to run this code. !!!")
    print("!!! >>> pip install catboost                                       !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    CatBoostRegressor = None
    Pool = None

# --- Settings ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

# --- Data Splitting (Unchanged) ---
def split_data(df: pd.DataFrame,
               target_name: str,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Splits the dataframe into training and testing sets.
    Handles missing values in target by dropping rows.
    Handles missing values in numeric features by median imputation (simplified).
    Returns X as DataFrame to preserve column names/types before potential pipeline steps.
    """
    print(f"\n--- Splitting Data (Test Size: {test_size}, Random State: {random_state}) ---")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target_name not in df.columns:
        raise KeyError(f"Target column '{target_name}' not found in DataFrame columns: {df.columns.tolist()}")
    if not pd.api.types.is_numeric_dtype(df[target_name]):
        raise ValueError(f"Target column '{target_name}' must be numeric for regression.")
    if df[target_name].isnull().any():
         print(f"Warning: Target column '{target_name}' contains missing values. Dropping rows with NaN target.")
         df_cleaned = df.dropna(subset=[target_name]).copy()
    else:
         df_cleaned = df.copy()

    if df_cleaned.empty:
         raise ValueError("No data left after dropping rows with missing target.")
    if len(df_cleaned) < 2:
         raise ValueError("Not enough samples left after handling missing target values to perform a split.")


    X = df_cleaned.drop(columns=[target_name])
    y = df_cleaned[target_name]

    # Impute NaNs in numeric features using median on the *full* data before split.
    numeric_cols_with_nan = X.select_dtypes(include=np.number).columns[X.select_dtypes(include=np.number).isnull().any()]
    if not numeric_cols_with_nan.empty:
         print(f"  Imputing {len(numeric_cols_with_nan)} numeric columns with median (simplified approach for tuning).")
         imputation_values = X[numeric_cols_with_nan].median()
         X[numeric_cols_with_nan] = X[numeric_cols_with_nan].fillna(imputation_values)
         if X.isnull().sum().sum() > 0:
              print("Warning: NaNs still exist after numeric imputation. Check non-numeric features if using them.")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    original_feature_names = list(X_train.columns)

    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test, original_feature_names


# --- Tuning Function (Adapted for Grid Search) ---
def tune_catboost_grid(X_train: pd.DataFrame, # Keep as DataFrame for potential cat feature handling later
                       y_train: pd.Series,
                       pipeline_param_grid: dict, # Expects prefixed keys, list/array values
                       cv: int,
                       scoring: str,
                       random_state: int,
                       apply_scaling: bool = False,
                       apply_pca: bool = False,
                       pca_n_components=None,
                       catboost_base_params: dict = None
                       ):
    """
    Performs hyperparameter tuning for CatBoostRegressor using GridSearchCV within a Pipeline.
    Expects parameter values in pipeline_param_grid to be lists or arrays.
    """
    print(f"\n--- Starting Grid Search (Pipeline: Scale={apply_scaling}, PCA={apply_pca}) ---")
    print(f"   (cv={cv}, scoring='{scoring}')")
    search_start_time = time.time()

    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost library is not installed. Cannot perform tuning.")

    # 1. Define Pipeline Steps
    steps = []
    if apply_scaling:
        steps.append(('scaler', StandardScaler()))
        print("   Adding StandardScaler step.")

    if apply_pca:
        if isinstance(pca_n_components, int):
             if pca_n_components <= 0:
                 raise ValueError(f"pca_n_components must be > 0 for int input, got {pca_n_components}.")
             # PCA handles n_components > n_features gracefully
        elif isinstance(pca_n_components, float) and not (0 < pca_n_components <= 1):
             raise ValueError(f"pca_n_components must be between 0 and 1 for float input, got {pca_n_components}.")
        elif pca_n_components is None:
             warnings.warn("apply_pca is True, but pca_n_components is None. PCA will use default n_components, which might be high.", UserWarning)

        steps.append(('pca', PCA(n_components=pca_n_components, random_state=random_state)))
        print(f"   Adding PCA step (n_components={pca_n_components}).")

    catboost_base_params_final = catboost_base_params.copy() if catboost_base_params is not None else {}
    catboost_base_params_final.setdefault('random_state', random_state)
    catboost_base_params_final['verbose'] = 0 # Keep verbosity low during CV fitting
    catboost_base_params_final.setdefault('loss_function', 'RMSE')
    catboost_base_params_final.setdefault('eval_metric', 'RMSE')

    steps.append(('catboost', CatBoostRegressor(**catboost_base_params_final)))
    pipeline = Pipeline(steps)
    print(f"   Pipeline steps: {[s[0] for s in pipeline.steps]}")

    # 2. Verify Parameter Grid Keys and Values
    print("   Hyperparameter grid:")
    valid_step_names = [s[0] for s in pipeline.steps]
    prefixed_param_grid = {}
    n_combinations = 1
    for param_key, values in pipeline_param_grid.items():
         if '__' not in param_key:
              # Assuming it's for the final estimator (catboost)
              prefixed_key = f'catboost__{param_key}'
              print(f"Warning: Parameter key '{param_key}' lacks a pipeline prefix. Assuming it's for the 'catboost' step.")
         else:
              step_name = param_key.split('__')[0]
              if step_name not in valid_step_names:
                  warnings.warn(f"Parameter '{param_key}' targets step '{step_name}', which is NOT in the current pipeline configuration: {valid_step_names}. This parameter will be ignored by GridSearchCV.", UserWarning)
                  continue
              prefixed_key = param_key

         if not isinstance(values, (list, np.ndarray)):
             raise ValueError(f"Parameter grid value for '{param_key}' must be a list or numpy array, got {type(values)}.")

         prefixed_param_grid[prefixed_key] = values
         n_combinations *= len(values)
         print(f"     {prefixed_key}: {values}")

    if not prefixed_param_grid:
         raise ValueError("Hyperparameter grid is empty after processing prefixes and pipeline steps. Cannot perform tuning.")

    print(f"   Total parameter combinations: {n_combinations}")


    # 3. Setup and Run Grid Search
    cv_strategy = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=prefixed_param_grid, # Use the validated/prefixed dict
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1, # Use all available cores for the search iterations
        return_train_score=True,
        verbose=1 # Set verbosity level for GridSearchCV itself
    )

    print(f"   Fitting GridSearchCV...")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", FitFailedWarning)

            grid_search.fit(X_train, y_train)

            if w:
                print(f"   Captured {len(w)} FitFailedWarning(s) during GridSearchCV fit.")

        print("   GridSearchCV fitting complete.")
    except Exception as e:
        print(f"\nError during GridSearchCV fit: {e}")
        traceback.print_exc()
        raise RuntimeError(f"GridSearchCV fit failed: {e}")

    search_end_time = time.time()
    search_time = search_end_time - search_start_time
    print(f"Search finished in {search_time:.2f} seconds.")

    best_estimator = grid_search.best_estimator_
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.columns = results_df.columns.str.replace(":", "_", regex=False)
    results_df = results_df.sort_values(by=f'rank_test_score', ascending=True)

    print("\n--- Best Pipeline Results ---")
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print(f"Best CV Score ({scoring}): {best_score:.4f}")
    print("Best Parameters Found:")
    valid_step_names_in_best_pipe = [s[0] for s in best_estimator.steps]
    params_to_print = {k: v for k, v in best_params.items()
                       if k.split('__')[0] in valid_step_names_in_best_pipe}

    for param, value in params_to_print.items():
        print(f"  {param}: {value}")

    try:
        best_result_row = results_df[results_df['rank_test_score'] == 1].iloc[0]
        mean_train_score = best_result_row.get('mean_train_score')
        std_test_score = best_result_row.get('std_test_score')
        std_train_score = best_result_row.get('std_train_score')

        print(f"\nMean Train Score ({scoring}): {mean_train_score:.4f} (+/- {std_train_score:.4f})")
        print(f"Mean Test Score ({scoring}): {best_score:.4f} (+/- {std_test_score:.4f})")

    except Exception as e:
         print(f"Warning: Could not extract detailed scores for best model from cv_results_: {e}")


    return best_estimator, results_df, search_time

# --- Saving Results (Adapted filename) ---
def save_tuning_results(results_df: pd.DataFrame, best_params: dict, best_score: float,
                        scoring: str, cv: int, n_iter: int, search_time: float,
                        random_state: int, results_dir: str, search_type: str = "Grid"): # Added search_type
    """Saves the detailed tuning results (CSV) and best parameters (TXT)."""
    print("\n--- Saving Tuning Results ---")
    os.makedirs(results_dir, exist_ok=True)
    # Use search_type in filename
    results_filename = os.path.join(results_dir, f'catboost_{search_type.lower()}_tuning_results.csv')
    try:
        results_df.to_csv(results_filename, index=False)
        print(f"Detailed tuning results saved to: {results_filename}")
    except Exception as e: print(f"Error saving results CSV: {e}")

    best_params_filename = os.path.join(results_dir, f'catboost_{search_type.lower()}_best_params.txt') # Use search_type
    try:
        with open(best_params_filename, 'w') as f:
            f.write(f"--- CatBoost Pipeline {search_type} Tuning Summary ---\n") # Use search_type
            f.write(f"Scoring Metric: {scoring}\n")
            f.write(f"Best CV Score ({scoring}): {best_score:.4f}\n")
            f.write("\nBest Pipeline Parameters:\n")
            for param, value in best_params.items():
                 f.write(f"  {param}: {value}\n")

            f.write("\n--- Search Configuration ---\n")
            f.write(f"CV Folds: {cv}\n")
            # Include n_iter for RandomizedSearch, not directly applicable/useful for GridSearch
            if search_type.lower() == 'randomized':
                 f.write(f"Search Iterations (n_iter): {n_iter}\n")
            f.write(f"Random State Seed: {random_state}\n")
            f.write(f"Tuning Time: {search_time:.2f} seconds\n")
        print(f"Best parameters summary saved to: {best_params_filename}")
    except Exception as e: print(f"Error saving best parameters text file: {e}")


# --- Plotting Results (Unchanged from previous version) ---
def plot_tuning_results(results_df: pd.DataFrame,
                        pipeline_param_distributions: dict, # This dict structure is from RandomizedSearch. For GridSearch, it's a grid.
                        scoring: str,
                        results_dir: str,
                        max_unique_for_boxplot: int = 10,
                        search_type: str = "Grid"): # Added search_type
    """
    Generates and saves plots (boxplots or scatterplots) for tuning results.
    Chooses plot type based on the number of unique parameter values.
    Handles data from both RandomizedSearchCV and GridSearchCV results_df.
    """
    print("\n--- Generating Plots ---")
    try:
        # For GridSearchCV, the 'param_*' columns directly represent the grid points.
        # We just need to identify which ones were varied.
        # For RandomizedSearchCV, we used pipeline_param_distributions to guide plotting.
        # Let's iterate through the 'param_*' columns in the results_df and plot any that varied.

        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        plotted_params = [] # Store the original prefixed keys ('param_param_key' -> 'param_key')
        for param_col in param_cols:
             # Remove the 'param_' prefix to get the original prefixed key
             original_prefixed_key = param_col[len('param_'):]
             if results_df[param_col].nunique() > 1:
                  plotted_params.append(original_prefixed_key)

        if not plotted_params:
            print("No parameters varied significantly in the search results. Skipping plots.")
            return

        n_params = len(plotted_params)
        n_cols = 3; n_rows = (n_params + n_cols - 1) // n_cols
        score_col = 'mean_test_score'
        plt.figure(figsize=(n_cols * 6, n_rows * 5))

        for i, param_key in enumerate(plotted_params): # Use the original prefixed keys
            ax = plt.subplot(n_rows, n_cols, i + 1)
            param_col = f'param_{param_key}' # Reconstruct the results_df column name
            plot_data = results_df[[param_col, score_col]].copy()

            plot_data = plot_data.dropna(subset=[param_col, score_col])

            unique_vals = plot_data[param_col].nunique()
            param_values = plot_data[param_col].values

            # Determine plot type
            # For GridSearch, unique_vals will match the grid size. Boxplot is usually better.
            # For RandomizedSearch, unique_vals can be large. Scatter is better for numeric.
            # Let's use the threshold logic, but boxplot is preferred for GridSearch if possible.
            use_boxplot = (unique_vals <= max_unique_for_boxplot) or (search_type.lower() == 'grid')
            # Still need to check if the data type is plottable for boxplot
            if use_boxplot:
                 # Check data type suitability for boxplot
                 try:
                      # Attempt to sort to check for type compatibility
                      sorted_values = sorted(plot_data[param_col].unique())
                      is_suitable_for_boxplot_sort = True
                 except TypeError:
                      is_suitable_for_boxplot_sort = False

                 if not is_suitable_for_boxplot_sort and unique_vals > max_unique_for_boxplot:
                      # If not sortable and too many unique values, force scatter if numeric
                      is_numeric_param = pd.api.types.is_numeric_dtype(param_values)
                      if is_numeric_param:
                           use_boxplot = False
                           print(f"Warning: Parameter '{param_key}' not sortable but numeric with many unique values ({unique_vals}). Using scatterplot.")
                      else:
                           # Still not suitable for scatter either
                           print(f"Warning: Parameter '{param_key}' has {unique_vals} unique non-numeric values. Skipping plot.")
                           use_boxplot = False # Force skip plot

            # --- Perform Plotting ---
            if use_boxplot:
                plot_data[param_col] = plot_data[param_col].astype(str)
                unique_str_vals = plot_data[param_col].unique()
                plot_order = sorted(unique_str_vals) if all(isinstance(v, str) for v in unique_str_vals) else np.unique(plot_data[param_col]).tolist()

                sns.boxplot(x=param_col, y=score_col, data=plot_data, ax=ax,
                            palette="viridis", order=plot_order)
                ax.set_title(f"CV Score vs {param_key}") # Title without plot type for consistency
                ax.set_xlabel(param_key)
                ax.set_ylabel(f'Mean CV Score ({scoring})')
                if len(unique_str_vals) > 7 or any(len(str(v)) > 12 for v in unique_str_vals):
                    ax.tick_params(axis='x', rotation=60)
                else: ax.tick_params(axis='x', rotation=0)

            elif pd.api.types.is_numeric_dtype(param_values): # Scatter plot for numeric with many unique values or forced
                ax.scatter(plot_data[param_col], plot_data[score_col], alpha=0.6)
                ax.set_title(f"CV Score vs {param_key}")
                ax.set_xlabel(param_key)
                ax.set_ylabel(f'Mean CV Score ({scoring})')
                sns.regplot(x=param_col, y=score_col, data=plot_data, scatter=False, color='red', ax=ax, ci=None)
                ax.grid(True)

            else:
                # Fallback if plot skipped
                ax.text(0.5, 0.5, 'Plot Skipped\n(Too many unique non-numeric values\nor unsortable)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title(f"CV Score vs {param_key}")
                ax.set_xticks([])
                ax.set_yticks([])


        plt.suptitle(f'CatBoost {search_type} Tuning Results (Score: {scoring})', fontsize=16, y=1.02) # Title with search type
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plot_filename = os.path.join(results_dir, f'catboost_{search_type.lower()}_tuning_plots.png') # Filename with search type
        plt.savefig(plot_filename, bbox_inches='tight'); print(f"Plots saved to: {plot_filename}")
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules: plt.show()
        else: plt.close()
    except Exception as e:
        print(f"\nError generating plots: {e}")
        traceback.print_exc()

# --- Saving Model (Unchanged) ---
def save_best_model(estimator, filepath: str):
    """Saves the trained model/pipeline to a file using joblib."""
    print(f"\n--- Saving Best Model/Pipeline ---")
    try:
        joblib.dump(estimator, filepath)
        print(f"Best model/pipeline saved to: {filepath}")
    except Exception as e: print(f"Error saving best model/pipeline to {filepath}: {e}")

# --- Evaluating Model (Unchanged) ---
def evaluate_model(estimator, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluates the final model/pipeline on the hold-out test set."""
    print("\n--- Evaluating Best Model/Pipeline on Test Set ---")
    try:
        y_pred = estimator.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Test Set R2 Score: {r2:.4f}")
        print(f"Test Set RMSE: {rmse:.4f}")
        print(f"Test Set MAE: {mae:.4f}")
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    except Exception as e:
        print(f"Error during test set evaluation: {e}")
        traceback.print_exc()
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}

# --- Feature Importance (Unchanged) ---
def get_feature_importances(estimator, feature_names: list = None, top_n: int = None):
    """Extracts feature importances from the final estimator. Adapts for pipelines."""
    print("\n--- Extracting Feature Importances ---")
    final_estimator = None; transformer_steps = []
    if isinstance(estimator, Pipeline):
        final_estimator = estimator.steps[-1][1]
        transformer_steps = estimator.steps[:-1]
    else: final_estimator = estimator

    if not isinstance(final_estimator, CatBoostRegressor):
        print(f"Warning: Final estimator is not a CatBoostRegressor ({type(final_estimator).__name__}). Cannot use CatBoost's feature importance method.")
        if hasattr(final_estimator, 'feature_importances_'):
            print("  Using generic 'feature_importances_' attribute.")
            importances = final_estimator.feature_importances_
            pca_step = next((step for name, step in transformer_steps if isinstance(step, PCA)), None)
            if pca_step:
                 n_components = len(importances)
                 current_feature_names = [f'PC_{i+1}' for i in range(n_components)]
            else:
                 current_feature_names = feature_names

            if current_feature_names is None or len(importances) != len(current_feature_names):
                 print(f"Warning: Mismatch between generic importances ({len(importances)}) and available feature names ({len(current_feature_names) if current_feature_names else 0}). Using generic names.")
                 feature_col = [f'feature_{i}' for i in range(len(importances))]
            else:
                 feature_col = current_feature_names

            importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            return importance_df.head(top_n) if top_n else importance_df
        else:
            print("  Final estimator lacks feature importance attribute/method.")
            return None

    # --- CatBoost Specific Importance Extraction ---
    try:
        importances = final_estimator.get_feature_importance()

        pca_step = next((step for name, step in transformer_steps if isinstance(step, PCA)), None)

        if pca_step:
            print("Note: PCA applied before CatBoost. Importances relate to Principal Components.")
            n_components = len(importances)
            component_names = [f'PC_{i+1}' for i in range(n_components)]
            importance_df = pd.DataFrame({'Feature/Component': component_names, 'Importance': importances})
        elif feature_names:
             if len(importances) != len(feature_names):
                 print(f"Warning: Mismatch between CatBoost importances ({len(importances)}) and provided original feature names ({len(feature_names)}). Using generic names.")
                 feature_col = [f'feature_{i}' for i in range(len(importances))]
             else:
                 feature_col = feature_names

             importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})
        else:
             print("Warning: Feature names not provided and no PCA. Cannot map importances to original names. Using generic names.")
             feature_col = [f'feature_{i}' for i in range(len(importances))]
             importance_df = pd.DataFrame({'Feature/Component': feature_col, 'Importance': importances})

        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        return importance_df.head(top_n) if top_n else importance_df

    except Exception as e:
        print(f"Error extracting CatBoost feature importance: {e}")
        traceback.print_exc()
        return None


# --- Orchestrator Function for Grid Search ---
def orchestrate_catboost_grid_tuning(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    param_grid_config: dict, # Hyperparameter grid WITHOUT pipeline prefixes
    cv_folds: int,
    scoring_metric: str,
    random_state: int,
    results_dir: str,
    best_model_filename: str = 'best_catboost_grid_pipeline.joblib', # Distinct filename
    apply_scaling: bool = False,
    apply_pca: bool = False,
    pca_n_components = None,
    catboost_base_params: dict = None,
    save_importance: bool = True,
    top_n_features: int = None,
):
    """
    Runs the entire CatBoost Grid Search tuning pipeline.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str): Name of the target variable.
        test_size (float): Proportion for the test set split.
        param_grid_config (dict): Hyperparameter grid WITHOUT pipeline prefixes.
                                  Values must be lists or arrays.
        cv_folds (int): Number of cross-validation folds.
        scoring_metric (str): Metric for tuning evaluation.
        random_state (int): Seed for reproducibility.
        results_dir (str): Directory to save outputs.
        best_model_filename (str): Filename for the saved best model/pipeline.
        apply_scaling (bool): Whether to apply a scaler step (StandardScaler).
        apply_pca (bool): Whether to apply PCA step.
        pca_n_components: n_components for PCA (if apply_pca=True).
                           Can be int (>0) or float (0,1].
        catboost_base_params (dict, optional): Base parameters for the CatBoostRegressor
                                             estimator (e.g., loss_function, eval_metric).
        save_importance (bool): Whether to calculate and save feature importances.
        top_n_features (int, optional): Number of top features to save/print importance for.

    Returns:
        tuple: (best_pipeline, test_scores)
               best_pipeline: The best fitted pipeline object.
               test_scores: Dictionary of scores from evaluating on the test set.
               Returns (None, None) if CatBoost is not available or a critical error occurs.
    """
    print("===== Starting CatBoost Grid Search Tuning Pipeline =====")
    print(f"Input data shape: {df.shape}")
    pipeline_start_time = time.time()

    if CatBoostRegressor is None:
        print("CatBoostRegressor not available. Aborting pipeline.")
        return None, None

    os.makedirs(results_dir, exist_ok=True)

    # 1. Split Data
    try:
        X_train, X_test, y_train, y_test, original_feature_names = split_data(
            df, target_column, test_size=test_size, random_state=random_state
        )
        print("Step 1: Data splitting complete.")

        if X_train.shape[1] == 0:
             print("\nCRITICAL ERROR: 0 features available after splitting/imputation. Cannot proceed.")
             return None, None

        if apply_pca and pca_n_components is not None:
             max_features_available = X_train.shape[1]
             if isinstance(pca_n_components, int):
                  if pca_n_components <= 0:
                       raise ValueError(f"PCA n_components must be > 0 for integer input, got {pca_n_components}.")
                  if pca_n_components > max_features_available:
                       warnings.warn(f"PCA n_components ({pca_n_components}) > number of available features ({max_features_available}). PCA will automatically adjust to {max_features_available}.", UserWarning)
             elif isinstance(pca_n_components, float) and not (0 < pca_n_components <= 1):
                  raise ValueError(f"PCA n_components must be between 0 and 1 for float input, got {pca_n_components}.")
             elif pca_n_components is None:
                 warnings.warn("apply_pca is True, but pca_n_components is None. PCA will use default n_components, which might be high.", UserWarning)

    except Exception as e:
        print(f"\nCRITICAL ERROR during data splitting: {e}")
        traceback.print_exc()
        return None, None

    # 2. Prepare Parameter Grid for Pipeline
    pipeline_param_grid = {}
    # param_grid_config values must be lists or arrays
    for key, values in param_grid_config.items():
        if '__' not in key:
            pipeline_param_grid[f'catboost__{key}'] = values
        else:
            pipeline_param_grid[key] = values


    # 3. Tune Hyperparameters (Grid Search)
    best_pipeline = None
    tuning_results_df = pd.DataFrame()
    search_time = 0

    try:
        best_pipeline, tuning_results_df, search_time = tune_catboost_grid(
            X_train=X_train,
            y_train=y_train,
            pipeline_param_grid=pipeline_param_grid, # Pass the prefixed grid
            cv=cv_folds,
            scoring=scoring_metric,
            random_state=random_state,
            apply_scaling=apply_scaling,
            apply_pca=apply_pca,
            pca_n_components=pca_n_components,
            catboost_base_params=catboost_base_params
        )
        print("Step 3: Hyperparameter grid search complete.")

        n_failed_fits = tuning_results_df['mean_test_score'].isnull().sum()
        if n_failed_fits > 0:
            print(f"\nNote: {n_failed_fits} out of {len(tuning_results_df)} parameter combinations resulted in failed CV fits.")


    except Exception as e:
        print(f"\nCRITICAL ERROR during hyperparameter grid search: {e}")
        traceback.print_exc()
        if not tuning_results_df.empty:
             print("Attempting to save partial tuning results...")
             try:
                 save_tuning_results(
                     results_df=tuning_results_df,
                     best_params={},
                     best_score=np.nan,
                     scoring=scoring_metric,
                     cv=cv_folds,
                     n_iter=0, # n_iter is not applicable for GridSearch
                     search_time=search_time,
                     random_state=random_state,
                     results_dir=results_dir,
                     search_type="Grid" # Specify search type
                 )
                 print("Partial tuning results saved.")
             except Exception as save_e:
                 print(f"Error saving partial tuning results: {save_e}")
        return None, None

    # 4. Save Tuning Results
    if not tuning_results_df.empty and 'rank_test_score' in tuning_results_df.columns and tuning_results_df['rank_test_score'].min() == 1:
        best_cv_score = tuning_results_df['mean_test_score'].iloc[tuning_results_df['rank_test_score'].idxmin()]
        best_params_found = best_pipeline.get_params() if best_pipeline else {}
    else:
        print("Warning: No successful fits found during tuning. Cannot determine best score or parameters.")
        best_cv_score = np.nan
        best_params_found = {}

    save_tuning_results(
        results_df=tuning_results_df,
        best_params=best_params_found,
        best_score=best_cv_score,
        scoring=scoring_metric,
        cv=cv_folds,
        n_iter=0, # n_iter not applicable for GridSearch
        search_time=search_time,
        random_state=random_state,
        results_dir=results_dir,
        search_type="Grid" # Specify search type
    )
    print("Step 4: Tuning results saved.")

    # 5. Plot Tuning Results
    try:
        # Plotting for GridSearch results is usually boxplots for discrete values.
        # We can reuse the plot_tuning_results but pass search_type="Grid"
        # The plot function needs to handle the difference in results_df structure (no n_iter).
        # The plot function already checksnunique, so it should work.
        plot_tuning_results(
            results_df=tuning_results_df,
            pipeline_param_distributions=pipeline_param_grid, # Pass the grid, plotting uses it to find keys
            scoring=scoring_metric,
            results_dir=results_dir,
            max_unique_for_boxplot=1000, # Essentially force boxplot for GridSearch
            search_type="Grid" # Specify search type
        )
        print("Step 5: Tuning plots generated.")
    except Exception as e:
         print(f"\nError during plotting results: {e}")
         traceback.print_exc()

    # 6. Save the Best Pipeline
    if best_pipeline:
        model_filepath = os.path.join(results_dir, best_model_filename)
        save_best_model(best_pipeline, model_filepath)
        print("Step 6: Best pipeline saved.")
    else:
        print("Step 6: No best pipeline found (tuning failed). Skipping model saving.")

    # 7. Evaluate the Final Pipeline on the Test Set
    test_scores = {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}
    if best_pipeline and not X_test.empty:
        test_scores = evaluate_model(best_pipeline, X_test, y_test)
        print("Step 7: Test set evaluation complete.")
    else:
        print("Step 7: Skipping test set evaluation (No best pipeline or empty test set).")

    # 8. Get and Save Feature/Component Importances (Optional)
    if save_importance and best_pipeline:
        try:
            print("\nStep 8: Extracting Feature/Component Importances...")
            importances_df = get_feature_importances(
                best_pipeline,
                feature_names=original_feature_names,
                top_n=top_n_features
            )
            if importances_df is not None and not importances_df.empty:
                fi_filename = os.path.join(results_dir, 'catboost_grid_feature_component_importances.csv') # Distinct filename
                try:
                    full_importances_df = get_feature_importances(best_pipeline, original_feature_names)
                    if full_importances_df is not None:
                         full_importances_df.to_csv(fi_filename, index=False)
                         print(f"Full importances saved to: {fi_filename}")

                    print(f"Top {top_n_features if top_n_features else 'All'} Features/Components:")
                    print(importances_df.to_string())

                except Exception as e:
                    print(f"Error saving/printing importances: {e}")
                    traceback.print_exc()
            else:
                 print("Could not extract or found no valid feature importances.")

        except Exception as e:
             print(f"\nError during feature importance step: {e}")
             traceback.print_exc()
    elif save_importance:
        print("Step 8: Skipping feature importance extraction (No best pipeline found).")


    pipeline_end_time = time.time()
    print("\n===== CatBoost Grid Search Tuning Pipeline Finished =====")
    print(f"Total execution time: {pipeline_end_time - pipeline_start_time:.2f} seconds")

    return best_pipeline, test_scores





