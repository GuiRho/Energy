# TARGET_COLUMN = "SiteEUIWN(kBtu/sf)"


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


# Notebook 1 - Chapitre 1
"Caractéristiques"
"Datatype"

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


"Modalités"
def get_modalities(df):
    info_from_unique_column = []
    rank=[]

    for feat in df.columns:
        if df[feat].nunique()<2:
            info_from_unique_column.append((feat, df[feat].unique()))
        else:
            rank.append((feat, df[feat].nunique()))

    info_from_unique_column = tuple(info_from_unique_column)    
    rank = sorted(rank, key=lambda x:x[1])

    print(f'Il y a {len(rank)} colonnes avec au moins 2 valeurs, voici leurs nombres de modalités respectives')
    print(f'{rank}')
    print(f'Il y a {len(info_from_unique_column)} colonnes inutiles pour le dataframe, voici leurs contenus ')
    print(f'{info_from_unique_column}')
    return info_from_unique_column


def drop_low_modalities(df, low_info_col):
    print(f"Before drop : {df.shape}")
    non_usefull_feature_gen = (low_info_col[i][0] for i in range(len(low_info_col))) 

    for feat in list(non_usefull_feature_gen):
        df=df.drop(columns=feat, axis=1)

    print(f"After drop : {df.shape}")
    return df
    

"Chaines de caractères"

"""
- astype(str) : Converts the column to strings.
- x.split() : Splits each string value in the column into a list of substrings, using commas as delimiters.
- s.strip() : Removes leading and trailing whitespace from each substring.
- x.join() : Joins the cleaned substrings back into a single string, using commas as separators.
- .lower() Converts all characters to lowercase
"""

def clean_strings(df: pd.DataFrame, cat_col) -> pd.DataFrame:

    for col in df.columns :
        if col in df[cat_col]:  # Check if the col exists in the DataFrame
            df[col] = df[col].astype(str).apply(lambda x: ','.join([s.strip() for s in x.split(',')]).lower()) 
            df[col] = df[col].str.replace(r'\(.*?\)', '', regex=True)
            df[col] = df[col].str.strip()
    return df

def find_error_col(df, col, errors):

    df_col_error = df[df[col].isin(errors)]

    index_to_drop = df_col_error.index.tolist()

    print(f"Shape of df: {df.shape}")
    df_clean = df.drop(index=index_to_drop)
    print(f"Shape of df cleaned: {df_clean.shape}")
    return df_clean

def keep_value_col(df,col,values):
    df_clean = df[df[col].isin(values)].copy()
    print(f"Shape of df: {df.shape}")
    print(f"Shape of df cleaned: {df_clean.shape}")
    return df_clean

def keep_unique(df, pkey, keep='first'):
    num_duplicates = df.duplicated(subset=pkey, keep=keep).sum()
    print(f"Number of duplicate rows based on primary key {pkey}: {num_duplicates}")

    dup_mask = df.duplicated(subset=pkey, keep=keep)
    df_unique = df[~dup_mask].copy()
    print(f"Shape of df with unique primary key {pkey}: {df_unique.shape}")

    return df_unique

def get_duplicate(df, pkey, keep='first'):
    num_duplicates = df.duplicated(subset=pkey, keep=keep).sum()
    print(f"Number of duplicate rows based on primary key {pkey}: {num_duplicates}")
    return 



# 2 - Nettoyage

def check_na_row(df):
    na_per_row = df.isnull().sum(axis=1).sort_values(ascending=False)
    distrib_na_row = na_per_row.value_counts().sort_index()
    total_na = df.isnull().sum().sum()
    return distrib_na_row, total_na

def plot_na_row(distrib_na_row=None, totalna_row=None):
    if distrib_na_row is None:
        print("Error: distrib_na_row is not defined. Please run check_na_values_rows() first.")
        return

    if totalna_row is None:
        print("Error: totalna_row is not defined. Please run check_na_values_rows() first.")
        return

    plt.figure(figsize=(14, 6)) 
    bars = plt.bar(distrib_na_row.index, distrib_na_row.values, color='blue')  
    plt.xlabel("Number of NaN Values per Row")
    plt.ylabel("Number of Rows")
    plt.title("Distribution of NA Values per Row")
    plt.xticks(distrib_na_row.index)  # Set x-axis ticks to the NA counts
    plt.ylim(0, distrib_na_row.values.max() * 1.1)  # Set the y-axis range
    # Add labels to the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom') #Place the number above the bar

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()

def fill_na_values(df, NA_filling_rules):
    for feature, rule in NA_filling_rules.items():
        if feature in df.columns:
            df[feature].fillna(rule(df), inplace=True)
    return df

def conditional_fill_na(df, rules):

    df = df.copy()  # Avoid modifying the original DataFrame
    for col, rule_list in rules.items():
        for condition, fill_value in rule_list:
            na_mask = df[col].isna()
            cond_mask = condition(df)  # Evaluate the condition (should return a boolean Series)
            mask = na_mask & cond_mask
            df[col] = np.where(mask, fill_value, df[col]) # apply each condition one after the other
    return df


# 3 - Outlier

def outlier_stat(df_copy, num_col, cat_col):
    
    # Prepare the output DataFrame
    output_df = pd.DataFrame(index=df_copy.columns)
    output_df.index.name = "feature"

    # Calculate outlier statistics and counts for numeric columns
    for feat in num_col:
        try:
            q1 = df_copy[feat].quantile(0.25)
            q3 = df_copy[feat].quantile(0.75)
            iqr = q3 - q1
            outlier_max_iqr = q3 + 1.5 * iqr
            outlier_min_iqr = max(q1 - 1.5 * iqr, 0)
            mean = df_copy[feat].mean()
            std = df_copy[feat].std()
            outlier_max_zscore = mean + 3 * std
            outlier_min_zscore = mean - 3 * std

            # Calculate IQR outlier count
            iqr_outliers = df_copy[(df_copy[feat] < outlier_min_iqr) | (df_copy[feat] > outlier_max_iqr)][feat].count()
            output_df.loc[feat, 'IQR_OUT_NB'] = iqr_outliers

            # Calculate Z-score outlier count
            zscore_outliers = df_copy[(df_copy[feat] < outlier_min_zscore) | (df_copy[feat] > outlier_max_zscore)][feat].count()
            output_df.loc[feat, 'Z_OUT_NB'] = zscore_outliers

            output_df.loc[feat, 'outlier_max_iqr'] = outlier_max_iqr
            output_df.loc[feat, 'outlier_min_iqr'] = outlier_min_iqr
            output_df.loc[feat, 'outlier_max_zscore'] = outlier_max_zscore
            output_df.loc[feat, 'outlier_min_zscore'] = outlier_min_zscore

        except Exception as e:
            output_df.loc[feat, 'outlier_max_iqr'] = "Error"
            output_df.loc[feat, 'outlier_min_iqr'] = "Error"
            output_df.loc[feat, 'outlier_max_zscore'] = "Error"
            output_df.loc[feat, 'outlier_min_zscore'] = "Error"
            output_df.loc[feat, 'IQR_OUT_NB'] = "Error"
            output_df.loc[feat, 'Z_OUT_NB'] = "Error"
            print(f"Error calculating outlier statistics for {feat}: {e}")

    # Calculate value counts for categorical columns
    for feat in cat_col:
        try:
            mode = df_copy[feat].mode()  # Get the mode(s)
            if not mode.empty:
                mode_value = mode[0]  # Get the first mode if it exists
                mode_count = df_copy[feat].value_counts().get(mode_value, 0) # get the count, default 0 if not present
                output_df.loc[feat, 'mode'] = mode_value
                output_df.loc[feat, 'mode_occurrence'] = mode_count  # Store the count
        except Exception as e:
            output_df.loc[feat, 'mode'] = "Error"
            output_df.loc[feat, 'mode_occurrence'] = "Error"
            print(f"Error calculating mode and occurrence for {feat}: {e}")

    # Reorder the index to put categorical columns last
    numeric_index = [col for col in output_df.index if col not in cat_col]
    categorical_index = cat_col
    output_df = output_df.reindex(index=numeric_index + categorical_index)
    return output_df

def remove_z_outlier(df, num_col):
    df_cleaned = df.copy()  # Create a copy to avoid modifying the original
    all_outlier_indices = [] # list to gather all the outliers

    for feat in num_col: # Iterate through column names
        mean_val = df_cleaned[feat].mean()
        std_val = df_cleaned[feat].std()
        outlier_max_zscore = mean_val + 3 * std_val
        outlier_min_zscore = mean_val - 3 * std_val

        # Identify outlier rows
        outlier_rows = df_cleaned[(df_cleaned[feat] < outlier_min_zscore) | (df_cleaned[feat] > outlier_max_zscore)]

        # Get outlier indices
        outlier_indices = outlier_rows.index.tolist()

        #Add these outlier indexes to the list for removal
        all_outlier_indices.extend(outlier_indices)

    #Remove duplicates
    all_outlier_indices = list(set(all_outlier_indices))
    print(f"Number of total outliers = {len(all_outlier_indices)}")

    # Drop outlier rows from the copy
    df_cleaned = df_cleaned.drop(index=all_outlier_indices)
    return df_cleaned # Return the modified DataFrame

def remove_1percent_outliers (df,num_col):
    df_cleaned = df.copy()
    all_outlier_indices = []
    for feat in num_col:
        top_99_val = df_cleaned[feat].quantile(0.99)
        outlier_rows = df_cleaned[df_cleaned[feat] > top_99_val]
        outlier_indices = outlier_rows.index.tolist()
        all_outlier_indices.extend(outlier_indices)

    all_outlier_indices = list(set(all_outlier_indices))
    print(f"Number of total outliers = {len(all_outlier_indices)}")

    df_cleaned = df_cleaned.drop(index=all_outlier_indices)
    return df_cleaned



# 4 - Analyse desciptive

def graphebarre(dataset: pd.DataFrame) -> None:

    categorical_features = dataset.select_dtypes(exclude=np.number).columns.tolist()

    if not categorical_features:
        print("No categorical features found in the dataset.")
        return

    for feature in categorical_features:
        value_counts = dataset[feature].value_counts().nlargest(10)  # Get top 10
        total = len(dataset[feature])  # Total for percentage calculation
        max_value = value_counts.max()

        if max_value < 20:
            print(f"Skipping feature '{feature}' because the maximum value count ({max_value}) is less than 20.")
            continue

        x = np.arange(len(value_counts))  # Create numeric x-axis values
        truncated_labels = [label[:25] for label in value_counts.index] # truncate labels

        plt.figure(figsize=(10, 6))  # Adjust figure size for better readability
        bars = plt.bar(x, value_counts.values)  # Store the bars for labeling
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(f"Bar Graph of Top 10 Values for {feature}")
        plt.xticks(x, truncated_labels, rotation=45, ha="right")  # Rotate x-axis labels for readability

        # Add labels to the bars (both count and percentage)
        for bar in bars:
            yval = bar.get_height()
            percentage = '{:.1f}%'.format(100 * yval / total) # Calculate percentage
            plt.text(bar.get_x() + bar.get_width()/2, yval + max_value*0.01, f'{int(yval)}\n({percentage})', ha='center', va='bottom') # Place the number and percentage above the bar

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.show()

def histogramme(dataset: pd.DataFrame) -> None:
    
    numerical_features = dataset.select_dtypes(include=np.number).columns.tolist()

    if not numerical_features:
        print("No numerical features found in the dataset.")
        return

    for feature in numerical_features:
        plt.figure(figsize=(12, 6))
        ax = sns.histplot(dataset[feature], bins=30, kde=True, color='blue')  # Use dataset, keep NA

        # Calculate percentages and add labels (unless count is 0)
        total = len(dataset[feature])
        max_height = 0 # Initialize variable to store the maximum height to display it dynamically

        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                percentage = '{:.1f}%'.format(100 * height/total)
                x = p.get_x() + p.get_width() / 2
                y = height
                ax.text(x, y + max_height * 0.01, percentage, ha='center', va='bottom')
            if height > max_height:
                max_height = height

        # Add titles and labels
        plt.title(f'Histogram of {feature}', fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Nombre de lignes', fontsize=14)

        # Set Y-axis limits manually
        ax.set_ylim(0, max_height * 1.05)  # Adjust 1.05 as needed for padding

        # Display the plot
        plt.show()

def univariate_analysis(df):

    # Prepare the output DataFrame
    output_df = pd.DataFrame(index=df.columns)
    output_df.index.name = "feature"
    output_df["type"] = df.dtypes
    output_df["count"] = df.count()
    output_df["missing"] = df.isna().sum()
    output_df["unique"] = df.nunique()

    # Calculate mode for all columns
    try:
        output_df["mode"] = df.astype(str).mode().iloc[0] #All columns to string to handle mixed data types
    except Exception as e:
        print(f"Warning: Could not calculate mode for all columns: {e}")
        output_df["mode"] = "N/A"

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    # Calculate numerical statistics
    if not numerical_cols.empty:
        numerical_stats = df[numerical_cols].agg(['min', 'mean', 'max', 'std', 'skew', 'kurt',
                                                     lambda x: x.quantile(0.25),
                                                     lambda x: x.quantile(0.5),
                                                     lambda x: x.quantile(0.75)]).T
        numerical_stats.columns = ['min', 'mean', 'max', 'std', 'skew', 'kurt', 'q1', 'median', 'q3']
        output_df = output_df.combine_first(numerical_stats) #Combine with existing output_df

    # Plotting
    histogramme(df)
    graphebarre(df)

    return output_df


def print_top_correlations(dataset: pd.DataFrame, n: int = 5, threshold: float = 0.85) -> list:
    
    numerical_features = dataset.select_dtypes(include=np.number).columns.tolist()

    if not numerical_features:
        print("No numerical features found in the dataset.")
        return [], set()  # Return empty list and set

    correlation_matrix = dataset[numerical_features].corr()
    high_correlations = []
    added_pairs = set()

    for feature in numerical_features:
        correlations = correlation_matrix[feature].drop(feature, errors='ignore') #Drop is now inside of loop to prevent error

        # Sort correlations by absolute value in descending order
        abs_correlations = abs(correlations).sort_values(ascending=False)
        top_correlations = abs_correlations.head(n)

        print(f"Top {n} Correlations (Positive and Negative) for {feature}:")
        for other_feature, abs_correlation in top_correlations.items(): #Iterate over absolute values so we can assess threshold properly
            correlation = correlations[other_feature]  # Get the original correlation value (can be negative)
            print(f"  {other_feature}: {correlation:.2f}")

            if abs(correlation) > threshold:
                pair = tuple(sorted((feature, other_feature)))  # Create a sorted tuple to avoid duplicates
                if pair not in added_pairs:
                    high_correlations.append((feature, other_feature, correlation))
                    added_pairs.add(pair)

        print("-" * 40)

    return high_correlations, added_pairs

from collections import Counter

def count_string_occurrences(data):
    # Initialize a counter
    counter = Counter()
    
    # Determine the maximum length of the tuples
    max_length = max(len(item) for item in data)
    
    # Iterate over each tuple in the data
    for item in data:
        # Iterate over the first two items in each tuple
        for i in range(min(2, len(item))):
            counter[item[i]] += 1
    
    return counter


import seaborn as sns
def corr_matrix(df):

    correlation_matrix = df.corr()

    # Create the heatmap with larger cells
    plt.figure(figsize=(18, 14))  # Increase figure size for bigger cells
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5,
                annot_kws={"size": 6},  # Increase annotation font size
                cbar_kws={'shrink': 0.7})   #Shrink the colorbar so annotations don't overlap

    # Add a title
    plt.title("Correlation Matrix of X1", fontsize=16)  #Increase title fontsize

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()


import scipy.stats as ss
def cramers_v(contingency_matrix):
    """
    Calculates Cramer's V, a measure of association between two nominal variables.

    Args:
        contingency_matrix: A Pandas DataFrame representing the contingency table.

    Returns:
        Cramer's V statistic.
    """
    chi2 = ss.chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.sum().sum()  # Corrected: Sum all values in the table
    phi2 = chi2 / n
    r, k = contingency_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def analyse_cat_cat(df, alpha=0.05):

    categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

    if not categorical_columns:
        print("No categorical features found in the dataset.")
        return None

    results = []  # List to store results

    # Iterate through all unique pairs of categorical columns
    for i in range(len(categorical_columns)):
        for j in range(i + 1, len(categorical_columns)):
            col1 = categorical_columns[i]
            col2 = categorical_columns[j]

            # Check if mode occurs less than 20 times in either column
            if df[col1].value_counts().iloc[0] < 15 or df[col2].value_counts().iloc[0] < 15:
                print(f"Skipping {col1} vs. {col2}: One or both columns have a mode occurring less than 15 times.")
                continue

            try:
                # Create a contingency table
                contingency_table = pd.crosstab(df[col1], df[col2])

                # Perform the chi-squared test
                stat, p, dof, expected = ss.chi2_contingency(contingency_table)

                # Check if the p-value is less than the significance level
                if p < alpha:
                    # Calculate Cramer's V
                    cramers_v_value = cramers_v(contingency_table)

                    # Store the results
                    results.append({
                        'col1': col1,
                        'col2': col2,
                        'p_value': p,
                        'cramers_v': cramers_v_value
                    })

            except Exception as e:
                print(f"Error during analysis of {col1} vs. {col2}: {e}")

    # Create a Pandas DataFrame from the results
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No significant associations found.")
        return None

    return results_df


def bar_chart(df, feature, target_variable, roundto=4, p_threshold=0.05, sig_ttest_only=True, min_group_size=2, max_t_tests=5):

    plt.figure(figsize=(10, 6))

    # Ensure target_variable is numeric and feature is categorical
    if pd.api.types.is_numeric_dtype(df[feature]):
        cat = target_variable
        num = feature
    else:
        cat = feature
        num = target_variable

    # Create barplot
    sns.barplot(x=cat, y=num, data=df, ci=None)

    # Perform ANOVA
    groups = df[cat].unique()
    group_lists = []
    valid_groups = []
    for g in groups:
        group_data = df[df[cat] == g][num]
        if len(group_data) >= min_group_size:
            group_lists.append(group_data)
            valid_groups.append(g)  # Keep track of valid group names
        else:
            print(f"Skipping group '{g}' due to insufficient data (n < {min_group_size}).")

    # Perform ANOVA only if there are at least two valid groups
    if len(group_lists) >= 2:
        try:
            f, p = stats.f_oneway(*group_lists)
        except Exception as e:
            print(f"Error during ANOVA: {e}")
            f, p = np.nan, np.nan  # Assign NaN values if ANOVA fails
    else:
        print("Not enough groups with sufficient data for ANOVA.")
        f, p = np.nan, np.nan

    # Perform t-tests for each pair of valid groups
    ttests = []
    num_comparisons = 0
    for i1, g1 in enumerate(valid_groups):
        for i2 in range(i1 + 1, len(valid_groups)):  # Avoid redundant comparisons and comparing a group to itself.
            g2 = valid_groups[i2]  # Correctly access group name based on index in valid_groups
            list1 = df[df[cat] == g1][num]
            list2 = df[df[cat] == g2][num]
            try:
                t, tp = stats.ttest_ind(list1, list2)
                ttests.append([f'{g1} - {g2}', round(t, roundto), round(tp, roundto)])
                num_comparisons += 1  # To calculate the Bonferroni correction
            except Exception as e:
                print(f"Error during t-test between '{g1}' and '{g2}': {e}")
                ttests.append([f'{g1} - {g2}', np.nan, np.nan])

    # Bonferroni correction
    if num_comparisons > 0:
        bonferroni = p_threshold / num_comparisons
    else:
        bonferroni = np.nan

    # Sort t-tests by absolute t-value and select top N
    ttests.sort(key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0, reverse=True)  # Sort by absolute t-value
    top_ttests = ttests[:max_t_tests]  # Select top N

    # Create annotation string
    textstr = f'ANOVA:\nF = {round(f, roundto)}\np = {round(p, roundto)}\n'
    textstr += f'Bonferroni p threshold: {round(bonferroni, roundto)}\n'

    # Add significant t-tests to the annotation
    sig_ttest_count = 0
    for ttest in top_ttests:  # Iterate only through the top t-tests
        if len(ttest) == 3:
            if not np.isnan(bonferroni) and ttest[2] <= bonferroni:
                textstr += f'\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}'
                sig_ttest_count += 1
            elif not sig_ttest_only:
                textstr += f'\n{ttest[0]}: t:{ttest[1]}, p:{ttest[2]}'
    if sig_ttest_only and sig_ttest_count == 0:
        textstr += f'\nNo significant t-tests'

    # Rotate x-axis labels if there are many categories
    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)

    # Add annotation to the plot (adjust position for better visibility)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white', alpha=0.8))

    plt.title(f'Bar Chart: {target_variable} by {feature}')
    plt.xlabel(target_variable)
    plt.ylabel(feature)
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()



def run_ols_with_two_features(df, target_col, feat_col):

    feature_names = [col for col in feat_col if col != target_col and pd.api.types.is_numeric_dtype(df[col])]

    if len(feature_names) < 2:
        print("Error: Need at least two numerical features (excluding the target) to run combinations.")
        return []

    results_ols2 = []
    for feat1, feat2 in itertools.combinations(feature_names, 2):
        try:
            X = df[[feat1, feat2]]
            X = sm.add_constant(X)
            y = df[target_col]
            model = sm.OLS(y, X).fit()
            results_ols2.append({'feature1': feat1, 'feature2': feat2, 'r_squared': model.rsquared})
        except Exception as e:
            print(f"Error during OLS2 regression for features {feat1}, {feat2}: {e}")

    return results_ols2

def compare_r_squared_2(results):
    for i, res in enumerate(sorted(results, key=lambda x: x['r_squared'], reverse=True)):
        print(f"Result {i}: Features=({res['feature1']}, {res['feature2']}), R-squared = {res['r_squared']:.4f}")

def compare_r_squared_3(results):
    for i, res in enumerate(sorted(results, key=lambda x: x['r_squared'], reverse=True)):
        print(f"Result {i}: Features=({res['feature1']}, {res['feature2']}, {res['feature3']}), R-squared = {res['r_squared']:.4f}")

def run_ols_with_3_features(df, target_col, feat_col):

    feature_names = [col for col in feat_col if col != target_col and pd.api.types.is_numeric_dtype(df[col])]

    if len(feature_names) < 3:
        print("Error: Need at least three numerical features (excluding the target) to run combinations.")
        return []

    results_ols3 = []
    for feat1, feat2, feat3 in itertools.combinations(feature_names, 3):
        try:
            X = df[[feat1, feat2, feat3]]
            X = sm.add_constant(X)
            y = df[target_col]
            model = sm.OLS(y, X).fit()
            results_ols3.append({'feature1': feat1, 'feature2': feat2, 'feature3': feat3, 'r_squared': model.rsquared})
        except Exception as e:
            print(f"Error during OLS2 regression for features {feat1}, {feat2}: {e}")

    return results_ols3



# 5 - Feature Engineering
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import warnings
from typing import List, Dict, Optional, Union, Tuple, Callable

def _calculate_correlation(
    transformed_series: pd.Series, target_series: pd.Series
) -> Optional[float]:
    """Helper function to calculate Pearson correlation and handle potential issues."""
    if transformed_series.isnull().all() or target_series.isnull().all():
        return np.nan # Avoids issues with all-NaN series
    if transformed_series.nunique() <= 1:
         return np.nan # Correlation is undefined for constant series

    try:
        # Replace inf/-inf with NaN before calculating correlation
        transformed_series = transformed_series.replace([np.inf, -np.inf], np.nan)

        # Use pairwise deletion for NaNs by default in .corr()
        corr = transformed_series.corr(target_series, method='pearson')

        # Ensure float or NaN is returned
        return float(corr) if pd.notna(corr) else np.nan
    except Exception:
        # Catch any unexpected errors during correlation calculation
        return np.nan
    
def _apply_power_transformation(series: pd.Series, power: float) -> pd.Series:
    """Applies power transformation to a series, handling non-positive values."""
    if not isinstance(series, pd.Series): return pd.Series([np.nan] * len(series)) # robustness
    transformed = series.copy()
    try:
        if power < 0: # Negative powers require non-zero values
             non_zero_mask = transformed != 0
             if non_zero_mask.any():
                 transformed.loc[non_zero_mask] = transformed.loc[non_zero_mask] ** power
                 transformed.loc[~non_zero_mask] = np.nan # Zeros become NaN
             else:
                 transformed[:] = np.nan
        elif 0 < power < 1: # Fractional powers require positive values
            positive_mask = transformed > 0
            if positive_mask.any():
                transformed.loc[positive_mask] = transformed.loc[positive_mask] ** power
                transformed.loc[~positive_mask] = np.nan # Non-positives become NaN
            else:
                transformed[:] = np.nan
        else: # Integer powers >= 1 and power == 0 (becomes 1)
            # Handle potential overflow for large powers
            with np.errstate(over='ignore'): # Ignore overflow, result will be inf
                 transformed = transformed ** power
                 transformed.replace([np.inf, -np.inf], np.nan, inplace=True) # Treat overflow as NaN

    except (TypeError, ValueError):
        transformed[:] = np.nan # Handle potential type errors if data isn't numeric
    return transformed

def _apply_log_transformation(series: pd.Series, log_func: callable) -> pd.Series:
    """Applies log transformation to a series, handling non-positive values."""
    if not isinstance(series, pd.Series): return pd.Series([np.nan] * len(series))
    transformed = series.copy()
    try:
        positive_mask = transformed > 0
        if positive_mask.any():
            transformed.loc[positive_mask] = log_func(transformed.loc[positive_mask])
            transformed.loc[~positive_mask] = np.nan # Non-positives become NaN
        else:
            transformed[:] = np.nan
    except (TypeError, ValueError):
         transformed[:] = np.nan
    return transformed

def _apply_scaling_transformation(
    series: pd.Series, scaler_instance: Union[StandardScaler, RobustScaler, PowerTransformer]
) -> pd.Series:
    """Applies a fitted or fresh scaling transformation to a series."""
    if not isinstance(series, pd.Series): return pd.Series([np.nan] * len(series))
    col_data = series.values.reshape(-1, 1)

    # Check for invalid input for scalers
    if pd.isna(col_data).all():
        return pd.Series(np.nan, index=series.index)
    if np.nanstd(col_data.astype(float)) == 0 and isinstance(scaler_instance, (StandardScaler, PowerTransformer)):# Reverted to standardscaler
        # Constant data leads to NaN/errors in StandardScaler/PowerTransformer
        return pd.Series(np.nan, index=series.index)

    try:
        # Always fit the scaler to the current data in this iterative context
        scaler_instance.fit(col_data)
        scaled_data = scaler_instance.transform(col_data)
        return pd.Series(scaled_data.flatten(), index=series.index)
    except (ValueError, TypeError) as e:
        # Catch errors during fit/transform
        return pd.Series(np.nan, index=series.index)
    except Exception as e: # added
        return pd.Series(np.nan, index=series.index)


# --- Transformation Definitions ---

TRANSFORMATIONS = {
    # Powers
    'power_0_25': lambda s: _apply_power_transformation(s, 0.25),
    'power_0_33': lambda s: _apply_power_transformation(s, 1/3),
    'power_0_50': lambda s: _apply_power_transformation(s, 0.5),
    'power_2_00': lambda s: _apply_power_transformation(s, 2),
    'power_3_00': lambda s: _apply_power_transformation(s, 5),
    'power_5_00': lambda s: _apply_power_transformation(s, 5), #changed
    'power_8_00': lambda s: _apply_power_transformation(s, 8), #added

    #'reciprocal': lambda s: _apply_power_transformation(s, -1), 
    #'reciprocal_2': lambda s: _apply_power_transformation(s, -2),
    #'reciprocal_3': lambda s: _apply_power_transformation(s, -3),
    
    # Logs
    'log2': lambda s: _apply_log_transformation(s, np.log2),
    'log10': lambda s: _apply_log_transformation(s, np.log10),
    # Scalers - use fresh instances each time
    'standard_scale': lambda s: _apply_scaling_transformation(s, StandardScaler()),
    'robust_scale': lambda s: _apply_scaling_transformation(s, RobustScaler()),
    'yeo_johnson': lambda s: _apply_scaling_transformation(s, PowerTransformer(method='yeo-johnson', standardize=True)) #removed robust scaling
}

# --- Iterative Analysis Function ---
def apply_and_test_all_transformations(
    current_series: pd.Series,
    target_series: pd.Series,
    transformations: Dict[str, Callable[[pd.Series], pd.Series]]
) -> Dict[str, float]:
    """Applies all defined transformations to the current series and returns correlations."""
    results = {}
    for name, func in transformations.items():
        # Apply transformation
        transformed_series = func(current_series)
        # Calculate correlation
        results[name] = _calculate_correlation(transformed_series, target_series)
    return results


def analyze_iterative_transformations(
    df: pd.DataFrame,
    target: str,
    max_turns: int = 5,
    min_improvement: float = 5e-4 # Minimum absolute correlation increase to continue
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[None, None]]:
    """
    Iteratively applies the best transformation to numerical features to maximize
    correlation with the target, tracking the history.

    Args:
        df: The input pandas DataFrame.
        target: The name of the target column in the DataFrame.
        max_turns: Maximum number of transformation iterations per feature.
        min_improvement: Minimum absolute improvement in correlation required
                         to apply the next transformation.

    Returns:
        A tuple containing two DataFrames:
        1. summary_df: Index = features, Columns = ['Initial Correlation',
                       'Final Correlation', 'Num Transformations'].
        2. history_df: Index = features, Columns = ['Transform_1', 'Corr_1', ...,
                      'Transform_N', 'Corr_N'] (up to max_turns).
        Returns (None, None) if input validation fails.
    """
    # --- Input Validation ---
    if not isinstance(df, pd.DataFrame):
        print("Error: Input 'df' must be a pandas DataFrame.")
        return None, None
    if target not in df.columns:
        print(f"Error: Target column '{target}' not found in DataFrame.")
        return None, None
    if not pd.api.types.is_numeric_dtype(df[target]):
        print(f"Error: Target column '{target}' must be numeric for correlation analysis.")
        return None, None
    if df[target].isnull().all():
        print(f"Error: Target column '{target}' contains only NaN values.")
        return None, None

    # --- Feature Selection & Initial Setup ---
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)

    if not numerical_cols:
        print("Error: No numerical features found (excluding target).")
        return None, None

    # Use copies and handle NaNs in target
    target_series = df[target].copy()
    valid_target_idx = target_series.dropna().index
    df_numeric = df[numerical_cols].copy().loc[valid_target_idx]
    target_series = target_series.loc[valid_target_idx]

    if df_numeric.empty:
        print("Error: No valid data remaining after removing NaNs in target.")
        return None, None

    print(f"Starting iterative transformation analysis for {len(numerical_cols)} features...")

    # --- Data Structures for Tracking ---
    feature_states = {} # Stores current series and best correlation per feature
    feature_history = {} # Stores the sequence of (transform, corr) per feature

    # Initialize with original features
    for col in numerical_cols:
        original_series = df_numeric[col]
        initial_corr = _calculate_correlation(original_series, target_series)
        if pd.isna(initial_corr): initial_corr = 0.0 # Treat NaN initial corr as 0 for comparison

        feature_states[col] = {
            'current_series': original_series,
            'best_corr_abs': abs(initial_corr),
            'best_corr_signed': initial_corr,
            'stopped': False
        }
        # History starts with the 'original' state
        feature_history[col] = [('original', initial_corr)]


    # --- Iterative Transformation Loop ---
    for turn in range(1, max_turns + 1):
        print(f"--- Turn {turn}/{max_turns} ---")
        num_improved_this_turn = 0

        for col in numerical_cols:
            state = feature_states[col]
            if state['stopped']:
                continue # Skip if improvement has stalled for this feature

            current_series = state['current_series']

            # Apply all transformations to the current state and get correlations
            turn_correlations = apply_and_test_all_transformations(
                current_series, target_series, TRANSFORMATIONS
            )

            # Find the best transformation *in this turn*
            best_turn_transform = None
            best_turn_corr_signed = np.nan
            best_turn_corr_abs = -1.0 # Initialize lower than any possible correlation abs value

            for transform_name, corr_signed in turn_correlations.items():
                 if pd.notna(corr_signed):
                      corr_abs = abs(corr_signed)
                      # Check if this is the best in *this turn*
                      if corr_abs > best_turn_corr_abs:
                           best_turn_corr_abs = corr_abs
                           best_turn_corr_signed = corr_signed
                           best_turn_transform = transform_name

            # Check for Improvement vs. Overall Best
            if best_turn_transform is not None and \
               (best_turn_corr_abs > state['best_corr_abs'] + min_improvement):

                num_improved_this_turn += 1
                print(f"  Feature '{col}': Improved corr from {state['best_corr_signed']:.4f} "
                      f"to {best_turn_corr_signed:.4f} using '{best_turn_transform}'")

                # Apply the best transformation to get the next series state
                next_series = TRANSFORMATIONS[best_turn_transform](current_series)

                # Update state and history
                state['current_series'] = next_series
                state['best_corr_abs'] = best_turn_corr_abs
                state['best_corr_signed'] = best_turn_corr_signed
                feature_history[col].append((best_turn_transform, best_turn_corr_signed))

            else:
                # No significant improvement found in this turn
                 # print(f"  Feature '{col}': No improvement found. Stopping.") # Optional: for verbose logging
                state['stopped'] = True

        if num_improved_this_turn == 0:
            print(f"No features improved in turn {turn}. Stopping iterations early.")
            break # Exit outer loop if no feature improved

    print("Iterative analysis complete.")

    # --- Prepare Output DataFrames ---

    # 1. Summary DataFrame
    summary_data = []
    for col in numerical_cols:
        initial_corr = feature_history[col][0][1] # Corr from 'original'
        final_corr = feature_states[col]['best_corr_signed']
        num_transforms = len(feature_history[col]) - 1 # Exclude 'original'
        summary_data.append([col, initial_corr, final_corr, num_transforms])

    summary_df = pd.DataFrame(
        summary_data,
        columns=['Feature', 'Initial Correlation', 'Final Correlation', 'Num Transformations']
    ).set_index('Feature')

    # 2. History DataFrame
    history_data = {col: {} for col in numerical_cols}
    history_cols = []
    for i in range(1, max_turns + 1):
        history_cols.extend([f'Transform_{i}', f'Corr_{i}'])

    for col in numerical_cols:
        history = feature_history[col]
        # Skip the 'original' entry at index 0
        for i in range(1, max_turns + 1):
            transform_col = f'Transform_{i}'
            corr_col = f'Corr_{i}'
            if i < len(history):
                history_data[col][transform_col] = history[i][0]
                history_data[col][corr_col] = history[i][1]
            else:
                history_data[col][transform_col] = None # Or np.nan
                history_data[col][corr_col] = np.nan

    history_df = pd.DataFrame.from_dict(history_data, orient='index')
    # Ensure correct column order if dict insertion order isn't guaranteed (Python < 3.7)
    history_df = history_df.reindex(columns=history_cols)

    return summary_df, history_df


def create_transformed_dataframe(
    df: pd.DataFrame, history_results: pd.DataFrame) -> pd.DataFrame:

    df_transformed = df.copy()  # Start with a copy of the original DataFrame

    # Identify columns to transform
    col_to_transform = [col for col in df.columns if not col.startswith('%_GFA')]
    features = history_results.index.tolist()  # Ensure the correct type

    # Helper function to apply a single transformation
    def apply_transformation(series, transform_name):
        if transform_name == 'original' or transform_name is None:
            return series
        if not isinstance(transform_name, str):
            return series  # Skip non-string transform names to be robust

        if transform_name in TRANSFORMATIONS:
            return TRANSFORMATIONS[transform_name](series)
        else:
            print(f"Warning: Transformation '{transform_name}' not found. Skipping.")
            return series  # Skip and issue warning

    try:
        for feature in features:
            # Only process if the feature is in the list of columns to transform
            if feature in col_to_transform:  # Important change: Check if feature is in col_to_transform
                current_series = df_transformed[feature].copy()

                for i in range(1, 6):  # Assuming max_turns = 5
                    transform_col = f'Transform_{i}'
                    if transform_col not in history_results.columns:
                        break  # No more transforms for this feature

                    transform_name = history_results.loc[feature, transform_col]
                    current_series = apply_transformation(current_series, transform_name)

                df_transformed[feature] = current_series  # Update ONLY transformed columns

    except KeyError as e:
        print(f"Error: KeyError during transformation application: {e}")
        return pd.DataFrame()  # Return empty DataFrame
    except Exception as e:  # Catch remaining issues
        print(f"Unexpected error during transformation application: {e}")
        return pd.DataFrame()

    return df_transformed


def bin_categories(df, features=[], cutoff = 0.007, replace_with='other_grouped', message = True):
    # Regroupe toutes les valeurs d'une récurrence de moins de 0,7% dans 'other_grouped'
    for feat in features:
        if feat in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feat]):
                other_list = df[feat].value_counts()[df[feat].value_counts() / df.shape[0] < cutoff].index
                df.loc[df[feat].isin(other_list), feat] = replace_with
        else:
            if message: print(f'{feat} not found in the dataframe provided. No binning performed')

    return df



# 6 - Preprocessing

def special_encoding(df: pd.DataFrame) -> pd.DataFrame:
    
    col_special_encoding = ["LargestPropertyUseType", "SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"]
    sufix_name = "%_GFA"

    for col in col_special_encoding:
        # Middle part of the colname created :
        modalities = df[col].dropna().unique()

        # Column value to consider to fill the new column created if the mask is respected
        value_col = f"{sufix_name}_{col.replace('PropertyUseType', '_Use')}"

        for name in modalities:
            # Valeur dans ces colonnes :
            final_name = f"{col}_{name}_{sufix_name}"  # Added the column as a prefix
            mask = df[col] == name
            df[final_name] = 0  # Create and initialize column
            df.loc[mask, final_name] = df.loc[mask, value_col]  # Assign values

    return df



def sum_special_encoding_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sums the columns created by the special_encoding function, grouping them
    by the middle name (modality) corresponding to the PropertyUseType.

    Args:
        df: The DataFrame with the columns created by the special_encoding function.

    Returns:
        The DataFrame with the new aggregated columns.  The original columns
        created by special_encoding are also retained.
    """
    df_copy = df.copy()

    # Identify the columns created by the special_encoding function.
    #  We assume they have the format "PropertyUseType_Modality_%_GFA".
    special_cols = [col for col in df_copy.columns if "%_GFA" in col and any(use_type in col for use_type in ["LargestPropertyUseType", "SecondLargestPropertyUseType", "ThirdLargestPropertyUseType"])]

    # Extract the modalities (middle names) from the column names.
    modalities = set() # Use a set to avoid duplicates
    for col in special_cols:
      parts = col.split("_")
      if len(parts) > 2: # Ensure there are enough parts
        modalities.add(parts[1]) # Modality is the second element after splitting

    # Create new columns for each modality, summing the corresponding columns.
    for modality in modalities:
        # Build a list of the columns to sum for the current modality.
        cols_to_sum = [col for col in special_cols if f"_{modality}_" in col]

        # Sum the columns and create a new column with the modality name.
        if cols_to_sum:  # Check if there are any columns to sum for this modality
            df_copy[f"{modality}_Total_%_GFA"] = df_copy[cols_to_sum].sum(axis=1)
        else:
            print(f"No columns found to sum for modality: {modality}")

    return df_copy


def onehot_encode_column(df, column_name):
 
    # 1. Create a OneHotEncoder instance.  handle_unknown='ignore' prevents errors if new categories appear later
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False returns a numpy array, not a sparse matrix

    # 2. Fit and transform the data
    encoder.fit(df[[column_name]]) # Fit to the *column* (as a DataFrame)

    transformed_data = encoder.transform(df[[column_name]])

    # 3. Create new column names
    feature_names = encoder.get_feature_names_out([column_name]) # Get the new feature names

    # 4. Create a new DataFrame from the transformed data
    encoded_df = pd.DataFrame(transformed_data, columns=feature_names, index=df.index) #preserve index

    # 5. Concatenate the new DataFrame with the original
    df = pd.concat([df, encoded_df], axis=1)

    # 6. Drop the original column
    df.drop(columns=[column_name], inplace=True)

    return df

