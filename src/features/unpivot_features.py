# src/features/unpivot_features.py
#
# Feature extraction module for Unpivot prediction based on Section 4.4
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Extracts features that help predict which columns to unpivot/melt
# 2. Builds compatibility matrices that capture relationships between columns
# 3. Processes unpivot/melt samples from Jupyter notebooks for training/testing

import pandas as pd
import numpy as np
import re
import os
from typing import List, Dict, Tuple, Any, Set
import warnings

# Filter to explicitly ignore the numpy divide warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

def detect_pattern_similarity(col1: str, col2: str) -> float:
    """
    Detect pattern similarity between two column names for unpivot prediction.

    This function identifies patterns in column names that suggest they should be
    unpivoted together. It looks for patterns like:
    - Columns with the same prefix/suffix: "sales_2019", "sales_2020"
    - Columns with the same non-numeric parts: "jan_revenue", "feb_revenue"
    - Columns that follow similar naming conventions

    Higher similarity scores indicate columns are more likely to be unpivoted together.

    Example:
        ```
        # Same non-numeric part with different years
        detect_pattern_similarity("revenue_2019", "revenue_2020")  # Returns: 0.9

        # Same prefix
        detect_pattern_similarity("sales_north", "sales_south")  # Returns: 0.8

        # Same suffix
        detect_pattern_similarity("jan_sales", "feb_sales")  # Returns: 0.8

        # Partial matching patterns
        detect_pattern_similarity("revenue_north", "north_sales")  # Returns: 0.6

        # Different patterns
        detect_pattern_similarity("customer_id", "revenue_2020")  # Returns: 0.2

        # Identical columns (not good for unpivot)
        detect_pattern_similarity("revenue", "revenue")  # Returns: 0.5
        ```

    Args:
        col1: First column name
        col2: Second column name

    Returns:
        Similarity score between 0 and 1, where:
        - 0.9: Columns have identical non-numeric parts (e.g., "year_2018", "year_2019")
        - 0.8: Columns have common prefix or suffix
        - 0.6: Columns have partially matching patterns
        - 0.5: Columns are identical (not ideal for unpivot)
        - 0.2: Columns have different patterns

    Note:
        This function is a key component for unpivot prediction, as it helps
        identify columns that follow similar patterns and should be collapsed
        together during unpivot operations.
    """
    # If columns are identical, they're similar but not good unpivot candidates
    if col1 == col2:
        return 0.5

    # Extract non-numeric parts
    non_numeric1 = re.sub(r'\d+', '', col1).strip('_')
    non_numeric2 = re.sub(r'\d+', '', col2).strip('_')

    # If they have the same non-numeric part, they likely follow a pattern
    if non_numeric1 and non_numeric2 and non_numeric1 == non_numeric2:
        return 0.9

    # Check common prefix/suffix (common patterns)
    prefix = os.path.commonprefix([col1, col2])
    if len(prefix) > 2 and prefix != col1 and prefix != col2:
        return 0.8

    # Check reversed to find common suffix
    col1_rev = col1[::-1]
    col2_rev = col2[::-1]
    suffix = os.path.commonprefix([col1_rev, col2_rev])[::-1]
    if len(suffix) > 2 and suffix != col1 and suffix != col2:
        return 0.8

    # Moderate similarity for partially matching patterns
    if non_numeric1 and non_numeric2 and (non_numeric1 in non_numeric2 or non_numeric2 in non_numeric1):
        return 0.6

    # Low similarity for different patterns
    return 0.2


def calculate_column_compatibility_features(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, float]:
    """
    Calculate compatibility features between two columns for unpivot prediction.

    These features help determine if two columns should be unpivoted together
    (i.e., collapsed into a single key-value pair).

    Args:
        df: Input DataFrame
        col1: First column name
        col2: Second column name

    Returns:
        Dictionary of compatibility features
    """
    features = {}

    # Feature 1: Data type compatibility
    # Columns to be unpivoted together should have the same or compatible data types
    features['same_type'] = float(df[col1].dtype == df[col2].dtype)

    # For numeric columns, check if they have similar ranges
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        try:
            min1, max1 = df[col1].min(), df[col1].max()
            min2, max2 = df[col2].min(), df[col2].max()

            # Calculate overlap of ranges
            range1 = max1 - min1 if not pd.isna(max1 - min1) else 0
            range2 = max2 - min2 if not pd.isna(max2 - min2) else 0

            if range1 > 0 and range2 > 0:
                range_ratio = min(range1, range2) / max(range1, range2)
                features['range_similarity'] = range_ratio
            else:
                features['range_similarity'] = 0.0
        except:
            features['range_similarity'] = 0.0
    else:
        features['range_similarity'] = 0.0

    # Feature 2: Name pattern similarity
    # Columns to be unpivoted often have similar naming patterns (year_2018, year_2019, etc.)
    features['name_pattern_similarity'] = detect_pattern_similarity(col1, col2)

    # Feature 3: Column position
    # Columns to be unpivoted are often adjacent or near each other
    col1_pos = list(df.columns).index(col1)
    col2_pos = list(df.columns).index(col2)
    pos_diff = abs(col1_pos - col2_pos)

    # Convert to similarity score (closer = higher)
    features['position_similarity'] = 1.0 / (1.0 + pos_diff)

    # Feature 4: Correlation for numeric columns
    # Highly correlated columns might be related and good candidates for unpivoting
    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        try:
            # Completely suppress warnings for this calculation
            with np.errstate(all='ignore'):
                # First check if columns have any variation
                std1 = df[col1].std(skipna=True)
                std2 = df[col2].std(skipna=True)

                has_values = df[col1].notna().sum() > 1 and df[col2].notna().sum() > 1

                # Only calculate correlation if both columns have variation and sufficient non-NA values
                if std1 > 0 and std2 > 0 and has_values:
                    # Find rows where both columns have non-null values
                    common_indices = df[col1].notna() & df[col2].notna()
                    if common_indices.sum() > 1:  # Need at least 2 values for correlation
                        # Calculate correlation safely
                        correlation = abs(df.loc[common_indices, col1].corr(
                            df.loc[common_indices, col2]))

                        # Check if correlation is valid
                        if not np.isnan(correlation) and np.isfinite(correlation):
                            features['correlation'] = correlation
                        else:
                            features['correlation'] = 0.0
                    else:
                        features['correlation'] = 0.0
                else:
                    features['correlation'] = 0.0

        except:
            features['correlation'] = 0.0
    else:
        features['correlation'] = 0.0

    # Feature 5: Similar null patterns
    # Columns to be unpivoted often have similar null patterns
    try:
        null1 = df[col1].isna()
        null2 = df[col2].isna()

        # Check if there are any nulls before calculating overlap
        if null1.sum() > 0 or null2.sum() > 0:
            null_overlap = (null1 & null2).sum() / max(null1.sum(), null2.sum(), 1)
            features['null_pattern_similarity'] = null_overlap
        else:
            # If no nulls in either column, they have "similar" null patterns (both have none)
            features['null_pattern_similarity'] = 1.0
    except:
        features['null_pattern_similarity'] = 0.0

    return features


def build_compatibility_matrix(input_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Build a compatibility matrix for the given columns for unpivot prediction.

    The compatibility matrix captures how likely columns are to be unpivoted together.
    High values indicate columns should be unpivoted together.

    Args:
        input_df: Input DataFrame
        columns: List of columns to consider

    Returns:
        Compatibility matrix as a pandas DataFrame
    """
    n = len(columns)
    compatibility_matrix = pd.DataFrame(0.0, index=columns, columns=columns)

    # Suppress numpy warnings globally during matrix construction
    with np.errstate(all='ignore'):
        for i in range(n):
            for j in range(i + 1, n):
                col1 = columns[i]
                col2 = columns[j]

                # Calculate features for this column pair
                features = calculate_column_compatibility_features(input_df, col1, col2)

                # Calculate the overall compatibility score
                # Weighted average of features (weights based on importance)
                # Here, again, instead of using the same regression model as in Pivot, we heuristically define the weights
                compatibility_score = (
                        0.3 * features['same_type'] +
                        0.1 * features.get('range_similarity', 0.0) +
                        0.3 * features['name_pattern_similarity'] +
                        0.2 * features['position_similarity'] +
                        0.05 * features.get('correlation', 0.0) +
                        0.05 * features.get('null_pattern_similarity', 0.0)
                )

                # Set the compatibility score (symmetric matrix)
                compatibility_matrix.loc[col1, col2] = compatibility_score
                compatibility_matrix.loc[col2, col1] = compatibility_score

        # Fill diagonal with 1.0 (maximum compatibility with self)
        for col in columns:
            compatibility_matrix.loc[col, col] = 1.0

    return compatibility_matrix


def process_unpivot_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process unpivot/melt samples to get input tables and unpivot parameters.

    This function extracts the ground truth from Jupyter notebooks by:
    1. Identifying which columns were kept as-is (id_vars)
    2. Identifying which columns were unpivoted/melted (value_vars)

    Args:
        samples: List of raw samples loaded from the data directory

    Returns:
        List of processed samples with input_table and unpivot parameters identified
    """
    processed_samples = []

    print(f"Processing {len(samples)} unpivot samples...")

    for sample_idx, sample in enumerate(samples):
        # Skip samples without necessary data
        if 'input_table' not in sample or 'params' not in sample:
            continue

        input_table = sample['input_table']
        params = sample['params']

        # Skip if table is empty
        if input_table.empty:
            continue

        # Extract unpivot parameters
        id_vars = []
        value_vars = []
        var_name = 'variable'
        value_name = 'value'

        # Check for id_vars parameter (columns to keep as-is)
        if 'id_vars' in params:
            id_vars_param = params['id_vars']
            if isinstance(id_vars_param, list):
                id_vars = id_vars_param
            elif isinstance(id_vars_param, str):
                id_vars = [id_vars_param]

        # Check for value_vars parameter (columns to unpivot)
        if 'value_vars' in params:
            value_vars_param = params['value_vars']
            if isinstance(value_vars_param, list):
                value_vars = value_vars_param
            elif isinstance(value_vars_param, str):
                value_vars = [value_vars_param]

        # If value_vars is not specified, it typically means "all columns not in id_vars"
        if not value_vars and id_vars:
            value_vars = [col for col in input_table.columns if col not in id_vars]

        # Check for var_name parameter (name for the new "variable" column)
        if 'var_name' in params:
            var_name = params['var_name']

        # Check for value_name parameter (name for the new "value" column)
        if 'value_name' in params:
            value_name = params['value_name']

        # Check if columns are valid in the table
        valid_columns = list(input_table.columns)
        id_vars = [col for col in id_vars if col in valid_columns]
        value_vars = [col for col in value_vars if col in valid_columns]

        # Only add sample if it has valid unpivot parameters
        if id_vars and value_vars and len(value_vars) >= 2:
            processed_samples.append({
                'input_table': input_table,
                'id_vars': id_vars,
                'value_vars': value_vars,
                'var_name': var_name,
                'value_name': value_name,
                'sample_id': sample.get('sample_id', f"sample_{sample_idx}")
            })

    print(f"Processed {len(processed_samples)} valid unpivot samples")
    return processed_samples