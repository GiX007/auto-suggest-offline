# src/features/pivot_features.py
#
# Feature extraction module for Pivot prediction based on Section 4.3
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Extracts features that help predict how to split dimension columns in pivot tables
# 2. Builds affinity matrices that capture relationships between columns
# 3. Processes pivot samples from Jupyter notebooks for training/testing

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def calculate_emptiness_reduction_ratio(df: pd.DataFrame, col1: str, col2: str) -> float:
    """
    Calculate the emptiness reduction ratio (ERR) between two columns.

    This ratio measures how much "emptier" a pivot table would be if the two columns
    are placed on different sides (one in index, one in header) vs. same side.
    High values indicate the columns should be on the same side.

    Args:
        df: Input DataFrame.
        col1: First column name.
        col2: Second column name.

    Returns:
        Emptiness reduction ratio.
    """
    try:
        # Count distinct values in each column
        unique_values1 = df[col1].nunique()
        unique_values2 = df[col2].nunique()

        # Count of distinct pairs (actual combinations that exist in data)
        unique_pairs = df[[col1, col2]].drop_duplicates().shape[0]

        # Calculate reduction ratio as described in the paper
        # This is the ratio of theoretical cells (unique1 * unique2) to actual cells (unique_pairs)
        # Higher values indicate more "emptiness" savings by keeping columns together
        reduction_ratio = (unique_values1 * unique_values2) / unique_pairs if unique_pairs > 0 else 1.0

        return reduction_ratio
    except Exception as e:
        print(f"Error calculating emptiness reduction ratio for {col1} and {col2}: {e}")
        return 1.0


def calculate_column_affinity_features(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, float]:
    """
    Calculate affinity features between two columns for pivot prediction.

    These features help determine if two columns should be on the same side
    (both in index or both in header) in a pivot table.

    Args:
        df: Input DataFrame.
        col1: First column name.
        col2: Second column name.

    Returns:
        Dictionary of affinity features.
    """
    features = {}

    # Feature 1: Emptiness-reduction-ratio
    # This is the main feature mentioned in the paper
    features['emptiness_reduction_ratio'] = calculate_emptiness_reduction_ratio(df, col1, col2)

    # Feature 2: Column-position-difference
    # The paper mentions that columns close to each other are often related
    col1_pos = list(df.columns).index(col1)
    col2_pos = list(df.columns).index(col2)
    features['position_difference'] = abs(col1_pos - col2_pos)
    features['relative_position_difference'] = features['position_difference'] / len(df.columns) if len(
        df.columns) > 0 else 0

    # Additional features that might be useful
    # Check if types of columns match (same types often belong together)
    features['same_type'] = float(df[col1].dtype == df[col2].dtype)

    # Check name similarity (columns with similar names often belong together)
    name_similarity = 0.0
    if col1.lower() in col2.lower() or col2.lower() in col1.lower():
        name_similarity = 0.5
    elif col1.lower() == col2.lower():
        name_similarity = 1.0
    features['name_similarity'] = name_similarity

    # Datatype correlation between columns (for numeric columns)
    try:
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            features['correlation'] = df[col1].corr(df[col2])
        else:
            features['correlation'] = 0.0
    except:
        features['correlation'] = 0.0

    return features


def build_affinity_matrix(input_df: pd.DataFrame, dimension_columns: List[str]) -> pd.DataFrame:
    """
    Build an affinity matrix for the given dimension columns.

    The affinity matrix captures how likely columns are to be on the same side
    in a pivot table. High values indicate columns should be kept together.

    Args:
        input_df: Input DataFrame.
        dimension_columns: List of dimension columns to consider.

    Returns:
        Affinity matrix as a pandas DataFrame.
    """
    n = len(dimension_columns)
    affinity_matrix = pd.DataFrame(0.0, index=dimension_columns, columns=dimension_columns) # (e.g. for a 5-column table, we build a 5x5 affinity matrix)

    for i in range(n):
        for j in range(i + 1, n):
            col1 = dimension_columns[i]
            col2 = dimension_columns[j]

            # Calculate features for this column pair
            features = calculate_column_affinity_features(input_df, col1, col2)

            # Calculate affinity score - we'll use a simple heuristic based on the paper
            # Higher emptiness reduction ratio means higher affinity
            affinity_score = np.tanh(features['emptiness_reduction_ratio'] / 10.0)

            # Adjust based on position difference (columns close to each other have higher affinity)
            # Use tanh to normalize high ERR values to [0,1] and dampen outliers — ensures affinity stays bounded
            position_factor = 1.0 / (1.0 + features['relative_position_difference'])

            # We apply a manual adjustment using relative column positions
            # (instead of learning a pivot classifier), as the paper opts for a heuristic affinity approach.
            affinity_score = affinity_score * position_factor

            # Consider name similarity (columns with similar names often belong together)
            if features['name_similarity'] > 0:
                affinity_score = max(affinity_score, features['name_similarity'])

            # Set the affinity score (symmetric matrix)
            affinity_matrix.loc[col1, col2] = affinity_score
            affinity_matrix.loc[col2, col1] = affinity_score

    # Fill diagonal with 1.0 (maximum affinity with self)
    for col in dimension_columns:
        affinity_matrix.loc[col, col] = 1.0

    return affinity_matrix


def process_pivot_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process pivot samples to get input tables and pivot parameters.

    This function extracts the ground truth from Jupyter notebooks by:
    1. Identifying which columns were used as index in the pivot
    2. Identifying which columns were used as header in the pivot
    3. Extracting the columns used for values and the aggregation function

    Args:
        samples: List of raw samples loaded from the data directory.

    Returns:
        List of processed samples with input_table and pivot parameters identified.
    """
    processed_samples = []

    print(f"Processing {len(samples)} pivot samples...")

    for sample_idx, sample in enumerate(samples):
        try:
            # Skip samples without necessary data
            if 'input_table' not in sample or 'params' not in sample:
                # print(f"Sample {sample_idx} missing input_table or params")
                continue

            input_table = sample['input_table']
            params = sample['params']

            # Skip if table is empty
            if input_table.empty:
                # print(f"Sample {sample_idx} has empty input table")
                continue

            # Extract pivot parameters
            index_columns = []
            header_columns = []
            values_columns = []
            aggfunc = 'mean'  # Default aggregation function

            # Check different parameter names that could contain index columns
            if 'index' in params:
                index_param = params['index']
                if isinstance(index_param, list):
                    index_columns = index_param
                elif isinstance(index_param, str):
                    index_columns = [index_param]
            elif 'rows' in params:
                rows_param = params['rows']
                if isinstance(rows_param, list):
                    index_columns = rows_param
                elif isinstance(rows_param, str):
                    index_columns = [rows_param]

            # Check different parameter names that could contain header columns
            # IMPORTANT: First check for 'column' (singular) which is used in your samples
            if 'column' in params:
                column_param = params['column']
                if isinstance(column_param, list):
                    header_columns = column_param
                elif isinstance(column_param, str):
                    header_columns = [column_param]
            elif 'columns' in params:
                columns_param = params['columns']
                if isinstance(columns_param, list):
                    header_columns = columns_param
                elif isinstance(columns_param, str):
                    header_columns = [columns_param]
            elif 'cols' in params:
                cols_param = params['cols']
                if isinstance(cols_param, list):
                    header_columns = cols_param
                elif isinstance(cols_param, str):
                    header_columns = [cols_param]
            elif 'header' in params:
                header_param = params['header']
                if isinstance(header_param, list):
                    header_columns = header_param
                elif isinstance(header_param, str):
                    header_columns = [header_param]

            # Extract values columns
            if 'values' in params:
                values_param = params['values']
                if isinstance(values_param, list):
                    values_columns = values_param
                elif isinstance(values_param, str):
                    values_columns = [values_param]
            elif 'value' in params:
                value_param = params['value']
                if isinstance(value_param, list):
                    values_columns = value_param
                elif isinstance(value_param, str):
                    values_columns = [value_param]

            # Extract aggregation function
            if 'aggfunc' in params:
                aggfunc = params['aggfunc']
            elif 'agg' in params:
                aggfunc = params['agg']
            elif 'function' in params:
                aggfunc = params['function']

            # Debug print for visibility - comment these out to reduce verbosity
            # Uncomment for detailed debugging when needed
            """
            print(f"Sample {sample_idx}:")
            print(f"  - Input table shape: {input_table.shape}")
            print(f"  - Raw params: {params}")
            print(f"  - Extracted index_columns: {index_columns}")
            print(f"  - Extracted header_columns: {header_columns}")
            print(f"  - Extracted values_columns: {values_columns}")
            """

            # Check if columns are valid in the table
            valid_columns = list(input_table.columns)

            valid_index_columns = [col for col in index_columns if col in valid_columns]
            if len(valid_index_columns) != len(index_columns):
                # print(f"  Warning: Some index columns don't exist in table: {set(index_columns) - set(valid_index_columns)}")
                pass

            valid_header_columns = [col for col in header_columns if col in valid_columns]
            if len(valid_header_columns) != len(header_columns):
                # print(f"  Warning: Some header columns don't exist in table: {set(header_columns) - set(valid_header_columns)}")
                pass

            valid_values_columns = [col for col in values_columns if col in valid_columns]
            if len(valid_values_columns) != len(values_columns):
                # print(f"  Warning: Some values columns don't exist in table: {set(values_columns) - set(valid_values_columns)}")
                pass

            # Only add sample if it has valid pivot parameters
            # We need at least one column for index AND one for header
            if valid_index_columns and valid_header_columns:
                processed_samples.append({
                    'input_table': input_table,
                    'index_columns': valid_index_columns,
                    'header_columns': valid_header_columns,
                    'values_columns': valid_values_columns,
                    'aggfunc': aggfunc,
                    'sample_id': sample.get('sample_id', f"sample_{sample_idx}")
                })
                # print(f"  Added valid sample {sample_idx}")
            else:
                # print(f"  Skipping sample {sample_idx}: Missing valid index or header columns")
                pass

        except Exception as e:
            import traceback
            # print(f"Error processing sample {sample_idx}:")
            # traceback.print_exc()
            pass

    print(f"Processed {len(processed_samples)} valid pivot samples out of {len(samples)} total")
    return processed_samples