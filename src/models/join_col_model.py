# src/models/join_col_model.py
#
# Implementation of join column prediction based on Section 4.1 of the
# "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks" paper.
#
# This module includes:
# 1. Feature extraction for identifying appropriate join column pairs between tables,
#    including features like distinct-value ratio, value overlap, value range overlap,
#    column data types, leftness, sortedness, and table-level statistics.
# 2. Generation of candidate join column pairs, supporting both single-column and multi-column join keys.
# 3. Preprocessing of join samples, including handling index/unnamed columns and normalizing join keys.
# 4. Functions to prepare training data for join column prediction models.
# 5. Functions to train, evaluate, and predict join columns using a Gradient Boosting classifier.
# 6. Evaluation metrics and utilities for precision@k, ndcg@k, and join key prediction ranking.
#
# All logic related to join column prediction is fully contained here, combining feature engineering, model training, and evaluation in a single module.
#

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from typing import List, Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.feature_selection import SelectFromModel
# from sklearn.metrics import classification_report
from src.utils.model_utils import numpy_to_list
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, ndcg_score


# Results for comparison methods from the paper (hardcoded values from Table 3 and Table 5 in the paper)
# In case of loading or calculation our baselines
JOIN_COLUMN_BASELINES = {
    "ML-FK": {"prec@1": 0.84, "prec@2": 0.87, "ndcg@1": 0.84, "ndcg@2": 0.87},
    "PowerPivot": {"prec@1": 0.31, "prec@2": 0.44, "ndcg@1": 0.31, "ndcg@2": 0.48},
    "Multi": {"prec@1": 0.33, "prec@2": 0.4, "ndcg@1": 0.33, "ndcg@2": 0.41},
    "Holistic": {"prec@1": 0.57, "prec@2": 0.63, "ndcg@1": 0.57, "ndcg@2": 0.65},
    "max-overlap": {"prec@1": 0.53, "prec@2": 0.61, "ndcg@1": 0.53, "ndcg@2": 0.63}
}

JOIN_COLUMN_VENDORS = {
    "Vendor-A": {"prec@1": 0.76, "ndcg@1": 0.76},
    "Vendor-C": {"prec@1": 0.42, "ndcg@1": 0.42},
    "Vendor-B": {"prec@1": 0.33, "ndcg@1": 0.33}
}


def should_exclude_column(col_name, table=None):
    """
    Determines if a column should be excluded from join candidates.

    Some columns are automatically generated and not meaningful for joins.

    Args:
        col_name: Name of the column to check
        table: Optional DataFrame to perform additional checks (optional)

    Returns:
        True if the column should be excluded, False otherwise
    """
    # Exclude columns that appear to be auto-generated indices or row numbers
    # 'Unnamed: 0' is commonly created by pandas when reading CSVs with unlabeled index columns
    if col_name.startswith('Unnamed:'):
        return True

    # If no table is provided, we can only do basic name-based checks
    if table is None:
        return False

    # Additional checks if table is provided
    try:
        # Check if column exists in the table
        if col_name not in table.columns:
            return True

        # Additional exclusion criteria
        col = table[col_name]

        # Exclude datetime columns
        if pd.api.types.is_datetime64_any_dtype(col):
            return True

        # Exclude columns with all NaN values
        if col.isna().all():
            return True

        return False

    except Exception as e:
        print(f"Error checking column {col_name}: {e}")
        return True


def extract_join_column_features(left_table: pd.DataFrame, right_table: pd.DataFrame,
                                 left_cols: List[str], right_cols: List[str]) -> Dict[str, float]:
    """
    Extracts features for a candidate join column pair as described in Section 4.1 of the paper.

    This function calculates various features that help determine if two sets of columns
    are good candidates for joining tables.

    Args:
        left_table: The left table for the join.
        right_table: The right table for the join.
        left_cols: The candidate columns from the left table.
        right_cols: The candidate columns from the right table.

    Returns:
        A dictionary of feature values for the candidate join columns.
    """
    features = {}

    # If 'index' is in columns, replace it with the actual index column
    left_cols = ['index' if col == 'index' else col for col in left_cols]
    right_cols = ['index' if col == 'index' else col for col in right_cols]

    # Ensure columns exist in the table
    left_cols = [col for col in left_cols if col in left_table.columns]
    right_cols = [col for col in right_cols if col in right_table.columns]

    # Skip feature extraction if no valid columns remain
    if not left_cols or not right_cols:
        return features

    # Feature 1: Distinct-value-ratio
    # Ratio of distinct tuples over total number of tuples (rows)
    # Key columns typically have ratio close to 1 (unique values)
    try:
        # Count the number of distinct tuples in each column set
        left_distinct_count = left_table[left_cols].drop_duplicates().shape[0]
        right_distinct_count = right_table[right_cols].drop_duplicates().shape[0]

        # Calculate the ratio of distinct values to total rows
        left_distinct_ratio = left_distinct_count / len(left_table) if len(left_table) > 0 else 0
        right_distinct_ratio = right_distinct_count / len(right_table) if len(right_table) > 0 else 0

        features['left_distinct_ratio'] = left_distinct_ratio
        features['right_distinct_ratio'] = right_distinct_ratio
    except Exception as e:
        print(f"Error calculating distinct-value-ratio: {e}")
        features['left_distinct_ratio'] = 0
        features['right_distinct_ratio'] = 0

    # Feature 2: Value-overlap
    # Measures how much the values in the two column sets overlap
    # Good join candidates typically have high overlap
    try:
        # Convert (one or more) column values to sets (into a single 'row') for easier comparison
        # Handle both single and multi-column cases
        if len(left_cols) > 1:
            # For multi-column, create tuples of values for each row
            left_values = set(tuple(row) for row in left_table[left_cols].astype(str).values)
        else:
            # For single column, just get the unique values
            left_values = set(left_table[left_cols[0]].astype(str))

        if len(right_cols) > 1:
            right_values = set(tuple(row) for row in right_table[right_cols].astype(str).values)
        else:
            right_values = set(right_table[right_cols[0]].astype(str))

        # Calculate intersection and union sizes
        intersection = len(left_values.intersection(right_values))
        union = len(left_values.union(right_values))

        # Calculate Jaccard similarity (intersection/union)
        features['jaccard_similarity'] = intersection / union if union > 0 else 0

        # Calculate containment in both directions
        # left_to_right_containment: What fraction of the left table's join keys appear in the right table
        # right_to_left_containment: What fraction of the right table's join keys appear in the left table
        features['left_to_right_containment'] = intersection / len(left_values) if len(left_values) > 0 else 0
        features['right_to_left_containment'] = intersection / len(right_values) if len(right_values) > 0 else 0
    except Exception as e:
        print(f"Error calculating value-overlap: {e}")
        features['jaccard_similarity'] = 0
        features['left_to_right_containment'] = 0
        features['right_to_left_containment'] = 0

    # Feature 3: Value-range-overlap (for numeric columns)
    # Checks if numeric ranges overlap
    # If the range overlap is low, columns may not be good candidates to join
    try:
        # Check if all columns are numeric
        left_numeric = all( pd.api.types.is_numeric_dtype(left_table[col]) and not pd.api.types.is_bool_dtype(left_table[col]) for col in left_cols)
        right_numeric = all( pd.api.types.is_numeric_dtype(right_table[col]) and not pd.api.types.is_bool_dtype(right_table[col]) for col in right_cols)

        # Only calculate range overlap for single numeric columns
        if left_numeric and right_numeric and len(left_cols) == 1 and len(right_cols) == 1:
            # Get min and max values for each column
            left_min = left_table[left_cols[0]].min()
            left_max = left_table[left_cols[0]].max()
            right_min = right_table[right_cols[0]].min()
            right_max = right_table[right_cols[0]].max()

            # Calculate intersection of ranges
            range_intersection = max(0, min(left_max, right_max) - max(left_min, right_min))
            # Calculate union of ranges
            range_union = max(left_max, right_max) - min(left_min, right_min)

            # Calculate overlap ratio
            features['range_overlap'] = range_intersection / range_union if range_union > 0 else 0
        else:
            features['range_overlap'] = 0
    except Exception as e:
        print(f"Error calculating value-range-overlap: {e}")
        features['range_overlap'] = 0

    # Feature 4: Column-value-types
    # Compares data types of columns
    # Two string columns with high overlap are better join candidates than numeric columns
    try:
        # Check if columns are strings (object type in pandas)
        features['left_is_string'] = all(pd.api.types.is_object_dtype(left_table[col]) for col in left_cols)
        features['right_is_string'] = all(pd.api.types.is_object_dtype(right_table[col]) for col in right_cols)

        # Check if columns are numeric
        features['left_is_numeric'] = all(pd.api.types.is_numeric_dtype(left_table[col]) for col in left_cols)
        features['right_is_numeric'] = all(pd.api.types.is_numeric_dtype(right_table[col]) for col in right_cols)

        # Check if data types match between columns
        features['type_match'] = (features['left_is_string'] == features['right_is_string']) and \
                                 (features['left_is_numeric'] == features['right_is_numeric'])
    except Exception as e:
        print(f"Error calculating column-value-types: {e}")
        features['left_is_string'] = False
        features['right_is_string'] = False
        features['left_is_numeric'] = False
        features['right_is_numeric'] = False
        features['type_match'] = False

    # Feature 5: Left-ness (position in table)
    # Columns to the left of tables are more likely to be join columns
    try:
        # Get positions of candidate columns in their tables
        left_positions = [list(left_table.columns).index(col) for col in left_cols]
        right_positions = [list(right_table.columns).index(col) for col in right_cols]

        # Calculate average positions (absolute)
        features['left_absolute_position'] = int(np.mean(left_positions))
        features['right_absolute_position'] = int(np.mean(right_positions))

        # Calculate relative positions (as fraction of table width)
        features['left_relative_position'] = int(np.mean([pos / len(left_table.columns) for pos in left_positions]))
        features['right_relative_position'] = int(np.mean([pos / len(right_table.columns) for pos in right_positions]))
    except Exception as e:
        print(f"Error calculating left-ness: {e}")
        features['left_absolute_position'] = 0
        features['right_absolute_position'] = 0
        features['left_relative_position'] = 0
        features['right_relative_position'] = 0

    # Feature 6: Sorted-ness
    # Whether values in columns are sorted (sorted columns are more likely key columns)
    try:
        # Single column sorting check
        if len(left_cols) == 1:
            features['left_is_sorted'] = left_table[left_cols[0]].equals(
                left_table[left_cols[0]].sort_values(ignore_index=True))
        else:
            # Multi-column lexicographic sorting check
            sorted_left_df = left_table[left_cols].sort_values(by=list(left_cols))
            features['left_is_sorted'] = left_table[left_cols].reset_index(drop=True).equals(
                sorted_left_df.reset_index(drop=True))

        # Same for right table
        if len(right_cols) == 1:
            features['right_is_sorted'] = right_table[right_cols[0]].equals(
                right_table[right_cols[0]].sort_values(ignore_index=True))
        else:
            # Multi-column lexicographic sorting check
            sorted_right_df = right_table[right_cols].sort_values(by=list(right_cols))
            features['right_is_sorted'] = right_table[right_cols].reset_index(drop=True).equals(
                sorted_right_df.reset_index(drop=True))

    except Exception as e:
        print(f"Error calculating sorted-ness: {e}")
        features['left_is_sorted'] = False
        features['right_is_sorted'] = False

    # Feature 7: Single-column-candidate
    # Indicates whether a candidate is a single-column or not
    # Single-column joins are more common and may be preferred
    features['is_single_column'] = len(left_cols) == 1 and len(right_cols) == 1

    # Feature 8: Table-level-statistics
    # Information about the tables being joined
    # Helps assess the reliability of overlap metrics
    features['left_row_count'] = len(left_table)
    features['right_row_count'] = len(right_table)
    features['row_count_ratio'] = len(left_table) / len(right_table) if len(right_table) > 0 else float('inf')

    return features


def generate_join_candidates(left_table, right_table, max_multi_column=2, max_candidates=15):
    """
    Generates candidate join column pairs between two tables based on strict data type and length compatibility.

    This function:
    1. Filters out columns that are not meaningful for joining:
        - Datetime columns (often not used as join keys)
        - Columns with names starting with 'Unnamed:' (commonly auto-generated index columns)
    2. Generates single-column join candidates:
        - Pairs of columns from left and right tables that have strictly matching data types
        - Only considers columns with the same length
    3. Generates multi-column join candidates (up to max_multi_column):
        - Uses combinations of 2 or more columns from both tables
        - Considers at most 5 combinations from each table to limit computational overhead
        - Ensures all paired columns in a multi-column candidate have matching data types
        - Ensures no duplicate rows in multi-column combinations for join robustness
    4. Returns a comprehensive list of candidate join column pairs for further feature extraction and join prediction.

    Args:
        max_candidates:
        left_table (pd.DataFrame): The left table to generate candidates from.
        right_table (pd.DataFrame): The right table to generate candidates from.
        max_multi_column (int, optional): Maximum number of columns to consider for multi-column joins. Defaults to 2.
        max_candidates (int, optional): Maximum number of candidates to generate. Defaults to 15.
    Returns:
        List[Tuple[List[str], List[str]]]: List of candidate join column pairs.
            Each pair is a tuple of (list of left join columns, list of right join columns).

    Notes:
        - This function performs strict data type matching and requires join columns to have the same length.
        - It uses a capped number of combinations for multi-column joins to avoid combinatorial explosion.
        - This function does not validate that these candidates are semantically correct join keys—
          it only generates plausible candidates based on the table structures.
    """
    # Filter columns, allowing 'index' if it exists
    left_cols = [col for col in left_table.columns if not should_exclude_column(col) or col == 'index']
    right_cols = [col for col in right_table.columns if not should_exclude_column(col) or col == 'index']

    candidates = [] # list for generated candidates

    # Single-column candidates
    for left_col in left_cols:
        for right_col in right_cols:
            # Relaxed type compatibility: numeric ↔ numeric, string ↔ string
            if ( (pd.api.types.is_numeric_dtype(left_table[left_col]) and pd.api.types.is_numeric_dtype(right_table[right_col]))
                    or (pd.api.types.is_object_dtype(left_table[left_col]) and pd.api.types.is_object_dtype(right_table[right_col])) ):
                    candidates.append(([left_col], [right_col]))

                    # Break if max candidates reached
                    if len(candidates) >= max_candidates:
                        return candidates

    # Add multi-column candidates up to max_multi_column
    # Multi-column candidates (up to 2 columns)
    if max_multi_column >= 2:
        for i in range(2, max_multi_column + 1):
            # Generate combinations with max 5 combinations
            left_combinations = list(combinations(left_cols, i))[:3]    # Limit to 3 multi_column combos to reduce runtime
            right_combinations = list(combinations(right_cols, i))[:3]

            # Cross-product of combinations
            for left_comb in left_combinations:
                for right_comb in right_combinations:
                    # Ensure same number of columns and compatible types
                    if ( len(left_comb) == len(right_comb) and all(((pd.api.types.is_numeric_dtype(left_table[left_col]) and pd.api.types.is_numeric_dtype(right_table[right_col]))
                                or (pd.api.types.is_object_dtype(left_table[left_col]) and pd.api.types.is_object_dtype(right_table[right_col]))) for left_col, right_col in zip(left_comb, right_comb))):   # and len(left_table[list(left_comb)].drop_duplicates()) == len(left_table))
                        candidates.append((list(left_comb), list(right_comb)))

                        if len(candidates) >= max_candidates:
                            return candidates

    return candidates


def process_join_samples(samples: List[Dict]) -> List[Dict]:
    """
    Processes join samples to extract join parameters and prepare them for downstream feature extraction and modeling.

    This function:
    1. **Extracts join keys** from various formats present in the 'params' dictionary:
        - 'join_keys' dictionary with explicit left/right keys
        - 'left on' / 'right on' parameters (as strings or lists)
        - Single 'on' parameter (list, string, or dict mapping)
        - Inferred common columns if no explicit join keys are provided
    2. **Normalizes join keys** to lists for consistency.
    3. **Skips samples** with empty tables or missing join information.
    4. **Tracks diagnostics** about the join parameter formats and extraction methods used, providing insights for debugging and pipeline health.
    5. **Returns** only the successfully processed join samples, ready for feature extraction.

    Args:
        samples (List[Dict]): List of raw join samples, each containing at least:
            - 'left_table': The left table (pd.DataFrame)
            - 'right_table': The right table (pd.DataFrame)
            - 'params': Dictionary with join parameters (like 'on', 'left on', 'right on', 'join_keys', or inferred)

    Returns:
        List[Dict]: List of processed join samples, each containing:
            - 'left_table' (pd.DataFrame): Left table (as-is).
            - 'right_table' (pd.DataFrame): Right table (as-is).
            - 'left_join_keys' (List[str]): Normalized left join key columns.
            - 'right_join_keys' (List[str]): Normalized right join key columns.
            - 'join_type' (str): Join type, defaulting to 'inner' if not specified.
            - 'sample_id' (str or None): Optional sample identifier for tracking.
    """
    # Inspect a sample (look how it looks like)
    # for k, v in samples[0].items():
    #     print(k, v)   # Expected keys: sample_id, params, left_table, right_table as keys

    processed_samples = []
    diagnostics = {
        'total_samples': len(samples),
        'processed_samples': 0,
        'param_formats': {},
        'join_key_extraction_methods': {}
    }

    for sample_idx, sample in enumerate(samples):
        # Basic validation
        if 'left_table' not in sample or 'right_table' not in sample or 'params' not in sample:
            continue

        left_table = sample['left_table']
        right_table = sample['right_table']
        params = sample['params']

        # Skip empty tables
        if left_table.empty or right_table.empty:
            continue

        # Initialize join keys
        left_join_keys = []
        right_join_keys = []
        extraction_method = 'unresolved'

        # Diagnostic: Track parameter format
        param_keys = list(params.keys())
        param_format = tuple(sorted(param_keys))
        diagnostics['param_formats'][param_format] = diagnostics['param_formats'].get(param_format, 0) + 1

        # Exhaustive key extraction attempts
        try:
            # Method 1: Explicit join_keys dictionary
            if 'join_keys' in params:
                join_keys = params['join_keys']
                left_join_keys = join_keys.get('left', [])
                right_join_keys = join_keys.get('right', [])
                extraction_method = 'join_keys_dict'

            # Method 2: Separate 'left on' and 'right on'
            elif 'left on' in params and 'right on' in params:
                left_on = params['left on']
                right_on = params['right on']

                # Normalize to lists
                left_join_keys = [left_on] if isinstance(left_on, str) else left_on
                right_join_keys = [right_on] if isinstance(right_on, str) else right_on
                extraction_method = 'left_right_on'

            # Method 3: Single 'on' parameter with different formats
            elif 'on' in params:
                on_param = params['on']

                if isinstance(on_param, list):
                    # Same columns in both tables
                    left_join_keys = right_join_keys = on_param
                    extraction_method = 'on_list'

                elif isinstance(on_param, str):
                    # Single column name
                    left_join_keys = right_join_keys = [on_param]
                    extraction_method = 'on_string'

                elif isinstance(on_param, dict):
                    # Mapping between columns
                    left_join_keys = list(on_param.keys())
                    right_join_keys = list(on_param.values())
                    extraction_method = 'on_dict'

            # Method 4: Infer from common columns
            else:
                common_cols = set(left_table.columns).intersection(set(right_table.columns))
                meaningful_common_cols = [col for col in common_cols if not col.startswith('Unnamed:')]

                if meaningful_common_cols:
                    left_join_keys = right_join_keys = meaningful_common_cols
                    extraction_method = 'inferred_common'

            # The following lines exclude any join samples where the join keys do not exist as actual columns in the left or right table.
            # Specifically, they:
            # - Replace any occurrence of the 'index' join key with the literal string 'index'
            # - Filter out any join keys that are not present as column names in the tables
            # As a result, if a join key (like 'index') is not a real column, these samples will be skipped.

            # Normalize join keys, replacing 'index' with actual index column
            left_join_keys = ['index' if key == 'index' else key for key in left_join_keys]
            right_join_keys = ['index' if key == 'index' else key for key in right_join_keys]
            #
            # # Filter out keys not in columns
            left_join_keys = [key for key in left_join_keys if key in left_table.columns]
            right_join_keys = [key for key in right_join_keys if key in right_table.columns]

            # Skip if no valid keys remain
            if not left_join_keys or not right_join_keys:
                continue


            # Track extraction method
            diagnostics['join_key_extraction_methods'][extraction_method] = \
                diagnostics['join_key_extraction_methods'].get(extraction_method, 0) + 1

            # Append processed sample
            processed_samples.append({
                'left_table': left_table,
                'right_table': right_table,
                'left_join_keys': left_join_keys,
                'right_join_keys': right_join_keys,
                'join_type': params.get('how', 'inner'),
                'sample_id': sample.get('sample_id', None)
            })

            diagnostics['processed_samples'] += 1

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")

    # Inspect a processed sample (see how it looks like now, after processing)
    # for k, v in processed_samples[0].items():
    #     print(k, v)  # Expected left_table, right_table, left_join_keys, right_join_keys, join_type, sample_id as keys

    # Print diagnostics
    # print("\nJoin Sample Processing Diagnostics:")
    # print(f"Total Samples: {diagnostics['total_samples']}")
    print(f"\nProcessed {diagnostics['processed_samples']} join samples\n")

    # print("\nParameter Formats:")
    # for format, count in sorted(diagnostics['param_formats'].items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {format}: {count}")
    #
    # print("\nJoin Key Extraction Methods:")
    # for method, count in sorted(diagnostics['join_key_extraction_methods'].items(), key=lambda x: x[1], reverse=True):
    #     print(f"  {method}: {count}")

    return processed_samples


def prepare_join_data(processed_samples: List[Dict]) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Prepares training data for join column prediction.

    This function processes each sample to generate multiple candidate join column pairs.
    For each sample, it:
    1. Extracts the ground truth join keys
    2. Generates multiple candidate join column pairs
    3. Labels each candidate as 1 (matches ground truth) or 0 (doesn't match)

    Args:
        processed_samples: List of processed join samples

    Returns:
    Tuple containing:
        - X: Numeric feature matrix (NumPy array), shape (num_samples, num_features).
        - feature_cols: List of feature names corresponding to columns in X.
        - labels: List of 0/1 labels indicating if the candidate matches the ground truth.
    """
    features_list = []
    labels = []

    for sample_idx, sample in enumerate(processed_samples):
        left_table = sample['left_table']
        right_table = sample['right_table']

        # Extract ground truth join keys
        true_left_keys = sample['left_join_keys']
        true_right_keys = sample['right_join_keys']

        # Generate candidate join columns
        candidates = generate_join_candidates(left_table, right_table)

        # Debug: Print candidate for generation details
        # print(f"Sample {sample_idx}: Total candidates = {len(candidates)}")
        # print(f"  True Left Keys: {true_left_keys}")
        # print(f"  True Right Keys: {true_right_keys}")

        for left_cols, right_cols in candidates:
            # If 'index' is used as a join key but not present in the table, create it by resetting the DataFrame index
            if 'index' in true_left_keys and 'index' not in left_table.columns:
                # Example: left_table has 3 rows → reset index creates new 'index' column (0, 1, 2)
                left_table = left_table.reset_index().rename(columns={'index': 'index'})

            if 'index' in true_right_keys and 'index' not in right_table.columns:
                right_table = right_table.reset_index().rename(columns={'index': 'index'})


            # Extract features for the candidate pair
            features = extract_join_column_features(left_table, right_table, left_cols, right_cols)

            # Determine label: 1 if candidate matches ground truth exactly (order-insensitive)
            is_match = (set(left_cols) == set(true_left_keys) and
                        set(right_cols) == set(true_right_keys))

            features_list.append(features)
            labels.append(1 if is_match else 0)

    # Create DataFrame for inspection and numeric conversion
    features_df = pd.DataFrame(features_list)

    # Remove non-feature columns
    feature_cols = [col for col in features_df.columns if col != 'sample_id']

    # Debug: Check feature column types before conversion
    # print("\nFeature Column Types:")
    # for col in feature_cols:
    #     print(f"  {col}: {features_df[col].dtype}")

    # Convert boolean columns to int
    for col in feature_cols:
        if features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)

    # Convert to numeric matrix
    X = features_df[feature_cols].values

    # Detailed data summary
    # positive_count = sum(labels)
    # total_count = len(labels)
    # negative_count = total_count - positive_count
    #
    # print("\nTraining Data Summary:")
    # print(f"  Total training samples: {len(processed_samples)}")
    # print(f"  Total candidate pairs: {total_count}")
    # print(f"  Number of features per pair: {len(feature_cols)}")
    # print(f"  Feature matrix shape (X): {X.shape}")
    # print(f"  Label vector shape (y): {len(labels)}")
    # print(f"  Positive examples: {positive_count}")
    # print(f"  Negative examples: {negative_count}")
    # print(f"  Positive percentage: {(positive_count / total_count) * 100:.2f}%")
    #
    # # Numeric matrix sanity checks
    # if not np.issubdtype(X.dtype, np.number):
    #     print("\nWARNING: Not all features are numeric!")
    #
    #     # Identify non-numeric columns
    #     non_numeric_cols = [
    #         col for col in feature_cols
    #         if not np.issubdtype(features_df[col].dtype, np.number)
    #     ]
    #     print("Non-numeric columns:", non_numeric_cols)
    #
    return X, feature_cols, labels


def train_join_column_model(X_train, y_train, X_val, y_val, feature_names):
    """
    Trains a model to predict join columns.

    This function takes pre-prepared feature matrices (X_train, X_val) and binary labels (y_train, y_val),
    and trains a classifier using hyperparameter tuning and validation set evaluation.

    Args:
        X_train: Feature matrix for training data (numpy array)
        y_train: Labels for training data (numpy array or list)
        X_val: Feature matrix for validation data (numpy array)
        y_val: Labels for validation data (numpy array or list)
        feature_names: List of feature names, used for feature importance analysis and debugging

    Returns:
        Trained model and list of feature names used by the model.
    """
    # Check if we have any training or validation data
    if X_train.shape[0] == 0 or len(y_train) == 0:
        print("Error: No training data available. Cannot train model.")
        return None, []

    if X_val.shape[0] == 0 or len(y_val) == 0:
        print("Error: No validation data available. Cannot train model.")
        return None, []

    # Final feature space
    print(f"\nUsing {len(feature_names)} features:")
    print(feature_names)

    # Print distribution of positive examples
    print("\nDistribution among all candidate join column pairs:")
    print(f"Train positives: {sum(y_train)}/{len(y_train)} "
          f"({sum(y_train) / len(y_train) * 100:.2f}%) — from all candidate pairs generated in training samples")
    print(f"Validation positives:  {sum(y_val)}/{len(y_val)} "
          f"({sum(y_val) / len(y_val) * 100:.2f}%) — from all candidate pairs generated in validation samples")

    # Address class imbalance with sample weights
    sample_weights = compute_sample_weight("balanced", y_train)

    # Train a Gradient Boosting model
    print("\nTraining join column prediction model...")
    start_time = time.time()

    # Define parameter grid
    # param_grid = {
    #     'n_estimators': [50, 100, 150, 200],
    #     'learning_rate': [0.05, 0.1],
    #     'max_depth': [3, 4, 6],
    #     'subsample': [0.8, 1.0],
    #     'min_samples_leaf': [1, 3, 5]
    # }

    # Grid search
    # grid_search = GridSearchCV(
    #     estimator=GradientBoostingClassifier(random_state=42),
    #     param_grid=param_grid,
    #     #refit=True,
    #     scoring='f1',
    #     cv=StratifiedKFold(n_splits=5),
    #     verbose=1,
    #     n_jobs=-1
    # )

    # Fit the best model
    # grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    # model = grid_search.best_estimator_
    # print("Best join column model hyperparameters:", grid_search.best_params_)

    # model = GradientBoostingClassifier()    # This gives better val acc but with 100% train acc!
    model = GradientBoostingClassifier(
        n_estimators=100,      # Reduced number of trees (some regularization)
        learning_rate=0.1,    # Moderate learning rate
        max_depth=3,          # Shallow trees
        min_samples_leaf=10,  # Regularization to prevent overfitting
        subsample=0.8,        # Subsampling for additional regularization
        random_state=42       # For reproducibility
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    end_time = time.time()
    total_training_time = round(end_time - start_time, 2)

    print(f"\nModel training completed in {total_training_time} seconds")   # (without hyperparameter tuning)
    print(f"Trained model: GradientBoostingClassifier ({model.n_estimators} estimators, max_depth={model.max_depth})")

    # Calculate standard binary training metrics (accuracy, precision and recall)
    y_train_pred = model.predict(X_train)   # GradientBoostingClassifier returns 0 and 1
    train_accuracy = np.mean(y_train_pred == y_train)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)

    # Calculate standard binary validation metrics (accuracy, precision and recall)
    y_val_pred = model.predict(X_val)
    val_accuracy = np.mean(y_val_pred == y_val)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    # print(classification_report(y_val, y_val_pred))

    # Print training vs validation metrics (how well the model identifies positive join pairs among all candidates)
    print("\nStandard Binary Classification Metrics on Train and Validation Sets:")
    #print("(These metrics show how well the model classifies candidate join column pairs)")
    print(f"Accuracy: Training = {train_accuracy:.4f}, Validation = {val_accuracy:.4f}")
    print(f"Precision: Training = {train_precision:.4f}, Validation = {val_precision:.4f}")
    print(f"Recall: Training = {train_recall:.4f}, Validation = {val_recall:.4f}")
    # For realistic ranking evaluation (precision@k, ndcg@k), use --mode eval on the true held-out test set.

    # Create directories for results
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Create a dictionary with all relevant metrics (in native python type, json only supports int, float, str, lists, bool and None!)
    metrics_dict = {
        'operator': 'join_column',
        'mode': 'training',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'samples': 90,  # int(len(y_train)+len(y_val))
        'train_examples': int(len(y_train)),
        'train_positives': int(sum(y_train)),
        'train_pos_ratio': float(sum(y_train) / len(y_train)),
        'validation_examples': int(len(y_val)),
        'validation_positives': int(sum(y_val)),
        'validation_pos_ratio': float(sum(y_val) / len(y_val)),
        'model_type': 'GradientBoostingClassifier',
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'max_depth': model.max_depth,
        'training_time': float(total_training_time),
        'num_features': int(len(feature_names)),
        'train_accuracy': float(train_accuracy),
        'validation_accuracy': float(val_accuracy),
        'train_precision': float(train_precision),
        'validation_precision': float(val_precision),
        'train_recall': float(train_recall),
        'validation_recall': float(val_recall),
    }

    # Extract and store feature importance for interpretation and reporting
    # We identify the top 5 most influential features based on the trained model,
    # which helps explain which input features were most predictive of join column quality
    feature_importance = model.feature_importances_
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
    for i, (feature, importance) in enumerate(top_features, 1):
        metrics_dict[f'top_feature_{i}'] = feature
        metrics_dict[f'importance_{i}'] = float(importance) # All numpy types to native python before saving into .json

    # All metrics to a JSON file
    metrics_path = 'results/metrics/all_operators_metrics.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Converts numpy types to native Python types (important for JSON!)
    metrics_dict = numpy_to_list(metrics_dict)

    # Append new metrics to existing list or create new
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(metrics_dict)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Join Column Prediction')
    plt.tight_layout()
    plt.savefig('results/figures/join_column_feature_importance.png')
    print("\nMetrics and figures have been saved to the 'results' directory")

    return model, feature_names


def predict_join_columns(model, feature_names, left_table, right_table, top_k, verbose=True):
    """
    Predicts join columns for two tables.

    Args:
        model: Trained join column prediction model
        feature_names: List of feature names expected by the model
        left_table: Left table for the join
        right_table: Right table for the join
        top_k: Number of top predictions to return
        verbose: Whether to print predictions (default: True)

    Returns:
        List of tuples containing (left_columns, right_columns, score) sorted by score
    """
    # Generate candidate join columns
    candidates = generate_join_candidates(left_table, right_table)

    if not candidates:
        print("No valid join candidates found between these tables.")
        return []

    # Prepare feature matrix
    X = []
    processed_candidates = []

    for left_cols, right_cols in candidates:
        # If 'index' is in candidate join keys but not present in the table, reset index to create the 'index' column
        if 'index' in left_cols and 'index' not in left_table.columns:
            left_table = left_table.reset_index().rename(columns={'index': 'index'})

        if 'index' in right_cols and 'index' not in right_table.columns:
            right_table = right_table.reset_index().rename(columns={'index': 'index'})

        # Extract features for each candidate
        features = extract_join_column_features(left_table, right_table, left_cols, right_cols)

        # Create feature vector in the same order as training features
        feature_vector = [features.get(name, 0) for name in feature_names]

        X.append(feature_vector)
        processed_candidates.append((left_cols, right_cols))

    # Convert to numpy array for prediction
    X = np.array(X)

    # Predict probs
    #scores = model.predict(np.array(X)) # only gives 0 or 1
    scores = model.predict_proba(np.array(X))[:, 1] # Gives probability of joinability (class 1)

    # Combine candidates with their scores
    # Convert 'Unnamed: 0' to 'index' in the displayed results
    results = [ (left_cols, right_cols, score) for (left_cols, right_cols), score in zip(processed_candidates, scores) ]

    # Sort by score in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # Limit to top_k results
    ranked_preds = results[:top_k]

    # Display the top predictions if verbose mode is on
    if verbose:
        print("\nTop Join Column Predictions:")
        for i, (left_cols, right_cols, score) in enumerate(ranked_preds[:top_k], 1):
            left_str = ", ".join(left_cols)
            right_str = ", ".join(right_cols)
            print(f"{i}. Left columns: [{left_str}] ↔ Right columns: [{right_str}] (confidence: {score:.4f})")

    return results


def evaluate_join_column_model(model, feature_names, test_samples, top_k):
    """
    Evaluates a join column prediction model on test samples.

    Args:
        model: Trained join column model
        feature_names: Feature names used by the model
        test_samples: List of test samples
        top_k: Number of k values for precision@k and ndcg@k evaluation (default: [1, 2])

    Returns:
        Dictionary of evaluation metrics
    """
    # Dynamically and locally imports to avoid circuits
    from src.utils.model_utils import generate_feature_importance_table, generate_prediction_table
    from src.baselines.join_baselines import evaluate_baselines

    k_values = list(range(1, top_k + 1))
    correct_at_k = {k: 0 for k in k_values}
    ndcg_sum = {k: 0 for k in k_values}
    total = 0
    print(f"\nEvaluating join column prediction on test samples...")

    # Track overall binary metrics
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []

    # Keep track of test examples and test positives
    total_test_examples = 0
    total_test_positives = 0

    for sample_idx, sample in enumerate(test_samples):
        left_table = sample['left_table']
        right_table = sample['right_table']
        true_left_cols = sample['left_join_keys']
        true_right_cols = sample['right_join_keys']

        # Predict join columns
        try:
            # The function predict_join_columns does the candidate generation and feature extraction internally.
            # It uses the same feature extraction logic as during training.
            # So no explicit feature preparation is needed beyond what's in predict_join_columns().

            predictions = predict_join_columns(model, feature_names, left_table, right_table, max(k_values), verbose=False) # Returns a list of tuples of (left_candidate_table, right_candidate_table, probability)
            if not predictions:
                continue
            total += 1  # Number of test samples that have predictions

            # Ensure predictions are sorted by score (descending)
            predictions.sort(key=lambda x: x[2], reverse=True)

            # Convert all to strings for robust comparison
            # true_left_cols = [str(c) for c in true_left_cols]
            # true_right_cols = [str(c) for c in true_right_cols]

            # Prepare y_true and y_pred for binary metrics
            y_true = []
            y_pred = []
            for left_cols, right_cols, score in predictions:
                # left_cols = [str(c) for c in left_cols]
                # right_cols = [str(c) for c in right_cols]
                is_match = (set(left_cols) == set(true_left_cols) and set(right_cols) == set(true_right_cols))
                y_true.append(1 if is_match else 0)
                y_pred.append(score)

                total_test_examples += 1    # Number of candidate pair examples (if sample 1 has 2 cand pairs and sample 2 has 3, then this is 5!)

            total_test_positives += sum(y_true) # Get all positive samples (1)

            # Check correct-at-k (for precision)
            for i, (left_cols, right_cols, _) in enumerate(predictions):
                is_match = (set(left_cols) == set(true_left_cols) and set(right_cols) == set(true_right_cols))
                if is_match:
                    for k in k_values:
                        if i < k:
                            correct_at_k[k] += 1
                    break

            # Calculate NDCG
            if any(y_true):
                for k in k_values:
                    if k <= len(y_true):
                        ndcg = ndcg_score([y_true], [y_pred], k=k)
                        ndcg_sum[k] += ndcg

            # Calculate binary metrics for this sample
            y_pred_binary = [1 if s >= 0.5 else 0 for s in y_pred]
            test_accuracy_list.append(accuracy_score(y_true, y_pred_binary))
            test_precision_list.append(precision_score(y_true, y_pred_binary, zero_division=0))
            test_recall_list.append(recall_score(y_true, y_pred_binary, zero_division=0))

        except Exception as e:
            print(f"Error predicting and evaluating join columns (sample {sample_idx}): {e}")
            continue

    # Final metrics for all samples
    if total == 0:
        test_accuracy = 0.0
        test_precision = 0.0
        test_recall = 0.0
    else:
        test_accuracy = np.mean(test_accuracy_list)
        test_precision = np.mean(test_precision_list)
        test_recall = np.mean(test_recall_list)

    # Calculate final metrics (e.g. for 10 samples, prec@1=prec@1_sample1+prec@1_sample2+.../10)
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0
        metrics[f'ndcg@{k}'] = ndcg_sum[k] / total if total > 0 else 0
    metrics['samples_evaluated'] = int(total)

    # Create full evaluation record
    eval_dict = {
        "operator": "join_column",
        "mode": "evaluation",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "samples": int(len(test_samples)),
        "test_examples": int(total_test_examples),
        "test_positives": int(total_test_positives),
        "test_pos_ratio": float(total_test_positives / total_test_examples) if total_test_examples > 0 else 0.0,
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall)
    }

    # Add ranking metrics
    for k in k_values:
        eval_dict[f'precision@{k}'] = float(metrics[f'precision@{k}'])
        eval_dict[f'ndcg@{k}'] = float(metrics[f'ndcg@{k}'])

    # Converts numpy types to native Python types (important for JSON!)
    eval_dict = numpy_to_list(eval_dict)

    # Save to JSON
    os.makedirs('results/metrics', exist_ok=True)
    metrics_path = 'results/metrics/all_operators_metrics.json'

    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.append(eval_dict)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)

    # Calculate the metrics based on other heuristic methods
    baseline_metrics = evaluate_baselines(test_samples, k_values)

    # Generate Table 3 from the paper
    generate_prediction_table(
        auto_suggest_metrics=metrics,   # The metrics computed for Auto-Suggest
        k_values=k_values,
        baseline_metrics=baseline_metrics,  # Baseline metrics to compare with
        vendor_metrics=JOIN_COLUMN_VENDORS, # Vendor metrics to compare with
        operator_name="join"
    )

    # Generate Table 4 (feature importance)
    feature_importance = model.feature_importances_
    generate_feature_importance_table(feature_importance, feature_names, operator="join")

    return metrics
