# src/features/join_features.py
#
# Implementation of feature extraction for join column prediction based on Section 4.1 of the
# "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Extracts features for identifying appropriate join columns between two tables
# 2. Includes features like distinct-value ratio, value overlap, value range overlap, column value types, leftness, shortness, single column candidate, table statistics
# 3. Generates candidate join column pairs between tables
# 4. Provides preprocessing functions for join samples

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations, product


def is_sorted(df: pd.DataFrame) -> bool:
    """
    Check if the given DataFrame columns are sorted.

    This function determines whether values in the DataFrame are sorted in ascending order.
    For single-column DataFrames, it checks if values are sorted. For multi-column DataFrames,
    it checks if the DataFrame is sorted by all columns (lexicographically).

    Sorted columns are a signal that they might be key columns and good join candidates.

    Example: df1 = pd.DataFrame({'A': [1, 2, 3, 4]}) # Returns: True (Single column sorted)
             df2 = pd.DataFrame({'A': [1, 3, 2, 4]}) # Returns: False (Single column unsorted)
             df3 = pd.DataFrame({ 'A': [1, 1, 2, 2], 'B': [1, 2, 1, 2] }) # Returns: (True Multi-column sorted, lexicographically)
             (Lexicographic sorting means rows are ordered as they would be in a dictionary, comparing first column → second column)

    Args: 
        df: DataFrame to check.

    Returns:
        True if the DataFrame is sorted by the given columns, False otherwise.

    Note:
        This function compares the DataFrame with a sorted version of itself,
        which means it creates a copy of the data. For very large DataFrames,
        this might be memory-intensive.
    """
    # For single column
    if df.shape[1] == 1:
        col = df.columns[0]
        return df[col].equals(df[col].sort_values(ignore_index=True))
    # For multiple columns
    else:
        sorted_df = df.sort_values(by=list(df.columns))
        return df.reset_index(drop=True).equals(sorted_df.reset_index(drop=True))


def is_strict_numeric(col):
    """
    Determine whether a pandas Series is strictly numeric (excluding boolean types).

    This function checks if a column is of a numeric type such as int, float, or datetime,
    but explicitly excludes boolean columns, which are technically numeric in pandas
    but not suitable for range-based calculations like min/max.

    Args:
        col (pd.Series): A column from a DataFrame to be checked.

    Returns:
        bool: True if the column is numeric and not boolean, False otherwise.
    """
    return pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)


def extract_join_column_features(left_table: pd.DataFrame, right_table: pd.DataFrame,
                                 left_cols: List[str], right_cols: List[str]) -> Dict[str, float]:
    """
    Extract features for a candidate join column pair as described in Section 4.1 of the paper.

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
        left_numeric = all(is_strict_numeric(left_table[col]) for col in left_cols)
        right_numeric = all(is_strict_numeric(right_table[col]) for col in right_cols)

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
        features['left_is_sorted'] = is_sorted(left_table[left_cols])
        features['right_is_sorted'] = is_sorted(right_table[right_cols])
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


def should_exclude_column(col_name):
    """
    Determine if a column should be excluded from join candidates.

    Some columns are automatically generated and not meaningful for joins (like the first column without a name in .csv files).

    Args:
        col_name: Name of the column to check

    Returns:
        True if the column should be excluded, False otherwise
    """
    # Exclude columns that appear to be auto-generated indices or row numbers
    # 'Unnamed: 0' is commonly created by pandas when reading CSVs with unlabeled index columns
    return col_name.startswith('Unnamed:')


def generate_join_candidates(left_table, right_table, sample_id=None, max_multi_column=2):
    """
    Generate candidate column pairs for join operations between two tables.

    This function identifies potential join columns based on data types and creates both single-column and multi-column join candidates. It includes:
    1. Pruning columns that should be excluded (like auto-generated indices)
    2. Creating single-column candidates with matching data types
    3. Creating multi-column candidates up to max_multi_column columns
    4. Limiting the number of combinations to avoid combinatorial explosion

    Example:
        Consider two simple tables:

        ```
        # Left table:
        left_df = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie']
        })

        # Right table:
        right_df = pd.DataFrame({
            'cust_id': [1, 2, 4],
            'order_date': ['2023-01-01', '2023-01-15', '2023-02-01']
        })

        candidates = generate_join_candidates(left_df, right_df)
        # Returns: [(['customer_id'], ['cust_id'])]
        ```

        The function correctly identifies customer_id and cust_id as potential join columns
        despite their different names, because they have the same data type and aren't excluded.

    Args:
        left_table: DataFrame representing the left table
        right_table: DataFrame representing the right table
        sample_id: Optional identifier for the sample
        max_multi_column: Maximum number of columns to consider in multi-column joins (default: 2)

    Returns:
        List of candidate column pairs, where each pair is a tuple of
        ([left_columns], [right_columns])

    Note:
        For multi-column joins, the function limits the number of combinations to avoid
        exponential growth. It considers at most 5 combinations from each table when
        max_multi_column > 1.
    """
    candidates = []

    # Filter out columns that should be excluded (like auto-generated indices)
    left_cols = [col for col in left_table.columns if not should_exclude_column(col)]
    right_cols = [col for col in right_table.columns if not should_exclude_column(col)]

    # Add single-column candidates (most common join scenario)
    # For each combination of columns from both tables
    for left_col in left_cols:
        for right_col in right_cols:
            # Skip obvious type mismatches to reduce candidate space
            if left_table[left_col].dtype != right_table[right_col].dtype:
                continue
            candidates.append(([left_col], [right_col]))

    # Add multi-column candidates up to max_multi_column
    if max_multi_column >= 2:
        # Get combinations of columns from each table
        for i in range(2, max_multi_column + 1):
            left_combinations = list(combinations(left_cols, i))
            right_combinations = list(combinations(right_cols, i))

            # Only consider a subset of combinations to avoid explosion
            max_combs = 5  # Limit number of combinations to try
            if len(left_combinations) > max_combs:
                left_combinations = left_combinations[:max_combs]
            if len(right_combinations) > max_combs:
                right_combinations = right_combinations[:max_combs]

            # For each possible combination of left and right columns
            for left_comb, right_comb in product(left_combinations, right_combinations):
                # Only consider pairs with the same number of columns
                if len(left_comb) == len(right_comb):
                    candidates.append((list(left_comb), list(right_comb)))

    return candidates


def process_join_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process join samples to extract join parameters and prepare for model training.

    Takes raw samples loaded from data files and extracts the tables and join columns.
    Handles various formats of join parameters found in the samples.

    Note on special column handling:
    - 'Unnamed' columns: These are typically auto-generated by pandas when reading
      CSVs with index columns. We avoid using these as join keys when inferring joins,
      as they usually don't represent meaningful data relationships.

    - 'index' column/parameter: When 'index' is specified as a join key (e.g., 'right on': 'index')
      but doesn't exist as an actual column (column with no name), we treat it as a reference to the DataFrame's row index.
      In this case, we create a temporary column named 'index' containing the row indices to enable the join.

      Important: When generating join candidates, these 'index' columns must be properly
      handled to ensure they match with the ground truth. If not handled correctly, this
      can lead to situations where ground truth join keys appear as empty lists ([]) in
      the feature extraction phase, resulting in fewer positive examples than expected.

      Example:
        Left Table:               Right Table:
           Sample ID  Value1        Data1  Data2
        0         A      10      0    100    200
        1         B      15      1    150    250
        2         C      20      2    200    300

        With parameters: {'left on': 'Sample ID', 'right on': 'index'}

        We add an 'index' column to the right table:
           Data1  Data2  index
        0    100    200      0
        1    150    250      1
        2    200    300      2

        Then join using 'Sample ID' and 'index' columns, matching A→0, B→1, C→2 (ariane-lozachmeur_capstone example)

    Args:
        samples: List of raw samples loaded from the data directory.

    Returns:
        List of processed samples with left_table, right_table, and join columns identified.
    """
    processed_samples = []

    # Count different issues
    missing_tables = 0
    empty_tables = 0
    missing_join_keys = 0
    invalid_columns = 0

    for sample_idx, sample in enumerate(samples):
        # Skip samples without necessary data
        if 'left_table' not in sample or 'right_table' not in sample or 'params' not in sample:
            missing_tables += 1
            continue

        # Create copies to avoid modifying original tables
        left_table = sample['left_table'].copy()
        right_table = sample['right_table'].copy()
        params = sample['params']

        # Skip if tables are empty
        if left_table.empty or right_table.empty:
            empty_tables += 1
            continue

        #print(f"\nSample {sample_idx + 1}: Tables shapes: {left_table.shape}, {right_table.shape}")
        #print(f"Parameters: {params}")

        # Extract join columns from parameters
        left_join_keys = []
        right_join_keys = []

        # Check different possible formats in param.json
        if 'join_keys' in params:
            # Format: {"join_keys": {"left": ["col1"], "right": ["col2"]}}
            join_keys = params['join_keys']
            left_join_keys = join_keys.get('left', [])
            right_join_keys = join_keys.get('right', [])
            print(f"Found join_keys format: {left_join_keys} ↔ {right_join_keys}")

        # Check for 'left on' and 'right on'
        elif 'left on' in params and 'right on' in params:
            left_on = params['left on']
            right_on = params['right on']

            # Convert to list if string
            if isinstance(left_on, str):
                left_join_keys = [left_on]
            elif isinstance(left_on, list):
                left_join_keys = left_on

            if isinstance(right_on, str):
                right_join_keys = [right_on]
            elif isinstance(right_on, list):
                right_join_keys = right_on

            #print(f"Found left/right on format: {left_join_keys} ↔ {right_join_keys}")

        elif 'on' in params:
            # Different formats for 'on' parameter
            on_param = params['on']
            print(f"Found 'on' parameter: {on_param}")

            if isinstance(on_param, list):
                # Format: {"on": ["col1", "col2"]} - same column names in both tables
                left_join_keys = right_join_keys = on_param
                print(f"List format: {left_join_keys} ↔ {right_join_keys}")
            elif isinstance(on_param, str):
                # Format: {"on": "col"} - single column with same name
                left_join_keys = right_join_keys = [on_param]
                print(f"String format: {left_join_keys} ↔ {right_join_keys}")
            elif isinstance(on_param, dict):
                # Format: {"on": {"left_col": "right_col"}} - mapping between columns
                left_join_keys = list(on_param.keys())
                right_join_keys = list(on_param.values())
                print(f"Dict format: {left_join_keys} ↔ {right_join_keys}")
        else:
            missing_join_keys += 1
            print("No explicit join keys found, attempting to infer")

            # Try to infer join columns by looking at common column names
            # Avoid using "Unnamed: X" columns for inference
            common_cols = set(left_table.columns).intersection(set(right_table.columns))
            meaningful_common_cols = [col for col in common_cols if not col.startswith('Unnamed:')]

            if meaningful_common_cols:
                left_join_keys = right_join_keys = meaningful_common_cols
                #print(f"Inferred common columns: {meaningful_common_cols}")
            else:
                #print("Could not infer join columns, skipping sample")
                continue  # Can't determine join keys

        # Special handling for 'index' which refers to the DataFrame's row index
        # This handles cases like Sample 2 where one table should join on its row numbers
        if 'index' in left_join_keys and 'index' not in left_table.columns:
            #print("Converting 'index' to DataFrame index for left table")
            # Add a column containing the row indices (0, 1, 2, ...) to enable joining on row numbers
            left_table['index'] = left_table.index

        if 'index' in right_join_keys and 'index' not in right_table.columns:
            #print("Converting 'index' to DataFrame index for right table")
            # Add a column containing the row indices (0, 1, 2, ...) to enable joining on row numbers
            right_table['index'] = right_table.index

        # Validate that columns exist in tables
        # After handling 'index', all specified columns should exist
        valid_left_keys = [col for col in left_join_keys if col in left_table.columns]
        valid_right_keys = [col for col in right_join_keys if col in right_table.columns]

        if len(valid_left_keys) != len(left_join_keys) or len(valid_right_keys) != len(right_join_keys):
            # print(f"Warning: Some join keys don't exist in tables")
            # print(f"Left keys: {left_join_keys} -> {valid_left_keys}")
            # print(f"Right keys: {right_join_keys} -> {valid_right_keys}")

            if not valid_left_keys or not valid_right_keys:
                #print("No valid join keys, skipping sample")
                invalid_columns += 1
                continue

            left_join_keys = valid_left_keys
            right_join_keys = valid_right_keys

        # Final validation - ensure we have the same number of columns on both sides
        if len(left_join_keys) != len(right_join_keys):
            #print(f"Error: Mismatched number of join columns: {len(left_join_keys)} vs {len(right_join_keys)}")
            invalid_columns += 1
            continue

        #print(f"Final join keys: {left_join_keys} ↔ {right_join_keys}")

        # Add this sample to our processed samples
        processed_samples.append({
            'left_table': left_table,
            'right_table': right_table,
            'left_join_keys': left_join_keys,
            'right_join_keys': right_join_keys,
            'join_type': params.get('how', 'inner'),  # Default to 'inner' if not specified
            'sample_id': sample.get('sample_id', None)
        })

    print(f"\nProcessed {len(processed_samples)} valid join samples\n")

    # Print diagnostic information if no valid samples
    if len(processed_samples) == 0:
        print("\nDiagnostic information:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Missing tables: {missing_tables}")
        print(f"  Empty tables: {empty_tables}")
        print(f"  Missing join keys: {missing_join_keys}")
        print(f"  Invalid columns: {invalid_columns}")

    return processed_samples