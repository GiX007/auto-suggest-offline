# src/features/groupby_features.py
#
# This module implements feature extraction for GroupBy prediction based on Section 4.2 of the
# "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks" paper.
#
# The GroupBy operator requires predicting which columns should be used for grouping (dimensions)
# versus which columns should be used for aggregation (measures). This module extracts
# features from columns that help make this prediction.
#
# Key functionality:
# 1. extract_column_features: Computes 17 features for a column to determine if it's suitable for GroupBy:
#    - Distinct-value-count: Number of unique values and ratio to total rows
#    - Column-data-type: Whether the column is string, integer, float, etc.
#    - Left-ness: Position of column in the table (columns to the left are often dimensions)
#    - Emptiness: Ratio of null values
#    - Value-range: Range of values for numeric columns
#    - Peak-frequency: How frequent the most common value is
#    - Column-name-related: Whether names contain common GroupBy or Aggregation terms
#
# 2. process_groupby_samples: Processes samples from Jupyter notebooks to extract the
#    input tables and the columns that were chosen for GroupBy and Aggregation
#
# The features extracted in this module are used to train a model that can predict
# which columns are likely to be used for grouping versus aggregation, allowing the
# system to make intelligent recommendations to users.

import pandas as pd
import traceback
from typing import Dict, List, Any


def extract_column_features(table: pd.DataFrame, column_name: str) -> Dict[str, float]:
    """
    Extract features for a column in the context of GroupBy/Aggregation prediction.

    Args:
        table: The input table.
        column_name: The name of the column to extract features for.

    Returns:
        A dictionary of feature values.
    """
    features = {}

    # Skip if column doesn't exist
    if column_name not in table.columns:
        return {}

    try:
        column = table[column_name]

        # Feature 1: Distinct-value-count
        distinct_values = column.nunique()
        features['distinct_count'] = distinct_values
        features['distinct_ratio'] = distinct_values / len(table) if len(table) > 0 else 0

        # Feature 2: Column-data-type
        features['is_string'] = pd.api.types.is_string_dtype(column) or pd.api.types.is_object_dtype(column)
        features['is_int'] = pd.api.types.is_integer_dtype(column)
        features['is_float'] = pd.api.types.is_float_dtype(column)
        features['is_bool'] = pd.api.types.is_bool_dtype(column)
        features['is_datetime'] = pd.api.types.is_datetime64_dtype(column)

        # Feature 3: Left-ness
        position = list(table.columns).index(column_name)
        features['absolute_position'] = position
        features['relative_position'] = position / len(table.columns) if len(table.columns) > 0 else 0

        # Feature 4: Emptiness
        null_count = column.isna().sum()
        features['null_ratio'] = null_count / len(table) if len(table) > 0 else 0

        # Feature 5: Value-range (for numeric columns)
        if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
            try:
                col_min = column.min()
                col_max = column.max()
                column_range = col_max - col_min if not column.empty else 0
                features['value_range'] = float(column_range) if not pd.isna(column_range) else 0
                features['distinct_to_range_ratio'] = distinct_values / float(column_range) if float(
                    column_range) > 0 else 0
            except TypeError:
                # Handle case where subtraction isn't supported
                features['value_range'] = 0
                features['distinct_to_range_ratio'] = 0
        else:
            features['value_range'] = 0
            features['distinct_to_range_ratio'] = 0

        # Feature 6: Peak-frequency
        value_counts = column.value_counts(dropna=False)
        most_common_count = value_counts.iloc[0] if not value_counts.empty else 0
        features['peak_frequency'] = most_common_count
        features['peak_frequency_ratio'] = most_common_count / len(table) if len(table) > 0 else 0

        # Feature 7: Column-name-related features
        common_groupby_terms = ['id', 'category', 'type', 'group', 'class', 'gender', 'year', 'month', 'day',
                                'region', 'country', 'state', 'city', 'quarter', 'segment', 'sector']
        common_agg_terms = ['amount', 'count', 'sum', 'revenue', 'profit', 'sales', 'quantity', 'price',
                            'total', 'average', 'score', 'value', 'rate', 'ratio', 'percentage']

        col_name_lower = column_name.lower()

        features['groupby_term_in_name'] = any(term in col_name_lower for term in common_groupby_terms)
        features['agg_term_in_name'] = any(term in col_name_lower for term in common_agg_terms)

    except Exception as e:
        print(f"Error extracting features for column {column_name}: {e}")
        # Set default values for failed features
        features = {
            'distinct_count': 0,
            'distinct_ratio': 0,
            'is_string': False,
            'is_int': False,
            'is_float': False,
            'is_bool': False,
            'is_datetime': False,
            'absolute_position': 0,
            'relative_position': 0,
            'null_ratio': 0,
            'value_range': 0,
            'distinct_to_range_ratio': 0,
            'peak_frequency': 0,
            'peak_frequency_ratio': 0,
            'groupby_term_in_name': False,
            'agg_term_in_name': False
        }

    return features


def process_groupby_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process groupby samples to get input tables and groupby parameters.

    This function extracts ground truth from Jupyter notebooks by:
    1. Identifying which columns were used for GroupBy in actual notebook usage
    2. Identifying which columns were used for Aggregation
    3. Filtering out invalid or incomplete samples

    The output is used for training and evaluating the GroupBy prediction model.

    Args:
        samples: List of raw samples loaded from the data directory.

    Returns:
        List of processed samples with input_table and groupby/aggregation columns identified.
    """
    processed_samples = []

    # Add verbose logging
    print(f"Starting to process {len(samples)} GroupBy samples...")
    problematic_samples = 0

    for sample_idx, sample in enumerate(samples):
        try:
            # Skip samples without necessary data
            if 'input_table' not in sample:
                print(f"Sample {sample_idx} missing 'input_table' key")
                problematic_samples += 1
                continue

            if 'params' not in sample:
                print(f"Sample {sample_idx} missing 'params' key")
                problematic_samples += 1
                continue

            input_table = sample['input_table']
            params = sample['params']

            # Skip if table is empty
            if input_table.empty:
                print(f"Sample {sample_idx} has empty input table")
                problematic_samples += 1
                continue

            # Extract groupby columns from parameters with more robust extraction
            groupby_columns = []

            # Check all possible parameter names for groupby columns
            if 'by' in params:
                by_param = params['by']
                if isinstance(by_param, list):
                    groupby_columns = by_param
                elif isinstance(by_param, str):
                    groupby_columns = [by_param]
            elif 'columns' in params:
                cols_param = params['columns']
                if isinstance(cols_param, list):
                    groupby_columns = cols_param
                elif isinstance(cols_param, str):
                    groupby_columns = [cols_param]
            elif 'groupby_cols' in params:
                gb_param = params['groupby_cols']
                if isinstance(gb_param, list):
                    groupby_columns = gb_param
                elif isinstance(gb_param, str):
                    groupby_columns = [gb_param]

            # If we still don't have groupby columns, check a few more options
            if not groupby_columns:
                # Sometimes groupby params are nested
                if 'groupby' in params:
                    gb_data = params['groupby']
                    if isinstance(gb_data, dict) and 'by' in gb_data:
                        by_param = gb_data['by']
                        if isinstance(by_param, list):
                            groupby_columns = by_param
                        elif isinstance(by_param, str):
                            groupby_columns = [by_param]

                # Last resort - try to infer from key column names
                if not groupby_columns:
                    # Look for columns with these typical dimension names
                    dimension_keywords = ['id', 'category', 'year', 'month', 'quarter', 'region', 'state', 'city']
                    for col in input_table.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in dimension_keywords):
                            groupby_columns.append(col)

            # Extract aggregation columns from parameters
            agg_columns = []
            if 'agg_function' in params:
                agg_function = params['agg_function']
                if isinstance(agg_function, dict):
                    for col, func in agg_function.items():
                        if col not in groupby_columns:
                            agg_columns.append(col)
            elif 'aggfunc' in params:
                agg_func = params['aggfunc']
                if isinstance(agg_func, dict):
                    for col, func in agg_func.items():
                        if col not in groupby_columns:
                            agg_columns.append(col)

            # If we still don't have aggregation columns, try to infer
            if not agg_columns:
                # Look for typical measure columns (numeric non-grouped columns)
                measure_keywords = ['amount', 'count', 'sum', 'revenue', 'profit', 'sales', 'quantity', 'price']
                for col in input_table.columns:
                    if col in groupby_columns:
                        continue

                    # Check if it's numeric (typical for aggregation)
                    if pd.api.types.is_numeric_dtype(input_table[col]):
                        # If it has a typical measure name, add it
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in measure_keywords):
                            agg_columns.append(col)
                        # Or if it has few unique values compared to rows, likely not a dimension
                        elif input_table[col].nunique() / len(input_table) > 0.5:
                            agg_columns.append(col)

            # Check if columns are valid in the table
            valid_columns = list(input_table.columns)

            valid_groupby_columns = [col for col in groupby_columns if col in valid_columns]
            valid_agg_columns = [col for col in agg_columns if col in valid_columns]

            if valid_groupby_columns:  # Only add sample if it has valid groupby columns
                processed_samples.append({
                    'input_table': input_table,
                    'groupby_columns': valid_groupby_columns,   # ( this is the sample's ground truth! )
                    'agg_columns': valid_agg_columns,
                    'sample_id': sample.get('sample_id', f"sample_{sample_idx}")
                })
            else:
                print(f"Sample {sample_idx} has no valid groupby columns. Original: {groupby_columns}")
                problematic_samples += 1

        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            traceback.print_exc()
            problematic_samples += 1

    print(f"Processed {len(processed_samples)} valid groupby samples out of {len(samples)} total")
    print(f"Encountered {problematic_samples} problematic samples")

    return processed_samples