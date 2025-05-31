# src/models/groupby_model.py
#
# Implementation of the GroupBy prediction model based on Section 4.2
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks".
#
# This module includes:
# 1. Processing and Preparation of training data for groupby prediction
# 2. Training of a gradient boosting model to predict groupby columns
# 3. Prediction of groupby columns for new input tables
# 4. Full groupby recommendation pipeline with confidence scores
# 5. Visualization and display of groupby recommendations
# 6. Evaluation of groupby prediction performance with precision and recall metrics
#
# This module combines both model and pipeline functionality for streamlined usage
# compared to the join operator (which separates join column and join type prediction).
#
# All logic related to groupby column prediction is contained here.
#

import os
import random
import time
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, ndcg_score
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.utils.model_utils import generate_prediction_table, generate_feature_importance_table
from sklearn.utils.class_weight import compute_sample_weight
from src.utils.model_utils import numpy_to_list


# Define Paths
test_data = r"C:\Users\giorg\Auto_Suggest\data\test_data"

# Results for comparison with vendors from the paper (Table 6)
# These are hardcoded values from Table 6 in the paper
GROUPBY_VENDORS = {
    "Vendor-B": {"prec@1": 0.56, "prec@2": 0.71, "ndcg@1": 0.56, "ndcg@2": 0.75, "full-accuracy": 0.45},
    "Vendor-C": {"prec@1": 0.71, "prec@2": 0.82, "ndcg@1": 0.71, "ndcg@2": 0.85, "full-accuracy": 0.67}
}


def process_groupby_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process GroupBy samples to extract input tables, groupby columns, and aggregation columns.

    This function:
    1. Validates input samples.
    2. Extracts groupby and aggregation columns from parameters.
    3. Falls back to typical dimension and measure column heuristics if explicit parameters are missing.
    4. Filters out invalid or incomplete samples.

    Args:
        samples: List of raw samples loaded from the data directory.

    Returns:
        List of processed samples (dicts) containing input_table, groupby_columns, agg_columns, and sample_id.
    """
    # Inspect a sample (look how it looks like)
    # print("Input sample:")
    # for k, v in samples[0].items():
    #     print(k, v)

    processed_samples = []
    problematic_samples = 0

    #print(f"Starting to process {len(samples)} GroupBy samples...")

    for idx, sample in enumerate(samples):
        try:
            # Skip samples without necessary data
            if 'input_table' not in sample or 'params' not in sample:
                print(f"Sample {idx} missing 'input_table' or 'params'")
                problematic_samples += 1
                continue

            input_table = sample['input_table']

            # Remove noisy columns
            columns_to_keep = [col for col in input_table.columns if not (col.startswith('Unnamed:') or col == 'index' or col == '__dummy__')]
            input_table = input_table[columns_to_keep]

            params = sample['params']

            # Skip if table is empty
            if input_table.empty:
                print(f"Sample {idx} has empty input table")
                problematic_samples += 1
                continue

            # print(f"\nSample {idx + 1}: Table's' shapes: {input_table.shape}")
            # print(f"Parameters: {params}")

            # Extract groupby columns by checking all possible parameter names for groupby columns
            groupby_columns = []
            for key in ['by', 'columns', 'groupby_cols']:
                if key in params:
                    val = params[key]
                    groupby_columns = val if isinstance(val, list) else [val]
                    break
            else:
                # Check nested structures
                gb_data = params.get('groupby', {})
                if isinstance(gb_data, dict) and 'by' in gb_data:
                    by_param = gb_data['by']
                    groupby_columns = by_param if isinstance(by_param, list) else [by_param]

                # Try to infer from common key column names
                if not groupby_columns:
                    #print(f"Sample {idx}: No explicit grouping columns, inferring from column names")
                    dimension_keywords = ['id', 'category', 'year', 'month', 'quarter', 'region', 'state', 'city']
                    groupby_columns = [col for col in input_table.columns
                                       if any(kw in col.lower() for kw in dimension_keywords)]

            # Extract aggregation columns
            agg_columns = []
            for key in ['agg_function', 'aggfunc']:
                agg_func = params.get(key, {})
                if isinstance(agg_func, dict):
                    for col in agg_func.keys():
                        if col not in groupby_columns:
                            agg_columns.append(col)

            # Check for 'value' key in params
            if not agg_columns and 'value' in params:
                val = params['value']
                if isinstance(val, list):
                    agg_columns = [col for col in val if col not in groupby_columns]
                elif isinstance(val, str):
                    if val not in groupby_columns:
                        agg_columns = [val]

            # If we still don't have aggregation columns, try to infer
            if not agg_columns:
                #print("No explicit aggr columns found, attempting to infer")
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

            # Validate columns
            valid_columns = list(input_table.columns)
            valid_groupby_columns = [col for col in groupby_columns if col in valid_columns]
            valid_agg_columns = [col for col in agg_columns if col in valid_columns]

            # Remove known 'Unnamed' columns
            noise_columns = {'Unnamed: 0', 'index'}
            valid_groupby_columns = [col for col in valid_groupby_columns if col not in noise_columns]
            valid_agg_columns = [col for col in valid_agg_columns if col not in noise_columns]

            # Add this valid sample to our processed samples
            if valid_groupby_columns:
                processed_samples.append({
                    'input_table': input_table,
                    'groupby_columns': valid_groupby_columns,   # ( this is the sample's ground truth! )
                    'agg_columns': valid_agg_columns,
                    'sample_id': sample.get('sample_id', f"sample_{idx}")
                })
            else:
                print(f"Sample {idx} has no valid groupby columns. Original: {groupby_columns}")
                problematic_samples += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            traceback.print_exc()
            problematic_samples += 1

    # Inspect a processed sample (see how it looks like now, after processing)
    # print("Processed sample:")
    # for k, v in processed_samples[0].items():
    #     print(k, v)

    print(f"\nProcessed {len(processed_samples)} valid groupby samples")  # out of {len(samples)} total
    #print(f"Encountered {problematic_samples} problematic samples")

    return processed_samples


def extract_groupby_pair_features(table: pd.DataFrame, groupby_cols, agg_cols) -> Dict[str, float]:
    """
    Extracts features for a pair of column sets: dimension/groupby columns and measure/aggregation columns.

    Args:
        table: The input table.
        groupby_cols: Single column name or list of dimension/groupby columns.
        agg_cols: Single column name or list of aggregation/measure columns.

    Returns:
        A dictionary of feature values for this dimension-measure pair.
    """
    features = {}

    # Ensure column inputs are lists (either they are a single column or a list of columns)
    groupby_cols = [groupby_cols] if isinstance(groupby_cols, str) else groupby_cols
    agg_cols = [agg_cols] if isinstance(agg_cols, str) else agg_cols

    try:

        # Feature 1: Distinct-value-count
        gb_distinct_values = table[groupby_cols].drop_duplicates().shape[0]
        features['gb_distinct_count'] = gb_distinct_values
        features['gb_distinct_ratio'] = gb_distinct_values / len(table) if len(table) > 0 else 0

        agg_distinct_values = table[agg_cols].drop_duplicates().shape[0]
        features['agg_distinct_count'] = agg_distinct_values
        features['agg_distinct_ratio'] = agg_distinct_values / len(table) if len(table) > 0 else 0

        # Feature 2: Column-data-type
        # For groupby columns: average of column types
        gb_types = table[groupby_cols].dtypes
        features['gb_is_string'] = gb_types.apply(lambda dt: pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt)).mean()
        features['gb_is_int'] = gb_types.apply(pd.api.types.is_integer_dtype).mean()
        features['gb_is_float'] = gb_types.apply(pd.api.types.is_float_dtype).mean()
        features['gb_is_bool'] = gb_types.apply(pd.api.types.is_bool_dtype).mean()
        features['gb_is_datetime'] = gb_types.apply(pd.api.types.is_datetime64_dtype).mean()

        # For aggregation columns: average of column types
        agg_types = table[agg_cols].dtypes
        features['agg_is_string'] = agg_types.apply(lambda dt: pd.api.types.is_string_dtype(dt) or pd.api.types.is_object_dtype(dt)).mean()
        features['agg_is_int'] = agg_types.apply(pd.api.types.is_integer_dtype).mean()
        features['agg_is_float'] = agg_types.apply(pd.api.types.is_float_dtype).mean()
        features['agg_is_bool'] = agg_types.apply(pd.api.types.is_bool_dtype).mean()
        features['agg_is_datetime'] = agg_types.apply(pd.api.types.is_datetime64_dtype).mean()

        # Feature 3: Left-ness (position-based)
        gb_positions = [list(table.columns).index(col) for col in groupby_cols]
        features['gb_absolute_position_mean'] = sum(gb_positions) / len(gb_positions)
        features['gb_relative_position_mean'] = features['gb_absolute_position_mean'] / len(table.columns) if len(table.columns) > 0 else 0

        agg_positions = [list(table.columns).index(col) for col in agg_cols]
        features['agg_absolute_position_mean'] = sum(agg_positions) / len(agg_positions)
        features['agg_relative_position_mean'] = features['agg_absolute_position_mean'] / len(table.columns) if len(table.columns) > 0 else 0

        # Feature 4: Emptiness
        features['gb_null_ratio_mean'] = table[groupby_cols].isna().mean().mean()
        features['agg_null_ratio_mean'] = table[agg_cols].isna().mean().mean()

        # Feature 5: Value-range (for numeric columns)
        # For groupby columns: no typical value range, but for completeness
        features['gb_value_range_mean'] = table[groupby_cols].apply(lambda col: col.max() - col.min() if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col) else 0).mean()
        features['agg_value_range_mean'] = table[agg_cols].apply(lambda col: col.max() - col.min() if (pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)) else 0).mean()

        features['gb_distinct_to_range_ratio'] = features['gb_distinct_count'] / features['gb_value_range_mean'] if features['gb_value_range_mean'] > 0 else 0
        features['agg_distinct_to_range_ratio'] = features['agg_distinct_count'] / features['agg_value_range_mean'] if features['agg_value_range_mean'] > 0 else 0

        # Feature 6: Peak-frequency
        common_groupby_terms = ['id', 'category', 'type', 'group', 'class', 'gender', 'year', 'month', 'day',
                                'region', 'country', 'state', 'city', 'quarter', 'segment', 'sector']
        common_agg_terms = ['amount', 'count', 'sum', 'revenue', 'profit', 'sales', 'quantity', 'price',
                            'total', 'average', 'score', 'value', 'rate', 'ratio', 'percentage']

        gb_col_names = ' '.join(groupby_cols).lower()
        agg_col_names = ' '.join(agg_cols).lower()

        features['groupby_term_in_name'] = any(term in gb_col_names for term in common_groupby_terms)
        features['agg_term_in_name'] = any(term in agg_col_names for term in common_agg_terms)


    except Exception as e:
        print(f"Error extracting features for groupby={groupby_cols} & agg={agg_cols}: {e}")
        traceback.print_exc()

    return features


def extract_column_features(table: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Extracts features for a single column to assess its groupby-likelihood.

    Args:
        table: The input table.
        column: Column name.

    Returns:
        A dictionary of feature values for this column.
    """
    features = {}

    try:
        col_data = table[column]
        col_index = table.columns.get_loc(column)
        row_count = len(table)

        # Feature 1: Distinct-value-count
        distinct_count = col_data.nunique(dropna=True)
        features['gb_distinct_count'] = distinct_count
        features['gb_distinct_ratio'] = distinct_count / row_count if row_count > 0 else 0

        # Feature 2: Data-type
        features['gb_is_string'] = int(pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data))
        features['gb_is_int'] = int(pd.api.types.is_integer_dtype(col_data))
        features['gb_is_float'] = int(pd.api.types.is_float_dtype(col_data))
        features['gb_is_bool'] = int(pd.api.types.is_bool_dtype(col_data))
        features['gb_is_datetime'] = int(pd.api.types.is_datetime64_dtype(col_data))

        # Feature 3: Column position
        features['gb_absolute_position_mean'] = col_index
        features['gb_relative_position_mean'] = col_index / len(table.columns) if len(table.columns) > 0 else 0

        # Feature 4: Emptiness
        features['gb_null_ratio_mean'] = col_data.isna().mean()

        # Feature 5: Value-range (for numeric columns)
        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            value_range = col_data.max() - col_data.min()
        else:
            value_range = 0
        features['gb_value_range_mean'] = value_range

        features['gb_distinct_to_range_ratio'] = distinct_count / value_range if value_range > 0 else 0

        # Feature 6: Name-based signals (typical groupby terms)
        common_groupby_terms = ['id', 'category', 'type', 'group', 'class', 'gender', 'year', 'month', 'day',
                                'region', 'country', 'state', 'city', 'quarter', 'segment', 'sector']
        col_name_lower = column.lower()
        features['groupby_term_in_name'] = int(any(term in col_name_lower for term in common_groupby_terms))

        # Debug print to ensure the feature extraction works
        #print(f"Features for column '{column}': {features}")


    except Exception as e:
        print(f"Error extracting features for column '{column}': {e}")
        traceback.print_exc()

    return features



def prepare_groupby_pair_data(processed_samples: List[Dict]) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Prepares training data for GroupBy column prediction.

    This function processes each sample to create feature vectors for each column in the input table.
    For each column, it:
    1. Extracts relevant features describing the column's properties
    2. Labels each column as 1 if it was used as a GroupBy column in the ground truth, 0 otherwise

    This results in a balanced dataset of positive and negative examples to train a classifier.

    Args:
        processed_samples: List of processed groupby samples with ground truth groupby and aggregation columns.

    Returns:
        Tuple containing:
            - X: Numeric feature matrix (NumPy array), shape (num_samples, num_features).
            - feature_cols: List of feature names corresponding to columns in X.
            - labels: List of 0/1 labels indicating if the column is a ground-truth groupby column.
    """
    features_list = []
    labels = []

    #print("\nPreparing GroupBy training data from {} samples".format(len(processed_samples)))
    for sample_idx, sample in enumerate(processed_samples):

        input_table = sample['input_table']
        groupby_columns = sample['groupby_columns']
        agg_columns = sample['agg_columns']
        all_columns = list(input_table.columns)

        # print(f"\nSample {sample_idx} has {len(all_columns)} columns: {all_columns}")
        # print(f"  Groupby columns are {len(groupby_columns)}: {groupby_columns}")
        # print(f"  Agg columns are {len(agg_columns)}: {agg_columns}")
        # print(f"  All possible combinations are: {len(groupby_columns * len(agg_columns))}: {groupby_columns}")

        # Retrieve ground-truth positive groupby pairs
        true_pairs = set((g, a) for g in groupby_columns for a in agg_columns)
        pos_pairs = list(true_pairs)

        # -------------------------------------------------------------------
        # Negative sampling: balance dataset for better learning
        # -------------------------------------------------------------------
        # In GroupBy prediction, most column pairs are not actual groupby/agg pairs.
        # If we included ALL negative pairs, the dataset would be dominated by them
        # (like 99% negatives), causing the model to learn to always predict negative.
        # Instead, we sample 3x as many negatives as positives for each sample.
        # This balances the dataset (~25–30% positives) so that:
        #  - The model learns how to separate true groupby/agg pairs from false ones
        #  - We avoid overwhelming the classifier with useless negatives
        #  - We still include enough negatives to learn the difference
        # The sampling multiplier (3x) is a heuristic for this tradeoff — it can be tuned.
        # -------------------------------------------------------------------

        # collect negative pairs
        neg_pairs = []
        for g_col in all_columns:
            # Check if groupby column is categorical
            if pd.api.types.is_numeric_dtype(input_table[g_col]):
                continue
            for a_col in all_columns:
                # Check if aggregation column is numeric
                if not pd.api.types.is_numeric_dtype(input_table[a_col]):
                    continue
                # Skip self-pair
                if g_col == a_col:
                    continue
                # if pd.api.types.is_datetime64_any_dtype(input_table[a_col]):
                #     continue  # Skip datetime as measure
                # if input_table[a_col].nunique() <= 5:
                #     continue  # Skip low-cardinality numeric column as aggregation column
                if (g_col, a_col) not in true_pairs:
                    neg_pairs.append((g_col, a_col))

        # sample 3 negatives for each positive
        sampled_neg_pairs = random.sample(neg_pairs, min(len(neg_pairs), len(pos_pairs) * 3))

        # add positive samples
        for g_col, a_col in pos_pairs:
            features = extract_groupby_pair_features(input_table, g_col, a_col)
            features_list.append(features)
            labels.append(1)

        # add negative samples
        for g_col, a_col in sampled_neg_pairs:
            features = extract_groupby_pair_features(input_table, g_col, a_col)
            features_list.append(features)
            labels.append(0)

    # Create the DataFrame after collecting all features
    features_df = pd.DataFrame(features_list)

    # Remove non-feature columns
    non_feature_cols = ['sample_id', 'groupby_column', 'agg_column']
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    # Convert all boolean columns to int (1/0) for compatibility with GradientBoostingClassifier
    for col in feature_cols:
        if features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)

    # After processing all samples, check the features matrix
    # print("\nChecking feature matrix types...")
    # for col in feature_cols:
    #     dtype = features_df[col].dtype
    #     print(f"Feature '{col}': {dtype}")

    # Convert features DataFrame to numeric matrix
    X = features_df[feature_cols].values
    # print(f"\nX matrix dtype: {X.dtype}")
    # print(f"X matrix shape: {X.shape}")

    # if np.issubdtype(X.dtype, np.number):
    #     print("All features in X are numeric and ready for training!")
    # else:
    #     print("Warning: Some features are not numeric!")

    # Final summary

    # Note: For each sample, there may be multiple candidate pairs (N candidates),
    # so the final feature matrix X (X samples) can have X*N rows (candidates) and many columns (features).

    # print("\nSummary of generated training data:")
    # print(f"  Total training samples: {len(processed_samples)}")
    # print(f"  Number of feature vectors (total pairs): {len(features_list)}")
    # print(f"  Number of features per pair: {len(feature_cols)}")
    # print(f"  Shape of X: {X.shape}")
    # print(f"  Shape of y: {len(labels)}")
    #
    # # Show counts of positives and negatives
    # positive_count = sum(labels)
    # negative_count = len(labels) - positive_count
    # print(f"  Number of positive examples: {positive_count}")
    # print(f"  Number of negative examples: {negative_count}")
    # print(f"  Positive percentage: {(positive_count / len(labels)) * 100:.2f}%")

    return X, feature_cols, labels


def prepare_groupby_data(processed_samples: List[Dict], upsample=True) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Prepares training data for GroupBy column prediction (per-column approach).
    Optionally balances the dataset using upsampling.

    For each sample:
    1. Extracts relevant features for each column.
    2. Labels each column as 1 if it was used as a GroupBy column in the ground truth, 0 otherwise.

    Args:
        processed_samples: List of processed groupby samples with ground truth groupby columns.
        upsample: Whether to balance data by upsampling positive examples.

    Returns:
        - X: Numeric feature matrix (NumPy array), shape (num_columns, num_features).
        - feature_cols: List of feature names.
        - labels: List of 0/1 labels indicating if the column is a true groupby column.
    """
    features_list = []
    labels = []

    for sample_idx, sample in enumerate(processed_samples):
        input_table = sample['input_table']
        groupby_columns = sample['groupby_columns']
        all_columns = list(input_table.columns)

        for column in all_columns:
            col_data = input_table[column]

            # Exclude datetime-like columns
            if pd.api.types.is_datetime64_any_dtype(col_data):
                continue  # skip!

            # Exclude numeric columns with high cardinality (distinct ratio > 0.9)
            if pd.api.types.is_numeric_dtype(col_data):
                distinct_ratio = col_data.nunique(dropna=True) / len(col_data)
                if distinct_ratio > 0.8:
                    continue  # skip!

            # Exclude numeric columns altogether (groupby to be mostly categorical)
            if pd.api.types.is_numeric_dtype(col_data):
                continue

            # Extract features
            features = extract_column_features(input_table, column)
            features_list.append(features)

            # Label as groupby if it's in the ground truth
            label = 1 if column in groupby_columns else 0
            labels.append(label)

    # Create the features DataFrame
    features_df = pd.DataFrame(features_list)

    # Remove non-feature columns (if any)
    non_feature_cols = ['sample_id', 'groupby_column', 'agg_column']
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    # Convert booleans to int (1/0)
    for col in feature_cols:
        if features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)

    if upsample:
        from sklearn.utils import resample

        # Combine features and labels into one DataFrame
        features_df['label'] = labels

        # Separate classes
        df_positive = features_df[features_df['label'] == 1]
        df_negative = features_df[features_df['label'] == 0]

        # Upsample positive class to match negatives
        if len(df_positive) > 0:
            df_positive_upsampled = resample(df_positive,
                                             replace=True,
                                             n_samples=len(df_negative),
                                             random_state=42)
            df_upsampled = pd.concat([df_negative, df_positive_upsampled])
            df_upsampled = df_upsampled.sample(frac=1, random_state=42)  # Shuffle
            features_df = df_upsampled
            # print("\nUpsampling performed:")
            # print(f"  Positives: {sum(features_df['label'])}")
            # print(f"  Negatives: {len(features_df) - sum(features_df['label'])}")

    # Final feature matrix
    X = features_df[feature_cols].values
    y = features_df['label'].values

    # Debug print
    # print(f"Total feature vectors: {len(X)}")
    # print(f"Number of features: {len(feature_cols)}")

    # Check for all-zero feature vectors
    # zero_feature_vectors = np.sum(X, axis=1) == 0
    # num_zero_vectors = np.sum(zero_feature_vectors)
    #
    # if num_zero_vectors > 0:
    #     print(f"\nFound {num_zero_vectors} feature vectors that are all zeros!")
    # else:
    #     print("\nNo all-zero feature vectors found during training.")

    return X, feature_cols, y.tolist()


def train_groupby_model(X_train, y_train, X_val, y_val, feature_names):
    """
    Trains a model to predict groupby columns.

    This function takes feature lists (dictionaries) and binary labels for train and validation sets,
    extracts numeric feature matrices and feature names, and trains a classifier using hyperparameter tuning.

    Example:
    - Sample 1 has 3 columns: 'customer_id', 'signup_date', 'amount'
    - Only 'customer_id' is a groupby column (label=1), others are not (label=0)
    - The features for each column include stats (distinct ratio, position, type, etc.)
    - This repeats for all samples to build the full training dataset.

    Args:
        X_train: List of feature dictionaries (train)
        y_train: List of 0/1 labels (train)
        X_val: List of feature dictionaries (validation)
        y_val: List of 0/1 labels (validation)
        feature_names: List of feature names, used for feature importance analysis and debugging

    Returns:
        Trained model and list of feature names used by the model.
    """
    # Check if we have data
    if X_train is None or len(X_train) == 0 or y_train is None or len(y_train) == 0:
        print("Error: No training data available for groupby prediction.")
        return None, []

    if X_val is None or len(X_val) == 0 or y_val is None or len(y_val) == 0:
        print("Error: No validation data available for groupby prediction.")
        return None, []

    # Final feature space
    print(f"\nUsing {len(feature_names)} features:")
    print(feature_names)

    # Convert to DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_names)
    val_df = pd.DataFrame(X_val, columns=feature_names)

    # Handle booleans
    for col in feature_names:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(int)
            val_df[col] = val_df[col].astype(int)

    # Handle missing values
    train_df = train_df.fillna(0)
    val_df = val_df.fillna(0)

    # Final feature matrices
    X_train = train_df[feature_names].values
    X_val = val_df[feature_names].values
    y_train = np.array(y_train)
    y_val = np.array(y_val)


    # Print train/test split statistics in the same format as Join module
    train_positives = sum(y_train)
    train_total = len(y_train)
    train_pos_pct = (train_positives / train_total) * 100 if train_total > 0 else 0

    val_positives = sum(y_val)
    val_total = len(y_val)
    val_pos_pct = (val_positives / val_total) * 100 if val_total > 0 else 0

    print("\nDistribution among all candidate training and validation data:")
    print(f"Train positives: {train_positives}/{train_total} ({train_pos_pct:.2f}%)")
    print(f"Validation positives: {val_positives}/{val_total} ({val_pos_pct:.2f}%)")

    # Address class imbalance with sample weights
    sample_weights = compute_sample_weight("balanced", y_train)

    # Train a Gradient Boosting model
    print("\nTraining groupby column prediction model...")
    start_time = time.time()

    # Define parameter grid
    # param_grid = {
    #     'n_estimators': [100, 150, 200],
    #     'learning_rate': [0.05, 0.1],
    #     'max_depth': [3, 4, 5],
    #     'subsample': [0.8, 1.0],
    #     'min_samples_leaf': [1, 3, 5]
    # }

    # Grid search
    # grid_search = GridSearchCV(
    #     estimator=GradientBoostingClassifier(random_state=42),
    #     param_grid=param_grid,
    #     refit=True,
    #     scoring='recall',
    #     cv=StratifiedKFold(n_splits=5),
    #     verbose=1,
    #     n_jobs=-1
    # )

    # Fit the best model
    # grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    # model = grid_search.best_estimator_
    # print("Best groupby column model hyperparameters:", grid_search.best_params_)

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=135,
        subsample=0.6,
        random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Random Forest model
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier(
    #     n_estimators=200,
    #     max_depth=8,
    #     min_samples_leaf=2,
    #     class_weight='balanced',  # balance classes
    #     random_state=42,
    #     n_jobs=-1
    # )
    # model.fit(X_train, y_train, sample_weight=sample_weights)

    end_time = time.time()
    total_training_time = round(end_time - start_time, 2)

    print(f"\nModel training (with hyperparameter tuning) completed in {total_training_time} seconds")
    print(f"Trained model: GradientBoostingClassifier ({model.n_estimators} estimators, max_depth={model.max_depth})")

    # Calculate training metrics
    y_train_pred = model.predict(X_train)  # GradientBoostingClassifier returns 0 and 1 (is groupby or not)
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
    # print("(These metrics show how well the model classifies each column as a groupby or not)")
    print(f"Accuracy: Training = {train_accuracy:.4f}, Validation = {val_accuracy:.4f}")
    print(f"Precision: Training = {train_precision:.4f}, Validation = {val_precision:.4f}")
    print(f"Recall: Training = {train_recall:.4f}, Validation = {val_recall:.4f}")
    # For realistic ranking evaluation (precision@k, ndcg@k), use --mode eval on the true held-out test set.

    # Create directories for results, if not already exist
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Create a dictionary with all relevant metrics (in native python type, json only supports int, float, str, lists, bool and None!)
    metrics_dict = {
        'operator': 'groupby',
        'mode': 'training',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'samples': 90,  # This is hardcoded as we take candidates as inputs here and not the real number of train samples
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
    feature_importance = model.feature_importances_
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
    for i, (feature, importance) in enumerate(top_features, 1):
        metrics_dict[f'top_feature_{i}'] = feature
        metrics_dict[f'importance_{i}'] = float(importance)  # All numpy types to native python before saving into .json

    # All metrics to the JSON file
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
    plt.title('Feature Importance for GroupBy Column Prediction')
    plt.tight_layout()
    plt.savefig('results/figures/groupby_feature_importance.png')
    print("\nMetrics and figures have been saved to the 'results' directory")

    return model, feature_names


def predict_groupby_column_pairs(model, feature_names, table, top_k, verbose=True):
    """
    Predicts scores for each possible (groupby_column, aggregation_column) pair in a table.

    Args:
        model: Trained groupby column prediction model.
        feature_names: List of feature names used by the model.
        table: Input table.
        top_k: Number of top predictions to return
        verbose: Whether to print predictions (default: True)

    Returns:
        List of tuples: (groupby_column, agg_column, score), sorted by score descending.
    """
    pairs_features = []
    all_columns = list(table.columns)

    for g_col in all_columns:
        # Check if groupby column is categorical
        if pd.api.types.is_numeric_dtype(table[g_col]):
            continue  # Skip numeric groupby columns

        for a_col in all_columns:
            # Check if aggregation column is numeric
            if not pd.api.types.is_numeric_dtype(table[a_col]):
                continue  # Skip non-numeric measures
            if pd.api.types.is_datetime64_any_dtype(table[a_col]):
                continue  # Skip datetime as measure
            if g_col == a_col:
                continue  # Skip self-pair
            if table[a_col].nunique() <= 5:
                continue  # Skip low-cardinality numeric column

            # Extract features for this pair
            features = extract_groupby_pair_features(table, g_col, a_col)
            pairs_features.append({
                'groupby_column': g_col,
                'agg_column': a_col,
                'features': features
            })

    # Prepare feature matrix
    X = []
    for pair in pairs_features:
        features = pair['features']
        X.append([features.get(name, 0) for name in feature_names])

    # Avoid empty input to model
    if len(X) == 0:
        return []

    # Predict scores
    scores = model.predict_proba(np.array(X))[:, 1]

    # Combine results
    results = []
    for pair, score in zip(pairs_features, scores):
        results.append((pair['groupby_column'], pair['agg_column'], score))

    # Sort by score (descending)
    results.sort(key=lambda x: x[2], reverse=True)

    # Limit to top_k predictions
    ranked_preds = results[:top_k]

    # Display top predictions if verbose
    if verbose:
        print("\nTop GroupBy-aggregation Predictions:")
        for i, (groupby_col, agg_col, score) in enumerate(ranked_preds, 1):
            print(f"{i}. GroupBy: {groupby_col} ↔ Aggregation: {agg_col} (confidence: {score:.4f})")

    return results


def predict_column_groupby_scores(model, feature_names, input_table):
    """
    Predicts groupby-likelihood scores for each column in the input table.

    Args:
        model: Trained model to predict groupby-likelihood.
        feature_names: List of feature names expected by the model.
        input_table: Input DataFrame.

    Returns:
        List of tuples: (column_name, groupby_score), sorted by descending score.
    """
    column_scores = []

    for column in input_table.columns:
        # Extract features for this column
        features = extract_column_features(input_table, column)

        # Order features according to expected model feature names
        feature_vector = [features.get(fname, 0) for fname in feature_names]
        # print(f"Feature vector for column '{column}': {feature_vector}")

        # Debug: print the extracted feature dictionary
        # print(f"\n=== Column: '{column}' ===")
        # print("Extracted features dict:")
        # for k, v in features.items():
        #     print(f"  {k}: {v}")

        # Debug: print feature names expected by the model
        # print("\nModel expects feature names:")
        # print(feature_names)

        # Predict groupby-likelihood score

        # Fix: Replace NaNs in feature_vector (for next_operation_predictor case)
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)

        # For each sample, we have as many predictions as its columns
        score = model.predict_proba([feature_vector])[0][1]  # probability for class "1" (groupby)

        #print(f"Column: {column}, Score: {score:.4f}")

        column_scores.append((column, score))

    # Sort by descending groupby-likelihood score
    column_scores.sort(key=lambda x: x[1], reverse=True)

    return column_scores


def evaluate_groupby_pairs_model(model, feature_names, test_samples, top_k):
    """
    Evaluates a GroupBy column prediction model on test samples.

    Args:
        model: Trained GroupBy column model
        feature_names: Feature names used by the model
        test_samples: List of test samples
        top_k: Number of k values for precision@k and ndcg@k evaluation (default: [1, 2])

    Returns:
        Dictionary of evaluation metrics
    """
    # Dynamic local Import to avoid a loop (circular import)
    from src.baselines.groupby_baselines import evaluate_baselines

    k_values = list(range(1, top_k + 1))
    correct_at_k = {k: 0 for k in k_values}
    ndcg_sum = {k: 0 for k in k_values}
    full_accuracy_count = 0
    total = 0
    print(f"\nEvaluating Groupby column prediction on {len(test_samples)} test samples...")

    # Track overall binary metrics
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []

    # Keep track of test examples and test positives
    total_test_examples = 0
    total_test_positives = 0

    for sample_idx, sample in enumerate(test_samples):
        input_table = sample['input_table']
        true_groupby_cols = sample['groupby_columns']
        # true_agg_cols = sample['agg_columns']

        # Predict GroupBy columns
        try:
            predictions = predict_groupby_column_pairs(model, feature_names, input_table, max(k_values), verbose=False)
            if not predictions:
                continue
            total += 1

            # Ensure predictions are sorted by score (descending)
            predictions.sort(key=lambda x: x[2], reverse=True)

            # ---------------------------------------------------------------------------
            # Full-accuracy logic (Case 1):
            # A prediction is considered fully correct (full-accuracy=1) only if:
            # 1) All true groupby columns are predicted as groupby columns.
            # 2) No true groupby columns are mistakenly predicted as aggregation columns.
            # 3) All true aggregation columns are predicted as aggregation columns.
            # 4) No true aggregation columns are mistakenly predicted as groupby columns.
            # This matches the baseline evaluation logic and ensures a strict check that
            # both sets (groupby and aggregation) are perfectly separated and matched.
            # ---------------------------------------------------------------------------
            predicted_groupby = [groupby_col for groupby_col, _, score in predictions if score >= 0.5]
            predicted_agg = [agg_col for _, agg_col, score in predictions if score < 0.5]

            all_true_cols_predicted = all(col in predicted_groupby for col in true_groupby_cols)
            no_true_cols_in_agg = not any(col in predicted_agg for col in true_groupby_cols)

            # all_true_aggs_predicted = all(col in predicted_agg for col in true_agg_cols)
            # no_true_aggs_in_groupby = not any(col in predicted_groupby for col in true_agg_cols)

            # print(f"Sample {sample_idx}: full-accuracy condition met? ",
            #       all_true_cols_predicted and no_true_cols_in_agg and
            #       all_true_aggs_predicted and no_true_aggs_in_groupby)

            # Case 1
            # if (all_true_cols_predicted and no_true_cols_in_agg and
            #         all_true_aggs_predicted and no_true_aggs_in_groupby):
            #     full_accuracy_count += 1


            # Case 2: check only groupby columns for full-accuracy (like paper Table 6)
            if all_true_cols_predicted and no_true_cols_in_agg:
                full_accuracy_count += 1

            # # Debug print
            # print(f"\nSample {sample_idx}")
            # print(f"True GroupBy: {true_groupby_cols}")
            # print(f"True Agg: {true_agg_cols}")
            # print(f"Pred GroupBy: {predicted_groupby_cols}")
            # print(f"Pred Agg: {predicted_agg_cols}")
            # print("Match?", predicted_groupby_cols == set(true_groupby_cols) and predicted_agg_cols == set(true_agg_cols))

            # Create relevance labels: 1 if predicted column in ground-truth, 0 otherwise
            y_true = []
            y_pred = []
            for groupby_col, _, score in predictions:
                relevance = 1 if groupby_col in true_groupby_cols else 0
                y_true.append(relevance)
                y_pred.append(score)

                total_test_examples += 1
                total_test_positives += relevance  # counts 1 if correct groupby col

            # Check correct-at-k (for precision) ONLY for grouping columns
            for k in k_values:
                top_k_preds = list(predicted_groupby)[:k]
                if any(pred in true_groupby_cols for pred in top_k_preds):
                    correct_at_k[k] += 1

            # Calculate per-sample NDCG@k ONLY for grouping columns
            if any(y_true):
                for k in k_values:
                    if k <= len(y_true):
                        ndcg = ndcg_score([y_true], [y_pred], k=k)
                        ndcg_sum[k] += ndcg

            # Calculate binary metrics
            y_pred_binary = [1 if s >= 0.5 else 0 for s in y_pred]
            test_accuracy_list.append(accuracy_score(y_true, y_pred_binary))
            test_precision_list.append(precision_score(y_true, y_pred_binary, zero_division=0))
            test_recall_list.append(recall_score(y_true, y_pred_binary, zero_division=0))

        except Exception as e:
            print(f"Error predicting and evaluating groupby columns: {e}")
            traceback.print_exc()
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
    full_accuracy = full_accuracy_count / total

    # Calculate final metrics
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0
        metrics[f'ndcg@{k}'] = ndcg_sum[k] / total if total > 0 else 0

    # Calculate full accuracy (matches "full-accuracy" in Table 6)
    metrics['full-accuracy'] = full_accuracy
    metrics['samples_evaluated'] = total

    # Create full evaluation record
    eval_dict = {
        "operator": "groupby",
        "mode": "evaluation",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "samples": int(len(test_samples)),               # total test samples
        "test_examples": int(total_test_examples),       # total predicted pairs
        "test_positives": int(total_test_positives),     # total correct groupby predictions
        "test_pos_ratio": float(total_test_positives / total_test_examples) if total_test_examples > 0 else 0,
        "test_accuracy": float(test_accuracy),  # averaged binary accuracy (over samples)
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "full-accuracy": float(full_accuracy)  # table-level full accuracy
    }

    # Add ranking metrics
    for k in k_values:
        eval_dict[f'precision@{k}'] = float(metrics[f'precision@{k}'])
        eval_dict[f'ndcg@{k}'] = float(metrics[f'ndcg@{k}'])

    # Converts numpy types to native Python types (important for JSON!)
    eval_dict = numpy_to_list(eval_dict)

    # Save to the JSON
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

    # Print summary of results
    # print("\nGroupBy Column Prediction Results:")
    # print(f"  Samples evaluated: {total}")
    # for metric, value in metrics.items():
    #     if isinstance(value, float):
    #         print(f"  {metric}: {value:.4f}")
    #     else:
    #         print(f"  {metric}: {value}")

    # Calculate the metrics based on other heuristic methods
    baseline_metrics = evaluate_baselines(test_samples, k_values)

    # Generate Table 6
    generate_prediction_table(
        auto_suggest_metrics=metrics,       # The metrics computed for Auto-Suggest
        k_values=k_values,
        baseline_metrics=baseline_metrics,  # Baseline metrics to compare with
        vendor_metrics=GROUPBY_VENDORS,     # Vendor metrics to compare with
        include_full_accuracy=True,
        operator_name="groupby"
    )

    # Generate Table 7 (feature importance)
    feature_importance = model.feature_importances_
    generate_feature_importance_table(feature_importance, feature_names, operator="groupby")

    return metrics


def evaluate_groupby_model(model, feature_names, test_samples, top_k):
    """
    Evaluates a GroupBy column prediction model on test samples.

    Args:
        model: Trained GroupBy column model.
        feature_names: Feature names used by the model.
        test_samples: List of test samples.
        top_k: Maximum k for precision@k and ndcg@k metrics.

    Returns:
        Dictionary of evaluation metrics.
    """
    from src.baselines.groupby_baselines import evaluate_baselines

    k_values = list(range(1, top_k + 1))
    correct_at_k = {k: 0 for k in k_values}
    ndcg_sum = {k: 0 for k in k_values}
    full_accuracy_count = 0
    total = 0

    print(f"\nEvaluating Groupby column prediction on {len(test_samples)} test samples...")

    # Track overall binary metrics
    test_accuracy_list = []
    test_precision_list = []
    test_recall_list = []

    total_test_examples = 0
    total_test_positives = 0

    for sample_idx, sample in enumerate(test_samples):
        input_table = sample['input_table']
        true_groupby_cols = sample['groupby_columns']

        if not true_groupby_cols:
            continue
        total += 1

        try:
            # Predict per-column groupby-likelihood scores
            predictions = predict_column_groupby_scores(model, feature_names, input_table)

            # Create relevance labels for each column
            y_true = []
            y_pred = []
            for col, score in predictions:
                relevance = 1 if col in true_groupby_cols else 0
                y_true.append(relevance)
                y_pred.append(score)

                total_test_examples += 1
                total_test_positives += relevance

            # Check correct-at-k
            ranked_cols = [col for col, _ in predictions]
            for k in k_values:
                top_k_cols = ranked_cols[:k]
                has_true_col = any(col in true_groupby_cols for col in top_k_cols)
                if has_true_col:
                    correct_at_k[k] += 1

            # Full-accuracy: perfect match of predicted groupby columns
            predicted_groupby_cols = [col for col, score in predictions if score >= 0.5]
            all_true_cols_predicted = all(col in predicted_groupby_cols for col in true_groupby_cols)
            no_extra_predicted = all(col in true_groupby_cols for col in predicted_groupby_cols)

            if all_true_cols_predicted and no_extra_predicted:
                full_accuracy_count += 1

            # NDCG@k
            if any(y_true):
                for k in k_values:
                    if k <= len(y_true):
                        ndcg = ndcg_score([y_true], [y_pred], k=k)
                        ndcg_sum[k] += ndcg

            # Binary metrics
            y_pred_binary = [1 if s >= 0.5 else 0 for s in y_pred]
            test_accuracy_list.append(accuracy_score(y_true, y_pred_binary))
            test_precision_list.append(precision_score(y_true, y_pred_binary, zero_division=0))
            test_recall_list.append(recall_score(y_true, y_pred_binary, zero_division=0))

        except Exception as e:
            print(f"Error predicting and evaluating groupby columns for sample {sample_idx}: {e}")
            traceback.print_exc()
            continue

    # Final metrics
    if total == 0:
        test_accuracy = 0.0
        test_precision = 0.0
        test_recall = 0.0
    else:
        test_accuracy = np.mean(test_accuracy_list)
        test_precision = np.mean(test_precision_list)
        test_recall = np.mean(test_recall_list)
    full_accuracy = full_accuracy_count / total

    # Final metrics dict
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0
        metrics[f'ndcg@{k}'] = ndcg_sum[k] / total if total > 0 else 0
    metrics['full-accuracy'] = full_accuracy
    metrics['samples_evaluated'] = total

    # Save results to JSON
    eval_dict = {
        "operator": "groupby",
        "mode": "evaluation",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "samples": int(len(test_samples)),
        "test_examples": int(total_test_examples),
        "test_positives": int(total_test_positives),
        "test_pos_ratio": float(total_test_positives / total_test_examples) if total_test_examples > 0 else 0,
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "full-accuracy": float(full_accuracy)
    }
    for k in k_values:
        eval_dict[f'precision@{k}'] = float(metrics[f'precision@{k}'])
        eval_dict[f'ndcg@{k}'] = float(metrics[f'ndcg@{k}'])

    eval_dict = numpy_to_list(eval_dict)
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

    # Generate tables and feature importance plot
    baseline_metrics = evaluate_baselines(test_samples, k_values)
    generate_prediction_table(
        auto_suggest_metrics=metrics,
        k_values=k_values,
        baseline_metrics=baseline_metrics,
        vendor_metrics=GROUPBY_VENDORS,
        include_full_accuracy=True,
        operator_name="groupby"
    )
    feature_importance = model.feature_importances_
    generate_feature_importance_table(feature_importance, feature_names, operator="groupby")
    print("\nMetrics and figures have been saved to the 'results' directory")

    return metrics



def display_groupby_pair_recommendations(recommendations, top_k, table=None):
    """
    Displays top-k GroupBy-Aggregation column pairs in a readable format.

    Args:
        recommendations: List of (groupby_column, agg_column, score) tuples
        top_k: Number of top predictions to return
        table: Optional input table to show sample values
    """
    if not recommendations:
        print("No recommendations found.")
        return

    # Only display top_k recommendations
    recommendations= recommendations[:top_k]

    # Print recommendations in a readable format
    print("\n=== GroupBy-Aggregation Pair Recommendations ===")
    print("=" * 80)

    for i, (groupby_col, agg_col, score) in enumerate(recommendations, 1):
        print(f"{i}. GroupBy: {groupby_col} ↔ Aggregation: {agg_col} (confidence: {score:.3f})")

        # Show sample values for the groupby column
        if table is not None and groupby_col in table.columns:
            unique_vals = table[groupby_col].nunique()
            sample_vals = table[groupby_col].dropna().unique()[:3]  # Up to 3 samples
            print(
                f"   - GroupBy Column: {unique_vals} unique values, samples: {', '.join(str(v) for v in sample_vals)}")

        # Show numeric statistics for the aggregation column
        if table is not None and agg_col in table.columns and pd.api.types.is_numeric_dtype(table[agg_col]):
            stats = table[agg_col].describe()
            print(
                f"   - Aggregation Column: Range: {stats['min']:.2f} to {stats['max']:.2f}, Mean: {stats['mean']:.2f}")

    # Provide example pandas code for the top recommendation
    if recommendations:
        print("\n=== Example Pandas Code ===")
        print("=" * 80)

        top_groupby_col, top_agg_col, _ = recommendations[0]

        # Determine the best aggregation function based on column type
        if table is not None:
            col_type = str(table[top_agg_col].dtype)
            if 'int' in col_type or 'float' in col_type:
                agg_func = 'sum'  # For numeric columns
            else:
                agg_func = 'count'  # For other types
        else:
            agg_func = 'sum'  # Default to sum

        print("# Using pandas to perform the GroupBy operation:")
        print(f"result = df.groupby('{top_groupby_col}')['{top_agg_col}'].{agg_func}()\n")
        # print("\n# Alternative with agg() for more control:")
        # print(f"result = df.groupby('{top_groupby_col}').agg({{'{top_agg_col}': '{agg_func}'}})")


def display_groupby_recommendations(recommendations, top_k, table=None):
    """
    Displays top-k GroupBy column recommendations in a readable format.

    Args:
        recommendations: List of (column_name, groupby_score) tuples
        top_k: Number of top predictions to return
        table: Optional input table to show sample values
    """
    if not recommendations:
        print("No recommendations found.")
        return

    # Only display top_k recommendations
    recommendations = recommendations[:top_k]

    # Print recommendations
    print("\n=== Top-k GroupBy Column Recommendations ===")
    print("=" * 80)

    for i, (col, score) in enumerate(recommendations, 1):
        print(f"{i}. GroupBy Column: {col} (confidence: {score:.3f})")

        if table is not None and col in table.columns:
            # Show number of unique values and up to 3 sample values for quick inspection
            unique_vals = table[col].nunique()
            sample_vals = table[col].dropna().unique()[:3]
            print(f"   - Unique values: {unique_vals}, samples: {', '.join(str(v) for v in sample_vals)}")

    # Provide example pandas code for the top recommendation
    if recommendations:
        print("\n=== Example Pandas Code ===")
        print("=" * 80)

        top_groupby_col, _ = recommendations[0]

        # Suggest a default aggregation function
        if table is not None:
            # Suggest a numeric column for aggregation
            numeric_cols = [col for col in table.columns if pd.api.types.is_numeric_dtype(table[col])]
            if numeric_cols:
                suggested_agg_col = numeric_cols[0]
                print("# Using pandas to perform the GroupBy operation:")
                print(f"result = df.groupby('{top_groupby_col}')['{suggested_agg_col}'].sum()\n")
            else:
                print(f"# GroupBy example (no numeric columns found for aggregation):")
                print(f"result = df.groupby('{top_groupby_col}').size()")
        else:
            print(f"# GroupBy example (no table loaded):")
            print(f"result = df.groupby('{top_groupby_col}').size()")
