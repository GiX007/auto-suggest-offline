# src/models/join_col_model.py
#
# Implementation of join column prediction based on Section 4.1 of the
# "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Processes extracted join samples to prepare training data
# 2. Trains a gradient boosting model to predict join columns
# 3. Evaluates model performance with standard metrics
# 4. Provides functions to predict join columns for new tables
# 5. Handles various join column formats and special cases

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Import from our package structure
from src.features.join_features import extract_join_column_features, generate_join_candidates, should_exclude_column
from src.utils.evaluation import evaluate_per_sample_ranking, calculate_accuracy


def prepare_join_training_data(processed_samples: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Prepare training data for join column prediction.

    This function processes each sample to generate multiple candidate join column pairs.
    For each sample, it:
    1. Extracts the ground truth join keys
    2. Generates multiple candidate join column pairs
    3. Labels each candidate as 1 (matches ground truth) or 0 (doesn't match)

    Note that the number of positive examples may be less than the number of samples
    if some samples don't generate any candidates matching the ground truth.

    Args:
        processed_samples: List of processed join samples with ground truth join keys

    Returns:
        Tuple containing list of feature dictionaries and corresponding labels (1 for correct join columns, 0 otherwise)
    """
    features_list = []
    labels = []
    positive_examples_count = 0
    sample_contributing_positives = 0

    for sample_idx, sample in enumerate(processed_samples):
        left_table = sample['left_table']
        right_table = sample['right_table']

        # Get the original ground truth keys
        orig_left_keys = sample['left_join_keys'].copy()
        orig_right_keys = sample['right_join_keys'].copy()

        # Drop 'Unnamed: 0' if 'index' is present in the ground truth
        if 'index' in orig_left_keys and 'Unnamed: 0' in orig_left_keys:
            orig_left_keys.remove('Unnamed: 0')
        if 'index' in orig_right_keys and 'Unnamed: 0' in orig_right_keys:
            orig_right_keys.remove('Unnamed: 0')

        # Deduplicate index keys if needed
        if 'index' in orig_left_keys and orig_left_keys.count('index') > 1:
            orig_left_keys = ['index']
        if 'index' in orig_right_keys and orig_right_keys.count('index') > 1:
            orig_right_keys = ['index']

        # Check if ground truth ONLY consists of index or unnamed columns
        left_has_real_cols = any(not col.startswith('Unnamed:') and col != 'index' for col in orig_left_keys)
        right_has_real_cols = any(not col.startswith('Unnamed:') and col != 'index' for col in orig_right_keys)

        # If ground truth only has index/unnamed columns
        if not left_has_real_cols and not right_has_real_cols:
            # For matching candidates, map 'index' to actual column name in table
            match_left = []
            for col in orig_left_keys:
                if col == 'index' and 'Unnamed: 0' in left_table.columns:
                    match_left.append('Unnamed: 0')
                else:
                    match_left.append(col)

            match_right = []
            for col in orig_right_keys:
                if col == 'index' and 'Unnamed: 0' in right_table.columns:
                    match_right.append('Unnamed: 0')
                else:
                    match_right.append(col)

            # For display, keep original keys
            display_left = orig_left_keys
            display_right = orig_right_keys
        else:
            # Filter out unnamed/index columns if we have real columns
            match_left = [col for col in orig_left_keys if not col.startswith('Unnamed:') and col != 'index']
            match_right = [col for col in orig_right_keys if not col.startswith('Unnamed:') and col != 'index']

            # Display the same columns
            display_left = match_left
            display_right = match_right

        # Generate candidate join columns
        candidates = generate_join_candidates(left_table, right_table, sample_id=sample_idx)

        # Get counts
        single_col_candidates = [c for c in candidates if len(c[0]) == 1 and len(c[1]) == 1]
        multi_col_candidates = [c for c in candidates if len(c[0]) > 1 or len(c[1]) > 1]

        # If there are empty join keys, we need to handle this special case
        # Empty join keys in the processed sample usually indicate 'index' joins
        empty_keys = len(match_left) == 0 or len(match_right) == 0

        # Fix for the case where index-based joins resulted in empty key lists
        if empty_keys:
            # Use the original keys from the sample, which should include 'index'
            match_left = orig_left_keys
            match_right = orig_right_keys

            # Make sure these actually exist in the tables after processing
            match_left = [col for col in match_left if col in left_table.columns]
            match_right = [col for col in match_right if col in right_table.columns]

            if not match_left or not match_right:
                print(f"Warning: Could not find valid join columns for index-based join")
                continue

        # Track if this sample contributes any positive examples
        found_match_for_sample = False

        for left_cols, right_cols in candidates:
            # Extract features
            features = extract_join_column_features(left_table, right_table, left_cols, right_cols)

            if 'sample_id' in sample:
                features['sample_id'] = sample['sample_id']
            else:
                features['sample_id'] = sample_idx

            features_list.append(features)

            # Check if this candidate matches the ground truth
            is_match = (set(left_cols) == set(match_left) and
                        set(right_cols) == set(match_right))

            if is_match:
                labels.append(1)
                positive_examples_count += 1
                found_match_for_sample = True
            else:
                labels.append(0)

        # Count samples that contribute at least one positive example
        if found_match_for_sample:
            sample_contributing_positives += 1

    return features_list, labels


def train_join_column_model(features_list: List[Dict], labels: List[int]):
    """
    Train a model to predict join columns.

    Args:
        features_list: List of feature dictionaries for join column candidates
        labels: List of 0/1 labels (1 for correct join columns, 0 otherwise)

    Returns:
        Trained model and list of feature names used by the model
    """
    # Check if we have any training data
    if len(features_list) == 0 or len(labels) == 0:
        print("Error: No training data available. Cannot train model.")
        return None, []

    # Convert list of dictionaries to DataFrame
    features_df = pd.DataFrame(features_list)

    # Convert boolean features to integers (0/1)
    bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)

    # Convert any object/string columns to numeric if possible
    # Skip sample_id as we don't need to convert it
    for col in features_df.select_dtypes(include=['object']).columns:
        if col != 'sample_id':  # Skip sample_id
            try:
                features_df[col] = pd.to_numeric(features_df[col])
            except:
                print(f"Warning: Could not convert column '{col}' to numeric")

    # Include all columns except sample_id
    feature_cols = [col for col in features_df.columns if col != 'sample_id']

    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

    # Convert features DataFrame to numpy array
    X = features_df[feature_cols].values
    y = np.array(labels)

    feature_names = feature_cols

    # Split data into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Get sample IDs for test set (needed for correct evaluation)
    test_indices = np.arange(len(X))[len(X_train):]
    test_sample_ids = features_df.iloc[test_indices]['sample_id'].values

    # Add debug prints to verify distribution of positive examples
    print("\nDistribution among all candidate join column pairs:")
    print(f"Train positives: {sum(y_train)}/{len(y_train)} "
          f"({sum(y_train) / len(y_train) * 100:.2f}%) — from all candidate pairs generated in training samples")
    print(f"Test positives:  {sum(y_test)}/{len(y_test)} "
          f"({sum(y_test) / len(y_test) * 100:.2f}%) — from all candidate pairs generated in test samples")

    # Address class imbalance with sample weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    # Train a Gradient Boosting model
    print("\nTraining join column prediction model...")
    start_time = time.time()

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"\nModel training completed in {time.time() - start_time:.2f} seconds")
    print("Trained model: GradientBoostingClassifier (100 estimators, max_depth=3)")

    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    y_train_pred_binary = (y_train_pred >= 0.5).astype(int)
    train_accuracy = np.mean(y_train_pred_binary == y_train)

    # Calculate precision for training set
    from sklearn.metrics import precision_score
    train_precision = precision_score(y_train, y_train_pred_binary, zero_division=0)

    # Calculate test metrics
    y_test_pred = model.predict(X_test)
    y_test_pred_binary = (y_test_pred >= 0.5).astype(int)
    test_accuracy = np.mean(y_test_pred_binary == y_test)

    # Calculate precision for test set
    test_precision = precision_score(y_test, y_test_pred_binary, zero_division=0)

    # Print comparison of training vs test metrics (how well the model identifies positive join pairs among all candidates)
    print("\nBinary Classification Metrics on Train and Test Sets (using threshold 0.5):")
    #print("(These metrics show how well the model classifies candidate join column pairs)")
    print(f"Accuracy: Training = {train_accuracy:.4f}, Test = {test_accuracy:.4f}")
    print(f"Precision: Training = {train_precision:.4f}, Test = {test_precision:.4f}")

    # Evaluate on test set
    y_pred = model.predict(X_test)

    # Use threshold of 0.5 for binary predictions
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Calculate basic accuracy
    accuracy = np.mean(y_pred_binary == y_test)

    # Use evaluate_per_sample_ranking to get per-sample metrics
    # This groups predictions by their sample ID and calculates top-k accuracy per sample
    eval_metrics = evaluate_per_sample_ranking(test_sample_ids, y_test, y_pred, k_values=[1, 2])

    # NOTE: This ranking evaluation is on the internal test split used during training.
    # It often shows perfect scores (e.g. precision@1 = 1.0) due to overfitting (or data leakage).
    # For realistic metrics, use --mode eval on a fresh test split.

    # print("\nRanking Evaluation on Test Set (as in the Auto-Suggest paper):")
    # print("precision@k: Proportion of correct join columns in the top-k recommendations")
    # for metric, value in eval_metrics.items():
    #     print(f"  {metric}: {value:.4f}")

    # Store feature importance
    feature_importance = model.feature_importances_

    # Create directories for results
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Create a dictionary with all relevant metrics
    metrics_dict = {
        'operator': 'join',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'train_examples': len(y_train),
        'train_positives': sum(y_train),
        'train_pos_ratio': sum(y_train) / len(y_train),
        'test_examples': len(y_test),
        'test_positives': sum(y_test),
        'test_pos_ratio': sum(y_test) / len(y_test),
        'model_type': 'GradientBoostingRegressor',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'training_time': time.time() - start_time,
        'num_features': len(feature_names),
        'accuracy': accuracy,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision
    }

    # Add all evaluation metrics to the dictionary
    for metric, value in eval_metrics.items():
        metrics_dict[metric] = value

    # Add top 5 more important features
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
    for i, (feature, importance) in enumerate(top_features, 1):
        metrics_dict[f'top_feature_{i}'] = feature
        metrics_dict[f'importance_{i}'] = importance

    # Convert to DataFrame for easy CSV export
    metrics_df = pd.DataFrame([metrics_dict])

    # Save to a combined metrics file for all operators
    combined_metrics_file = f'results/metrics/all_operators_metrics.csv'
    if os.path.exists(combined_metrics_file):
        metrics_df.to_csv(combined_metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(combined_metrics_file, index=False)

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


def predict_join_columns(model, feature_names, left_table, right_table, top_k=3, verbose=True):
    """
    Predict join columns for two tables.

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

    # Extract features for each candidate
    candidate_features = []
    for left_cols, right_cols in candidates:
        features = extract_join_column_features(left_table, right_table, left_cols, right_cols)
        candidate_features.append((left_cols, right_cols, features))

    # Prepare features for prediction
    X = []
    for _, _, features in candidate_features:
        X.append([features.get(name, 0) for name in feature_names])

    # Predict scores
    scores = model.predict(np.array(X))

    # Combine candidates with their scores
    # Convert 'Unnamed: 0' to 'index' in the displayed results
    results = []
    for (left_cols, right_cols, _), score in zip(candidate_features, scores):
        # For display, convert 'Unnamed: 0' to 'index'
        display_left = ['index' if col == 'Unnamed: 0' else col for col in left_cols]
        display_right = ['index' if col == 'Unnamed: 0' else col for col in right_cols]
        results.append((display_left, display_right, score))

    # Sort by score in descending order
    results.sort(key=lambda x: x[2], reverse=True)

    # Limit to top_k results
    top_results = results[:top_k]

    # Display the top predictions if verbose mode is on
    if verbose:
        print("\nTop Join Column Predictions:")
        for i, (left_cols, right_cols, score) in enumerate(top_results[:top_k], 1):
            left_str = ", ".join(left_cols)
            right_str = ", ".join(right_cols)
            print(f"{i}. Left columns: [{left_str}] ↔ Right columns: [{right_str}] (confidence: {score:.4f})")

    return results