# src/models/groupby_model.py
#
# This module implements the GroupBy model training, evaluation, and prediction functionality
# of the "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks" paper.
#
# The GroupBy model is responsible for predicting which columns in a table should be used for
# grouping (dimensions) versus which columns should be used for aggregation (measures).
# This is a central component of the Auto-Suggest system that helps users with data preparation.
#
# Key functionality:
# 1. prepare_groupby_training_data: Prepares training data for the model
# 2. train_groupby_model: Trains a gradient boosting model to predict GroupBy columns
# 3. predict_groupby_columns: Predicts groupby/aggregation columns for a new table
# 4. recommend_groupby: Provides complete recommendations with confidence scores
# 5. display_groupby_recommendations: Formats and displays the recommendations
# 6. predict_on_file: Runs the prediction pipeline on a CSV file
#
# This module combines both model and pipeline functionality in one place
# to provide a streamlined implementation compared to the Join operator which
# requires separate models for column and type prediction.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import List, Dict, Tuple, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Import from our package structure
from src.features.groupby_features import extract_column_features
from src.utils.evaluation import evaluate_predictions, calculate_accuracy


def prepare_groupby_training_data(processed_samples: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """
    Prepare training data for groupby column prediction.

    Args:
        processed_samples: List of processed groupby samples.

    Returns:
        Tuple of (features_list, labels) where features_list is a list of feature dictionaries
        and labels is a list of 0/1 indicating if the column is a groupby column.
    """
    features_list = []
    labels = []

    print("\nPreparing GroupBy training data from {} samples".format(len(processed_samples)))

    for sample in processed_samples:
        input_table = sample['input_table']
        groupby_columns = sample['groupby_columns']
        agg_columns = sample['agg_columns']

        # Extract features for all columns in the input table
        for column_name in input_table.columns:
            # Extract features for this column
            features = extract_column_features(input_table, column_name)

            # Add sample ID for reference
            features['sample_id'] = sample['sample_id']
            features['column_name'] = column_name

            features_list.append(features)

            # Check if this column is a groupby column
            is_groupby = column_name in groupby_columns
            labels.append(1 if is_groupby else 0)

    # Print stats similar to Join module format
    positive_count = sum(labels)
    total_count = len(labels)
    positive_percentage = (positive_count / total_count) * 100 if total_count > 0 else 0

    print(f"Prepared {total_count} training instances, {positive_count} positive examples ({positive_percentage:.2f}%)")

    return features_list, labels


def train_groupby_model(features_list: List[Dict], labels: List[int]):
    """
    Train a model to predict groupby columns.

    Args:
        features_list: List of feature dictionaries.
        labels: List of 0/1 labels.

    Returns:
        Trained model and feature names.
    """
    # Convert list of dictionaries to DataFrame
    features_df = pd.DataFrame(features_list)

    # Remove non-feature columns
    non_feature_cols = ['sample_id', 'column_name']
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    # Handle missing values
    features_df = features_df.fillna(0)

    # Feature names for model
    feature_names = feature_cols

    # Print feature names in a similar format to Join module
    print("\nUsing {} features: {}".format(len(feature_names), feature_names))

    # Convert features DataFrame to numpy array
    X = features_df[feature_cols].values
    y = np.array(labels)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Print train/test split statistics in the same format as Join module
    train_positives = sum(y_train)
    train_total = len(y_train)
    train_pos_pct = (train_positives / train_total) * 100 if train_total > 0 else 0

    test_positives = sum(y_test)
    test_total = len(y_test)
    test_pos_pct = (test_positives / test_total) * 100 if test_total > 0 else 0

    print("\nDistribution among all candidate join column pairs:")
    print(f"\nTrain positives: {train_positives}/{train_total} ({train_pos_pct:.2f}%) — from all candidate pairs generated in training samples")
    print(f"Test positives: {test_positives}/{test_total} ({test_pos_pct:.2f}%) — from all candidate pairs generated in test samples")

    # Train a Gradient Boosting model
    print("\nTraining groupby column prediction model...")
    start_time = time.time()

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

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

    # Print comparison of training vs test metrics (like in Join module)
    print("\nBinary Classification Metrics on Train and Test Sets:")
    print(f"Accuracy: Training = {train_accuracy:.4f}, Test = {test_accuracy:.4f}")
    print(f"Precision: Training = {train_precision:.4f}, Test = {test_precision:.4f}")

    # Evaluate on test set for ranking metrics
    metrics = evaluate_predictions(y_test, y_test_pred, k_values=[1, 2])

    # Format ranking metrics similar to Join module
    # print("\nRanking Metrics (as in the paper):")
    # print("  precision@k: Proportion of correct GroupBy columns in the top-k recommendations")
    # for metric, value in metrics.items():
    #     print(f"  {metric}: {value:.4f}")

    # Calculate and store feature importance but don't print it
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Create directories for metrics and figures
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Visualize feature importance and save to file
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for GroupBy Column Prediction')
    plt.tight_layout()
    plt.savefig('results/figures/groupby_feature_importance.png')

    print("\nMetrics and figures have been saved to the 'results' directory")

    # Save to a combined metrics file for all operators
    combined_metrics_file = 'results/metrics/all_operators_metrics.csv'

    # Collect metrics in a compatible format
    groupby_metrics_df = pd.DataFrame([{
        'operator': 'groupby',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'train_examples': len(y_train),
        'train_positives': int(train_positives),
        'train_pos_ratio': train_pos_pct / 100,
        'test_examples': len(y_test),
        'test_positives': int(test_positives),
        'test_pos_ratio': test_pos_pct / 100,
        'model_type': 'GradientBoostingClassifier',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'training_time': time.time() - start_time,
        'num_features': len(feature_names),
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'precision@1': metrics.get('precision@1', 0),
        'ndcg@1': metrics.get('ndcg@1', 0),
        'precision@2': metrics.get('precision@2', 0),
        'ndcg@2': metrics.get('ndcg@2', 0),
    }])

    # Write or append to the combined CSV
    if os.path.exists(combined_metrics_file):
        groupby_metrics_df.to_csv(combined_metrics_file, mode='a', header=False, index=False)
    else:
        groupby_metrics_df.to_csv(combined_metrics_file, index=False)

    return model, feature_names


def predict_groupby_columns(model, feature_names, table):
    """
    Predict groupby and aggregation columns for a table.

    Args:
        model: Trained groupby column prediction model.
        feature_names: List of feature names used by the model.
        table: Input table.

    Returns:
        List of tuples containing (column, score) sorted by score.
    """
    # Extract features for each column
    column_features = []
    for column_name in table.columns:
        features = extract_column_features(table, column_name)
        column_features.append((column_name, features))

    # Prepare features for prediction
    X = []
    for _, features in column_features:
        X.append([features.get(name, 0) for name in feature_names])

    # Predict scores
    scores = model.predict(np.array(X))

    # Combine columns with their scores
    results = [(col, score) for (col, _), score in zip(column_features, scores)]

    # Sort by score (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def recommend_groupby(model, feature_names, table, top_k=3):
    """
    Generate GroupBy and Aggregation column recommendations for a table.

    Args:
        model: Trained GroupBy column model
        feature_names: Feature names used by the model
        table: Input table
        top_k: Number of top recommendations to return

    Returns:
        Dictionary with recommendations for GroupBy and Aggregation columns
    """
    # Predict scores for all columns
    column_scores = predict_groupby_columns(model, feature_names, table)

    # Split predictions into GroupBy and Aggregation columns
    # GroupBy columns have higher scores (closer to 1)
    # Aggregation columns have lower scores (closer to 0)

    # Sort by score (descending)
    ranked_columns = sorted(column_scores, key=lambda x: x[1], reverse=True)

    # Get column names and their actual table data types for better recommendations
    column_types = {col: str(table[col].dtype) for col, _ in ranked_columns}

    # Get top-k columns for GroupBy (dimensions)
    groupby_columns = [(col, score) for col, score in ranked_columns[:top_k] if score >= 0.5]

    # Get top-k columns for Aggregation (measures)
    # Prioritize numeric columns that have scores < 0.5
    agg_candidates = [(col, score) for col, score in ranked_columns if score < 0.5]
    agg_columns = []

    # Prioritize numeric columns for aggregation
    for col, score in agg_candidates:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(table[col]):
            agg_columns.append((col, score))
            if len(agg_columns) >= top_k:
                break

    # If we don't have enough numeric columns, add non-numeric ones
    if len(agg_columns) < top_k:
        for col, score in agg_candidates:
            if (col, score) not in agg_columns:
                agg_columns.append((col, score))
                if len(agg_columns) >= top_k:
                    break

    return {
        'groupby_columns': groupby_columns,
        'agg_columns': agg_columns,
        'column_types': column_types,
        'all_columns': ranked_columns
    }


def display_groupby_recommendations(recommendations, table=None):
    """
    Display GroupBy recommendations in a readable format.

    Args:
        recommendations: Dictionary with GroupBy recommendations
        table: Optional input table to show sample values
    """
    if not recommendations:
        print("No recommendations found.")
        return

    # Print recommendations in a readable format
    print("\n=== GroupBy Column Recommendations ===")
    print("=" * 80)

    # Display GroupBy column recommendations
    print("\nRecommended GroupBy Columns (Dimensions):")
    print("-" * 60)
    for i, (col, score) in enumerate(recommendations['groupby_columns'], 1):
        col_type = recommendations['column_types'].get(col, 'unknown')
        print(f"{i}. {col} (confidence: {score:.3f}, type: {col_type})")
        # Show sample values if table is provided
        if table is not None and col in table.columns:
            unique_vals = table[col].nunique()
            sample_vals = table[col].dropna().unique()[:3]  # Show up to 3 sample values
            print(f"   - {unique_vals} unique values, samples: {', '.join(str(v) for v in sample_vals)}")

    # Display Aggregation column recommendations
    print("\nRecommended Aggregation Columns (Measures):")
    print("-" * 60)
    for i, (col, score) in enumerate(recommendations['agg_columns'], 1):
        col_type = recommendations['column_types'].get(col, 'unknown')
        print(f"{i}. {col} (confidence: {1 - score:.3f}, type: {col_type})")
        # Show numeric statistics if table is provided
        if table is not None and col in table.columns and pd.api.types.is_numeric_dtype(table[col]):
            stats = table[col].describe()
            print(f"   - Range: {stats['min']:.2f} to {stats['max']:.2f}, Mean: {stats['mean']:.2f}")

    # Provide example pandas code for the top recommendation
    if recommendations['groupby_columns'] and recommendations['agg_columns']:
        print("\n=== Example Pandas Code ===")
        print("=" * 80)

        # Get top GroupBy columns
        groupby_cols = [col for col, _ in recommendations['groupby_columns']]
        # Get top Aggregation column
        agg_col = recommendations['agg_columns'][0][0] if recommendations['agg_columns'] else None

        # Generate pandas code based on column types
        if groupby_cols and agg_col:
            groupby_str = ", ".join(f"'{col}'" for col in groupby_cols)

            # Determine best aggregation function based on column type
            col_type = recommendations['column_types'].get(agg_col, 'unknown')
            if 'int' in col_type or 'float' in col_type:
                agg_func = 'sum'  # For numeric columns
            else:
                agg_func = 'count'  # For other column types

            print("# Using pandas to perform the GroupBy operation:")
            print(f"result = df.groupby([{groupby_str}])['{agg_col}'].{agg_func}()")
            print("\n# Alternative with agg() for more control:")
            print(f"result = df.groupby([{groupby_str}]).agg({{'{agg_col}': '{agg_func}'}})")

            # Show reset_index for better formatting
            print("\n# For a tabular result:")
            print(f"result = df.groupby([{groupby_str}])['{agg_col}'].{agg_func}().reset_index()\n")


def predict_on_file(file_path, model_dir='models', top_k=3):
    """
    Run GroupBy prediction on a CSV file.

    Args:
        file_path: Path to input table CSV
        model_dir: Directory containing trained models
        top_k: Number of top recommendations to return

    Returns:
        Dictionary with GroupBy recommendations
    """
    from src.utils.model_utils import load_model

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return None

    # Load table
    try:
        table = pd.read_csv(file_path)

        print(f"\nLoaded table: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"Column names: {', '.join(table.columns)}")

        # Show sample rows
        print("\nSample data (first 3 rows):")
        print(table.head(3),"\n")

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Load model
    try:
        model_path = os.path.join(model_dir, "groupby_column_model.pkl")
        model_data = load_model(model_path)
        model, feature_names = model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first.")
        return None

    # Generate GroupBy recommendations
    recommendations = recommend_groupby(model, feature_names, table, top_k)

    # Display recommendations
    display_groupby_recommendations(recommendations, table)

    return recommendations