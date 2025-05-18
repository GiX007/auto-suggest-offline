# src/models/join_type_model.py
#
# Implementation of join type prediction (inner, left, right, full) based on Section 4.1
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Extracts features from tables that signal which join type is appropriate
# 2. Trains a gradient boosting model to predict join types
# 3. Uses signals like table size ratios and column overlap for prediction
# 4. Complements join column prediction by determining how tables should be combined

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import warnings
from typing import List, Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore')

# Import from our package structure
from src.features.join_features import extract_join_column_features
from src.utils.evaluation import evaluate_predictions


def extract_join_type_features(left_table: pd.DataFrame, right_table: pd.DataFrame,
                               left_join_keys: List[str], right_join_keys: List[str]) -> Dict:
    """
    Extract features for join type prediction based on Section 4.1 of the paper.

    Args:
        left_table: The left table for the join
        right_table: The right table for the join
        left_join_keys: Join columns from the left table
        right_join_keys: Join columns from the right table

    Returns:
        Dictionary of features for join type prediction
    """
    features = {}

    # Table size features
    features['left_row_count'] = len(left_table)
    features['right_row_count'] = len(right_table)
    features['left_col_count'] = len(left_table.columns)
    features['right_col_count'] = len(right_table.columns)

    # Size ratios (important for determining join type as per the paper)
    features['row_count_ratio'] = len(left_table) / len(right_table) if len(right_table) > 0 else float('inf')
    features['col_count_ratio'] = len(left_table.columns) / len(right_table.columns) if len(
        right_table.columns) > 0 else float('inf')

    # Which table is "larger" overall - a key signal mentioned in the paper
    features['left_is_larger_table'] = (len(left_table) * len(left_table.columns)) > (
            len(right_table) * len(right_table.columns))

    # Determine if the join is likely a "filtering" join
    # The paper mentions if one table has few columns or its columns are contained in the other
    features['right_is_small_cols'] = len(right_table.columns) <= 3  # Arbitrary threshold
    features['left_is_small_cols'] = len(left_table.columns) <= 3  # Arbitrary threshold

    # Check if the join keys are the only columns in either table
    features['left_keys_only_cols'] = len(left_join_keys) == len(left_table.columns)
    features['right_keys_only_cols'] = len(right_join_keys) == len(right_table.columns)

    # Check column overlap (excluding join keys)
    left_non_key_cols = set(left_table.columns) - set(left_join_keys)
    right_non_key_cols = set(right_table.columns) - set(right_join_keys)
    common_non_key_cols = left_non_key_cols.intersection(right_non_key_cols)

    features['non_key_col_overlap'] = len(common_non_key_cols)
    features['non_key_col_overlap_ratio'] = len(common_non_key_cols) / len(left_non_key_cols) if len(
        left_non_key_cols) > 0 else 0

    # Value overlap in join columns (to determine if outer join is needed)
    try:
        # Handle multi-column joins
        if len(left_join_keys) > 1 or len(right_join_keys) > 1:
            # Create compound keys for comparison
            left_key_values = set(tuple(row) for row in left_table[left_join_keys].astype(str).values)
            right_key_values = set(tuple(row) for row in right_table[right_join_keys].astype(str).values)
        else:
            # Single column join
            left_key_values = set(left_table[left_join_keys[0]].astype(str))
            right_key_values = set(right_table[right_join_keys[0]].astype(str))

        # Count values that appear in both tables
        intersection = left_key_values.intersection(right_key_values)

        # Calculate containment in both directions
        features['left_in_right_containment'] = len(intersection) / len(left_key_values) if len(
            left_key_values) > 0 else 0
        features['right_in_left_containment'] = len(intersection) / len(right_key_values) if len(
            right_key_values) > 0 else 0

        # Low containment in either direction suggests the need for an outer join
        features['min_containment'] = min(features['left_in_right_containment'], features['right_in_left_containment'])
        features['max_containment'] = max(features['left_in_right_containment'], features['right_in_left_containment'])

    except Exception as e:
        # Silently handle errors
        features['left_in_right_containment'] = 0
        features['right_in_left_containment'] = 0
        features['min_containment'] = 0
        features['max_containment'] = 0

    # Also include some features from join column prediction that might be relevant
    # (as mentioned in the paper, features used for join-column-prediction can also be used here)
    join_col_features = extract_join_column_features(left_table, right_table, left_join_keys, right_join_keys)

    # Add selected features from join column prediction
    features['jaccard_similarity'] = join_col_features.get('jaccard_similarity', 0)
    features['left_is_sorted'] = join_col_features.get('left_is_sorted', False)
    features['right_is_sorted'] = join_col_features.get('right_is_sorted', False)

    return features


def prepare_join_type_training_data(processed_samples: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """
    Prepare training data for join type prediction.

    Args:
        processed_samples: List of processed samples with join information.

    Returns:
        Tuple of features list and join type labels.
    """
    features_list = []
    labels = []

    print(f"\nPreparing join type training data from {len(processed_samples)} samples")

    # NOTE: Unlike join column prediction, this step does not generate join key candidates.
    # It uses the ground truth join keys directly to extract features for predicting the join type.

    for sample_idx, sample in enumerate(processed_samples):
        left_table = sample['left_table']
        right_table = sample['right_table']
        left_join_keys = sample['left_join_keys']
        right_join_keys = sample['right_join_keys']
        join_type = sample['join_type']

        # Extract features for join type prediction
        features = extract_join_type_features(left_table, right_table, left_join_keys, right_join_keys)

        # Add sample_id if available
        if 'sample_id' in sample:
            features['sample_id'] = sample['sample_id']
        else:
            features['sample_id'] = sample_idx

        features_list.append(features)
        labels.append(join_type)

    # Count distribution of join types
    join_type_counts = {}
    for join_type in labels:
        join_type_counts[join_type] = join_type_counts.get(join_type, 0) + 1

    print("\nJoin type distribution:")
    for join_type, count in join_type_counts.items():
        print(f"  {join_type}: {count} ({count / len(labels) * 100:.1f}%)")

    return features_list, labels


def train_join_type_model(features_list: List[Dict], labels: List[str]):
    """
    Train a model to predict join types.

    Args:
        features_list: List of feature dictionaries
        labels: List of join type labels (e.g., 'inner', 'left', 'right', 'outer')

    Returns:
        Trained model, feature names, and label encoder
    """
    # Check if we have any training data
    if not features_list or not labels:
        print("Error: No training data available for join type prediction")
        return None, [], None

    # Convert list of dictionaries to DataFrame
    features_df = pd.DataFrame(features_list)

    # Convert boolean features to integers (0/1)
    bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)

    # Convert non-numeric columns
    for col in features_df.select_dtypes(include=['object']).columns:
        if col != 'sample_id':  # Skip sample_id
            try:
                features_df[col] = pd.to_numeric(features_df[col])
            except:
                # Silently convert columns
                pass

    # Include all columns except sample_id
    feature_cols = [col for col in features_df.columns if col != 'sample_id']
    feature_names = feature_cols

    # Convert features DataFrame to numpy array
    X = features_df[feature_cols].values

    # Encode join type labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    print(f"\nJoin type classes: {label_encoder.classes_}")

    # Count the labels for each class
    class_counts = np.bincount(y_encoded)

    # Check if any class has only 1 sample - this will cause stratification to fail
    has_singleton_class = any(count == 1 for count in class_counts)

    # Split data into train and test sets, using stratification only if appropriate
    # (i.e., if each class has at least 2 samples)
    if has_singleton_class:
        print("\nWARNING: At least one join type has only 1 example. Using random split instead of stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
    else:
        # Use stratified split as before
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    # Check class distribution
    train_class_counts = {label_encoder.classes_[i]: (y_train == i).sum() for i in range(len(label_encoder.classes_))}
    test_class_counts = {label_encoder.classes_[i]: (y_test == i).sum() for i in range(len(label_encoder.classes_))}

    print("\nTrain set distribution:")
    for label, count in train_class_counts.items():
        print(f"  {label}: {count} ({count / len(y_train) * 100:.1f}%)")

    print("\nTest set distribution:")
    for label, count in test_class_counts.items():
        print(f"  {label}: {count} ({count / len(y_test) * 100:.1f}%)")

    # Address class imbalance with sample weights
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    # Print the features being used (similar to join column model)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

    # Train a Gradient Boosting model
    print("\nTraining join type prediction model...")
    start_time = time.time()

    # Use deeper trees and more estimators here since join type prediction has fewer samples and harder class boundaries than join column prediction
    # More estimators and lower learning rate = best learning with finer updates, with higher depth model can learn more complex decision boundaries
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    print(f"\nModel training completed in {time.time() - start_time:.2f} seconds")
    print("Trained model: GradientBoostingClassifier (200 estimators, max_depth=8)")

    # Calculate training metrics
    y_train_pred = model.predict(X_train)
    y_train_pred_classes = np.round(y_train_pred).astype(int)
    y_train_pred_classes = np.clip(y_train_pred_classes, 0, len(label_encoder.classes_) - 1)

    # Convert to original labels for evaluation
    y_train_orig = label_encoder.inverse_transform(y_train)
    y_train_pred_labels = label_encoder.inverse_transform(y_train_pred_classes)
    train_accuracy = np.mean(y_train_pred_labels == y_train_orig)

    # Calculate test metrics
    y_test_pred = model.predict(X_test)
    y_test_pred_classes = np.round(y_test_pred).astype(int)
    y_test_pred_classes = np.clip(y_test_pred_classes, 0, len(label_encoder.classes_) - 1)

    # Convert to original labels for evaluation
    y_test_orig = label_encoder.inverse_transform(y_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred_classes)
    test_accuracy = np.mean(y_test_pred_labels == y_test_orig)

    # Print comparison of training vs test metrics
    print("\nBinary Classification Metrics on Train and Test Sets:")
    print(f"Accuracy: Training = {train_accuracy:.4f}, Test = {test_accuracy:.4f}")

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    # Get unique class labels that are present in either test set or predictions
    unique_labels = sorted(set(y_test_orig) | set(y_test_pred_labels))

    # Map these to indices
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}

    # Initialize confusion matrix with zeros
    cm_size = len(unique_labels)
    cm = np.zeros((cm_size, cm_size), dtype=int)

    # Fill in confusion matrix
    for true_label, pred_label in zip(y_test_orig, y_test_pred_labels):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    # Save confusion matrix as a figure
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=45)
    plt.yticks(tick_marks, unique_labels)

    # Add labels and counts to the cells
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the figure
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/join_type_confusion_matrix.png')
    plt.close()

    # Get feature importance
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importance))

    # Save metrics
    os.makedirs('results/metrics', exist_ok=True)

    # Create metrics dictionary
    metrics_dict = {
        'operator': 'join_type',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'train_examples': len(y_train),
        'test_examples': len(y_test),
        'model_type': 'GradientBoostingRegressor',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'training_time': time.time() - start_time,
        'num_features': len(feature_names),
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

    # Add class-specific metrics based on the classes that are present
    for label in label_encoder.classes_:
        # Check if this class is in the test set
        if label in y_test_orig:
            label_idx = label_to_idx.get(label, -1)
            if label_idx >= 0:
                # True positives, false positives, etc.
                tp = cm[label_idx, label_idx]
                fp = np.sum(cm[:, label_idx]) - tp
                fn = np.sum(cm[label_idx, :]) - tp

                # Precision and recall for this class
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                # Add to metrics
                metrics_dict[f'precision_{label}'] = precision
                metrics_dict[f'recall_{label}'] = recall
        else:
            # Class not in test set
            metrics_dict[f'precision_{label}'] = float('nan')
            metrics_dict[f'recall_{label}'] = float('nan')

    # Add top 5 most important features
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
    sorted_idx = np.argsort(feature_importance)[-15:]  # Top 15 features
    plt.barh(np.array(feature_names)[sorted_idx], feature_importance[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for Join Type Prediction')
    plt.tight_layout()
    plt.savefig('results/figures/join_type_feature_importance.png')

    # Add an empty line before the "Metrics" message
    print("\nMetrics and figures have been saved to the 'results' directory")

    return model, feature_names, label_encoder


def predict_join_type(model, feature_names, label_encoder, left_table, right_table, left_join_keys, right_join_keys):
    """
    Predict the join type for two tables.

    Args:
        model: Trained join type prediction model
        feature_names: List of feature names used by the model
        label_encoder: Label encoder used to convert class labels
        left_table: The left table for the join
        right_table: The right table for the join
        left_join_keys: Join columns from the left table
        right_join_keys: Join columns from the right table

    Returns:
        Predicted join type and confidence score
    """
    # Extract features
    features = extract_join_type_features(left_table, right_table, left_join_keys, right_join_keys)

    # Prepare features for prediction
    X = np.array([[features.get(name, 0) for name in feature_names]])

    # Get prediction scores
    prediction_score = model.predict(X)[0]

    # Convert the regression model's continuous output to a discrete class index
    # Since we're using a regression model for classification, we round to the nearest integer
    pred_class_idx = int(np.round(prediction_score))

    # Ensure the predicted class index is within valid bounds (0 to num_classes-1)
    # This handles edge cases where the model might predict values outside the valid range
    pred_class_idx = np.clip(pred_class_idx, 0, len(label_encoder.classes_) - 1)

    # Map the numeric class index back to the original join type label (e.g., 'inner', 'left')
    # using the LabelEncoder that was used during training
    predicted_join_type = label_encoder.classes_[pred_class_idx]

    # Calculate a confidence score based on how close the prediction is to the nearest integer
    # If prediction_score is exactly an integer (e.g., 1.0 for class 1), confidence is 1.0
    # If prediction_score is halfway between integers (e.g., 1.5), confidence is 0.5
    # We cap the difference at 1.0 so confidence is never negative
    confidence = 1.0 - min(abs(prediction_score - pred_class_idx), 1.0)

    # Get alternative join types with lower confidence
    alternatives = []
    probabilities = []

    # Loop through all possible join types to calculate alternative probabilities
    for i, join_type in enumerate(label_encoder.classes_):
        # Skip the already-predicted class since we've already calculated its confidence
        if i != pred_class_idx:
            # Calculate how far the raw prediction score is from this alternative class index
            # For example, if prediction_score is 2.3 and we're checking class 1, distance is 1.3
            distance = abs(prediction_score - i)

            # Convert distance to a probability-like score between 0 and 1
            # Classes that are closer to the predicted score get higher probabilities
            # If distance ≥ 1.0, the probability becomes 0
            probability = max(0, 1.0 - distance)

            alternatives.append(join_type)
            probabilities.append(probability)

    # Sort alternatives by probability
    sorted_alternatives = sorted(zip(alternatives, probabilities), key=lambda x: x[1], reverse=True)

    # Create a result dictionary
    result = {
        'predicted_join_type': predicted_join_type,
        'confidence': confidence,
        'alternatives': [alt for alt, prob in sorted_alternatives if prob > 0.2]
        # Only include alternatives with reasonable probability
    }

    return result