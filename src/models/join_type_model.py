# src/models/join_type_model.py
#
# Implementation of join type prediction (inner, left, right, full) based on Section 4.1 of the paper
# "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module includes:
# 1. Feature extraction for join type prediction, using signals like row/column counts, data type balance, etc.
# 2. Preparation of training data for join type prediction.
# 3. Training a gradient boosting model to classify join types (inner, left, right, outer).
# 4. Evaluation of join type prediction performance on test samples.
# 5. Generating formatted tables for precision@1 (like Table 5 in the paper).
# 6. Utility functions to make predictions on new table pairs using the trained join type model.
#
# All logic related to join type prediction is fully contained here.
#

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from src.models.join_col_model import extract_join_column_features
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score
# from imblearn.over_sampling import SMOTE
# from sklearn.feature_selection import SelectFromModel
# from sklearn.model_selection import GridSearchCV, StratifiedKFold


JOIN_TYPE_VENDORS = {
    "Vendor-A": {"test_accuracy": 0.78}
}

# Suppress warnings
warnings.filterwarnings('ignore')


def extract_join_type_features(left_table: pd.DataFrame, right_table: pd.DataFrame,
                               left_join_keys: List[str], right_join_keys: List[str]) -> Dict:
    """
    Extracts features for join type prediction based on Section 4.1 of the paper.

    Args:
        left_table: The left table for the join
        right_table: The right table for the join
        left_join_keys: Join columns from the left table
        right_join_keys: Join columns from the right table

    Returns:
        Dictionary of features for join type prediction
    """
    features = dict()

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

    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        print(f"Handled expected error: {e}")
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


def prepare_join_type_training_data(processed_samples: List[Dict]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Prepares training data for join type prediction.

    Args:
        processed_samples: List of processed samples with join information.

    Returns:
        Tuple containing:
            - X: Numeric feature matrix (NumPy array), shape (num_samples, num_features).
            - feature_cols: List of feature names corresponding to columns in X.
            - labels_encoded: Encoded join type labels.
            - label_encoder: LabelEncoder used for encoding labels.
    """
    features_list = []
    labels = []

    print(f"\nPreparing join type training data from {len(processed_samples)} samples")

    # NOTE: Unlike join column prediction, this step does not generate join key candidates.
    # It uses the ground truth join keys directly to extract features for predicting the join type.
    for sample_idx, sample in enumerate(processed_samples):
        try:
            left_table = sample['left_table']
            right_table = sample['right_table']
            left_join_keys = sample['left_join_keys']
            right_join_keys = sample['right_join_keys']
            join_type = sample['join_type']

            # If 'index' is used as a join key but not present in the table, create it by resetting the DataFrame index
            if 'index' in left_join_keys and 'index' not in left_table.columns:
                # Example: left_table has 3 rows → reset index creates new 'index' column (0, 1, 2)
                left_table = left_table.reset_index().rename(columns={'index': 'index'})

            if 'index' in right_join_keys and 'index' not in right_table.columns:
                right_table = right_table.reset_index().rename(columns={'index': 'index'})

            # Extract features for join type prediction
            features = extract_join_type_features(left_table, right_table, left_join_keys, right_join_keys)
            features_list.append(features)
            labels.append(join_type)

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {e}")
            continue

    # Count distribution of join types
    join_type_counts = Counter(labels)
    print("\nJoin type distribution:")
    for join_type, count in join_type_counts.items():
        print(f"  {join_type}: {count} ({count / len(labels) * 100:.1f}%)")

    # Create the feature DataFrame (similar to prepare_join_data)
    features_df = pd.DataFrame(features_list)

    # Remove non-feature columns
    non_feature_cols = ['sample_id']
    feature_cols = [col for col in features_df.columns if col not in non_feature_cols]

    # Convert boolean columns to int
    for col in feature_cols:
        if features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)

    # Convert features to numeric matrix
    X = features_df[feature_cols].values
    labels = np.array(labels)

    # Diagnostic prints
    # print("\nData Preparation Diagnostics:")
    # print(f"Number of samples: {len(processed_samples)}")
    # print(f"Feature matrix shape: {X.shape}")
    # print(f"Labels shape: {len(labels)}")

    # Ensure X and labels have the same number of samples
    # if X.shape[0] != len(labels):
    #     print("WARNING: Mismatch between X and labels!")

    return X, feature_cols, labels


def train_join_type_model(X_train, y_train, X_val, y_val, feature_names):
    """
    Trains a model to predict join types.

    This function takes pre-prepared feature matrices (X_train, X_val) and encoded labels (y_train, y_val),
    and trains a classifier to predict the join type for new join scenarios.

    Unlike join column prediction (binary classification), join type prediction is a multi-class classification
    that uses features describing the whole join operation (e.g., row counts, overlap ratios).

    Args:
        X_train: Feature matrix for training data (numpy array)
        y_train: Labels for training data (numpy array or list)
        X_val: Feature matrix for validation data (numpy array)
        y_val: Labels for validation data (numpy array or list)
        feature_names: List of feature names, used for feature importance analysis and debugging

    Returns:
        Trained model, list of feature names used by the model, and the label encoder.
    """
    from src.utils.model_utils import numpy_to_list

    # Check if we have any data
    if X_train.shape[0] == 0 or len(y_train) == 0:
        print("Error: No training data available for join type prediction.")
        return None, [], None

    if X_val.shape[0] == 0 or len(y_val) == 0:
        print("Error: No validation data available for join type prediction.")
        return None, [], None

    # Final feature space
    print(f"\nUsing {len(feature_names)} features:")
    print(feature_names)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)  # learn the mapping
    y_val_enc = label_encoder.transform(y_val)  # Apply the learned mapping

    # Check how many classes we have
    print(f"\nJoin type classes in training set: {label_encoder.classes_}")

    # Check class distribution
    train_class_counts = {label_encoder.classes_[i]: np.sum(y_train_enc == i) for i in range(len(label_encoder.classes_))}
    val_class_counts = {label_encoder.classes_[i]: np.sum(y_val_enc == i) for i in range(len(label_encoder.classes_))}

    print("\nTrain set distribution:")
    for label, count in train_class_counts.items():
        print(f"  {label}: {count} ({count / len(y_train) * 100:.1f}%)")

    print("\nValidation set distribution:")
    for label, count in val_class_counts.items():
        print(f"  {label}: {count} ({count / len(y_val) * 100:.1f}%)")

    # 1. Data Augmentation

    # Create SMOTE object
    # smote = SMOTE(random_state=42)
    # X_train, y_train_enc = smote.fit_resample(X_train, y_train_enc)
    # print("New class distribution after augmentation:")
    # print(Counter(y_train_enc))

    # 2. Address class imbalance with sample weights:
    # The compute_sample_weight function assigns higher weights to samples from under-represented classes,
    # so the model's loss function will give these rare classes more importance during training.
    # This helps mitigate imbalance by making rare-class errors more "costly" in the loss.
    # However, this technique alone does NOT generate additional samples or increase the actual diversity of examples,
    # so it is most effective when used together with explicit data augmentation (like SMOTE) or feature engineering.
    sample_weights = compute_sample_weight("balanced", y_train_enc)

    # 3. Feature Selection
    # Use a lightweight GB model to estimate feature importances
    # selector_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    # selector_model.fit(X_train, y_train_enc)

    # Select top features using importance threshold (e.g., median)
    # selector = SelectFromModel(selector_model, prefit=True, threshold='median')
    # X_train = selector.transform(X_train)
    # X_val = selector.transform(X_val)

    # Update feature names for reporting
    # selected_mask = selector.get_support()
    # feature_names = [f for f, s in zip(feature_names, selected_mask) if s]

    # print(f"\nSelected {len(feature_names)} features from {len(init_features)}:")
    # print("Selected features:", feature_names)

    # Train a Gradient Boosting model with some hyperparameter tuning
    print("\nTraining join type prediction model...")
    start_time = time.time()

    # 4. Hyperparameter tuning
    # Define a parameter grid to search over combinations of model settings
    # param_grid = {
    #     'n_estimators': [50, 100, 200, 300],       # More trees, more complex
    #     'learning_rate': [0.01, 0.05],        # Learning rates to balance convergence speed (Smaller values = slower, finer updates)
    #     'max_depth': [1, 3, 5],            # Controls tree complexity - Deeper trees for capturing more complex patterns
    #     'subsample': [0.6, 0.7],        # Bagging fraction (introduces stochasticity, reduces variance) - Subsample < 1.0 adds regularization
    #     'min_samples_leaf': [15, 20],   # Regularization for very small leaves (reduces overfitting)
    # }

    # Use deeper trees and more estimators here since join type prediction has fewer samples
    # and harder class boundaries than join column prediction.
    # Using more estimators with a lower learning rate allows finer, incremental updates.
    # A higher tree depth helps model complex interactions between join type features.

    # Run grid search to select best hyperparameters
    # grid_search = GridSearchCV(
    #     estimator=GradientBoostingClassifier(
    #         random_state=42,
    #         # Additional regularization parameters
    #         min_samples_split=15,    # Increase minimum samples to split an internal node
    #         max_features='sqrt',     # Limit features at each split
    #         n_iter_no_change=3,      # Early stopping
    #         validation_fraction=0.3,  # Use part of training data for early stopping
    #         tol=0.001  # Tolerance for early stopping
    #     ),
    #     param_grid=param_grid,
    #     #refit=True, # Train best model on full training set
    #     scoring='balanced_accuracy',  # # More appropriate for imbalanced classes
    #     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # Shuffled cross-validation
    #     verbose=1,
    #     n_jobs=-1  # Use all CPU cores
    # )

    # Simple Gradient Boosting Classifier
    # model = GradientBoostingClassifier()
    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,      # Reduced number of trees
        learning_rate=0.03,    # Moderate learning rate
        max_depth=5,          # Shallow trees
        min_samples_leaf=13,  # Prevent overfitting
        subsample=0.8,        # Some regularization
    )
    model.fit(X_train, y_train_enc, sample_weight=sample_weights)


    # GS Model (Fit with optional class/sample weights)
    # grid_search.fit(X_train, y_train_enc, sample_weight=sample_weights)
    # model = grid_search.best_estimator_ # Retrieve the best model from the search
    # print("Best join type model hyperparameters:", grid_search.best_params_)

    end_time = time.time()
    total_training_time = end_time - start_time

    print(f"\nModel training (with hyperparameter tuning) completed in {total_training_time:.2f} seconds")
    print(f"Trained model: GradientBoostingClassifier ({model.n_estimators} estimators, max_depth={model.max_depth})")

    # Calculate training metrics (accuracy, macro precision and recall)
    y_train_pred = model.predict(X_train)
    train_accuracy = np.mean(y_train_pred == y_train_enc)
    # Use macro-averaged to treat all join types equally, regardless of their frequency.
    # This is important for multi-class settings where some join types may be underrepresented.
    train_precision = precision_score(y_train_enc, y_train_pred, average='macro', zero_division=0)
    train_recall = recall_score(y_train_enc, y_train_pred, average='macro', zero_division=0)

    # Calculate validation metrics (accuracy, macro precision and recall)
    y_val_pred = model.predict(X_val)
    val_accuracy = np.mean(y_val_pred == y_val_enc)
    val_precision = precision_score(y_val_enc, y_val_pred, average='macro', zero_division=0)
    val_recall = recall_score(y_val_enc, y_val_pred, average='macro', zero_division=0)
    # print(classification_report(y_val, y_val_pred))

    # Print training vs validation metrics
    print("\nStandard Multi-Class Classification Metrics on Train and Validation Sets:")
    print(f"Accuracy:  Training = {train_accuracy:.4f}, Validation = {val_accuracy:.4f}")
    print(f"Precision: Training = {train_precision:.4f}, Validation = {val_precision:.4f}")
    print(f"Recall: Training = {train_recall:.4f}, Validation = {val_recall:.4f}")

    # Generate confusion matrix for validation set
    # This reveals how well the model distinguishes between each join type
    cm = confusion_matrix(y_val_enc, y_val_pred, labels=range(len(label_encoder.classes_)))

    # Map numeric labels back to string labels
    class_labels = label_encoder.classes_

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Annotate cells with counts
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save confusion matrix figure
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/join_type_confusion_matrix.png')
    plt.close()

    # Create directories for results, if they are already exist
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)

    # Create a dictionary with all relevant metrics (convert all to native Python types for JSON compatibility)
    metrics_dict = {
        'operator': 'join_type',
        'mode': 'training',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'samples': int(len(y_train)+len(y_val)),
        'train_examples': int(len(y_train)),
        'train_classes': len(np.unique(y_train)),
        'validation_examples': int(len(y_val)),
        'validation_classes': len(np.unique(y_val)),
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

    # Store feature importances
    feature_importance = model.feature_importances_
    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:5]
    for i, (feature, importance) in enumerate(top_features, 1):
        metrics_dict[f'top_feature_{i}'] = feature
        metrics_dict[f'importance_{i}'] = float(importance)

    # Add per-class precision/recall using confusion matrix if available
    cm = confusion_matrix(y_val_enc, y_val_pred)
    label_names = label_encoder.classes_
    for i, label in enumerate(label_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics_dict[f'precision_{label}'] = float(precision)
        metrics_dict[f'recall_{label}'] = float(recall)

    # Converts numpy types to native Python types (important for JSON!)
    metrics_dict = numpy_to_list(metrics_dict)

    # Save to a combined JSON metrics file for all operators
    metrics_path = 'results/metrics/all_operators_metrics.json'
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
    plt.title('Feature Importance for Join Type Prediction')
    plt.tight_layout()
    plt.savefig('results/figures/join_type_feature_importance.png')

    # Add an empty line before the "Metrics" message
    print("\nMetrics and figures have been saved to the 'results' directory")

    return model, feature_names, label_encoder


def predict_join_type(model, feature_names, label_encoder, left_table, right_table, left_join_keys, right_join_keys):
    """
    Predicts the join type for two tables.

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
    # Extract features for join type prediction
    features = extract_join_type_features(left_table, right_table, left_join_keys, right_join_keys)

    # Prepare feature vector
    X = np.array([[features.get(name, 0) for name in feature_names]])

    # Predict class probabilities
    probs = model.predict_proba(X)[0]

    # Determine the most likely class
    pred_class_idx = np.argmax(probs)
    predicted_join_type = label_encoder.classes_[pred_class_idx]
    confidence = probs[pred_class_idx]

    # Collect alternative join types with probabilities > 0.2 (optional threshold)
    alternatives = [
        (label_encoder.classes_[i], prob)
        for i, prob in enumerate(probs)
        if i != pred_class_idx and prob > 0.2
    ]

    # Sort alternatives by confidence
    sorted_alternatives = sorted(alternatives, key=lambda x: x[1], reverse=True)

    return {
        'predicted_join_type': predicted_join_type,
        'confidence': round(confidence, 2),
        'alternatives': [alt for alt, _ in sorted_alternatives]
    }


def evaluate_join_type_model(model, feature_names, label_encoder, test_samples):
    """
    Evaluates a join type prediction model on test samples.

    Args:
        model: Trained join type model
        feature_names: Feature names used by the model
        label_encoder: Label encoder used by the model
        test_samples: List of test samples

    Returns:
        Dictionary of evaluation metrics
    """
    # Dynamic local Import to avoid a loop (circular import)
    from src.utils.model_utils import generate_prediction_table

    total = 0
    total_test_positives = 0
    y_true = []
    y_pred = []

    unseen_labels = set()

    print("\nEvaluating join type prediction on test samples...")

    for sample in test_samples:
        left_table = sample['left_table']
        right_table = sample['right_table']
        left_cols = sample['left_join_keys']
        right_cols = sample['right_join_keys']
        true_join_type = sample['join_type']

        # Predict join type
        try:
            result = predict_join_type(model, feature_names, label_encoder, left_table, right_table, left_cols, right_cols)

            predicted_type = result['predicted_join_type']
            #confidence = result['confidence']

            if true_join_type in label_encoder.classes_:
                # Valid label for evaluation
                y_true.append(true_join_type)
                y_pred.append(predicted_type)

                total += 1
                if predicted_type == true_join_type:
                    total_test_positives += 1

            elif true_join_type not in unseen_labels:
                print(f"\nSkipping unseen label during evaluation: {true_join_type}")
                unseen_labels.add(true_join_type)

        except Exception as e:
            print(f"Error predicting and evaluating join type: {e}")
            continue

    if not y_true:
        print("\nNo valid samples with known labels found for evaluation.")
        return {}

    # Encode labels for scikit metrics
    true_encoded = label_encoder.transform(y_true)
    pred_encoded = label_encoder.transform(y_pred)

    # Compute metrics
    test_accuracy = accuracy_score(true_encoded, pred_encoded)
    test_precision = precision_score(true_encoded, pred_encoded, average='macro', zero_division=0)
    test_recall = recall_score(true_encoded, pred_encoded, average='macro', zero_division=0)

    # Create full evaluation record for saving to a json file
    eval_dict = {
        "operator": "join_type",
        "mode": "evaluation",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "samples": int(len(test_samples)),
        "test_examples": int(total),    # same as samples with valid predictions
        "test_positives": int(total_test_positives),
        "test_classes": len(label_encoder.classes_),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
    }

    # Add per-class metrics from confusion data
    cm = confusion_matrix(true_encoded, pred_encoded)
    #present_classes = list(label_encoder.transform(label_encoder.classes_))
    for i, class_name in enumerate(label_encoder.classes_):
        if i >= cm.shape[0]:  # Skip if class index is missing in confusion matrix
            continue
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        eval_dict[f"precision_{class_name}"] = float(precision)
        eval_dict[f"recall_{class_name}"] = float(recall)

    # Converts numpy types to native Python types (important for JSON!)
    eval_dict = numpy_to_list(eval_dict)

    # Save metrics to JSON file
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

    # Return just the basic metrics for print
    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "samples_evaluated": len(y_true)
    }

    # Generate Table 5 from the paper
    generate_prediction_table(
        auto_suggest_metrics=metrics, # The metrics computed for Auto-Suggest
        k_values=[],  # not needed for join type
        baseline_metrics=JOIN_TYPE_VENDORS, # Vendor metrics to compare with
        operator_name="join_type",
        include_accuracy_only=True
    )

    return metrics
