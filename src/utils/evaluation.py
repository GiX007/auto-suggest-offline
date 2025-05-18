# src/utils/evaluation.py
#
# Consolidated evaluation module for join and groupby operators in the Auto-Suggest system.
# This module contains all the functions needed to evaluate predictions for different data preparation operators (Join, GroupBy).
#
# This file centralizes all evaluation metrics and reporting functions to:
# 1. Calculate metrics like precision@k and ndcg@k for ranking tasks
# 2. Generate evaluation tables that match those in the original paper
# 3. Evaluate per-sample ranking predictions (for join column prediction)
# 4. Compare Auto-Suggest performance with baseline methods
# 5. Visualize feature importance
#
# The evaluation functions here are used by model training code to assess model
# performance and generate standardized reports for comparison with baselines.
#

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Any, Union, Optional
from sklearn.metrics import precision_score, ndcg_score, accuracy_score
from tabulate import tabulate  # Install with: pip install tabulate

# Results for comparison methods from the paper
# These are hardcoded values from Table 3 and Table 5 in the paper
# JOIN_COLUMN_BASELINES = {
#     "ML-FK": {"prec@1": 0.84, "prec@2": 0.87, "ndcg@1": 0.84, "ndcg@2": 0.87},
#     "PowerPivot": {"prec@1": 0.31, "prec@2": 0.44, "ndcg@1": 0.31, "ndcg@2": 0.48},
#     "Multi": {"prec@1": 0.33, "prec@2": 0.4, "ndcg@1": 0.33, "ndcg@2": 0.41},
#     "Holistic": {"prec@1": 0.57, "prec@2": 0.63, "ndcg@1": 0.57, "ndcg@2": 0.65},
#     "max-overlap": {"prec@1": 0.53, "prec@2": 0.61, "ndcg@1": 0.53, "ndcg@2": 0.63}
# }

JOIN_COLUMN_VENDORS = {
    "Vendor-A": {"prec@1": 0.76, "ndcg@1": 0.76},
    "Vendor-C": {"prec@1": 0.42, "ndcg@1": 0.42},
    "Vendor-B": {"prec@1": 0.33, "ndcg@1": 0.33}
}

JOIN_TYPE_BASELINES = {
    "Vendor-A": {"accuracy": 0.78}
}


def calculate_accuracy(y_true: List[Union[int, bool]], y_pred: List[Union[int, bool]]) -> float:
    """
    Calculate simple accuracy score between two lists of labels.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score between 0 and 1
    """
    return accuracy_score(y_true, y_pred)


def evaluate_predictions(y_true: List[int], y_pred: List[float], k_values: List[int] = [1, 2]) -> Dict[str, float]:
    """
    Evaluate the quality of predictions using ranking metrics.

    This implementation is NOT appropriate for join column prediction
    as it doesn't group candidates by sample.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted scores for each candidate.
        k_values: List of k values for top-k metrics.

    Returns:
        A dictionary containing evaluation metrics.
    """
    results = {}

    for k in k_values:
        if k > len(y_pred):
            continue

        # For precision@k, we need to convert scores to rankings
        y_pred_binary = np.zeros_like(y_pred)
        top_k_indices = np.argsort(y_pred)[::-1][:k]
        y_pred_binary[top_k_indices] = 1

        precision_at_k = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)

        # Reshape for ndcg calculation
        true_relevance = np.array(y_true).reshape(1, -1)
        predictions = np.array(y_pred).reshape(1, -1)

        ndcg_at_k = ndcg_score(true_relevance, predictions, k=k)

        results[f'precision@{k}'] = precision_at_k
        results[f'ndcg@{k}'] = ndcg_at_k

    return results

# =============================================================================
# Join Column Evaluation Functions
# =============================================================================


def evaluate_per_sample_ranking(sample_ids: List, y_true: List[int], y_pred: List[float],
                                k_values: List[int] = [1, 2]) -> Dict[str, float]:
    """
    Evaluate ranking metrics on a per-sample basis for join column prediction.

    This function computes ranking metrics (precision@k, ndcg@k) by grouping predictions
    by sample and evaluating each sample independently. This is the correct approach for
    join column prediction as described in the paper, where:

    "precision@k = (Number of samples with correct join in top-k) / (Total number of samples)"

    The function:
    1. Groups predictions by sample_id
    2. For each sample, sorts predictions by score
    3. Checks if any of the top-k predictions are correct (1 in y_true)
    4. Calculates precision@k and ndcg@k across all samples

    Example:
        Consider evaluating join column predictions for 2 samples:

        ```
        sample_ids = [1, 1, 1, 2, 2, 2]  # 3 candidates each for 2 samples
        y_true =     [0, 1, 0, 0, 0, 1]  # Ground truth (1 = correct join)
        y_pred =     [0.9, 0.5, 0.2, 0.8, 0.9, 0.6]  # Predicted scores

        metrics = evaluate_per_sample_ranking(sample_ids, y_true, y_pred, k_values=[1, 2])

        # For sample 1:
        # - Top prediction (score 0.9) is incorrect => not in top-1
        # - Second prediction (score 0.5) is correct => in top-2

        # For sample 2:
        # - Top prediction (score 0.9) is incorrect => not in top-1
        # - Second prediction (score 0.8) is incorrect => not in top-2
        # - Third prediction (score 0.6) is correct => not in top-2

        # precision@1 = 0/2 = 0.0 (no samples have correct join in top-1)
        # precision@2 = 1/2 = 0.5 (one sample has correct join in top-2)
        ```

    Args:
        sample_ids: Identifier for which sample each prediction belongs to
        y_true: Ground truth labels (0 or 1) for each candidate
        y_pred: Predicted scores for each candidate
        k_values: List of k values for top-k metrics

    Returns:
        A dictionary containing evaluation metrics (precision@k, ndcg@k)

    Note:
        This function is specifically designed for evaluating join column predictions
        where multiple candidates belong to the same sample. It implements the evaluation
        methodology described in the Auto-Suggest paper.
    """
    results = {}

    # Group predictions by sample_id
    sample_groups = {}
    for i, sample_id in enumerate(sample_ids):
        if sample_id not in sample_groups:
            sample_groups[sample_id] = []
        sample_groups[sample_id].append((y_true[i], y_pred[i]))

    # Calculate metrics for each k
    for k in k_values:
        correct_at_k = 0
        ndcg_sum = 0

        # For each sample
        for sample_id, predictions in sample_groups.items():
            # Sort predictions by score in descending order
            sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)

            # Check if any of the top-k predictions are correct
            # This implements the formula from the paper:
            # "precision@k = (Number of samples with correct join in top-k) / (Total number of samples)"
            top_k_preds = sorted_preds[:min(k, len(sorted_preds))]
            if any(label == 1 for label, _ in top_k_preds):
                correct_at_k += 1

            # Calculate NDCG@k for this sample
            dcg = 0
            idcg = 0

            # Calculate DCG (Discounted Cumulative Gain)
            for i, (label, _) in enumerate(sorted_preds[:k]):
                if label == 1:
                    # Position is 0-indexed, so add 1 for the log calculation
                    dcg += 1 / np.log2(i + 2)

            # Calculate ideal DCG (sort by true label for ideal ordering)
            ideal_sorted = sorted(predictions, key=lambda x: x[0], reverse=True)
            for i, (label, _) in enumerate(ideal_sorted[:k]):
                if label == 1:
                    idcg += 1 / np.log2(i + 2)

            # Calculate NDCG for this sample
            sample_ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_sum += sample_ndcg

        # Calculate final metrics
        num_samples = len(sample_groups)
        precision_at_k = correct_at_k / num_samples if num_samples > 0 else 0
        ndcg_at_k = ndcg_sum / num_samples if num_samples > 0 else 0

        results[f'precision@{k}'] = precision_at_k
        results[f'ndcg@{k}'] = ndcg_at_k

    return results


def evaluate_join_column_model(model, feature_names, test_samples, k_values=[1, 2]):
    """
    Evaluate a join column prediction model on test samples.

    Args:
        model: Trained join column model
        feature_names: Feature names used by the model
        test_samples: List of test samples
        k_values: List of k values for precision@k and ndcg@k evaluation (default: [1, 2])

    Returns:
        Dictionary of evaluation metrics
    """
    correct_at_k = {k: 0 for k in k_values}
    ndcg_sum = {k: 0 for k in k_values}
    total = 0

    max_k = max(k_values)
    print(f"\nEvaluating join column prediction on test samples...")

    # Store detailed prediction results for analysis
    all_predictions = []

    for sample in test_samples:
        left_table = sample['left_table']
        right_table = sample['right_table']
        true_left_cols = sample['left_join_keys']
        true_right_cols = sample['right_join_keys']

        # Predict join columns
        try:
            # Import here to avoid circular imports
            from src.models.join_col_model import predict_join_columns
            predictions = predict_join_columns(model, feature_names, left_table, right_table, max_k, verbose=False)

            if not predictions:
                continue

            total += 1

            # Check if ground truth is in top predictions for different k values
            found_correct = False
            for i, (left_cols, right_cols, score) in enumerate(predictions):
                # Check if this prediction matches the ground truth
                is_match = (set(left_cols) == set(true_left_cols) and
                            set(right_cols) == set(true_right_cols))

                # Store prediction details
                all_predictions.append({
                    'sample_id': sample.get('sample_id', f'sample_{total}'),
                    'rank': i + 1,
                    'left_cols': ','.join(left_cols),
                    'right_cols': ','.join(right_cols),
                    'score': score,
                    'is_correct': is_match,
                    'true_left_cols': ','.join(true_left_cols),
                    'true_right_cols': ','.join(true_right_cols)
                })

                if is_match:
                    found_correct = True
                    # For each k-value, check if the correct prediction is within top-k
                    for k in k_values:
                        if i < k:  # i is 0-indexed, so i < k means within top-k
                            correct_at_k[k] += 1

                    # No need to check further predictions
                    break

            # Calculate NDCG metrics
            # Create binary relevance vector (1 for correct match, 0 otherwise)
            y_true = []
            y_pred = []

            for i, (left_cols, right_cols, score) in enumerate(predictions):
                is_match = (set(left_cols) == set(true_left_cols) and
                            set(right_cols) == set(true_right_cols))
                y_true.append(1 if is_match else 0)
                y_pred.append(score)

            # If we have true predictions, calculate NDCG
            if any(y_true):
                for k in k_values:
                    if k <= len(y_true):
                        from sklearn.metrics import ndcg_score
                        ndcg = ndcg_score(
                            np.array([y_true]).astype(np.int64),
                            np.array([y_pred]).astype(np.float64),
                            k=k
                        )
                        ndcg_sum[k] += ndcg

        except Exception as e:
            print(f"Error predicting join columns: {e}")
            continue

    # Calculate metrics
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0
        metrics[f'ndcg@{k}'] = ndcg_sum[k] / total if total > 0 else 0

    # Add number of samples evaluated
    metrics['samples_evaluated'] = total

    # Print summary of results
    # print(f"Join Column Prediction: " +
    #       ", ".join([f"precision@{k}={metrics[f'precision@{k}']:.4f}" for k in k_values]) +
    #       ", " +
    #       ", ".join([f"ndcg@{k}={metrics[f'ndcg@{k}']:.4f}" for k in k_values]))

    # Save metrics to CSV
    os.makedirs('results/metrics', exist_ok=True)

    # Create a DataFrame with metrics
    metrics_df = pd.DataFrame([{
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'operator': 'join_column_eval',
        'samples_evaluated': total,
        **{f'precision@{k}': metrics[f'precision@{k}'] for k in k_values},
        **{f'ndcg@{k}': metrics[f'ndcg@{k}'] for k in k_values}
    }])

    # Save metrics to CSV
    metrics_file = 'results/metrics/join_column_eval_metrics.csv'
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    # Save detailed predictions to CSV for analysis
    # if all_predictions:
    #     predictions_df = pd.DataFrame(all_predictions)
    #     predictions_file = 'results/metrics/join_column_predictions.csv'
    #     predictions_df.to_csv(predictions_file, index=False)

    # Generate Table 3 from the paper
    generate_join_column_table(metrics, k_values, save_dir="results")

    # Generate Table 4 (feature importance) from the paper
    try:
        feature_importance = model.feature_importances_
        generate_feature_importance_table(feature_importance, feature_names, save_dir="results")
    except Exception as e:
        print(f"Warning: Could not generate feature importance table: {e}")

    return metrics


def evaluate_join_type_model(model, feature_names, label_encoder, test_samples):
    """
    Evaluate a join type prediction model on test samples.

    Args:
        model: Trained join type model
        feature_names: Feature names used by the model
        label_encoder: Label encoder used by the model
        test_samples: List of test samples

    Returns:
        Dictionary of evaluation metrics
    """
    from src.models.join_type_model import predict_join_type

    correct = 0
    total = 0

    # Store detailed prediction results
    all_predictions = []

    # Store confusion matrix data
    confusion_data = {}

    # Add an empty line before the evaluation message
    print("\nEvaluating join type prediction on test samples...")

    for sample in test_samples:
        left_table = sample['left_table']
        right_table = sample['right_table']
        left_cols = sample['left_join_keys']
        right_cols = sample['right_join_keys']
        true_join_type = sample['join_type']

        # Predict join type
        try:
            result = predict_join_type(
                model, feature_names, label_encoder,
                left_table, right_table, left_cols, right_cols
            )

            total += 1
            predicted_type = result['predicted_join_type']
            confidence = result['confidence']
            is_correct = predicted_type == true_join_type

            # Store prediction details
            all_predictions.append({
                'sample_id': sample.get('sample_id', f'sample_{total}'),
                'left_cols': ','.join(left_cols),
                'right_cols': ','.join(right_cols),
                'true_join_type': true_join_type,
                'predicted_join_type': predicted_type,
                'confidence': confidence,
                'is_correct': is_correct,
                'alternatives': ','.join(result.get('alternatives', []))
            })

            # Update confusion matrix data
            confusion_key = f"{true_join_type}→{predicted_type}"
            confusion_data[confusion_key] = confusion_data.get(confusion_key, 0) + 1

            if is_correct:
                correct += 1

        except Exception as e:
            print(f"Error predicting join type: {e}")
            continue

    accuracy = correct / total if total > 0 else 0

    # Calculate metrics
    metrics = {
        'accuracy': accuracy,
        'samples_evaluated': total
    }

    # Save metrics to CSV
    os.makedirs('results/metrics', exist_ok=True)

    # Create a DataFrame with metrics
    metrics_df = pd.DataFrame([{
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'operator': 'join_type_eval',
        'samples_evaluated': total,
        'accuracy': accuracy
    }])

    # Add confusion matrix data to metrics
    for confusion_pair, count in confusion_data.items():
        metrics_df[f'confusion_{confusion_pair}'] = count

    # Save metrics to CSV
    metrics_file = 'results/metrics/join_type_eval_metrics.csv'
    if os.path.exists(metrics_file):
        # Only save core columns to maintain the structure when appending
        save_cols = ['timestamp', 'operator', 'samples_evaluated', 'accuracy']
        metrics_df[save_cols].to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    # Save detailed predictions to CSV for analysis
    # if all_predictions:
    #     predictions_df = pd.DataFrame(all_predictions)
    #     predictions_file = 'results/metrics/join_type_predictions.csv'
    #     predictions_df.to_csv(predictions_file, index=False)

    # Save confusion matrix as separate CSV
    # confusion_df = pd.DataFrame([{'pair': k, 'count': v} for k, v in confusion_data.items()])
    # confusion_df.to_csv('results/metrics/join_type_confusion.csv', index=False)

    # Generate Table 5 from the paper
    generate_join_type_table(metrics, save_dir="results")

    return metrics


# TABLE GENERATION FUNCTIONS

def generate_join_column_table(auto_suggest_metrics, k_values=[1, 2], save_dir="results"):
    """
    Generate tables similar to Table 3 in the paper for join column prediction.

    If baseline_metrics is provided, it will use those metrics instead of the
    hardcoded values from the paper. This allows for direct comparison on the
    same dataset rather than using the paper's reported values.

    Args:
        auto_suggest_metrics: Dictionary of metrics from Auto-Suggest
        k_values: List of k values to include in the table (default: [1, 2])
        save_dir: Directory to save results
        baseline_metrics: Optional dictionary of metrics from baseline methods
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Check if baseline metrics are available
    try:
        # Dynamically import the baselines module
        from src.baselines.join_baselines import evaluate_baselines
        # Check if we have any test samples in the current context
        # This is a bit hacky, but it's the simplest way to check
        # if we're in an evaluation context
        import inspect
        frame = inspect.currentframe()
        test_samples_found = False

        # Look for test_samples in the caller's variables
        if frame:
            for frame_info in inspect.stack():
                if 'test_samples' in frame_info.frame.f_locals:
                    test_samples = frame_info.frame.f_locals.get('test_samples')
                    if test_samples:
                        # print("\nCalculating baseline metrics on the current dataset...")
                        baseline_metrics = evaluate_baselines(test_samples, k_values)
                        test_samples_found = True
                        break

        if not test_samples_found:
            # print("\nUsing hardcoded baseline metrics from the paper.")
            baseline_metrics = JOIN_COLUMN_BASELINES
    except (ImportError, Exception) as e:
        # print(f"\nNote: Baseline module not available or error occurred: {e}")
        # print("Using hardcoded baseline metrics from the paper.")
        baseline_metrics = JOIN_COLUMN_BASELINES

    # Part 1: Methods from literature (full comparison)
    methods = ["Auto-Suggest"] + list(baseline_metrics.keys())

    # Create rows for the table
    rows = []
    for method in methods:
        row = [method]
        for k in k_values:
            # Add precision@k
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get(f'precision@{k}', 0):.2f}")
            else:
                row.append(
                    f"{baseline_metrics[method].get(f'prec@{k}', baseline_metrics[method].get(f'precision@{k}', 0)):.2f}")

        for k in k_values:
            # Add ndcg@k
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get(f'ndcg@{k}', 0):.2f}")
            else:
                row.append(f"{baseline_metrics[method].get(f'ndcg@{k}', 0):.2f}")

        rows.append(row)

    # Create headers based on k_values
    headers = ["method (all data)"]
    for k in k_values:
        headers.append(f"prec@{k}")
    for k in k_values:
        headers.append(f"ndcg@{k}")

    # Print Table 3 (top part)
    print("\nTable 3: Join column prediction evaluation - Literature methods")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save to CSV
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(os.path.join(save_dir, "join_column_literature_comparison.csv"), index=False)

    # Part 2: Vendor comparison (if we have k=1)
    if 1 in k_values:
        methods = ["Auto-Suggest"] + list(JOIN_COLUMN_VENDORS.keys())

        # Create rows for the vendor comparison table
        rows = []
        for method in methods:
            row = [method]
            # Add precision@1 and ndcg@1
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get('precision@1', 0):.2f}")
                row.append(f"{auto_suggest_metrics.get('ndcg@1', 0):.2f}")
            else:
                row.append(f"{JOIN_COLUMN_VENDORS[method].get('prec@1', 0):.2f}")
                row.append(f"{JOIN_COLUMN_VENDORS[method].get('ndcg@1', 0):.2f}")

            rows.append(row)

        # Print Table 3 (bottom part)
        print("\nTable 3: Join column prediction evaluation - Commercial systems")
        print(tabulate(rows, headers=["method (sampled data)", "prec@1", "ndcg@1"], tablefmt="grid"))

        # Save to CSV
        # df = pd.DataFrame(rows, columns=["method (sampled data)", "prec@1"])
        # df.to_csv(os.path.join(save_dir, "join_column_vendor_comparison.csv"), index=False)


def generate_feature_importance_table(feature_importance, feature_names, save_dir="results"):
    """
    Generate and visualize feature importance (related to Table 4 in the paper).

    This function calculates "feature group importance" by aggregating individual
    feature importance values into logical groups (e.g., "val-overlap", "distinct-val-ratio").
    Feature group importance provides a higher-level view of which types of signals
    are most valuable for predictions, making the model more interpretable.

    The function:
    1. Groups individual features into logical categories (as in Table 4 of the paper)
    2. Aggregates importance values within each group
    3. Normalizes group importance to sum to 1
    4. Creates visualizations and CSV output of the results

    Example:
        Consider a join model with these features and importance values:

        ```
        feature_names = ['left_distinct_ratio', 'right_distinct_ratio',
                        'jaccard_similarity', 'left_to_right_containment',
                        'right_to_left_containment', 'left_absolute_position',
                        'right_absolute_position']

        feature_importance = [0.15, 0.15, 0.1, 0.05, 0.05, 0.25, 0.25]

        # The function would group these into:
        # - "distinct-val-ratio": 0.15 + 0.15 = 0.30
        # - "val-overlap": 0.1 + 0.05 + 0.05 = 0.20
        # - "left-ness": 0.25 + 0.25 = 0.50
        ```

        The output would identify "left-ness" as the most important feature group (50%),
        followed by "distinct-val-ratio" (30%) and "val-overlap" (20%).

    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names
        save_dir: Directory to save results

    Note:
        This function creates visualizations and CSV files in the specified directory,
        which can be useful for reporting and further analysis. It mimics Table 4
        in the paper, which shows the importance of different feature groups.
    """
    # Ensure the directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    # Group features into logical categories as in Table 4 of the paper
    # Each category represents a different type of signal used for prediction
    # Feature group importance helps understand which types of signals are most useful
    feature_groups = {
        "left-ness": ["left_absolute_position", "left_relative_position",
                      "right_absolute_position", "right_relative_position"],
        "val-range-overlap": ["range_overlap"],
        "distinct-val-ratio": ["left_distinct_ratio", "right_distinct_ratio"],
        "val-overlap": ["jaccard_similarity", "left_to_right_containment", "right_to_left_containment"],
        "single-col-candidate": ["is_single_column"],
        "col-val-types": ["left_is_string", "right_is_string", "left_is_numeric", "right_is_numeric", "type_match"],
        "table-stats": ["left_row_count", "right_row_count", "row_count_ratio"],
        "sorted-ness": ["left_is_sorted", "right_is_sorted"]
    }

    # Calculate importance for each group
    group_importance = {}

    for group, features in feature_groups.items():
        group_importance[group] = 0
        for feature in features:
            if feature in feature_names:
                idx = feature_names.index(feature)
                group_importance[group] += feature_importance[idx]

    # Normalize to sum to 1
    total_importance = sum(group_importance.values())
    if total_importance > 0:
        for group in group_importance:
            group_importance[group] /= total_importance

    # Sort by importance
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

    # Create rows for the table
    rows = []
    for i in range(0, len(sorted_groups), 4):
        row = []
        for j in range(4):
            if i + j < len(sorted_groups):
                group, importance = sorted_groups[i + j]
                row.extend([group, f"{importance:.2f}"])
            else:
                row.extend(["", ""])
        rows.append(row)

    # Print Table 4
    print("\nTable 4: Importance of Feature Groups for Join")
    headers = []
    for i in range(4):
        headers.extend([f"feature_{i + 1}", f"importance_{i + 1}"])
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save to CSV
    # feature_importance_data = [{"feature": group, "importance": importance}
    #                            for group, importance in sorted_groups]
    # df = pd.DataFrame(feature_importance_data)
    # df.to_csv(os.path.join(save_dir, "join_feature_importance.csv"), index=False)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    groups, importances = zip(*sorted_groups)
    plt.barh(groups, importances)
    plt.xlabel('Feature Group Importance')
    plt.title('Feature Group Importance for Join Column Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", "join_feature_group_importance.png"))
    plt.close()


def generate_join_type_table(auto_suggest_metrics, save_dir="results"):
    """
    Generate table similar to Table 5 in the paper for join type prediction.

    Args:
        auto_suggest_metrics: Dictionary of metrics from Auto-Suggest
        save_dir: Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    methods = ["Auto-Suggest"] + list(JOIN_TYPE_BASELINES.keys())

    # Create rows for the table
    rows = []
    for method in methods:
        row = [method]
        # Add accuracy
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('accuracy', 0):.2f}")
        else:
            row.append(f"{JOIN_TYPE_BASELINES[method].get('accuracy', 0):.2f}")

        rows.append(row)

    # Print Table 5
    # prec@1 here is equivalent to accuracy since only one label is predicted per sample
    print("\nTable 5: Join type prediction")
    print(tabulate(rows, headers=["method", "prec@1"], tablefmt="grid"))

    # Save to CSV
    # df = pd.DataFrame(rows, columns=["method", "prec@1"])
    # df.to_csv(os.path.join(save_dir, "join_type_comparison.csv"), index=False)


def extract_feature_importance(model, feature_names):
    """
    Extract feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names

    Returns:
        Dictionary mapping feature names to importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return {}

    feature_importance = model.feature_importances_
    importance_dict = dict(zip(feature_names, feature_importance))

    return importance_dict


# =============================================================================
# GroupBy Evaluation Functions
# =============================================================================

# Results for comparison with vendors from the paper (Table 6)
# These are hardcoded values from Table 6 in the paper
GROUPBY_VENDORS = {
    "Vendor-B": {"precision@1": 0.56, "precision@2": 0.71, "ndcg@1": 0.56, "ndcg@2": 0.75, "full-accuracy": 0.45},
    "Vendor-C": {"precision@1": 0.71, "precision@2": 0.82, "ndcg@1": 0.71, "ndcg@2": 0.85, "full-accuracy": 0.67}
}


def evaluate_groupby_model(model, feature_names, test_samples, k_values=[1, 2]):
    """
    Evaluate a GroupBy column prediction model on test samples.

    Args:
        model: Trained GroupBy column model
        feature_names: Feature names used by the model
        test_samples: List of test samples
        k_values: List of k values for precision@k and ndcg@k evaluation (default: [1, 2])

    Returns:
        Dictionary of evaluation metrics
    """
    from src.models.groupby_model import predict_groupby_columns

    correct_at_k = {k: 0 for k in k_values}
    full_accuracy_count = 0
    total = 0

    max_k = max(k_values)
    print(f"\nEvaluating GroupBy column prediction on {len(test_samples)} test samples...")

    # Store detailed prediction results for analysis
    all_predictions = []

    # Store all predictions for per-sample ranking metrics
    all_sample_ids = []
    all_y_true = []
    all_y_pred = []

    for sample_idx, sample in enumerate(test_samples):
        input_table = sample['input_table']
        true_groupby_cols = sample['groupby_columns']

        # Predict GroupBy columns
        try:
            predictions = predict_groupby_columns(model, feature_names, input_table)

            if not predictions:
                continue

            total += 1

            # For each column in the table, record its ground truth and predicted score
            for col_idx, col in enumerate(input_table.columns):
                is_groupby = 1 if col in true_groupby_cols else 0

                # Find the score for this column
                score = 0.0
                for pred_col, pred_score in predictions:
                    if pred_col == col:
                        score = pred_score
                        break

                # Record for per-sample ranking evaluation
                all_sample_ids.append(sample_idx)
                all_y_true.append(is_groupby)
                all_y_pred.append(score)

                # Store prediction details
                all_predictions.append({
                    'sample_id': sample.get('sample_id', f'sample_{sample_idx}'),
                    'column': col,
                    'is_groupby': is_groupby,
                    'predicted_score': score,
                    'column_idx': col_idx
                })

            # Get the top-k predicted GroupBy columns
            top_k_cols = [col for col, score in predictions[:max_k]]

            # Check if any ground truth columns are in top-k predictions
            for k in k_values:
                top_cols = top_k_cols[:k]
                has_true_col = any(col in true_groupby_cols for col in top_cols)
                if has_true_col:
                    correct_at_k[k] += 1

            # Check for full accuracy: all GroupBy columns ranked ahead of aggregation columns
            # This matches the "full-accuracy" metric in Table 6 of the paper
            predicted_groupby = [col for col, score in predictions if score >= 0.5]
            predicted_agg = [col for col, score in predictions if score < 0.5]

            # Full accuracy is achieved when all true GroupBy columns are in predicted_groupby
            # and no true GroupBy columns are in predicted_agg
            all_true_cols_predicted = all(col in predicted_groupby for col in true_groupby_cols)
            no_true_cols_in_agg = not any(col in predicted_agg for col in true_groupby_cols)

            if all_true_cols_predicted and no_true_cols_in_agg:
                full_accuracy_count += 1

        except Exception as e:
            print(f"Error evaluating sample {sample_idx}: {e}")
            continue

    # Convert to numpy arrays
    sample_ids_np = np.array(all_sample_ids)
    y_true_np = np.array(all_y_true)
    y_pred_np = np.array(all_y_pred)

    # Calculate precision@k
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0

    # Calculate ndcg@k using per-sample ranking
    if len(all_sample_ids) > 0:
        ranking_metrics = evaluate_per_sample_ranking(
            sample_ids_np,
            y_true_np,
            y_pred_np,
            k_values
        )

        # Add ndcg metrics
        for k in k_values:
            metrics[f'ndcg@{k}'] = ranking_metrics[f'ndcg@{k}']

    # Calculate full accuracy (matches "full-accuracy" in Table 6)
    metrics['full-accuracy'] = full_accuracy_count / total if total > 0 else 0

    # Add number of samples evaluated
    metrics['samples_evaluated'] = total

    # Print summary of results
    print("\nGroupBy Column Prediction Results:")
    print(f"  Samples evaluated: {total}")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

    # Save metrics to CSV
    os.makedirs('results/metrics', exist_ok=True)

    # Create a DataFrame with metrics
    metrics_df = pd.DataFrame([{
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'operator': 'groupby_eval',
        'samples_evaluated': total,
        **{metric: value for metric, value in metrics.items() if isinstance(value, (int, float))}
    }])

    # Save metrics to CSV
    metrics_file = 'results/metrics/groupby_eval_metrics.csv'
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    # Save detailed predictions to CSV for analysis
    # if all_predictions:
    #     predictions_df = pd.DataFrame(all_predictions)
    #     predictions_file = 'results/metrics/groupby_column_predictions.csv'
    #     predictions_df.to_csv(predictions_file, index=False)

    # Generate Table 6 from the paper
    generate_groupby_table(metrics, k_values, save_dir="results")

    # Generate Table 7 (feature importance) from the paper
    try:
        feature_importance = model.feature_importances_
        generate_groupby_feature_importance_table(feature_importance, feature_names, save_dir="results")
    except Exception as e:
        print(f"Warning: Could not generate feature importance table: {e}")

    return metrics


def generate_groupby_table(auto_suggest_metrics, k_values=[1, 2], save_dir="results"):
    """
    Generate a table similar to Table 6 in the paper for GroupBy column prediction.

    Args:
        auto_suggest_metrics: Dictionary of metrics from Auto-Suggest
        k_values: List of k values to include in the table (default: [1, 2])
        save_dir: Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Check if baseline metrics are available
    try:
        # Dynamically import the baselines module
        from src.baselines.groupby_baselines import evaluate_baselines
        # Check if we have any test samples in the current context
        import inspect
        frame = inspect.currentframe()
        test_samples_found = False

        # Look for test_samples in the caller's variables
        if frame:
            for frame_info in inspect.stack():
                if 'test_samples' in frame_info.frame.f_locals:
                    test_samples = frame_info.frame.f_locals.get('test_samples')
                    if test_samples:
                        # print("\nCalculating baseline metrics on the current dataset...")

                        # Set quiet=True to suppress detailed output from evaluate_baselines
                        baseline_metrics = evaluate_baselines(test_samples, k_values, quiet=True)
                        test_samples_found = True
                        break

        if not test_samples_found:
            # print("\nUsing hardcoded baseline metrics from the paper.")

            # If test samples aren't available, use hardcoded values from the paper
            baseline_metrics = {
                "SQL-history": {"precision@1": 0.58, "precision@2": 0.61, "ndcg@1": 0.58, "ndcg@2": 0.63,
                                "full-accuracy": 0.53},
                "Coarse-grained-types": {"precision@1": 0.47, "precision@2": 0.52, "ndcg@1": 0.47, "ndcg@2": 0.54,
                                         "full-accuracy": 0.46},
                "Fine-grained-types": {"precision@1": 0.31, "precision@2": 0.4, "ndcg@1": 0.31, "ndcg@2": 0.42,
                                       "full-accuracy": 0.38},
                "Min-Cardinality": {"precision@1": 0.68, "precision@2": 0.83, "ndcg@1": 0.68, "ndcg@2": 0.86,
                                    "full-accuracy": 0.68}
            }
    except Exception as e:
        # print(f"\nNote: Baseline module not available or error occurred: {e}")
        # print("Using hardcoded baseline metrics from the paper.")

        # Default metrics from the paper
        baseline_metrics = {
            "SQL-history": {"precision@1": 0.58, "precision@2": 0.61, "ndcg@1": 0.58, "ndcg@2": 0.63,
                            "full-accuracy": 0.53},
            "Coarse-grained-types": {"precision@1": 0.47, "precision@2": 0.52, "ndcg@1": 0.47, "ndcg@2": 0.54,
                                     "full-accuracy": 0.46},
            "Fine-grained-types": {"precision@1": 0.31, "precision@2": 0.4, "ndcg@1": 0.31, "ndcg@2": 0.42,
                                   "full-accuracy": 0.38},
            "Min-Cardinality": {"precision@1": 0.68, "precision@2": 0.83, "ndcg@1": 0.68, "ndcg@2": 0.86,
                                "full-accuracy": 0.68}
        }

    # All methods for comparison
    methods = ["Auto-Suggest"] + list(baseline_metrics.keys())

    # Create rows for the table
    rows = []
    for method in methods:
        row = [method]
        # Add precision@k
        for k in k_values:
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get(f'precision@{k}', 0):.2f}")
            else:
                row.append(f"{baseline_metrics[method].get(f'precision@{k}', 0):.2f}")

        # Add ndcg@k
        for k in k_values:
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get(f'ndcg@{k}', 0):.2f}")
            else:
                row.append(f"{baseline_metrics[method].get(f'ndcg@{k}', 0):.2f}")

        # Add full-accuracy
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('full-accuracy', 0):.0%}")
        else:
            row.append(f"{baseline_metrics[method].get('full-accuracy', 0):.0%}")

        rows.append(row)

    # Create headers based on k_values
    headers = ["method"]
    for k in k_values:
        headers.append(f"prec@{k}")
    for k in k_values:
        headers.append(f"ndcg@{k}")
    headers.append("full-accuracy")

    # Print Table 6 (methods from literature)
    print("\nTable 6: GroupBy column prediction evaluation")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save to CSV
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(os.path.join(save_dir, "groupby_literature_comparison.csv"), index=False)

    # Generate a comparison with commercial vendors (similar to in Table 6)
    methods = ["Auto-Suggest"] + list(GROUPBY_VENDORS.keys())

    # Create rows for the vendor comparison table
    vendor_rows = []
    for method in methods:
        row = [method]
        # Add precision@k
        for k in k_values:
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get(f'precision@{k}', 0):.2f}")
            else:
                row.append(f"{GROUPBY_VENDORS[method].get(f'precision@{k}', 0):.2f}")

        # Add ndcg@k
        for k in k_values:
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get(f'ndcg@{k}', 0):.2f}")
            else:
                row.append(f"{GROUPBY_VENDORS[method].get(f'ndcg@{k}', 0):.2f}")

        # Add full-accuracy
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('full-accuracy', 0):.0%}")
        else:
            row.append(f"{GROUPBY_VENDORS[method].get('full-accuracy', 0):.0%}")

        vendor_rows.append(row)

    # Print vendor comparison
    print("\nComparison with Commercial Systems:")
    print(tabulate(vendor_rows, headers=headers, tablefmt="grid"))

    # # Save to CSV
    # vendor_df = pd.DataFrame(vendor_rows, columns=headers)
    # vendor_df.to_csv(os.path.join(save_dir, "groupby_vendor_comparison.csv"), index=False)


def generate_groupby_feature_importance_table(feature_importance, feature_names, save_dir="results"):
    """
    Generate and visualize feature importance for GroupBy prediction (Table 7 in the paper).

    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names
        save_dir: Directory to save results
    """
    # Ensure the directories exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    # Group features into logical categories as in Table 7 of the paper
    feature_groups = {
        "col-type": ["is_string", "is_int", "is_float", "is_bool", "is_datetime"],
        "col-name-freq": ["groupby_term_in_name", "agg_term_in_name"],
        "distinct-val": ["distinct_count", "distinct_ratio"],
        "val-range": ["value_range", "distinct_to_range_ratio"],
        "left-ness": ["absolute_position", "relative_position"],
        "peak-freq": ["peak_frequency", "peak_frequency_ratio"],
        "emptiness": ["null_ratio"]
    }

    # Calculate importance for each group
    group_importance = {}

    for group, features in feature_groups.items():
        group_importance[group] = 0
        for feature in features:
            if feature in feature_names:
                idx = feature_names.index(feature)
                group_importance[group] += feature_importance[idx]

    # Normalize to sum to 1
    total_importance = sum(group_importance.values())
    if total_importance > 0:
        for group in group_importance:
            group_importance[group] /= total_importance

    # Sort by importance
    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

    # Create a simple table format like Table 7 in the paper
    rows = []
    for group, importance in sorted_groups:
        rows.append([group, f"{importance:.2f}"])

    # Print Table 7
    print("\nTable 7: Importance of Feature Groups for GroupBy")
    print(tabulate(rows, headers=["feature", "importance"], tablefmt="grid"))

    # Save to CSV
    # feature_importance_data = [{"feature": group, "importance": importance}
    #                            for group, importance in sorted_groups]
    # df = pd.DataFrame(feature_importance_data)
    # df.to_csv(os.path.join(save_dir, "groupby_feature_importance.csv"), index=False)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    groups, importances = zip(*sorted_groups)
    plt.barh(groups, importances)
    plt.xlabel('Feature Group Importance')
    plt.title('Feature Group Importance for GroupBy Column Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", "groupby_feature_group_importance.png"))
    plt.close()