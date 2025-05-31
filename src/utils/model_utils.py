# src/utils/model_utils.py
#
# Utility functions for saving and loading trained models.
#
# Includes:
# - save_model: Save a trained model to disk
# - load_model: Load a trained model from disk
# - generate_prediction_table: Generate precision@k, ndcg@k, and other comparison tables for predictions (like Table 3, 5, 6)
# - generate_feature_importance_table: Summarize and visualize feature group importance (like Table 4, 7 in the paper)
# - recommend_joins: Generate complete join recommendations for two tables using the trained join models
# - display_join_recommendations: Display the join recommendations in a clear and formatted output
# - evaluate_per_sample_ranking: Evaluate precision@k and ndcg@k on a per-sample (grouped) basis for ranking predictions
#

import os
import pickle
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from typing import List, Dict
from src.models.join_col_model import predict_join_columns
from src.models.join_type_model import predict_join_type


def numpy_to_list(obj):
    # It recursively converts NumPy arrays in a structure to plain Python lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj


def save_model(model, model_path: str):
    """
    Saves a trained model to the given path.

    Args:
        model: The trained model to save.
        model_path: Path where the model should be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")


def load_model(model_path: str):
    """
    Loads a trained model from the given path.

    Args:
        model_path: Path to the saved model.

    Returns:
        The loaded model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")

    return model


def generate_prediction_table(auto_suggest_metrics, k_values, save_dir="results/metrics", baseline_metrics=None, vendor_metrics=None, include_full_accuracy=False, operator_name="join", include_accuracy_only=False):
    """
    Generates prediction tables similar to:
    - Table 3 (Join)
    - Table 5 (Join Type)
    - Table 6 (GroupBy)
    Also saves it as a CSV to 'results/metrics/{operator_name}_methods_comparison.csv'.

    Args:
        auto_suggest_metrics: Metrics from Auto-Suggest
        k_values: List of k values (like [1, 2])
        save_dir: Directory to save results
        baseline_metrics: Baseline comparison metrics (literature methods)
        vendor_metrics: Vendor comparison metrics (commercial systems)
        include_full_accuracy: Adds full-accuracy column (for GroupBy)
        operator_name: 'join', 'groupby', or 'join_type'
        include_accuracy_only: If True, show only 'prec@1' (like Join Type Table 5)
    """
    os.makedirs(save_dir, exist_ok=True)

    if baseline_metrics is None:
        baseline_metrics = {}

    # ----- Part 1: Literature / Baseline Comparison -----
    methods = ["Auto-Suggest"] + list(baseline_metrics.keys())
    rows = []

    for method in methods:
        row = [method]

        if include_accuracy_only:
            # Table 5: Join Type (prec@1 is equivalent to accuracy here)
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get('test_accuracy', 0):.2f}")
            else:
                row.append(f"{baseline_metrics[method].get('test_accuracy', 0):.2f}")
        else:
            # For Join Column / GroupBy: precision@k
            for k in k_values:
                if method == "Auto-Suggest":
                    row.append(f"{auto_suggest_metrics.get(f'precision@{k}', 0):.2f}")
                else:
                    row.append(f"{baseline_metrics[method].get(f'prec@{k}', 0):.2f}")

            # For Join Column / GroupBy: ndcg@k
            for k in k_values:
                if method == "Auto-Suggest":
                    row.append(f"{auto_suggest_metrics.get(f'ndcg@{k}', 0):.2f}")
                else:
                    row.append(f"{baseline_metrics[method].get(f'ndcg@{k}', 0):.2f}")

            # For GroupBy only: full-accuracy
            if include_full_accuracy:
                if method == "Auto-Suggest":
                    row.append(f"{auto_suggest_metrics.get('full-accuracy', 0):.0%}")
                else:
                    row.append(f"{baseline_metrics[method].get('full-accuracy', 0):.0%}")

        rows.append(row)

    # Build headers
    headers = ["method"]
    if include_accuracy_only:
        # For Join Type Table 5
        headers.append("prec@1")
    else:
        for k in k_values:
            headers.append(f"prec@{k}")
        for k in k_values:
            headers.append(f"ndcg@{k}")
        if include_full_accuracy:
            headers.append("full-accuracy")

    # Print the Literature / Baseline Comparison Table
    print(f"\nTable: {operator_name.capitalize()} Prediction - Literature Comparison")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save to CSV
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(os.path.join(save_dir, f"{operator_name}_methods_comparison.csv"), index=False)

    # ----- Part 2: Vendor Comparison (only for join and groupby if k=1) -----
    # Note: Vendor Comparison is NOT needed for Join Type (Table 5) because vendors are included in the same table
    if vendor_metrics and not include_accuracy_only and 1 in k_values:
        methods = ["Auto-Suggest"] + list(vendor_metrics.keys())
        vendor_rows = []
        for method in methods:
            row = [method]
            if method == "Auto-Suggest":
                row.append(f"{auto_suggest_metrics.get('precision@1', 0):.2f}")
                row.append(f"{auto_suggest_metrics.get('ndcg@1', 0):.2f}")
            else:
                row.append(f"{vendor_metrics[method].get('prec@1', 0):.2f}")
                row.append(f"{vendor_metrics[method].get('ndcg@1', 0):.2f}")
            vendor_rows.append(row)

        print(f"\nTable: {operator_name.capitalize()} Prediction - Vendor Comparison")
        print(tabulate(vendor_rows, headers=["method", "prec@1", "ndcg@1"], tablefmt="grid"))

        # Save to CSV
        # vendor_df = pd.DataFrame(vendor_rows, columns=["method", "prec@1", "ndcg@1"])
        # vendor_df.to_csv(os.path.join(save_dir, f"{operator_name}_vendor_comparison.csv"), index=False)


def generate_feature_importance_table(feature_importance, feature_names, operator="join", save_dir="results"):
    """
    Generates and visualizes feature importance for either Join Column or GroupBy prediction (as in Tables 4 and 7 in the paper).

    This function calculates "feature group importance" by aggregating individual feature importance values into logical groups.
    Group-level importance provides a higher-level view of which types of signals are most important, making the model more interpretable.

    The function:
    1. Groups individual features into logical categories (like "left-ness", "distinct-val-ratio", etc.)
    2. Aggregates importance values within each group
    3. Normalizes group importance to sum to 1
    4. Creates visualizations and prints the resulting table

    Example (Join):
        feature_names = ['left_distinct_ratio', 'right_distinct_ratio', 'jaccard_similarity', 'left_absolute_position']
        feature_importance = [0.15, 0.15, 0.1, 0.6]
        # Groups:
        # - "distinct-val-ratio": 0.15 + 0.15 = 0.3
        # - "val-overlap": 0.1
        # - "left-ness": 0.6

    Example (GroupBy):
        feature_names = ['distinct_count', 'is_int', 'value_range', 'groupby_term_in_name']
        feature_importance = [0.1, 0.2, 0.3, 0.4]
        # Groups:
        # - "col-type": 0.2
        # - "val-range": 0.3
        # - "col-name-freq": 0.4
        # - "distinct-val": 0.1

    Args:
        feature_importance: Array of feature importance values
        feature_names: List of feature names
        operator: 'join' or 'groupby'
        save_dir: Directory to save results (figures and tables)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    if operator == "join":
        feature_groups = {
            "left-ness": ["left_absolute_position", "left_relative_position", "right_absolute_position", "right_relative_position"],
            "val-range-overlap": ["range_overlap"],
            "distinct-val-ratio": ["left_distinct_ratio", "right_distinct_ratio"],
            "val-overlap": ["jaccard_similarity", "left_to_right_containment", "right_to_left_containment"],
            "single-col-candidate": ["is_single_column"],
            "col-val-types": ["left_is_string", "right_is_string", "left_is_numeric", "right_is_numeric", "type_match"],
            "table-stats": ["left_row_count", "right_row_count", "row_count_ratio"],
            "sorted-ness": ["left_is_sorted", "right_is_sorted"]
        }
        table_title = "Table 4: Importance of Feature Groups for Join"
        figure_name = "join_feature_group_importance.png"


    elif operator == "groupby":

        feature_groups = {
            "col-type": ["gb_is_string", "gb_is_int", "gb_is_float", "gb_is_bool", "gb_is_datetime", "agg_is_string", "agg_is_int", "agg_is_float", "agg_is_bool", "agg_is_datetime"],
            "col-name-freq": ["groupby_term_in_name", "agg_term_in_name"],
            "distinct-val": ["gb_distinct_count", "gb_distinct_ratio", "agg_distinct_count", "agg_distinct_ratio"],
            "val-range": ["gb_value_range_mean", "agg_value_range_mean", "gb_distinct_to_range_ratio", "agg_distinct_to_range_ratio"],
            "left-ness": ["gb_absolute_position_mean", "gb_relative_position_mean", "agg_absolute_position_mean", "agg_relative_position_mean"],
            "emptiness": ["gb_null_ratio_mean", "agg_null_ratio_mean"],
        }
        table_title = "Table 7: Importance of Feature Groups for GroupBy"
        figure_name = "groupby_feature_group_importance.png"

    else:
        raise ValueError("operator must be 'join' or 'groupby'")

    # Calculate importance for each group
    group_importance = {group: 0 for group in feature_groups}
    for group, features in feature_groups.items():
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

    # Create table rows
    rows = []
    if operator == "join":
        # Format in 4-column table (like Table 4)
        for i in range(0, len(sorted_groups), 4):
            row = []
            for j in range(4):
                if i + j < len(sorted_groups):
                    group, importance = sorted_groups[i + j]
                    row.extend([group, f"{importance:.2f}"])
                else:
                    row.extend(["", ""])
            rows.append(row)
        # Prepare headers
        headers = []
        for i in range(4):
            headers.extend([f"feature_{i + 1}", f"importance_{i + 1}"])
    else:
        # GroupBy: simple two-column table
        rows = [[group, f"{importance:.2f}"] for group, importance in sorted_groups]
        headers = ["feature", "importance"]

    # Print table
    print(f"\n{table_title}")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    groups, importances = zip(*sorted_groups)
    plt.barh(groups, importances)
    plt.xlabel('Feature Group Importance')
    plt.title(f'Feature Group Importance for {operator.capitalize()} Column Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", figure_name))
    plt.close()


def recommend_joins(models, left_table, right_table, top_k):
    """
    Generates complete join recommendations for two tables.

    This function implements the two-step join recommendation process:
    1. First predicts which columns to join
    2. Then predicts what type of join to use for those columns

    Args:
        models: Dictionary of loaded models and related data
        left_table: Left table for join
        right_table: Right table for join
        top_k: Number of top recommendations to return

    Returns:
        List of recommendations, each containing join columns and join type
    """
    # Step 1: Predict Join Columns
    print("\n=== Step 1: Predicting Join Columns ===")
    # print("Generating join predictions for all candidates between tables...")
    join_column_cand_preds = predict_join_columns(
        models['col_model'],
        models['col_feature_names'],
        left_table, right_table, top_k
    )

    if not join_column_cand_preds:
        print("No viable join candidates found between these tables.")
        return []

    # Limit to top_k candidates with top_k probs
    join_column_preds = join_column_cand_preds[:top_k]

    # Step 2: For each top column candidate, predict join type
    print("\n=== Step 2: Predicting Join Types ===")
    print("\nJoin Type Predictions:")
    # print(f"Determining join types for top {top_k} column candidates...")

    recommendations = []

    for i, (left_cols, right_cols, column_score) in enumerate(join_column_preds):
        # Predict join type for this candidate
        join_type_result = predict_join_type(
            models['type_model'],
            models['type_feature_names'],
            models['type_label_encoder'],
            left_table, right_table,
            left_cols, right_cols
        )

        predicted_type = join_type_result['predicted_join_type']
        print(f"Processing candidate {i + 1}: {', '.join(left_cols)} ↔ {', '.join(right_cols)} → {predicted_type} join")

        # Combine the column and type predictions into a recommendation
        recommendation = {
            'rank': i + 1,
            'left_join_columns': left_cols,
            'right_join_columns': right_cols,
            'column_confidence': column_score,
            'join_type': predicted_type,
            'join_type_confidence': join_type_result['confidence'],
            'alternative_join_types': join_type_result['alternatives']
        }

        recommendations.append(recommendation)

    print(f"\nGenerated {len(recommendations)} complete join recommendations")
    return recommendations


def display_join_recommendations(recommendations, top_k):
    """
    Displays join recommendations in a readable format.

    Args:
        recommendations: List of join recommendations
        top_k: Number of top recommendations to display (default: 2)
    """
    if not recommendations:
        print("No recommendations found.")
        return

    # Print recommendations in a readable format
    print("\n=== Complete Join Recommendations ===")
    print("=" * 80)

    # Only display top_k recommendations
    for rec in recommendations[:top_k]:
        left_cols = ", ".join(rec['left_join_columns'])
        right_cols = ", ".join(rec['right_join_columns'])

        print(f"\nRecommendation {rec['rank']}: Join using")
        print(f"  Left columns: {left_cols}")
        print(f"  Right columns: {right_cols}")
        print(f"  Column confidence: {rec['column_confidence']:.3f}")
        print(f"  Recommended join type: {rec['join_type']} (confidence: {rec['join_type_confidence']:.3f})")

        if rec['alternative_join_types']:
            print(f"  Alternative join types: {', '.join(rec['alternative_join_types'])} ")

        print("-" * 40)

    # Provide example pandas code for the top recommendation
    if recommendations:
        top_rec = recommendations[0]
        left_cols_str = ", ".join([f"'{col}'" for col in top_rec['left_join_columns']])
        right_cols_str = ", ".join([f"'{col}'" for col in top_rec['right_join_columns']])

        print("\n=== Example Pandas Code for Top Recommendation ===")

        if len(top_rec['left_join_columns']) == 1 and len(top_rec['right_join_columns']) == 1:
            # Single column join
            print(f"result = pd.merge(left_table, right_table,")
            print(f"                  left_on='{top_rec['left_join_columns'][0]}',")
            print(f"                  right_on='{top_rec['right_join_columns'][0]}',")
            print(f"                  how='{top_rec['join_type']}')\n")
        else:
            # Multi-column join
            print(f"result = pd.merge(left_table, right_table,")
            print(f"                  left_on=[{left_cols_str}],")
            print(f"                  right_on=[{right_cols_str}],")
            print(f"                  how='{top_rec['join_type']}')\n")


def evaluate_per_sample_ranking(sample_ids: List, y_true: List[int], y_pred: List[float], k_values: List[int]) -> Dict[str, float]:
    """

    Purpose:
    --------
    Evaluates **precision@k** and **ndcg@k** on a per-sample basis for join column
    prediction (or groupby) tasks. This function is crucial for **baselines** where
    each sample has multiple candidate join-column pairs, and we want to know:
      1) Did the true join columns appear in the top-k predictions?
      2) How early in the ranked list do they appear (NDCG)?

    Core Concept:
    --------------
    In join (or groupby) prediction, a "sample" is typically a **pair of tables** (left table and right table).
    For each such pair, the baselines generate **multiple candidate join column pairs**, each with a score.
    We want to measure how well these predictions rank the **ground-truth join columns** within each sample.

    How it Works:
    --------------
    1) Groups all predictions by their sample ID (each table pair is a sample).
    2) For each sample:
        a) Sorts predictions by predicted score (highest first).
        b) Checks if the correct join column pair appears in the top-k predictions.
        c) Counts how many samples have the correct join in top-k -> precision@k.
        d) Calculates **NDCG@k** (Normalized Discounted Cumulative Gain):
           - Measures not just whether the correct join is in top-k, but also
             **how early** it appears in the ranking.
    3) Aggregates precision@k and ndcg@k across all samples.

    Importance:
    -----------
    Unlike typical classification metrics (like sklearn's precision, recall), this function
    is **ranking-aware** and respects the **per-sample grouping** of predictions.
    This makes it **essential for evaluating baselines** that produce ranked lists
    of join column candidates (e.g., ML-FK, PowerPivot).

        Difference with Auto-Suggest:
    -----------------------------
    Auto-Suggest does **not** use this function because it predicts a single
    **best join column pair per sample** (not a ranked list of multiple candidates).
    In other words, Auto-Suggest directly outputs a top-1 prediction for each table pair.

    Since there is only one prediction per sample (and no ranking of multiple candidates),
    evaluating it is straightforward: you can simply compare the predicted join
    with the ground truth using standard scikit's classification metrics (e.g., accuracy, precision).

    In contrast, this function is essential for evaluating **baselines** that produce
    a **ranked list of join candidates** (e.g., ML-FK, PowerPivot). These baselines
    don't just give one best guess — they output a list of possible joins, each with a score.
    To fairly evaluate them, we need ranking-aware metrics like precision@k and ndcg@k
    that measure **how early in the list** the true join columns appear.

    Therefore, Auto-Suggest's simpler prediction style avoids the need for per-sample
    ranking metrics like ndcg@k, whereas baseline evaluations rely on this function
    to capture the ranking quality of their multiple predictions.

    Debugging Note:
    ----------------
    If you find precision@1 = 0 in results:
      - It usually means that no baseline method predicted the correct join columns
        at the top-1 position (rank 1) for any sample.
      - To debug, add printouts **inside `evaluate_baselines`** to see:
            - What were the top-k predictions for each sample?
            - Did any of them exactly match the true join columns?
      - Check that baseline predictions are properly **ordered by score (descending)**,
        since precision@1 depends on correct ranking.

    Example Usage:
    ---------------
        sample_ids = [1, 1, 1, 2, 2, 2]
        y_true     = [0, 1, 0, 0, 0, 1]
        y_pred     = [0.9, 0.5, 0.2, 0.8, 0.9, 0.6]

        metrics = evaluate_per_sample_ranking(sample_ids, y_true, y_pred, k_values=[1, 2])
        # => precision@1 = 0.0, precision@2 = 0.5
        # => ndcg@1 and ndcg@2 reflect ranking quality

        # Explanation:
        #  - Each sample has 3 predictions (table pair join candidates).
        #  - The true join columns appear at rank-2 for sample 1 and rank-3 for sample 2.
        #  - Since no correct join is at top-1, precision@1 = 0.
        #  - One sample has correct join in top-2, so precision@2 = 0.5.
        #  - NDCG captures how early the true join appears in the ranked list.

    Returns:
    --------
    A dictionary with averaged precision@k and ndcg@k for all specified k_values.
    """
    results = {}

    # 1. Group predictions by sample_id.
    # For each sample (e.g., a pair of tables), collect all
    # predicted candidates and their true labels (0/1) along
    # with the associated prediction scores.
    # This groups predictions together to evaluate per-sample
    # precision@k and ndcg@k (ranking-based metrics).
    sample_groups = {}
    for i, sample_id in enumerate(sample_ids):
        if sample_id not in sample_groups:
            sample_groups[sample_id] = []
        sample_groups[sample_id].append((y_true[i], y_pred[i]))

    # Calculate metrics for each k
    for k in k_values:
        correct_at_k = 0
        ndcg_sum = 0

        # For each sample (i.e., table pair with join candidates)
        for sample_id, predictions in sample_groups.items():
            # 2. Sort predictions by predicted score in descending order for each sample.
            # This determines the model's ranking for each sample.
            sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)

            # 3. Check if the correct join is present in the top-k predictions.
            # If so, count this sample as a "correct at k" for precision@k.
            # This directly implements:
            #   precision@k = (# of samples with correct join in top-k) / (total samples)
            top_k_preds = sorted_preds[:min(k, len(sorted_preds))]
            if any(label == 1 for label, _ in top_k_preds):
                correct_at_k += 1

            # Debug prints
            # print(f"Sample {sample_id} predictions:")
            # print(f"  Top-{k} predictions:")
            # for idx, (label, score) in enumerate(top_k_preds):
            #     print(f"    {idx}: label={label}, score={score}")
            # # Check if any of top-k predictions is correct
            # if any(label == 1 for label, _ in top_k_preds):
            #     print(f"    Found correct prediction in top-{k} for Sample {sample_id}")
            # else:
            #     print(f"    No correct prediction in top-{k} for Sample {sample_id}")

            # 4. Calculate NDCG@k for this sample.
            # DCG (Discounted Cumulative Gain):
            #   Measures ranking quality — correct join appearing early in the sorted list.
            #   We compute DCG for this sample by summing the discounted gains of true join(s).
            dcg = 0
            idcg = 0

            for i, (label, _) in enumerate(sorted_preds[:k]):
                if label == 1:
                    dcg += 1 / np.log2(i + 2)   # +2 because position is 0-indexed.

            # IDCG (Ideal DCG):
            #   Represents perfect ranking (all correct joins at the top).
            #   Computed by sorting by true labels.
            ideal_sorted = sorted(predictions, key=lambda x: x[0], reverse=True)
            for i, (label, _) in enumerate(ideal_sorted[:k]):
                if label == 1:
                    idcg += 1 / np.log2(i + 2)

            # Sample NDCG = DCG / IDCG (normalized)
            sample_ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_sum += sample_ndcg

        # Calculate final metrics
        num_samples = len(sample_groups)
        precision_at_k = correct_at_k / num_samples if num_samples > 0 else 0
        ndcg_at_k = ndcg_sum / num_samples if num_samples > 0 else 0

        results[f'precision@{k}'] = precision_at_k
        results[f'ndcg@{k}'] = ndcg_at_k

    return results

# ---------------------------------------------------------
# References for manual DCG/IDCG (NDCG) calculation:
#
# - Python implementation (similar manual calculation of DCG/IDCG):
#   https://github.com/kmbnw/rank_metrics/blob/master/python/ndcg.py
#
# - Theoretical overview of DCG and NDCG:
#   https://en.wikipedia.org/wiki/Discounted_cumulative_gain
#
# Note:
# These sources outline the standard DCG/NDCG methodology, aligning
# with the implementation here for evaluating ranking quality
# (e.g., correct join columns appearing early in the sorted list).
# ---------------------------------------------------------
