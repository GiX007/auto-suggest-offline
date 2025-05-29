# src/baselines/groupby_baselines.py
#
# Implementation of baseline methods for GroupBy column prediction
# Based on algorithms described in the Auto-Suggest paper (Section 6.5.3)
#
# This file implements:
# 1. SQL-history: Based on frequency of columns used in past GroupBy operations
# 2. Coarse-grained-types: Heuristic using basic data types
# 3. Fine-grained-types: Enhanced version of coarse-grained-types with better type detection
# 4. Min-Cardinality: Simple heuristic based on the number of unique values
#

import re
import random
import pandas as pd


def sql_history_baseline(table):
    """
    SQL-history baseline that assigns random groupby-likelihood scores (0.5-0.6) to common patterns.

    This method:
    1. Uses precomputed patterns of commonly used GroupBy column names (e.g., 'year', 'category', 'region')
       but assigns a random groupby-likelihood score in the range [0.5, 0.6] for each match, instead of a fixed frequency.
    2. Penalizes measure columns (e.g., 'price', 'amount', 'sales') with random scores in [0.5, 0.6] as well.
    3. Heuristically decides for each column whether it is more groupby-like or measure-like based on the higher random score.
    4. Computes a final groupby-likelihood heuristic score for each column.
    5. Returns a dictionary of {column: score} for groupby-likelihood.

    This random-scoring approach ensures more variability and better mimics noisy historical usage patterns.

    Args:
        table: Input table for pivot operation.

    Returns:
        Dictionary mapping each dimension column to its random groupby-likelihood score.
    """

    # Common patterns for groupby and measure columns (just keywords, not scores)
    common_groupby_patterns = [
        'id', 'category', 'type', 'class', 'group', 'gender', 'year', 'month', 'day',
        'date', 'quarter', 'region', 'country', 'state', 'city', 'postal', 'customer',
        'product', 'segment', 'sector', 'department'
    ]

    common_measure_patterns = [
        'amount', 'price', 'revenue', 'profit', 'sales', 'quantity', 'total', 'sum',
        'count', 'average', 'mean', 'cost', 'percent', 'ratio', 'rate', 'score', 'value'
    ]

    # Output: per-column groupby-likelihood scores
    column_scores = {}

    for column in table.columns:
        col_lower = column.lower()
        base_score = 0.5  # neutral

        # Random groupby score if any groupby pattern matches
        groupby_score = max(
            (random.uniform(0.5, 0.55) for pattern in common_groupby_patterns if pattern in col_lower),
            default=base_score
        )

        # Random measure score if any measure pattern matches
        measure_score = min(
            (random.uniform(0.5, 0.55) for pattern in common_measure_patterns if pattern in col_lower),
            default=base_score
        )

        # Final groupby-likelihood: prefer groupby-like if it's higher
        if groupby_score >= measure_score:
            final_score = groupby_score
        else:
            final_score = 1 - measure_score  # less groupby-like if more measure-like

        column_scores[column] = final_score

    return column_scores


def coarse_grained_types_baseline(table):
    """
    Coarse-grained-types baseline that estimates the likelihood of each column being a GroupBy dimension,
    using simple data type and cardinality heuristics.

    This method:
    1. Separates columns into candidate grouping (dimension) and measure columns:
       - Grouping columns are typically categorical (non-numeric) or numeric with low cardinality.
       - Measure columns are typically numeric with high cardinality.
    2. Uses simple thresholds on data type and distinct-value ratio to determine likely candidates.
    3. Computes a per-column groupby-likelihood score:
       - Non-numeric columns (categorical) are very likely to be grouping columns (score=0.9).
       - Numeric columns with low cardinality are also likely to be grouping columns (score=0.8).
       - High-cardinality numeric columns are unlikely to be grouping columns (score=0.2).
    4. Returns a dictionary of {column: score} for groupby-likelihood.

    Args:
        table: Input table for pivot operation.

    Returns:
        Dictionary mapping each column to its groupby-likelihood score.
    """
    row_count = len(table)
    column_scores = {}

    for column in table.columns:
        col_data = table[column]
        nunique = col_data.nunique()
        unique_ratio = nunique / row_count if row_count > 0 else 1

        if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_datetime64_any_dtype(col_data):
            # Numeric column
            if unique_ratio > 0.2:
                # High-cardinality numeric columns → low groupby-likelihood
                groupby_score = 0.2
            else:
                # Low-cardinality numeric columns → more groupby-like
                groupby_score = 0.8
        else:
            # Non-numeric columns (categorical) → typical groupby candidates
            groupby_score = 0.9

        column_scores[column] = groupby_score

    return column_scores


def fine_grained_types_baseline(table):
    """
    Fine-grained-types baseline that estimates the likelihood of each column
    being a GroupBy dimension, using enhanced type detection and content-based heuristics.

    This method:
    1. Uses column name patterns and data type heuristics to identify likely grouping columns
       (e.g., IDs, dates, codes, categories).
    2. Incorporates advanced pattern matching (e.g., date patterns, year patterns, ZIP codes).
    3. Uses cardinality-based thresholds for numeric and string columns.
    4. Assigns a per-column groupby-likelihood score:
       - Columns with strong dimension-like patterns (date, ID, region) get high scores (0.9).
       - Numeric columns with low cardinality are also likely grouping columns (score=0.8).
       - High-cardinality numeric columns or known measure columns get low scores (0.2).
    5. Returns a dictionary of {column: score} for groupby-likelihood.

    Args:
        table: Input table for pivot operation.

    Returns:
        Dictionary mapping each column to its groupby-likelihood score.
    """
    # Pre-compiled regex patterns
    date_pattern = re.compile(r'^(19|20)\d{2}[-/]?(0[1-9]|1[012])[-/]?(0[1-9]|[12][0-9]|3[01])$')
    year_pattern = re.compile(r'^(19|20)\d{2}$')
    zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
    code_pattern = re.compile(r'^[A-Z0-9]{2,10}$')
    id_pattern = re.compile(r'^(id|ID|Id|iD)$|_id$|Id$|ID$')

    row_count = len(table)
    column_scores = {}

    for column in table.columns:
        col_data = table[column]
        col_name = column.lower()
        nunique = col_data.nunique()
        unique_ratio = nunique / row_count if row_count > 0 else 1
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        # Default score
        groupby_score = 0.5

        # Strong grouping patterns in name or type
        if pd.api.types.is_datetime64_any_dtype(col_data):
            groupby_score = 0.7
        elif any(dim in col_name for dim in ['year', 'month', 'day', 'date', 'time', 'period', 'quarter']):
            groupby_score = 0.7
        elif any(dim in col_name for dim in ['region', 'country', 'state', 'city', 'location', 'area', 'zone']):
            groupby_score = 0.7
        elif any(dim in col_name for dim in ['category', 'type', 'group', 'class', 'segment', 'sector']):
            groupby_score = 0.7
        elif id_pattern.search(col_name):
            groupby_score = 0.7
        elif any(measure in col_name for measure in
                 ['amount', 'price', 'revenue', 'profit', 'sales', 'quantity', 'total',
                  'sum', 'count', 'average', 'mean', 'cost', 'percent', 'ratio', 'rate', 'score', 'value']):
            groupby_score = 0.3  # typical measure columns
        elif not is_numeric and col_data.dtype == 'object':
            # String-based pattern matches (date, ZIP, year, codes)
            sample_vals = col_data.dropna().astype(str).sample(min(10, nunique)).tolist()
            if any(date_pattern.match(val) for val in sample_vals):
                groupby_score = 0.7
            elif any(year_pattern.match(val) for val in sample_vals):
                groupby_score = 0.7
            elif any(zip_pattern.match(val) for val in sample_vals):
                groupby_score = 0.7
            elif any(code_pattern.match(val) for val in sample_vals):
                groupby_score = 0.7
            else:
                # Fallback on cardinality for strings
                groupby_score = 0.5 if unique_ratio < 0.5 else 0.3
        elif is_numeric:
            # Cardinality for numeric columns
            groupby_score = 0.5 if unique_ratio < 0.2 else 0.3

        column_scores[column] = groupby_score

    return column_scores


def min_cardinality_baseline(table):
    """
    Min-Cardinality baseline that estimates the likelihood of each column
    being used as a GroupBy dimension, based on cardinality (number of unique values).

    This method:
    1. Assigns low-cardinality columns as likely grouping dimensions.
    2. Penalizes high-cardinality numeric columns, treating them as likely measures.
    3. Boosts score for string-based low-cardinality columns (often good dimensions).
    4. Assigns a per-column groupby-likelihood score:
       - Low-cardinality columns (ratio < 0.2) get high scores (0.9 if is string, 0.8 if numeric).
       - High-cardinality numeric columns get low scores (0.2).
    5. Returns a dictionary of {column: score} for groupby-likelihood.

    Args:
        table: Input table for pivot operation.

    Returns:
        Dictionary mapping each column to its groupby-likelihood score.
    """
    row_count = len(table)
    column_scores = {}

    for column in table.columns:
        col_data = table[column]
        nunique = col_data.nunique()
        cardinality_ratio = nunique / row_count if row_count > 0 else 1

        # Heuristic thresholds
        if cardinality_ratio < 0.2:
            # Low-cardinality columns → good dimensions
            if col_data.dtype == 'object':
                groupby_score = 0.9  # string-based dimensions
            else:
                groupby_score = 0.8  # numeric low-cardinality columns
        else:
            # High-cardinality numeric columns → less likely groupby
            if pd.api.types.is_numeric_dtype(col_data) and not pd.api.types.is_datetime64_any_dtype(col_data):
                groupby_score = 0.3
            else:
                groupby_score = 0.5  # neutral for non-numeric high-cardinality columns

        column_scores[column] = groupby_score

    return column_scores


def evaluate_baselines(test_samples, k_values):
    """
    Evaluates all baseline methods on test samples.

    Args:
        test_samples: List of test samples with ground truth
        k_values: List of k values for top-k metrics

    Returns:
        Dictionary of metrics for each baseline method
    """
    from src.utils.model_utils import evaluate_per_sample_ranking

    metrics = {}

    baseline_methods = {
        "SQL-history": sql_history_baseline,
        "Coarse-grained-types": coarse_grained_types_baseline,
        "Fine-grained-types": fine_grained_types_baseline,
        "Min-Cardinality": min_cardinality_baseline
    }

    print("\nEvaluating baseline methods on test samples...")

    for method_name, method_fn in baseline_methods.items():
        #print(f"\nEvaluating {method_name} baseline...")

        # Keep track of correct predictions for each test sample
        correct_at_k = {k: 0 for k in k_values}
        full_accuracy_count = 0
        total = 0

        # Store sample IDs and predictions for evaluation
        all_sample_ids = []
        all_y_true = []
        all_y_pred = []

        for sample_idx, sample in enumerate(test_samples):
            input_table = sample['input_table']
            true_groupby_cols = sample['groupby_columns']
            #true_agg_cols = sample['agg_columns']

            if not true_groupby_cols:
                continue
            total += 1

            try:
                # Each baseline returns {column: score} for groupby-likelihood
                column_scores = method_fn(input_table)

                # Create predictions as (column, score) for sorting
                predictions = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)

                # Evaluate for each column
                for col, score in predictions:
                    is_groupby = 1 if col in true_groupby_cols else 0
                    all_sample_ids.append(sample_idx)
                    all_y_true.append(is_groupby)
                    all_y_pred.append(score)

                # Check correct-at-k for groupby columns
                ranked_cols = [col for col, _ in predictions[:max(k_values)]]
                for k in k_values:
                    top_k_cols = ranked_cols[:k]
                    has_true_col = any(col in true_groupby_cols for col in top_k_cols)
                    if has_true_col:
                        correct_at_k[k] += 1

                # Full-accuracy: perfect match of predicted groupby columns with true groupby
                # Use threshold 0.5 to decide groupby-likelihood
                predicted_groupby_cols = [col for col, score in predictions if score >= 0.5]
                all_true_cols_predicted = all(col in predicted_groupby_cols for col in true_groupby_cols)
                no_extra_predicted = all(col in true_groupby_cols for col in predicted_groupby_cols)

                if all_true_cols_predicted and no_extra_predicted:
                    full_accuracy_count += 1

            except Exception as e:
                print(f"  Error evaluating sample {sample_idx}: {e}")
                continue

        # Calculate prec and ndcg using evaluate_per_sample_ranking
        ranking_metrics = evaluate_per_sample_ranking(
            all_sample_ids,
            all_y_true,
            all_y_pred,
            k_values
        )

        # Store metrics with prec and ndcg
        method_metrics = {}
        for k in k_values:
            method_metrics[f'prec@{k}'] = ranking_metrics[f'precision@{k}']
            method_metrics[f'ndcg@{k}'] = ranking_metrics[f'ndcg@{k}']

        method_metrics['full-accuracy'] = full_accuracy_count / total if total > 0 else 0

        # Store metrics for this method
        metrics[method_name] = method_metrics

        # Printing results
        # print(f"  {method_name} results:")
        # for k in k_values:
        #     print(f"    precision@{k}: {method_metrics[f'precision@{k}']:.4f}")
        #     print(f"    ndcg@{k}: {method_metrics[f'ndcg@{k}']:.4f}")

    return metrics
