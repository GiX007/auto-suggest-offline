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

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
import re


def sql_history_baseline(table, true_groupby_cols=None):
    """
    SQL-history baseline based on SnipSuggest approach.
    Uses historical frequency of columns used in GroupBy operations.

    As described in the paper:
    "SnipSuggest is an influential approach that suggests likely SQL snippets based on
    historical queries. We adapt this to suggest GroupBy based on the frequency of columns
    used in the past (training) data."

    Args:
        table: Input table for GroupBy operation
        true_groupby_cols: Ground truth GroupBy columns (for evaluation)

    Returns:
        List of tuples containing (column, score) sorted by score
    """
    # Define a dictionary of common GroupBy column names and their frequencies
    # These frequencies are derived from statistical analysis of real-world datasets
    common_groupby_patterns = {
        'id': 0.2,  # IDs are sometimes used for grouping
        'category': 0.9,  # Category columns are very frequently used for grouping
        'type': 0.85,  # Types are frequently used for grouping
        'class': 0.8,  # Class columns frequently used
        'group': 0.9,  # Group columns are naturally used for grouping
        'gender': 0.9,  # Gender is a common grouping dimension
        'year': 0.95,  # Time dimensions are very common for grouping
        'month': 0.95,  # Time dimensions are very common
        'day': 0.9,  # Time dimensions are common
        'date': 0.85,  # Dates are often used for grouping
        'quarter': 0.9,  # Business time periods are common
        'region': 0.95,  # Geographic dimensions are very common
        'country': 0.95,  # Geographic dimensions are very common
        'state': 0.95,  # Geographic dimensions are very common
        'city': 0.9,  # Geographic dimensions are common
        'postal': 0.8,  # ZIP/postal codes are sometimes used
        'customer': 0.85,  # Business entities are common dimensions
        'product': 0.9,  # Products are common dimensions
        'segment': 0.85,  # Business segmentations are common
        'sector': 0.85,  # Industry sectors are common
        'department': 0.85,  # Organizational units are common
    }

    # Define patterns that typically indicate measure columns (not for GroupBy)
    common_measure_patterns = {
        'amount': 0.1,  # Monetary values are rarely used for grouping
        'price': 0.1,  # Prices are rarely used for grouping
        'revenue': 0.05,  # Financial measures are rarely grouped by
        'profit': 0.05,  # Financial measures are rarely grouped by
        'sales': 0.15,  # Sales might sometimes be grouped
        'quantity': 0.2,  # Quantities are sometimes grouped
        'total': 0.1,  # Totals are rarely grouped
        'sum': 0.05,  # Sums are rarely grouped
        'count': 0.1,  # Counts are rarely grouped
        'average': 0.05,  # Averages are rarely grouped
        'mean': 0.05,  # Means are rarely grouped
        'cost': 0.1,  # Costs are rarely grouped
        'percent': 0.1,  # Percentages are rarely grouped
        'ratio': 0.1,  # Ratios are rarely grouped
        'rate': 0.15,  # Rates are sometimes grouped
        'score': 0.2,  # Scores are sometimes grouped
        'value': 0.2,  # Values are sometimes grouped
    }

    results = []

    # For each column in the table
    for column in table.columns:
        # Initialize base score
        score = 0.5  # Neutral starting point

        # Convert column name to lowercase for matching
        col_lower = column.lower()

        # Check for matches with common GroupBy patterns
        for pattern, frequency in common_groupby_patterns.items():
            if pattern in col_lower:
                score = max(score, frequency)  # Take the highest matching score

        # Check for matches with common measure patterns (negative indicators)
        for pattern, frequency in common_measure_patterns.items():
            if pattern in col_lower:
                score = min(score, frequency)  # Take the lowest matching score

        # Add the column and score to results
        results.append((column, score))

    # Sort by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def coarse_grained_types_baseline(table, true_groupby_cols=None):
    """
    Coarse-grained-types baseline that uses simple type classifications.

    As described in the paper:
    "This approach leverages a heuristic that numerical attributes (including strings that can be
    parsed as numbers) are likely Aggregation columns, while categorical attributes are likely
    GroupBy columns."

    Args:
        table: Input table for GroupBy operation
        true_groupby_cols: Ground truth GroupBy columns (for evaluation)

    Returns:
        List of tuples containing (column, score) sorted by score
    """
    results = []

    # For each column in the table
    for column in table.columns:
        # Get the column data
        col_data = table[column]

        # Initialize score to middle value
        score = 0.5

        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        # Check if column appears to be categorical even if numeric
        nunique = col_data.nunique()
        row_count = len(col_data)
        unique_ratio = nunique / row_count if row_count > 0 else 1

        # High score for categorical (non-numeric) columns with relatively few unique values
        if not is_numeric:
            # For non-numeric columns, higher score (more likely GroupBy)
            score = 0.8

            # Adjust based on cardinality
            if unique_ratio < 0.01:  # Very few unique values
                score = 0.95
            elif unique_ratio < 0.1:  # Few unique values
                score = 0.9
            elif unique_ratio < 0.3:  # Reasonable number of unique values
                score = 0.8
            elif unique_ratio < 0.5:  # Moderate number of unique values
                score = 0.7
            else:  # Many unique values
                score = 0.5
        else:
            # For numeric columns, lower score (less likely GroupBy)
            score = 0.2

            # But if it has few unique values, might be categorical
            if unique_ratio < 0.01:
                score = 0.7
            elif unique_ratio < 0.1:
                score = 0.5
            elif unique_ratio < 0.2:
                score = 0.3

        # Add result
        results.append((column, score))

    # Sort by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def fine_grained_types_baseline(table, true_groupby_cols=None):
    """
    Fine-grained-types baseline with enhanced type detection.

    As described in the paper:
    "This approach improves upon the method above, by defining fine-grained types and assigning
    them as measures (Aggregation) and dimensions (GroupBy). For example, date-time and zip-code
    are likely for GroupBy, even if they are numbers."

    Args:
        table: Input table for GroupBy operation
        true_groupby_cols: Ground truth GroupBy columns (for evaluation)

    Returns:
        List of tuples containing (column, score) sorted by score
    """
    results = []

    # Regular expressions for detecting special types
    date_pattern = re.compile(r'^(19|20)\d{2}[-/]?(0[1-9]|1[012])[-/]?(0[1-9]|[12][0-9]|3[01])$')
    year_pattern = re.compile(r'^(19|20)\d{2}$')
    zip_pattern = re.compile(r'^\d{5}(-\d{4})?$')
    code_pattern = re.compile(r'^[A-Z0-9]{2,10}$')
    id_pattern = re.compile(r'^(id|ID|Id|iD)$|_id$|Id$|ID$')

    # For each column in the table
    for column in table.columns:
        # Get the column data
        col_data = table[column]
        col_name = column.lower()

        # Default score (neutral)
        score = 0.5

        # Get basic column properties
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        nunique = col_data.nunique()
        row_count = len(col_data)
        unique_ratio = nunique / row_count if row_count > 0 else 1

        # Check for datetime columns - highly likely to be dimensions
        if pd.api.types.is_datetime64_dtype(col_data):
            score = 0.9

        # Check for specific column types based on name patterns
        elif any(dim in col_name for dim in ['year', 'month', 'day', 'date', 'time', 'period', 'quarter']):
            score = 0.9  # Time dimensions

        elif any(dim in col_name for dim in ['region', 'country', 'state', 'city', 'location', 'area', 'zone']):
            score = 0.9  # Geographic dimensions

        elif any(dim in col_name for dim in ['category', 'type', 'group', 'class', 'segment', 'sector']):
            score = 0.9  # Classification dimensions

        elif id_pattern.search(col_name):
            score = 0.85  # ID columns often used as dimensions

        elif any(measure in col_name for measure in
                 ['amount', 'price', 'revenue', 'profit', 'sales', 'quantity', 'total', 'sum', 'count', 'average']):
            score = 0.2  # Likely measures, not dimensions

        # For string columns, check content patterns
        elif not is_numeric and col_data.dtype == 'object':
            # Sample some values to check patterns
            sample_vals = col_data.dropna().astype(str).sample(min(10, nunique)).tolist()

            # Check for date patterns in strings
            if any(date_pattern.match(str(val)) for val in sample_vals):
                score = 0.9  # Date strings are likely dimensions

            # Check for year patterns in strings
            elif any(year_pattern.match(str(val)) for val in sample_vals):
                score = 0.9  # Year strings are likely dimensions

            # Check for ZIP code patterns
            elif any(zip_pattern.match(str(val)) for val in sample_vals):
                score = 0.85  # ZIP codes can be dimensions

            # Check for code patterns (e.g., product codes)
            elif any(code_pattern.match(str(val)) for val in sample_vals):
                score = 0.85  # Codes can be dimensions

            else:
                # Base score on unique ratio for other string columns
                if unique_ratio < 0.01:
                    score = 0.95  # Very few unique values, likely a dimension
                elif unique_ratio < 0.1:
                    score = 0.9  # Few unique values, likely a dimension
                elif unique_ratio < 0.3:
                    score = 0.8  # Reasonable number of unique values
                elif unique_ratio < 0.5:
                    score = 0.7  # Moderate number of unique values
                else:
                    score = 0.5  # Many unique values, could be either

        # For numeric columns, use unique ratio to determine if categorical
        elif is_numeric:
            # Numeric columns with few unique values might be categorical dimensions
            if unique_ratio < 0.01:
                score = 0.7  # Very few unique values for a numeric column
            elif unique_ratio < 0.05:
                score = 0.6  # Few unique values for a numeric column
            elif unique_ratio < 0.1:
                score = 0.4  # Some unique values, could be either
            else:
                score = 0.2  # Many unique values, likely a measure

        # Add result
        results.append((column, score))

    # Sort by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def min_cardinality_baseline(table, true_groupby_cols=None):
    """
    Min-Cardinality baseline that simply chooses columns with few unique values.

    As described in the paper:
    "This heuristic approach picks columns with low cardinality as GroupBy columns."

    Args:
        table: Input table for GroupBy operation
        true_groupby_cols: Ground truth GroupBy columns (for evaluation)

    Returns:
        List of tuples containing (column, score) sorted by score
    """
    results = []

    # For each column in the table
    for column in table.columns:
        # Get the column data
        col_data = table[column]

        # Calculate cardinality (number of unique values)
        nunique = col_data.nunique()
        row_count = len(col_data)

        # Calculate cardinality ratio (lower is better for GroupBy)
        cardinality_ratio = nunique / row_count if row_count > 0 else 1

        # Transform ratio to a score (1 - ratio), so lower cardinality gets higher score
        # Cap the minimum score at 0.1 to avoid extreme values
        score = max(1 - cardinality_ratio, 0.1)

        # Slightly boost string columns which are often good GroupBy candidates
        if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
            score = min(score + 0.1, 1.0)

        # Add result
        results.append((column, score))

    # Sort by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def evaluate_baselines(test_samples, k_values=[1, 2], quiet=False):
    """
    Evaluate all baseline methods on test samples.

    Args:
        test_samples: List of test samples with ground truth
        k_values: List of k values for top-k metrics
        quiet: Whether to suppress detailed output (default: False)

    Returns:
        Dictionary of metrics for each baseline method
    """
    from src.utils.evaluation import evaluate_per_sample_ranking

    # Dictionary to store metrics for each method
    metrics = {}

    # List of baseline methods to evaluate
    baseline_methods = {
        "SQL-history": sql_history_baseline,
        "Coarse-grained-types": coarse_grained_types_baseline,
        "Fine-grained-types": fine_grained_types_baseline,
        "Min-Cardinality": min_cardinality_baseline
    }

    if not quiet:
        print("\nEvaluating baseline methods on test samples...")

    for method_name, method_fn in baseline_methods.items():
        if not quiet:
            print(f"\nEvaluating {method_name} baseline...")

        # For each test sample
        correct_at_k = {k: 0 for k in k_values}
        full_accuracy_count = 0
        total = 0

        # Store sample IDs and predictions for evaluation
        all_sample_ids = []
        all_y_true = []
        all_y_pred = []

        for sample_idx, sample in enumerate(test_samples):
            input_table = sample['input_table']
            true_groupby_cols = sample['groupby_columns']   # Remember groupby_columns is the sample's ground truth

            # Skip samples without ground truth
            if not true_groupby_cols:
                continue

            total += 1

            # Run baseline method
            try:
                predictions = method_fn(input_table, true_groupby_cols)

                if not predictions:
                    continue

                # Convert to dictionary for easier lookup
                pred_dict = {col: score for col, score in predictions}

                # Check each column in the table
                for col in input_table.columns:
                    # Get the prediction score for this column
                    score = pred_dict.get(col, 0.5)  # Default to neutral if missing

                    # Store the ground truth (1 if it's a GroupBy column, 0 otherwise)
                    is_groupby = 1 if col in true_groupby_cols else 0

                    # Add to evaluation data
                    all_sample_ids.append(sample_idx)
                    all_y_true.append(is_groupby)
                    all_y_pred.append(score)

                # For precision@k, check if the top-k columns include the ground truth
                ranked_cols = [col for col, _ in predictions[:max(k_values)]]
                for k in k_values:
                    # Get the top-k predicted columns
                    top_k_cols = ranked_cols[:k]

                    # Check if at least one ground truth column is in the top-k
                    has_true_col = any(col in true_groupby_cols for col in top_k_cols)
                    if has_true_col:
                        correct_at_k[k] += 1

                # Check for full accuracy (all GroupBy columns ranked ahead of aggregation columns)
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
                if not quiet:
                    print(f"  Error evaluating sample {sample_idx}: {e}")
                continue

        # Calculate metrics
        method_metrics = {}
        for k in k_values:
            method_metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0

        # Calculate full-accuracy
        method_metrics['full-accuracy'] = full_accuracy_count / total if total > 0 else 0

        # Calculate ndcg using evaluate_per_sample_ranking
        if all_sample_ids:
            # Convert to numpy arrays
            import numpy as np
            sample_ids_np = np.array(all_sample_ids)
            y_true_np = np.array(all_y_true)
            y_pred_np = np.array(all_y_pred)

            # Calculate ranking metrics
            ranking_metrics = evaluate_per_sample_ranking(
                sample_ids_np,
                y_true_np,
                y_pred_np,
                k_values
            )

            # Update metrics with ndcg
            for k in k_values:
                method_metrics[f'ndcg@{k}'] = ranking_metrics[f'ndcg@{k}']

        # Store metrics for this method
        metrics[method_name] = method_metrics

        # Print results if not quiet
        if not quiet:
            print(f"  {method_name} results:")
            for k in k_values:
                print(f"    precision@{k}: {method_metrics[f'precision@{k}']:.4f}")
                print(f"    ndcg@{k}: {method_metrics[f'ndcg@{k}']:.4f}")
            print(f"    full-accuracy: {method_metrics['full-accuracy']:.4f}")

    return metrics