# src/baselines/unpivot_baselines.py
#
# Implementation of baseline methods for Unpivot column prediction
# Based on algorithms described in the Auto-Suggest paper (Section 6.5.5)
#
# This file implements:
# 1. Pattern-similarity: Uses pattern matching to identify similar columns
# 2. Col-name-similarity: Uses column name similarity to find related columns
# 3. Data-type: Uses data types to identify columns to unpivot
# 4. Contiguous-type: Uses data types with contiguity constraint
#

import re
import os


def detect_pattern_similarity(col1: str, col2: str) -> float:
    """
    Detects pattern similarity between two column names.

    This helps identify columns that follow a pattern like:
    - year_2018, year_2019, year_2020
    - jan_sales, feb_sales, mar_sales

    Args:
        col1: First column name
        col2: Second column name

    Returns:
        Similarity score between 0 and 1
    """
    # Normalize to lowercase for case-insensitive comparison
    col1 = col1.lower()
    col2 = col2.lower()

    # If columns are identical, they're not a good match for unpivot
    if col1 == col2:
        return 0.5

    # Extract non-numeric parts of column names (remove numeric parts)
    non_numeric1 = re.sub(r'\d+', '', col1).strip('_')
    non_numeric2 = re.sub(r'\d+', '', col2).strip('_')

    # If they have the same non-numeric part, they likely follow a pattern
    if non_numeric1 and non_numeric2 and non_numeric1 == non_numeric2:
        return 0.9

    # Check common meaningful prefix/suffix (common patterns)
    # (e.g., "year_2008" vs "year_2009" → "year_"; ignore if prefix == full column name)
    prefix = os.path.commonprefix([col1, col2]) # finds the longest character by character match from the start of 2 strings
    if len(prefix) > 2 and prefix != col1 and prefix != col2:
        return 0.8

    # Check reversed to find the longest common suffix
    col1_rev = col1[::-1]
    col2_rev = col2[::-1]
    suffix = os.path.commonprefix([col1_rev, col2_rev])[::-1]
    if len(suffix) > 2 and suffix != col1 and suffix != col2:
        return 0.8

    # Moderate similarity for partially matching patterns
    if non_numeric1 and non_numeric2 and (non_numeric1 in non_numeric2 or non_numeric2 in non_numeric1):
        return 0.6

    # Low similarity for different patterns
    return 0.2


def jaccard_similarity(str1: str, str2: str) -> float:
    """
    Calculates Jaccard similarity between two strings.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Jaccard similarity (intersection/union)
    """
    set1 = set(str1.lower())
    set2 = set(str2.lower())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0


def pattern_similarity_baseline(table, true_id_vars=None, true_value_vars=None):
    """
    Pattern-similarity baseline that uses pattern matching to identify similar columns.

    Based on Kabra & Saillet's approach for restructuring tables.

    Args:
        table: Input table for unpivot operation
        true_id_vars: Ground truth columns to keep as-is (for evaluation)
        true_value_vars: Ground truth columns to unpivot (for evaluation)

    Returns:
        Tuple of (id_vars, value_vars) where value_vars are columns to unpivot

    Note:
        If no strong pattern-based grouping is found (or too few columns to unpivot),
        this function automatically falls back to using the data-type-based approach.
    """
    all_columns = list(table.columns)

    if len(all_columns) < 3:  # Need at least 3 columns for meaningful unpivot
        return all_columns, []

    # Step 1: Detect pattern similarities between column names
    similarity_matrix = {}
    for i, col1 in enumerate(all_columns):
        for j, col2 in enumerate(all_columns):
            if i < j:  # Only calculate for unique pairs
                similarity = detect_pattern_similarity(col1, col2)
                similarity_matrix[(col1, col2)] = similarity

    # Step 2: Find groups of similar columns
    # Start with the pair having the highest similarity
    best_pair = max(similarity_matrix.items(), key=lambda x: x[1])
    if best_pair[1] < 0.6:  # If no pair has good similarity, use data type approach
        return data_type_baseline(table, true_id_vars, true_value_vars)

    # Initialize value_vars with the best pair
    value_vars = list(best_pair[0])
    remaining_cols = [col for col in all_columns if col not in value_vars]

    # Add columns with high similarity to the initial group
    for col in remaining_cols.copy():
        # Check similarity with existing value_vars
        avg_similarity = sum(similarity_matrix.get((min(col, v), max(col, v)), 0)
                             for v in value_vars) / len(value_vars)

        if avg_similarity > 0.6:  # Add if sufficiently similar
            value_vars.append(col)
            remaining_cols.remove(col)

    # If we have too few columns to unpivot, fall back to data type approach
    if len(value_vars) < 2:
        return data_type_baseline(table, true_id_vars, true_value_vars)

    # Remaining columns are id_vars
    id_vars = remaining_cols

    return id_vars, value_vars


def col_name_similarity_baseline(table, true_id_vars=None, true_value_vars=None):
    """
    Col-name-similarity baseline that uses column name similarity for unpivot.

    Based on approach described in the paper for data deduplication using
    column name similarity measured in Jaccard.

    Args:
        table: Input table for unpivot operation
        true_id_vars: Ground truth columns to keep as-is (for evaluation)
        true_value_vars: Ground truth columns to unpivot (for evaluation)

    Returns:
        Tuple of (id_vars, value_vars) where value_vars are columns to unpivot

            Note:
        If no strong cluster is found (or too few columns to unpivot),
        this function automatically falls back to using the data-type-based approach.

    Note:
        If no strong cluster is found (or too few columns to unpivot),
        this function automatically falls back to using the data-type-based approach.
    """
    all_columns = list(table.columns)

    if len(all_columns) < 3:  # Need at least 3 columns for meaningful unpivot
        return all_columns, []

    # Step 1: Look for semantic clusters based on name similarity
    name_clusters = []
    remaining = all_columns.copy()

    while remaining:
        # Start a new cluster with the first remaining column
        current = remaining[0]
        cluster = [current]
        remaining.remove(current)

        # Find columns similar to the current one
        i = 0
        while i < len(remaining):
            next_col = remaining[i]
            # Calculate Jaccard similarity between column names
            sim = jaccard_similarity(current, next_col)

            if sim > 0.4:  # Add to cluster if sufficiently similar
                cluster.append(next_col)
                remaining.remove(next_col)
            else:
                i += 1

        name_clusters.append(cluster)

    # Step 2: Find the largest cluster
    largest_cluster = max(name_clusters, key=len)

    # Only consider as value_vars if the cluster has at least 2 columns
    if len(largest_cluster) >= 2:
        value_vars = largest_cluster
        id_vars = [col for col in all_columns if col not in value_vars]
    else:
        # Fall back to data type approach if no good cluster found
        id_vars, value_vars = data_type_baseline(table, true_id_vars, true_value_vars)

    return id_vars, value_vars


def data_type_baseline(table, true_id_vars=None, true_value_vars=None):
    """
    Data-type baseline that uses data types to identify unpivot columns.

    This approach groups columns by their data types, with the assumption that
    columns of the same type that aren't key columns are good candidates for melting.

    Args:
        table: Input table for unpivot operation
        true_id_vars: Ground truth columns to keep as-is (for evaluation)
        true_value_vars: Ground truth columns to unpivot (for evaluation)

    Returns:
        Tuple of (id_vars, value_vars) where value_vars are columns to unpivot
    """
    all_columns = list(table.columns)

    if len(all_columns) < 3:  # Need at least 3 columns for meaningful unpivot
        return all_columns, []

    # Group columns by data type
    type_groups = {}
    for col in all_columns:
        dtype = str(table[col].dtype)
        if dtype not in type_groups:
            type_groups[dtype] = []
        type_groups[dtype].append(col)

    # Identify candidate groups: same-type columns that aren't likely keys
    candidate_groups = []

    for dtype, cols in type_groups.items():
        # Skip small groups
        if len(cols) < 2:
            continue

        # Check if these are likely key/identifier columns
        # Keys often have high distinct value counts
        is_key_group = True
        for col in cols:
            # If less than 80% values are unique, not likely a key
            is_key_group = all((table[col].nunique() / len(table)) >= 0.8 for col in cols if len(table) > 0)

        if not is_key_group:
            candidate_groups.append((dtype, cols))

    # Sort groups by size (descending)
    candidate_groups.sort(key=lambda x: len(x[1]), reverse=True)

    # If we found candidate groups, use the largest
    if candidate_groups:
        value_vars = candidate_groups[0][1]
        id_vars = [col for col in all_columns if col not in value_vars]
    else:
        # Default: assume first column is id, rest are values
        id_vars = [all_columns[0]]
        value_vars = all_columns[1:]

    return id_vars, value_vars


def contiguous_type_baseline(table, true_id_vars=None, true_value_vars=None):
    """
    Contiguous-type baseline that requires unpivot columns to be adjacent.

    This builds on the data-type approach but adds the constraint that
    melted columns must be contiguous in the table.

    Args:
        table: Input table for unpivot operation
        true_id_vars: Ground truth columns to keep as-is (for evaluation)
        true_value_vars: Ground truth columns to unpivot (for evaluation)

    Returns:
        Tuple of (id_vars, value_vars) where value_vars are columns to unpivot
    """
    all_columns = list(table.columns)

    if len(all_columns) < 3:  # Need at least 3 columns for meaningful unpivot
        return all_columns, []

    # Find contiguous sections of columns with the same type
    sections = []
    current_type = None
    current_section = []

    for col in all_columns:
        dtype = str(table[col].dtype)

        if dtype != current_type:
            # Start a new section
            if current_section:
                sections.append((current_type, current_section))
            current_type = dtype
            current_section = [col]
        else:
            # Continue current section
            current_section.append(col)

    # Add the last section if it exists
    if current_section:
        sections.append((current_type, current_section))

    # Filter sections to those with multiple columns
    candidate_sections = [s for s in sections if len(s[1]) >= 2]

    # Find the largest contiguous section
    if candidate_sections:
        candidate_sections.sort(key=lambda x: len(x[1]), reverse=True)
        value_vars = candidate_sections[0][1]
        id_vars = [col for col in all_columns if col not in value_vars]
    else:
        # Fall back to data type approach if no good contiguous section found
        id_vars, value_vars = data_type_baseline(table, true_id_vars, true_value_vars)

    return id_vars, value_vars


def evaluate_baselines(test_samples):
    """
    Evaluates all baseline methods on test samples.

    For each baseline:
      - Calculates full accuracy (exact match)
      - Calculates average precision, recall, and F1-score (column-level)

    Args:
        test_samples: List of test samples with ground truth.

    Returns:
        Dictionary of metrics for each baseline method.
    """
    # Dictionary to store metrics for each method
    metrics = {}

    # List of baseline methods to evaluate
    baseline_methods = {
        "Pattern-similarity": pattern_similarity_baseline,
        "Col-name-similarity": col_name_similarity_baseline,
        "Data-type": data_type_baseline,
        "Contiguous-type": contiguous_type_baseline
    }

    # print("\nEvaluating baseline methods for unpivot prediction...")

    for method_name, method_fn in baseline_methods.items():
        #print(f"\nEvaluating {method_name} baseline...")

        correct_predictions = 0
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        total_samples = 0

        for sample_idx, sample in enumerate(test_samples):
            try:
                input_table = sample['input_table']

                # Get ground truth columns
                true_id_vars = set(sample.get('id_vars', []))
                true_value_vars = set(sample.get('value_vars', []))

                if not true_id_vars or not true_value_vars:
                    # Skip if missing ground truth
                    continue

                total_samples += 1

                # Run baseline method
                pred_id_vars, pred_value_vars = method_fn(input_table, list(true_id_vars), list(true_value_vars))

                pred_id_vars = set(pred_id_vars)
                pred_value_vars = set(pred_value_vars)

                # Check if prediction matches ground truth (or flipped)
                is_correct = (pred_id_vars == true_id_vars and pred_value_vars == true_value_vars) or \
                             (pred_id_vars == true_value_vars and pred_value_vars == true_id_vars)

                if is_correct:
                    correct_predictions += 1

                # Calculate precision, recall, and f1-score
                intersection = len(pred_value_vars.intersection(true_value_vars))
                precision = intersection / len(pred_value_vars) if pred_value_vars else 0
                recall = intersection / len(true_value_vars) if true_value_vars else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                precision_sum += precision
                recall_sum += recall
                f1_sum += f1_score

                # Debug prints
                # if is_correct:
                #     print(f"  Sample {sample_idx}: ✓ (Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f})")
                # else:
                #     print(f"  Sample {sample_idx}: ✗ (Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_score:.2f})")

            except Exception as e:
                print(f"  Error evaluating sample {sample_idx}: {e}")
                continue

        # Calculate final metrics
        avg_precision = precision_sum / total_samples if total_samples > 0 else 0
        avg_recall = recall_sum / total_samples if total_samples > 0 else 0
        avg_f1 = f1_sum / total_samples if total_samples > 0 else 0
        full_accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        metrics[method_name] = {
            'full_accuracy': full_accuracy,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': avg_f1
        }

        # Summary
        # print(f"  {method_name} results:")
        # print(f"    full_accuracy: {full_accuracy:.4f}")
        # print(f"    column_precision: {avg_precision:.4f}")
        # print(f"    column_recall: {avg_recall:.4f}")
        # print(f"    column_F1: {avg_f1:.4f}")

    return metrics
