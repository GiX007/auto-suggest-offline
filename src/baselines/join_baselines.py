# src/baselines/join_baselines.py
#
# Implementation of baseline methods for join column prediction
# Based on algorithms described in the Auto-Suggest paper
#
# This file implements:
# 1. ML-FK: Machine learning approach for foreign-key discovery
# 2. PowerPivot: Heuristic approach used in Microsoft PowerPivot
# 3. Multi: Multi-column join discovery approach
# 4. Holistic: Combined approach using multiple signals
# 5. max-overlap: Simple value overlap heuristic

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier

# Import from our package structure
from src.features.join_features import extract_join_column_features, generate_join_candidates


def ml_fk_baseline(left_table, right_table, true_left_cols=None, true_right_cols=None):
    """
    ML-FK baseline that uses machine learning to identify foreign key relationships.
    Based on "Objective criteria for the evaluation of clustering methods" (Rostin et al.)

    This method:
    1. Generates candidates based on inclusion dependency
    2. Uses a random forest to classify candidates
    3. Ranks candidates by classifier confidence

    Args:
        left_table: Left table for join
        right_table: Right table for join
        true_left_cols: Ground truth left columns (for evaluation)
        true_right_cols: Ground truth right columns (for evaluation)

    Returns:
        Ranked list of join column predictions
    """
    # Generate candidates
    candidates = generate_join_candidates(left_table, right_table)

    # Filter candidates based on inclusion dependency
    filtered_candidates = []
    for left_cols, right_cols in candidates:
        # For multi-column candidates, join using compound key
        if len(left_cols) > 1 or len(right_cols) > 1:
            # Create compound keys (Combine join columns into single string key)
            if len(left_cols) > 1:
                left_key = left_table[left_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
            else:
                left_key = left_table[left_cols[0]].astype(str)

            if len(right_cols) > 1:
                right_key = right_table[right_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
            else:
                right_key = right_table[right_cols[0]].astype(str)

            # Calculate containment
            left_values = set(left_key)
            right_values = set(right_key)
            overlap = len(left_values.intersection(right_values))
            left_in_right = overlap / len(left_values) if len(left_values) > 0 else 0

            # Apply inclusion dependency threshold (80%)
            if left_in_right >= 0.8:
                filtered_candidates.append((left_cols, right_cols))

        # For single-column candidates
        else:
            # Calculate containment
            left_values = set(left_table[left_cols[0]].astype(str))
            right_values = set(right_table[right_cols[0]].astype(str))
            overlap = len(left_values.intersection(right_values))
            left_in_right = overlap / len(left_values) if len(left_values) > 0 else 0

            # Apply inclusion dependency threshold (80%)
            if left_in_right >= 0.8:
                filtered_candidates.append((left_cols, right_cols))

    # If no candidates pass the filter, use all candidates
    if len(filtered_candidates) == 0:
        filtered_candidates = candidates

    # Extract features for each candidate
    features_list = []
    for left_cols, right_cols in filtered_candidates:
        features = extract_join_column_features(left_table, right_table, left_cols, right_cols)
        features_list.append(features)

    # Convert to DataFrame for prediction
    if not features_list:
        return []

    features_df = pd.DataFrame(features_list)

    # Convert boolean features to integers
    bool_cols = features_df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        features_df[col] = features_df[col].astype(int)

    # Create a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)

    # Use a few key features for prediction
    feature_cols = [
        'jaccard_similarity',
        'left_to_right_containment',
        'right_to_left_containment',
        'left_is_string',
        'right_is_string',
        'left_is_numeric',
        'right_is_numeric'
    ]

    # Handle missing columns
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0

    # Generate synthetic training data
    X_synth = np.random.rand(100, len(feature_cols))
    # Bias toward high containment for positive examples
    y_synth = (X_synth[:, 1] > 0.8).astype(int)

    # Fit model on synthetic data
    model.fit(X_synth, y_synth)

    # Make predictions
    X_pred = features_df[feature_cols].values
    scores = model.predict_proba(X_pred)[:, 1]

    # Rank candidates by scores
    results = []
    for (left_cols, right_cols), score in zip(filtered_candidates, scores):
        results.append((left_cols, right_cols, score))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def power_pivot_baseline(left_table, right_table, true_left_cols=None, true_right_cols=None):
    """
    PowerPivot baseline that uses heuristics to identify join columns.
    Based on "Fast foreign-key detection in Microsoft SQL Server PowerPivot for Excel" (Chen et al.)

    This method:
    1. Prunes columns based on data type compatibility
    2. Prioritizes columns with names containing "id", "key", "code"
    3. Ranks by value overlap for remaining candidates

    Args:
        left_table: Left table for join
        right_table: Right table for join
        true_left_cols: Ground truth left columns (for evaluation)
        true_right_cols: Ground truth right columns (for evaluation)

    Returns:
        Ranked list of join column predictions
    """
    # Generate candidates - single column only (PowerPivot focus)
    candidates = []
    for left_col in left_table.columns:
        for right_col in right_table.columns:
            # Skip obvious type mismatches
            if left_table[left_col].dtype != right_table[right_col].dtype:
                continue

            # Skip boolean columns (rare join keys)
            if pd.api.types.is_bool_dtype(left_table[left_col]):
                continue

            candidates.append(([left_col], [right_col]))

    # Score candidates
    results = []
    for left_cols, right_cols in candidates:
        left_col = left_cols[0]
        right_col = right_cols[0]

        # Base score
        score = 0

        # Boost score for columns with key-like names
        key_terms = ['id', 'key', 'code', 'num', 'no']
        for term in key_terms:
            if term in left_col.lower():
                score += 0.2
            if term in right_col.lower():
                score += 0.2

        # Calculate value overlap
        left_values = set(left_table[left_col].astype(str))
        right_values = set(right_table[right_col].astype(str))

        # Jaccard similarity
        union = len(left_values.union(right_values))
        intersection = len(left_values.intersection(right_values))
        jaccard = intersection / union if union > 0 else 0

        # Add to score
        score += jaccard

        # Boost score if column names match
        if left_col.lower() == right_col.lower():
            score += 0.3

        results.append((left_cols, right_cols, score))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def multi_baseline(left_table, right_table, true_left_cols=None, true_right_cols=None):
    """
    Multi-column baseline for discovering multi-column foreign-keys.
    Based on "On multi-column foreign key discovery" (Zhang et al.)

    This method:
    1. Considers both single and multi-column join candidates
    2. Uses Earth Mover's Distance (EMD) for value distribution comparison
    3. Ranks by a combination of inclusion and EMD

    Explanation:
    - left_in_right captures value-level inclusion, a strong indicator of foreign-key relationships (foreign keys have the same name in both left and right tables).
    - distribution_similarity (approximate EMD) assesses whether the value frequencies align, helping distinguish between valid vs. misleading matches (e.g., overlapping keys with skewed distributions).
    - These two signals are complementary — one checks overlap, the other checks frequency shape.

    Args:
        left_table: Left table for join
        right_table: Right table for join
        true_left_cols: Ground truth left columns (for evaluation)
        true_right_cols: Ground truth right columns (for evaluation)

    Returns:
        Ranked list of join column predictions
    """
    # Generate both single and multi-column candidates
    candidates = generate_join_candidates(left_table, right_table, max_multi_column=2)

    # Score candidates
    results = []
    for left_cols, right_cols in candidates:
        # Simple approximation of EMD using value distributions
        # For single column case
        if len(left_cols) == 1 and len(right_cols) == 1:
            left_col = left_cols[0]
            right_col = right_cols[0]

            # Calculate value distributions
            left_value_counts = left_table[left_col].value_counts(normalize=True)
            right_value_counts = right_table[right_col].value_counts(normalize=True)

            # Find common values
            common_values = set(left_value_counts.index).intersection(set(right_value_counts.index))

            # Calculate a simple approximation of EMD
            distribution_diff = 0
            for value in common_values:
                diff = abs(left_value_counts.get(value, 0) - right_value_counts.get(value, 0))
                distribution_diff += diff

            # Normalize by number of common values
            if len(common_values) > 0:
                distribution_diff /= len(common_values)

            # Convert to a similarity score (lower diff = higher similarity)
            distribution_similarity = 1 - min(distribution_diff, 1)
        else:
            # For multi-column, use a simpler approach
            distribution_similarity = 0.5  # default value

        # Calculate containment (inclusion dependency)
        # Create compound keys for multi-column
        if len(left_cols) > 1:
            left_key = left_table[left_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
        else:
            left_key = left_table[left_cols[0]].astype(str)

        if len(right_cols) > 1:
            right_key = right_table[right_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
        else:
            right_key = right_table[right_cols[0]].astype(str)

        # Calculate containment
        left_values = set(left_key)
        right_values = set(right_key)
        intersection = len(left_values.intersection(right_values))
        left_in_right = intersection / len(left_values) if len(left_values) > 0 else 0

        # Combine containment and distribution similarity into a final score (70% containment + 30% distributional similarity)
        score = 0.7 * left_in_right + 0.3 * distribution_similarity

        results.append((left_cols, right_cols, score))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def holistic_baseline(left_table, right_table, true_left_cols=None, true_right_cols=None):
    """
    Holistic baseline that combines multiple approaches.
    Based on "Holistic primary key and foreign key detection" (Jiang and Naumann)

    This method:
    1. Combines signals from data types, value distributions, and inclusion dependencies
    2. Uses a weighted scoring approach
    3. Considers both structure and content-based evidence

    Args:
        left_table: Left table for join
        right_table: Right table for join
        true_left_cols: Ground truth left columns (for evaluation)
        true_right_cols: Ground truth right columns (for evaluation)

    Returns:
        Ranked list of join column predictions
    """
    # Generate candidates
    candidates = generate_join_candidates(left_table, right_table)

    # Score candidates
    results = []
    for left_cols, right_cols in candidates:
        # Extract features
        features = extract_join_column_features(left_table, right_table, left_cols, right_cols)

        # Calculate holistic score
        score = 0

        # Inclusion dependency (50% weight)
        containment = features.get('left_to_right_containment', 0)
        score += 0.5 * containment

        # Column name similarity (20% weight)
        name_similarity = 0
        if len(left_cols) == 1 and len(right_cols) == 1:
            left_col = left_cols[0]
            right_col = right_cols[0]

            # Simple name similarity
            if left_col.lower() == right_col.lower():
                name_similarity = 1.0
            elif left_col.lower() in right_col.lower() or right_col.lower() in left_col.lower():
                name_similarity = 0.5

        score += 0.2 * name_similarity

        # Data type compatibility (10% weight)
        type_match = features.get('type_match', 0)
        score += 0.1 * type_match

        # Value distribution similarity (20% weight)
        # Use jaccard similarity as a proxy
        jaccard = features.get('jaccard_similarity', 0)
        score += 0.2 * jaccard

        results.append((left_cols, right_cols, score))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def max_overlap_baseline(left_table, right_table, true_left_cols=None, true_right_cols=None):
    """
    Maximum overlap baseline that simply ranks by value overlap.
    A common approach used in many systems.

    Args:
        left_table: Left table for join
        right_table: Right table for join
        true_left_cols: Ground truth left columns (for evaluation)
        true_right_cols: Ground truth right columns (for evaluation)

    Returns:
        Ranked list of join column predictions
    """
    # Generate candidates
    candidates = generate_join_candidates(left_table, right_table)

    # Score candidates by overlap
    results = []
    for left_cols, right_cols in candidates:
        # Create compound keys for multi-column
        if len(left_cols) > 1:
            left_key = left_table[left_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
        else:
            left_key = left_table[left_cols[0]].astype(str)

        if len(right_cols) > 1:
            right_key = right_table[right_cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
        else:
            right_key = right_table[right_cols[0]].astype(str)

        # Calculate overlap (Jaccard similarity)
        left_values = set(left_key)
        right_values = set(right_key)
        intersection = len(left_values.intersection(right_values))
        union = len(left_values.union(right_values))
        jaccard = intersection / union if union > 0 else 0

        results.append((left_cols, right_cols, jaccard))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def evaluate_baselines(test_samples, k_values=[1, 2]):
    """
    Evaluate all baseline methods on test samples.

    Args:
        test_samples: List of test samples with ground truth
        k_values: List of k values for top-k metrics

    Returns:
        Dictionary of metrics for each baseline method
    """
    from src.utils.evaluation import evaluate_per_sample_ranking

    # Dictionary to store metrics for each method
    metrics = {}

    # List of baseline methods to evaluate
    baseline_methods = {
        "ML-FK": ml_fk_baseline,
        "PowerPivot": power_pivot_baseline,
        "Multi": multi_baseline,
        "Holistic": holistic_baseline,
        "max-overlap": max_overlap_baseline
    }

    # print("\nEvaluating baseline methods on test samples...")
    for method_name, method_fn in baseline_methods.items():
        # print(f"\nEvaluating {method_name} baseline...")

        # For each test sample
        correct_at_k = {k: 0 for k in k_values}
        ndcg_sum = {k: 0 for k in k_values}
        total = 0

        # Store sample IDs and predictions for evaluation
        all_sample_ids = []
        all_y_true = []
        all_y_pred = []

        for sample_idx, sample in enumerate(test_samples):
            left_table = sample['left_table']
            right_table = sample['right_table']
            true_left_cols = sample['left_join_keys']
            true_right_cols = sample['right_join_keys']

            # Skip samples without ground truth
            if not true_left_cols or not true_right_cols:
                continue

            total += 1

            # Run baseline method
            try:
                predictions = method_fn(left_table, right_table, true_left_cols, true_right_cols)

                if not predictions:
                    continue

                # Check if ground truth is in top predictions
                for i, (left_cols, right_cols, score) in enumerate(predictions):
                    # Check if this prediction matches the ground truth
                    is_match = (set(left_cols) == set(true_left_cols) and
                                set(right_cols) == set(true_right_cols))

                    # Store for evaluation
                    all_sample_ids.append(sample_idx)
                    all_y_true.append(1 if is_match else 0)
                    all_y_pred.append(score)

                    if is_match and i < max(k_values):
                        # Found correct prediction
                        for k in k_values:
                            if i < k:
                                correct_at_k[k] += 1
                        # No need to check more predictions for this sample
                        break

            except Exception as e:
                # print(f"  Error evaluating sample {sample_idx}: {e}")
                continue

        # Calculate metrics
        method_metrics = {}
        for k in k_values:
            method_metrics[f'precision@{k}'] = correct_at_k[k] / total if total > 0 else 0  # precision = corrects / total test samples

        # Calculate ndcg using evaluate_per_sample_ranking
        ranking_metrics = evaluate_per_sample_ranking(
            np.array(all_sample_ids),
            np.array(all_y_true),
            np.array(all_y_pred),
            k_values
        )

        # Update metrics with ndcg
        for k in k_values:
            method_metrics[f'ndcg@{k}'] = ranking_metrics[f'ndcg@{k}']

        # Store metrics for this method
        metrics[method_name] = method_metrics

        # Comment out result printing
        # print(f"  {method_name} results:")
        # for k in k_values:
        #     print(f"    precision@{k}: {method_metrics[f'precision@{k}']:.4f}")
        #     print(f"    ndcg@{k}: {method_metrics[f'ndcg@{k}']:.4f}")

    return metrics