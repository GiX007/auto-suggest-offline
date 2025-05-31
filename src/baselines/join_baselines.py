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
#

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.join_col_model import extract_join_column_features, generate_join_candidates
from src.utils.model_utils import save_model

# Define Paths
base_dir = "C:/Users/giorg/Auto_Suggest"

# ----------------------------
# Helper functions
# ----------------------------
def compute_inclusion_score(left_values, right_values):
    return len(left_values & right_values) / len(left_values) if left_values else 0

def compute_jaccard_score(left_values, right_values):
    union = len(left_values | right_values)
    return len(left_values & right_values) / union if union > 0 else 0

def compute_uniqueness_score(column_values):
    return len(set(column_values)) / len(column_values) if len(column_values) > 0 else 0

def make_key(df, cols):
    """
    Constructs a join key from one or more columns.

    Why is this useful?
    --------------------
    - In join prediction tasks, we need to compare "join keys" from different tables to calculate features like overlap and inclusion.
    - These join keys can be:
        1. Single-column (e.g., "user_id")
        2. Multi-column (e.g., ["first_name", "last_name"])
    - To **uniformly** handle both cases (single-column and multi-column join keys),
      we transform the multi-column keys into a single "composite key" string for each row.

    How it works:
    ---------------
    - If there's only one column (e.g., "user_id"), return it as a string Series.
    - If there are multiple columns, convert each row's values to strings, join them with a hyphen '-',
      and return the result as a single string key per row (e.g., "John-Smith").

    This ensures:
    - Join keys can be easily compared as **sets of strings** (even if originally multi-column).
    - Features like Jaccard similarity, inclusion, and uniqueness can be computed consistently,
      regardless of whether the join key involves one or many columns.

    Example usage:
    ----------------
    # Single-column join
    left_key = make_key(left_table, ['user_id'])

    # Multi-column join
    left_key = make_key(left_table, ['first_name', 'last_name'])

    Args:
        df: The DataFrame containing the join key columns.
        cols: List of columns (single or multiple) to use as join keys.

    Returns:
        A pandas Series of string join keys for each row.
    """
    if len(cols) > 1:
        # Multi-column join key: concatenate column values with '-' to create a composite key string
        return df[cols].apply(lambda x: '-'.join(x.astype(str)), axis=1)
    else:
        # Single-column join key: convert values to string directly
        return df[cols[0]].astype(str)


def ml_fk_baseline(left_table, right_table, model):
    """
    ML-FK baseline that identifies join columns using foreign key-style features
    and a machine learning classifier.

    Based on: "Objective criteria for the evaluation of clustering methods"
    (Rostin et al., 2009), adapted for Auto-Suggest's real-data setting.

    This method:
    1. Uses real training samples extracted from replayed notebooks, labeled with ground-truth join columns.
    2. Computes simple foreign key detection features for each candidate column pair:
       - inclusion_score: fraction of left key values found in the right key.
       - uniqueness_score: number of unique values in the left key divided by number of rows (approx. key-ness).
       - jaccard_score: Jaccard similarity between left and right key values (overlap measure).
    3. Trains a Gradient Boosting Classifier on these features using binary labels (1 = correct join, 0 = incorrect).
    4. Applies the trained model to score join candidates in the input tables.
    5. Returns a ranked list of predicted join column pairs, sorted by classifier confidence/score (left_cols, right_cols, score).

    This baseline simulates classic FK discovery behavior (as in ML-FK), with learned thresholds and combinations,
    and is particularly effective for structured join behavior in real-world notebook data.

    Args:
        left_table: Left input table.
        right_table: Right input table.
        model: Trained ML model to score column pairs.

    Returns:
        Ranked list of predicted join column pairs (left_cols, right_cols, score).
    """
    # Predict on input tables
    candidates = generate_join_candidates(left_table, right_table, max_multi_column=2)
    test_X, valid_candidates = [], []

    for left_cols, right_cols in candidates:
        try:
            lkey = make_key(left_table, left_cols)
            rkey = make_key(right_table, right_cols)
            lval = set(lkey)
            rval = set(rkey)

            features = [
                compute_inclusion_score(lval, rval),
                compute_uniqueness_score(lkey),
                compute_jaccard_score(lval, rval)
            ]
            test_X.append(features)
            valid_candidates.append((left_cols, right_cols))
        except Exception as e:
            print(f"Error processing candidate {left_cols}-{right_cols}: {e}")
            continue

    if not test_X:
        return []

    # Predict confidence scores and rank
    scores = model.predict_proba(test_X)[:, 1]

    results = [
        (left_cols, right_cols, score)
        for (left_cols, right_cols), score in zip(valid_candidates, scores)
    ]
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def power_pivot_baseline(left_table, right_table):
    """
    PowerPivot baseline that identifies likely join columns using simple heuristics.

    Based on: "Fast foreign-key detection in Microsoft SQL Server PowerPivot for Excel" (Chen et al.).

    This method:
    1. Prunes candidate column pairs based on data type compatibility and excludes boolean columns.
    2. Considers only single-column join candidates (multi-column joins are not supported).
    3. Assigns scores to candidates using the following heuristics:
       - +0.2 if the column name contains key-related terms (e.g., 'id', 'key', 'code', 'num', 'no').
       - +0.3 if column names match exactly (case-insensitive).
       - +jaccard_similarity based on value overlap between columns.
       - +containment score (fraction of left values also found in right).
    4. Returns a ranked list of single-column join candidates with their heuristic scores.

    Args:
        left_table: Left input table.
        right_table: Right input table.

    Returns:
        Ranked list of single-column join candidates with their heuristic scores.
    """
    # If 'index' is in the columns of either table, make sure it's a real column
    if 'index' not in left_table.columns:
        left_table = left_table.reset_index().rename(columns={'index': 'index'})

    if 'index' not in right_table.columns:
        right_table = right_table.reset_index().rename(columns={'index': 'index'})

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
            score += 0.5    # +score for name match

        # Compute containment score: what fraction of left values are also in right
        containment = len(left_values & right_values) / len(left_values) if left_values else 0
        score += containment

        # - × log - to penalize small - key joins ("Small-key join" = a join using a column with very few
        # distinct values. These joins are often incorrect, so our baseline penalizes them to improve robustness)
        # Adjust score using log-scaled sample size to penalize small candidate sets (penalizes small cardinality joins)
        score *= np.log1p(len(left_values)) / np.log1p(1000)

        results.append((left_cols, right_cols, score))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def multi_baseline(left_table, right_table):
    """
    Multi-column baseline for discovering join keys, including multi-column foreign keys.

    Based on: "On multi-column foreign key discovery" (Zhang et al.).

    This method:
    1. Considers both single-column and multi-column join candidates (up to 2 columns).
    2. Computes two complementary signals for each candidate:
       - Containment score (left_in_right): the fraction of distinct left key values that appear in the right key.
         This captures value-level inclusion — a strong indicator of true join relationships.
       - Distribution similarity (approx. Earth Mover's Distance): for single-column candidates, it compares the normalized
         value frequency distributions between the two columns and computes 1 - average absolute difference.
         For multi-column candidates, a default similarity score of 0.5 is used.
    3. Combines the two signals into a final score:
       - Final score = 0.5 * containment + 0.5 * distribution similarity.
       - The score is then adjusted by log-scaled key cardinality (based on distinct left values) to penalize small-key joins,
         reducing the chance of spurious matches on low-cardinality fields.
    4. Returns a ranked list of single- or multi-column join candidates with their computed scores.

    Args:
        left_table: Left input table.
        right_table: Right input table.

    Returns:
        Ranked list of single- or multi-column join candidates with their computed scores.
    """
    # If 'index' is in the columns of either table, make sure it's a real column
    if 'index' not in left_table.columns:
        left_table = left_table.reset_index().rename(columns={'index': 'index'})

    if 'index' not in right_table.columns:
        right_table = right_table.reset_index().rename(columns={'index': 'index'})

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


def holistic_baseline(left_table, right_table):
    """
    Holistic baseline that ranks join candidates by combining multiple signals,
    inspired by "Holistic primary key and foreign key detection" (Jiang and Naumann).

    This method:
    1. Considers all single- and multi-column join candidates.
    2. Computes a holistic score by combining structure-based and content-based evidence:
       - +0.4 × left-to-right containment: measures inclusion dependency.
       - +0.3 × column name similarity: exact name match scores 1.0; substring match scores 0.5.
       - +0.15 × type match: binary signal for whether data types are compatible.
       - +0.15 × Jaccard similarity: overlap of distinct values between candidate columns.
    3. Adjusts the score using log-scaled candidate size (based on number of join columns) to slightly penalize large join keys.
    4. Returns a ranked list of join column candidates based on the computed holistic score.

    Args:
        left_table: Left input table.
        right_table: Right input table.

    Returns:
        Ranked list of join column candidates with their holistic scores.
    """
    # If 'index' is in the columns of either table, make sure it's a real column
    if 'index' not in left_table.columns:
        left_table = left_table.reset_index().rename(columns={'index': 'index'})

    if 'index' not in right_table.columns:
        right_table = right_table.reset_index().rename(columns={'index': 'index'})

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
        score += 0.2 * containment

        # Column name similarity (20% weight)
        name_similarity = 0
        if len(left_cols) == 1 and len(right_cols) == 1:
            left_col = left_cols[0]
            right_col = right_cols[0]

            # Column name similarity (exact match: 1.0, substring match: 0.5)
            if left_col.lower() == right_col.lower():
                name_similarity = 1.0
            elif left_col.lower() in right_col.lower() or right_col.lower() in left_col.lower():
                name_similarity = 0.5

        score += 0.3 * name_similarity

        # Data type compatibility (10% weight)
        type_match = features.get('type_match', 0)
        score += 0.1 * type_match

        # Value distribution similarity (20% weight)
        # Use jaccard similarity as a proxy
        jaccard = features.get('jaccard_similarity', 0)
        score += 0.4 * jaccard

        results.append((left_cols, right_cols, score))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def max_overlap_baseline(left_table, right_table):
    """
    Maximum Overlap baseline that ranks join candidates purely based on
    Jaccard similarity between their distinct values.

    This method:
    1. Considers both single-column and multi-column join candidates.
    2. For each candidate pair, constructs compound keys (if needed) to support multi-column joins.
    3. Computes Jaccard similarity:
       - intersection size / union size of distinct values from left and right keys.
       - higher Jaccard indicates stronger value-level overlap between the two columns.
    4. Adjusts the score using log-scaled number of columns in the join key to lightly penalize wide keys.
    5. Returns a ranked list of join column candidates based on Jaccard similarity scores.

    Args:
        left_table: Left input table.
        right_table: Right input table.

    Returns:
        Ranked list of join column candidates with their Jaccard similarity scores.
    """
    # If 'index' is in the columns of either table, make sure it's a real column
    if 'index' not in left_table.columns:
        left_table = left_table.reset_index().rename(columns={'index': 'index'})

    if 'index' not in right_table.columns:
        right_table = right_table.reset_index().rename(columns={'index': 'index'})

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

        # Slightly reduce the impact of even high overlaps
        #jaccard *= 0.8  # soften scores a bit

        # If overlap is very low, apply a stronger penalty
        if intersection < 5:
            jaccard *= 0.5

        results.append((left_cols, right_cols, jaccard))

    # Sort by score
    results.sort(key=lambda x: x[2], reverse=True)

    return results


def evaluate_baselines(test_samples, k_values):
    """
    Evaluates all baseline methods on test samples.

    Args:
        test_samples: List of test samples with ground truth
        k_values: List of k values for top-k metrics

    Returns:
        Dictionary of metrics for each baseline method
    """
    # Dynamically import
    from src.utils.model_utils import evaluate_per_sample_ranking

    # ----------------------------
    # Train ML-FK model once
    # ----------------------------

    # Load preprocessed training samples
    train_path = "data/test_data/join_train_samples.pkl"
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")

    with open(train_path, "rb") as f:
        train_samples = pickle.load(f)

    # Step 2: Extract FK-style features from train samples
    X_train, y_train = [], []

    for n_sample, sample in enumerate(train_samples):
        #print(f"Processing sample: {n_sample+1}/{len(train_samples)}...")

        # Sample 34 has a right table with over 1 million rows, which causes significant slowdown
        # due to set-based operations (make_key, set()) in ML-FK. This is expected and normal.

        try:
            lt, rt = sample["left_table"], sample["right_table"]
            true_lk, true_rk = sample["left_join_keys"], sample["right_join_keys"]

            # If 'index' is in the join keys but not in the DataFrame, reset index to create the 'index' column
            if 'index' in true_lk and 'index' not in lt.columns:
                lt = lt.reset_index().rename(columns={'index': 'index'})

            if 'index' in true_rk and 'index' not in rt.columns:
                rt = rt.reset_index().rename(columns={'index': 'index'})

            # Sanity log sizes
            # print(f"   Left shape: {lt.shape}, Right shape: {rt.shape}")
            # print(f"   Join keys: {true_lk} ↔ {true_rk}")

        except Exception as e:
            print(f"Sample {n_sample + 1} failed to unpack: {e}")
            continue


        # Skip if join keys are missing (sanity check)
        if not true_lk or not true_rk:
            continue

        candidates = generate_join_candidates(lt, rt, max_multi_column=2)

        # Limit the number of candidates to 15 per test sample for speed
        max_candidates_per_sample = 15
        if len(candidates) > max_candidates_per_sample:
            candidates = candidates[:max_candidates_per_sample]

        for idx, (left_cols, right_cols) in enumerate(candidates):
            #print(f"[ML-FK] Processing candidate {idx + 1}/{len(candidates)}: {left_cols} <-> {right_cols}")
            try:
                lkey = make_key(lt, left_cols)
                rkey = make_key(rt, right_cols)
                lval = set(lkey)
                rval = set(rkey)

                features = [
                    compute_inclusion_score(lval, rval),
                    compute_uniqueness_score(lkey),
                    compute_jaccard_score(lval, rval)
                ]
                label = int(set(left_cols) == set(true_lk) and set(right_cols) == set(true_rk))
                X_train.append(features)
                y_train.append(label)
            except Exception as e:
                print(f"Sample {n_sample + 1} failed to unpack: {e}")
                continue

    if not X_train:
        print("Warning: No valid training samples for ML-FK. Skipping baseline.")
        return {}

    # Ensure X, y are numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)


    # Step 3: Train model
    print("\nTraining ml-fk prediction model...")
    start_time = time.time()

    clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    end_time = time.time()
    total_training_time = end_time - start_time

    print(f"\nModel training completed in {total_training_time:.2f} seconds")
    print(f"Trained model: RandomForestClassifier ({clf.n_estimators} estimators)")   # , max_depth={clf.max_depth}

    # Save the model
    model_path = os.path.join(base_dir, "models", "ml_fk_model.pkl")
    save_model(clf, model_path)

    # ----------------------------
    # Evaluate Baselines
    # ----------------------------

    # Dictionary to store metrics for each method
    metrics = {}

    # List of baseline methods to evaluate
    baseline_methods = {
        "ML-FK": lambda left_t, right_t, *_: ml_fk_baseline(left_t, right_t, clf),
        "PowerPivot": power_pivot_baseline,
        "Multi": multi_baseline,
        "Holistic": holistic_baseline,
        "max-overlap": max_overlap_baseline
    }

    # print("\nEvaluating baseline methods on test samples...")
    for method_name, method_fn in baseline_methods.items():
        # print(f"\nEvaluating {method_name} baseline...")

        # For each test sample
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
                predictions = method_fn(left_table, right_table)
                if not predictions:
                    continue

                # Check if ground truth is in top predictions
                for i, (left_cols, right_cols, score) in enumerate(predictions):
                    # Check if this prediction matches the ground truth
                    is_match = (set(left_cols) == set(true_left_cols) and
                                set(right_cols) == set(true_right_cols))

                    # Debug prints: check true vs. baseline predictions
                    # if sample_idx < 6:  # or adjust for more or less samples
                    #     print(f"Sample {sample_idx} - Method: {method_name} - Pred {i}: {left_cols}-{right_cols} "
                    #           f"score={score} (is_match={is_match})")
                    #     print(f"True: {true_left_cols} <-> {true_right_cols}")
                    #     print("Predictions:")
                    #     for idx, (lc, rc, sc) in enumerate(predictions[:5]):
                    #         print(f"  {idx}: {lc} - {rc} (score: {sc})")

                    # Store for evaluation
                    all_sample_ids.append(sample_idx)
                    all_y_true.append(1 if is_match else 0)
                    all_y_pred.append(score)

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

        # Store metrics for this method
        metrics[method_name] = method_metrics

        # Printing results
        # print(f"  {method_name} results:")
        # for k in k_values:
        #     print(f"    precision@{k}: {method_metrics[f'precision@{k}']:.4f}")
        #     print(f"    ndcg@{k}: {method_metrics[f'ndcg@{k}']:.4f}")

    return metrics
