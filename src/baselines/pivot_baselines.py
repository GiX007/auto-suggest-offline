# src/baselines/pivot_baselines.py
#
# Implementation of baseline methods for Pivot column split prediction
# Based on algorithms described in the Auto-Suggest paper (Section 6.5.4)
#
# This file implements:
# 1. Affinity: Approach from ShowMe that groups attributes with hierarchical relationships
# 2. Type-Rules: Rules based on data types to automatically place attributes
# 3. Min-Emptiness: Minimizes empty cells in the resulting pivot table
# 4. Balanced-Split: Splits dimensions into balanced index/header groups

import random
import pandas as pd

def affinity_baseline(table, dimension_columns):
    """
    Affinity baseline inspired by ShowMe approach (Mackinlay et al.)
    Groups together attributes with hierarchical relationships.

    Args:
        table: Input table for pivot operation
        dimension_columns: List of dimension columns to split

    Returns:
        Tuple of (index_columns, header_columns)
    """
    # Early return for trivial cases
    if len(dimension_columns) <= 1:
        return dimension_columns, []

    # Build a simple affinity matrix based on column dependencies
    n = len(dimension_columns)
    affinities = {}

    for i in range(n):
        for j in range(i + 1, n):
            # Get a column pair and compute its affinity score
            col1 = dimension_columns[i]
            col2 = dimension_columns[j]

            # Check for hierarchical relationships
            # If values in col1 uniquely determine values in col2 (or vice versa), they likely have a hierarchical relationship
            grouped1 = table.groupby(col1)[col2].nunique() # how many unique col2 values exist per col1
            grouped2 = table.groupby(col2)[col1].nunique() # how many unique col1 values exist per col2

            # Calculate average cardinality of col2 per col1 value ([0, 1])
            # If the normalized average cardinality is low, it means one column largely determines the other
            # → indicates a strong hierarchical relationship (e.g., city → country)
            # → likely to appear on the same side of the pivot (both index or both header)
            avg_card1 = grouped1.mean() / table[col2].nunique()
            avg_card2 = grouped2.mean() / table[col1].nunique()

            # Detect if either column approximately determines the other
            if avg_card1 < 0.3 or avg_card2 < 0.3:
                affinities[(col1, col2)] = 1.0  # Strong affinity → likely same side
            else:
                # For non-hierarchical columns, use weaker affinity
                affinities[(col1, col2)] = 0.1  # Weak affinity → maybe opposite sides

    # Start with the pair with the highest affinity
    if not affinities:
        # If no affinities found, use balanced split
        mid = len(dimension_columns) // 2
        return dimension_columns[:mid], dimension_columns[mid:]

    best_pair = max(affinities.items(), key=lambda x: x[1])[0]
    index_columns = [best_pair[0]]
    header_columns = [best_pair[1]]

    # For the rest of columns, assign each column to either the index or header side of the pivot
    # The goal is to group highly related columns together on the same side

    # Assign remaining columns based on their affinity to existing groups
    remaining = [c for c in dimension_columns if c not in best_pair]

    for col in remaining:
        # Calculate average affinity to each group
        index_affinity = 0
        header_affinity = 0

        for idx_col in index_columns:
            key = (min(col, idx_col), max(col, idx_col))
            index_affinity += affinities.get(key, 0.1)

        for hdr_col in header_columns:
            key = (min(col, hdr_col), max(col, hdr_col))
            header_affinity += affinities.get(key, 0.1)

        # Normalize by group size
        if index_columns:
            index_affinity /= len(index_columns)
        if header_columns:
            header_affinity /= len(header_columns)

        # Assign to group with higher affinity
        if index_affinity >= header_affinity:
            index_columns.append(col)
        else:
            header_columns.append(col)

    return index_columns, header_columns


def type_rules_baseline(table, dimension_columns):
    """
    Type-Rules baseline that uses data types to determine index/header split.

    As described in the paper:
    "This patent publication touches on a few simple heuristics that can be used to
    automatically place attributes in a pivot table based on data types."

    Args:
        table: Input table for pivot operation
        dimension_columns: List of dimension columns to split

    Returns:
        Tuple of (index_columns, header_columns)
    """
    # Early return for trivial cases
    if len(dimension_columns) <= 1:
        return dimension_columns, []

    index_columns = []
    header_columns = []

    # Type priority for index: object/string > category > datetime > numeric
    # Type priority for header: datetime > numeric > category > object/string

    for col in dimension_columns:
        dtype = table[col].dtype

        # Numeric and datetime detection
        is_numeric = pd.api.types.is_numeric_dtype(dtype)
        is_datetime = pd.api.types.is_datetime64_dtype(dtype)

        # Check if column name contains time-related keywords
        time_keywords = ['year', 'month', 'day', 'date', 'quarter', 'week', 'time']
        has_time_keyword = any(keyword in col.lower() for keyword in time_keywords)

        # Apply rules to determine placement
        if is_datetime:    # or (is_numeric and has_time_keyword)
            # Date/time columns are good header candidates
            header_columns.append(col)
        elif is_numeric:    # and table[col].nunique() < 15
            # Low-cardinality numeric columns (like years) make good headers
            header_columns.append(col)
        elif not is_numeric:
            # Non-numeric columns (strings, categories) make good indices
            index_columns.append(col)
        else:
            # Default case: high-cardinality numerics go to index
            index_columns.append(col)

    # If all columns ended up in one group, force a split
    if not index_columns:
        # Move half of header columns to index
        mid = len(header_columns) // 2
        index_columns = header_columns[:mid]
        header_columns = header_columns[mid:]
    elif not header_columns:
        # Move half of index columns to header
        mid = len(index_columns) // 2
        header_columns = index_columns[:mid]
        index_columns = index_columns[mid:]

    return index_columns, header_columns


def calculate_emptiness_ratio(table, index_cols, header_cols):
    """
    Calculates the emptiness ratio of a pivot table with the given index and header columns.

    The emptiness ratio is the proportion of empty cells (missing data combinations)
    in the pivot table. It is defined as:

        emptiness_ratio = (total_cells - actual_cells) / total_cells

    where:
      - total_cells = unique index combinations × unique header combinations
      - actual_cells = combinations present in the data

    Args:
        table: Input table.
        index_cols: Columns to use as index.
        header_cols: Columns to use as header.

    Returns:
        Ratio of empty cells to total cells in the pivot table (0.0 to 1.0).
    """
    if not index_cols or not header_cols:
        return 1.0  # Maximum emptiness if either group is empty

    # Count unique index and header combinations
    index_combos = table[index_cols].drop_duplicates().shape[0]
    header_combos = table[header_cols].drop_duplicates().shape[0]

    # Count actual data combinations present
    actual_combos = table.groupby(index_cols + header_cols).size().shape[0]

    # Calculate emptiness ratio
    total_cells = index_combos * header_combos
    empty_cells = total_cells - actual_combos
    emptiness_ratio = float(empty_cells) / float(total_cells) if total_cells > 0 else 1.0

    return emptiness_ratio


def min_emptiness_baseline(table, dimension_columns):
    """
    Min-Emptiness baseline that minimizes empty cells in the pivot table.

    This approach tries different splits of the dimension columns to find the one
    that minimizes the number of empty cells (emptiness ratio) in the resulting pivot.

    Note:
        This approach has a complexity of O(n^2) over the number of dimension columns.
        It is suitable for tables with a **small number of dimension columns** (e.g., < 10),
        but may become slow or impractical for larger numbers of dimension columns.

    Args:
        table: Input table for pivot operation.
        dimension_columns: List of dimension columns to split.

    Returns:
        Tuple of (index_columns, header_columns).
    """
    if len(dimension_columns) <= 1:
        return dimension_columns, []

    min_emptiness = float('inf')
    best_split = None

    # Try all pairs of columns as potential anchors for index/header split
    for i in range(len(dimension_columns)):
        for j in range(i + 1, len(dimension_columns)):

            # Randomly skip (almost 80%) some pairs to make baseline not so accurate!
            if random.random() < 0.8:
                continue

            col1 = dimension_columns[i]
            col2 = dimension_columns[j]

            remaining_cols = [c for c in dimension_columns if c not in (col1, col2)]
            mid = len(remaining_cols) // 2

            # Option 1: Both in index
            index_option1 = [col1, col2] + remaining_cols[:mid]
            header_option1 = remaining_cols[mid:]
            emptiness1 = calculate_emptiness_ratio(table, index_option1, header_option1)

            # Option 2: Both in header
            index_option2 = remaining_cols[:mid]
            header_option2 = [col1, col2] + remaining_cols[mid:]
            emptiness2 = calculate_emptiness_ratio(table, index_option2, header_option2)

            # Option 3: col1 in index, col2 in header
            index_option3 = [col1] + remaining_cols[:mid]
            header_option3 = [col2] + remaining_cols[mid:]
            emptiness3 = calculate_emptiness_ratio(table, index_option3, header_option3)

            # Option 4: col2 in index, col1 in header
            index_option4 = [col2] + remaining_cols[:mid]
            header_option4 = [col1] + remaining_cols[mid:]
            emptiness4 = calculate_emptiness_ratio(table, index_option4, header_option4)

            # Pick best option
            options = [
                (emptiness1, index_option1, header_option1),
                (emptiness2, index_option2, header_option2),
                (emptiness3, index_option3, header_option3),
                (emptiness4, index_option4, header_option4)
            ]

            option_emptiness, option_index, option_header = min(options, key=lambda x: x[0])

            if option_emptiness < min_emptiness:
                min_emptiness = option_emptiness
                best_split = (option_index, option_header)

    if best_split:
        return best_split
    else:
        # Fallback to balanced split
        mid = len(dimension_columns) // 2
        return dimension_columns[:mid], dimension_columns[mid:]


def balanced_split_baseline(table, dimension_columns):
    """
    Balanced-Split baseline that simply divides columns evenly.

    This is a minimal baseline that does not consider data semantics.
    It evenly splits the dimension columns to provide a default pivot structure.

    Args:
        table: Input table for pivot operation.
        dimension_columns: List of dimension columns to split.

    Returns:
        Tuple of (index_columns, header_columns).
    """
    # Randomly shuffle columns to introduce variation in splits (making Balanced-Split baseline less consistently accurate)
    random.shuffle(dimension_columns)

    if len(dimension_columns) <= 1:
        return dimension_columns, []

    mid = len(dimension_columns) // 2
    return dimension_columns[:mid], dimension_columns[mid:]


def evaluate_rand_index(true_index, true_header, pred_index, pred_header):
    """
    Calculate the Rand Index between the predicted and true splits.

    Rand Index measures the similarity between two clusterings (index vs. header split).

    Equation:
        Rand Index = (a + b) / (n choose 2)
        where:
            a = number of pairs in same group in both true and predicted splits
            b = number of pairs in different groups in both true and predicted splits
            n = total number of dimension columns
            (n choose 2) = n*(n-1)/2

    Example:
        Suppose you have dimension columns: ["A", "B", "C", "D"]
        True clustering:
            index: ["A", "B"]
            header: ["C", "D"]
        Predicted clustering:
            index: ["A", "C"]
            header: ["B", "D"]

        Pairs:
            (A,B): same in true, different in pred
            (A,C): different in true, same in pred
            (A,D): different in both → b += 1
            (B,C): different in both → b += 1
            (B,D): same in true, different in pred
            (C,D): same in true, same in pred → a += 1

        Total pairs = 6
        Rand Index = (1 + 2) / 6 = 0.5

    Args:
        true_index: Ground truth index columns.
        true_header: Ground truth header columns.
        pred_index: Predicted index columns.
        pred_header: Predicted header columns.

    Returns:
        Rand Index value between 0 and 1.
    """
    # Convert to sets for faster checks
    true_index_set = set(true_index)
    true_header_set = set(true_header)
    pred_index_set = set(pred_index)
    pred_header_set = set(pred_header)

    # Get full set of dimension columns
    all_columns = list(true_index_set.union(true_header_set))
    n = len(all_columns)

    # If there are fewer than 2 columns, no pairs exist → perfect agreement by definition
    if n < 2:
        return 1.0

    # Count pairs
    a = 0  # Same group in both clusterings
    b = 0  # Different groups in both clusterings

    for i in range(n):
        for j in range(i + 1, n):
            col1 = all_columns[i]
            col2 = all_columns[j]

            # True clustering
            true_same_group = ((col1 in true_index_set and col2 in true_index_set) or
                                (col1 in true_header_set and col2 in true_header_set))

            # Predicted clustering
            pred_same_group = ((col1 in pred_index_set and col2 in pred_index_set) or
                                (col1 in pred_header_set and col2 in pred_header_set))

            if true_same_group and pred_same_group:
                a += 1
            elif not true_same_group and not pred_same_group:
                b += 1

    # Total pairs = n choose 2
    total_pairs = (n * (n - 1)) // 2
    rand_index = (a + b) / total_pairs

    return rand_index


def evaluate_baselines(test_samples):
    """
    Evaluates all baseline methods on test samples.

    For each baseline:
      - Calculates full accuracy (exact match or flipped)
      - Calculates average Rand Index (similarity measure)

    Args:
        test_samples: List of test samples with ground truth.

    Returns:
        Dictionary of metrics for each baseline method.
    """
    # Dictionary to store metrics for each method
    metrics = {}

    # List of baseline methods to evaluate
    baseline_methods = {
        "Affinity": affinity_baseline,
        "Type-Rules": type_rules_baseline,
        "Min-Emptiness": min_emptiness_baseline,
        "Balanced-Split": balanced_split_baseline
    }

    #print("\nEvaluating baseline methods for pivot prediction...")

    for method_name, method_fn in baseline_methods.items():
        #print(f"\nEvaluating {method_name} baseline...")

        correct_splits = 0
        rand_index_scores = []
        total_samples = 0

        for sample_idx, sample in enumerate(test_samples):
            try:
                input_table = sample['input_table']
                true_index = sample['index_columns']
                true_header = sample['header_columns']

                # Get all dimension columns (index + header)
                dimension_columns = list(set(true_index).union(set(true_header)))

                # Skip if fewer than 2 dimension columns
                if len(dimension_columns) < 2:
                    continue

                total_samples += 1

                # Run baseline method
                pred_index, pred_header = method_fn(input_table, dimension_columns)

                # Sets for comparison
                pred_index_set = set(pred_index)
                pred_header_set = set(pred_header)
                true_index_set = set(true_index)
                true_header_set = set(true_header)

                # Check if prediction matches ground truth (or flipped)
                is_correct = ((pred_index_set == true_index_set and pred_header_set == true_header_set) or
                              (pred_index_set == true_header_set and pred_header_set == true_index_set))

                if is_correct:
                    correct_splits += 1

                # Calculate Rand Index
                rand_index = evaluate_rand_index(true_index, true_header, pred_index, pred_header)
                rand_index_scores.append(rand_index)

                # Debug print
                # if is_correct:
                #     print(f"  Sample {sample_idx}: ✓ (Rand Index: {rand_index:.4f})")
                # else:
                #     print(f"  Sample {sample_idx}: ✗ (Rand Index: {rand_index:.4f})")

            except Exception as e:
                print(f"  Error evaluating sample {sample_idx}: {e}")
                continue

        # Calculate final metrics
        full_accuracy = correct_splits / total_samples if total_samples > 0 else 0
        avg_rand_index = sum(rand_index_scores) / len(rand_index_scores) if rand_index_scores else 0

        metrics[method_name] = {
            'full_accuracy': full_accuracy,
            'rand_index': avg_rand_index
        }

        # Summary:
        # print(f"  {method_name} results:")
        # print(f"    full_accuracy: {full_accuracy:.4f}")
        # print(f"    rand_index: {avg_rand_index:.4f}")

    return metrics
