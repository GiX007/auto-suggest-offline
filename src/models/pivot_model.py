# src/models/pivot_model.py
#
# Implementation of the Pivot prediction model based on Section 4.3
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks".
#
# This module includes:
# 1. Processing and feature extraction for pivot samples (Section 4.3)
# 2. Affinity matrix construction for dimension columns
# 3. Solving the Affinity-Maximizing Pivot-Table (AMPT) problem
# 4. Heuristic and baseline model (groupby model) for pivot recommendations
# 5. Functions to train affinity-weight regression for improved affinity scores
# 6. Evaluation metrics and visualization of pivot prediction performance
# 7. Generation of sample pivot tables and example pivot code for data preparation tasks
#
# All logic related to pivot structure recommendation is contained here.
#

import os
import time
import json
import numpy as np
import pandas as pd
import networkx as nx
from tabulate import tabulate
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
from src.utils.model_utils import save_model, load_model
from src.baselines.pivot_baselines import evaluate_baselines
from src.models.groupby_model import predict_column_groupby_scores


# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
models_dir = os.path.join(base_dir, "models")

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


def process_pivot_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process Pivot samples to extract input tables and pivot parameters.

    This function:
    1. Validates input samples.
    2. Extracts index, header, values columns, and aggregation function from parameters.
    3. Filters out invalid or incomplete samples.

    Args:
        samples: List of raw samples loaded from the data directory.

    Returns:
        List of processed samples (dicts) containing input_table and pivot parameters.
    """
    # Inspect a sample (look how it looks like)
    # print("Input sample:")
    # for k, v in samples[0].items():
    #     print(k, v)

    processed_samples = []
    problematic_samples = 0

    #print(f"Starting to process {len(samples)} pivot samples...")

    for idx, sample in enumerate(samples):
        try:
            # Skip samples without necessary data
            if 'input_table' not in sample or 'params' not in sample:
                print(f"Sample {idx} missing 'input_table' or 'params'")
                problematic_samples += 1
                continue

            input_table = sample['input_table']

            # Remove noisy columns
            columns_to_keep = [col for col in input_table.columns if not (col.startswith('Unnamed:') )] # or col == 'index' or col == '__dummy__'
            input_table = input_table[columns_to_keep]

            params = sample['params']

            # Skip if table is empty
            if input_table.empty:
                print(f"Sample {idx} has empty input table")
                problematic_samples += 1
                continue

            # Extract index columns
            index_columns = []
            for key in ['index', 'rows']:
                if key in params:
                    val = params[key]
                    index_columns = val if isinstance(val, list) else [val]
                    break

            # Extract header columns
            header_columns = []
            for key in ['column', 'columns', 'cols', 'header']:
                if key in params:
                    val = params[key]
                    header_columns = val if isinstance(val, list) else [val]
                    break

            # Extract values columns
            values_columns = []
            for key in ['values', 'value']:
                if key in params:
                    val = params[key]
                    values_columns = val if isinstance(val, list) else [val]
                    break

            # Extract aggregation function
            aggfunc = params.get('aggfunc', params.get('agg', params.get('function', 'mean')))

            # Validate columns
            valid_columns = list(input_table.columns)
            valid_index_columns = [col for col in index_columns if col in valid_columns]
            valid_header_columns = [col for col in header_columns if col in valid_columns]
            valid_values_columns = [col for col in values_columns if col in valid_columns]

            # Remove known 'Unnamed' columns again: this is double safety net!
            noise_columns = {'Unnamed: 0', 'index'}
            valid_index_columns = [col for col in valid_index_columns if col not in noise_columns]
            valid_header_columns = [col for col in valid_header_columns if col not in noise_columns]
            valid_values_columns = [col for col in valid_values_columns if col not in noise_columns]

            # Add this valid sample to our processed samples
            if valid_index_columns and valid_header_columns:
                processed_samples.append({
                    'input_table': input_table,
                    'index_columns': valid_index_columns,
                    'header_columns': valid_header_columns,
                    'values_columns': valid_values_columns,
                    'aggfunc': aggfunc,
                    'sample_id': sample.get('sample_id', f"sample_{idx}")
                })
                # print(f"Sample {idx} processed successfully")
            else:
                #print(f"Sample {idx} has no valid index or header columns. Original index: {index_columns}, header: {header_columns}")
                problematic_samples += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            problematic_samples += 1

    # Inspect a processed sample (see how it looks like now, after processing)
    # print("Processed sample:")
    # for k, v in processed_samples[0].items():
    #     print(k, v)

    print(f"\nProcessed {len(processed_samples)} valid pivot samples")  # out of {len(samples)} total
    #print(f"Encountered {problematic_samples} problematic samples")

    return processed_samples


def calculate_column_affinity_features(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, float]:
    """
    Calculates the affinity features for a column pair based on the two key
    features from the paper.

    Features:
    - Emptiness Reduction Ratio (ERR): Measures how much empty space is saved
      when pivoting by this pair. Higher ERR means stronger affinity.
    - Relative Position Difference: Measures how far apart the columns are in the table.
      Columns closer together have higher affinity.

    Example of Emptiness Reduction Ratio:
    -------------------------------------
    Suppose:
      - col1: ['A', 'B', 'C']
      - col2: [1, 2, 1]
    There are 3 unique col1 values and 2 unique col2 values (3 * 2 = 6 possible pairs).
    If the actual distinct pairs present in the data is 3: (A,1), (B,2), (C,1), then:
      ERR = Theoretical Pairs / Actual Pairs = (3 * 2) / 3 = 2.0
    Higher ERR means there is more "empty space" avoided when combining these two columns in a pivot.
    It captures the potential sparsity of the full cross join vs the actual data, which reflects how well the columns
    compress the dataset when grouped together in a pivot table.

    Example of Relative Position Difference:
    ----------------------------------------
    If the table columns are:
      ['id', 'name', 'gender', 'age', 'city']
    For col1='name' (position 1) and col2='age' (position 3):
      position_difference = abs(1 - 3) = 2
      relative_position_difference = 2 / 5 = 0.4
    Closer columns (low relative position difference) have higher affinity.

    Args:
        df: Input DataFrame.
        col1: First column name.
        col2: Second column name.

    Returns:
        Dictionary with 'emptiness_reduction_ratio' and 'relative_position_difference'.
    """
    features = {}

    # Feature 1: Emptiness Reduction Ratio (ERR)
    try:
        unique_values1 = df[col1].nunique()
        unique_values2 = df[col2].nunique()
        unique_pairs = df[[col1, col2]].drop_duplicates().shape[0]

        # ERR = (theoretical number of pairs) / (actual number of distinct pairs)
        # Higher ERR means more "empty" combinations saved by grouping these together
        reduction_ratio = (unique_values1 * unique_values2) / unique_pairs if unique_pairs > 0 else 1.0
        features['emptiness_reduction_ratio'] = reduction_ratio
    except Exception as e:
        print(f"Error calculating emptiness reduction ratio for {col1} and {col2}: {e}")
        features['emptiness_reduction_ratio'] = 1.0

    # Feature 2: Relative Position Difference
    col1_pos = list(df.columns).index(col1)
    col2_pos = list(df.columns).index(col2)
    position_difference = abs(col1_pos - col2_pos)
    # Normalize by number of columns to get relative position (range [0,1])
    features['relative_position_difference'] = position_difference / len(df.columns) if len(df.columns) > 0 else 0.0

    return features


def train_affinity_weights(processed_samples, output_models_dir="models"):
    """
    Trains a linear regression model to learn weights for combining features for affinity matrix computation.

    Args:
        processed_samples: List of pivot samples with input_table, index_columns, and header_columns.
        output_models_dir: Directory to save the trained model (default: 'models').

    Returns:
        Tuple of (a, b, intercept) weights for final affinity calculation.
    """
    X = []
    y = []

    # Determine operator type based on the first sample
    if 'index_columns' in processed_samples[0] and 'header_columns' in processed_samples[0]:
        operator = 'pivot'
    elif 'id_vars' in processed_samples[0] and 'value_vars' in processed_samples[0]:
        operator = 'unpivot'
    else:
        raise ValueError("Unknown operator type in processed_samples. Expected pivot or unpivot samples.")

    for sample_idx, sample in enumerate(processed_samples):
        input_df = sample['input_table']

        # Determine which grouping keys exist in this sample.
        # Generalizes handling for both pivot samples (with 'index_columns' and 'header_columns')
        # and unpivot samples (with 'id_vars' and 'value_vars') so that the affinity model
        # can be trained on either type of sample seamlessly.
        if 'index_columns' in sample and 'header_columns' in sample:
            group_1 = set(sample['index_columns'])
            group_2 = set(sample['header_columns'])
        elif 'id_vars' in sample and 'value_vars' in sample:
            group_1 = set(sample['id_vars'])
            group_2 = set(sample['value_vars'])
        else:
            continue  # skip if neither found

        dimension_columns = list(group_1.union(group_2))  # unified!

        for i in range(len(dimension_columns)):
            for j in range(i + 1, len(dimension_columns)):
                col1, col2 = dimension_columns[i], dimension_columns[j]

                features = calculate_column_affinity_features(input_df, col1, col2)
                affinity_core = np.tanh(features['emptiness_reduction_ratio'] / 10.0)
                position_factor = 1.0 / (1.0 + features['relative_position_difference'])

                X.append([affinity_core, position_factor])

                same_group = ((col1 in group_1 and col2 in group_1) or
                              (col1 in group_2 and col2 in group_2))
                y.append(1 if same_group else 0)

    X = np.array(X)
    y = np.array(y)

    print("\nStarting affinity weights regression model training...")
    reg: LinearRegression = LinearRegression().fit(X, y)
    # print("Trained linear regression for affinity weights:")
    # print(f"a (affinity_core) = {reg.coef_[0]:.4f}")    # type: ignore
    # print(f"b (position_factor) = {reg.coef_[1]:.4f}")  # type: ignore
    # print(f"intercept = {reg.intercept_:.4f}")          # type: ignore

    # Save the model using your project’s save_model function
    os.makedirs(output_models_dir, exist_ok=True)
    model_path = os.path.join(output_models_dir, f"{operator}_affinity_weights_model.pkl")
    save_model(reg, model_path)
    print("\nAffinity weights regression model trained successfully!\n")

    # Return the trained weights for direct use
    # return reg.coef_[0], reg.coef_[1], reg.intercept_  # type: ignore


def solve_ampt(affinity_matrix: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Solves the Affinity-Maximizing Pivot-Table (AMPT) problem.

    This function implements the AMPT algorithm from Section 4.3 of the paper.
    AMPT aims to find the optimal split of dimension columns into index and header
    for a pivot table by maximizing the objective function:

    max Σ(c_i, c_j ∈ C) a(c_i, c_j) + Σ(c_i, c_j ∈ C̄) a(c_i, c_j) - Σ(c_i ∈ C, c_j ∈ C̄) a(c_i, c_j)

    where:
    - C and C̄ are the two partitions of columns
    - a(c_i, c_j) is the affinity between columns c_i and c_j

    The function uses the Stoer-Wagner min-cut algorithm to solve this problem
    optimally, with a fallback to a greedy approach if the algorithm fails.

    The first two terms reward placing related columns together in the same group
    (either index or header), and the third term penalizes separating strongly related
    columns across the split.

    Since the total pairwise affinity is constant regardless of partitioning, maximizing
    this full expression is equivalent to minimizing the third term alone:

        min Σ(c_i ∈ C, c_j ∈ C̄) a(c_i, c_j)

    This is exactly the formulation of a graph minimum cut problem. To solve it, we
    build a graph from the affinity matrix and negate all affinity scores to transform
    the objective into a min-cut problem. We then apply the Stoer-Wagner algorithm
    (which finds the minimum cut on a graph with non-negative weights) to identify
    the optimal column split.

    If the algorithm fails (e.g., due to negative weights), we fall back to a greedy
    partitioning strategy that approximates the split.

    Example:
        If we have columns ["Year", "Quarter", "Region", "Product"] with affinities
        showing "Year" and "Quarter" are related, and "Region" and "Product" are related:

        ```
        affinity_matrix = pd.DataFrame([
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0]
        ], index=["Year", "Quarter", "Region", "Product"],
           columns=["Year", "Quarter", "Region", "Product"])

        index_cols, header_cols = solve_ampt(affinity_matrix)
        # Returns: (["Year", "Quarter"], ["Region", "Product"])
        ```
        This would suggest creating a pivot table with Year and Quarter as row indices,
        and Region and Product as column headers.

    Args:
        affinity_matrix: Affinity matrix of columns.

    Returns:
        Tuple of (index_columns, header_columns) that maximizes the objective function.
    """
    # Convert the affinity matrix to a NetworkX graph
    G = nx.Graph()

    columns = list(affinity_matrix.columns)

    # Handle edge case with empty or singleton affinity matrix
    if len(columns) <= 1:
        # print("Warning: Not enough columns for meaningful pivot (need at least 2)")
        if len(columns) == 1:
            # Return the single column as index, with an empty header
            return columns, []
        else:
            # Return empty lists if no columns
            return [], []

    # Step 1: Build graph with nodes and edges (positive weights only)
    for i, col1 in enumerate(columns):
        G.add_node(col1)
        for j, col2 in enumerate(columns[i + 1:], i + 1):
            weight = affinity_matrix.loc[col1, col2]
            if weight > 0:
                G.add_edge(col1, col2, weight=weight)

    # Step 2: If graph has no edges, fallback to naive split (split the columns in half to avoid failure)
    if len(G.edges()) == 0:
        mid = len(columns) // 2
        return columns[:mid], columns[mid:]

    # Step 3: Check if all edge weights are strictly positive BEFORE negation
    all_weights_positive = all(d['weight'] > 0 for _, _, d in G.edges(data=True))

    if not all_weights_positive:
        # If some edge weights are non-positive, fall back to greedy (this means there are negative edges, so SW algo cannot be applied)
        # We are in case of having only the first 2 terms (Σ(c_i, c_j ∈ C) a(c_i, c_j) + Σ(c_i, c_j ∈ C̄) a(c_i, c_j)) and want to maximize it.
        return greedy_ampt_split(affinity_matrix)

    # Step 4: Negate weights to convert max-affinity to min-cut (Stoer-Wagner finds min-cut)
    for u, v in G.edges():
        G[u][v]['weight'] = -G[u][v]['weight']

    try:
        # Step 5: Solve min-cut using Stoer-Wagner on negative weights
        cut_value, partition = nx.stoer_wagner(G)   # returns: (5.0, (['A', 'B'], ['C', 'D']))
        #objective_value = -cut_value  # Invert cut value to interpret as max-affinity
        # print(f"DEBUG: Stoer-Wagner cut_value: {cut_value}")
        # print(f"DEBUG: Stoer-Wagner partition: {partition}")

        # Step 6: Handle degenerate partition (empty side)
        if not partition[0] or not partition[1]:
            mid = len(columns) // 2
            return columns[:mid], columns[mid:]

        return partition[0], partition[1]   # returns (index_pred_cols, header_pred_cols)

    except Exception as e:
        # Step 7: If Stoer-Wagner fails (e.g., due to disconnections), fallback to greedy
        # print(f"Warning: Stoer-Wagner min-cut failed with error: {e}")
        # print("Falling back to greedy AMPT split...")
        return greedy_ampt_split(affinity_matrix)


def greedy_ampt_split(affinity_matrix: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Greedy algorithm for solving the AMPT problem when Stoer-Wagner fails.

    This function uses a greedy approach to partition columns into two groups
    (index and header) for pivot tables. The algorithm:
    1. Starts with the pair of columns having the highest affinity
    2. Iteratively assigns remaining columns to the partition with higher affinity

    Example:
        Consider a simple affinity matrix for columns ["Year", "Quarter", "Region", "Product"]:

        ```
             Year  Quarter  Region  Product
        Year    1.0     0.9     0.1      0.2
        Quarter 0.9     1.0     0.2      0.1
        Region  0.1     0.2     1.0      0.8
        Product 0.2     0.1     0.8      1.0
        ```

        The algorithm would:
        1. Find the highest affinity pair: (Year, Quarter) with score 0.9
        2. Start with partition_1 = [Year], partition_2 = [Quarter]
        3. For "Region", calculate affinity to each partition:
           - Affinity to partition_1 (Year): 0.1
           - Affinity to partition_2 (Quarter): 0.2
           => Assign to partition_2
        4. For "Product", calculate affinity to each partition:
           - Affinity to partition_1 (Year): 0.2
           - Affinity to partition_2 (Quarter, Region): (0.1 + 0.8)/2 = 0.45
           => Assign to partition_2
        5. Final result: partition_1 = [Year], partition_2 = [Quarter, Region, Product]

    Args:
        affinity_matrix: Affinity matrix of columns

    Returns:
        Tuple of (index_columns, header_columns) representing the partitioning of columns
    """
    columns = list(affinity_matrix.columns)

    # If only one column, return it as index
    if len(columns) <= 1:
        return columns, []

    # Start with the two nodes having the highest affinity
    best_affinity = -float('inf')
    best_pair = None

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i + 1:], i + 1):
            affinity = affinity_matrix.loc[col1, col2]
            if affinity > best_affinity:
                best_affinity = affinity
                best_pair = (col1, col2)

    if best_pair:
        # Start with the best pair
        partition_1 = [best_pair[0]]
        partition_2 = [best_pair[1]]

        # Assign remaining columns to the partition with higher affinity
        remaining_cols = [col for col in columns if col not in (best_pair[0], best_pair[1])]
        for col in remaining_cols:
            # Calculate affinity to each partition
            total_affinity_to_p1 = sum(affinity_matrix.loc[col, p] for p in partition_1)
            total_affinity_to_p2 = sum(affinity_matrix.loc[col, p] for p in partition_2)

            # Debug: Show decision process
            #print(f"Assigning column '{col}' to {'partition_1' if total_affinity_to_p1 >= total_affinity_to_p2 else 'partition_2'}")

            # Assign to partition with higher affinity
            if total_affinity_to_p1 >= total_affinity_to_p2:
                partition_1.append(col)
            else:
                partition_2.append(col)

        return partition_1, partition_2
    else:
        # If no good pairs found, use simple split
        mid = len(columns) // 2
        return columns[:mid], columns[mid:]


def build_affinity_matrix(input_df: pd.DataFrame, dimension_columns: List[str], model) -> pd.DataFrame:
    """
    Builds an affinity matrix for the given dimension columns.

    The affinity matrix captures how likely columns are to be on the same side
    in a pivot table. High values indicate columns should be kept together.

    Args:
        input_df: Input DataFrame.
        dimension_columns: List of dimension columns to consider.
        model: The trained LinearRegression model (for affinity weights).

    Returns:
        Affinity matrix as a pandas DataFrame.
    """
    # These weights (a, b, intercept) calibrate the affinity matrix scoring
    a = model.coef_[0]
    b = model.coef_[1]
    intercept = model.intercept_  # 'intercept' (bias term) shifts the final affinity score up or down to calibrate baseline affinity

    n = len(dimension_columns)

    # Initialize matrix with identity (diagonal = 1.0)
    affinity_matrix = pd.DataFrame(np.eye(n), index=dimension_columns, columns=dimension_columns) # (e.g. for a 5-dimension column table (not measures!), we build a 5x5 affinity matrix)

    for i in range(n):
        for j in range(i + 1, n):
            col1, col2 = dimension_columns[i], dimension_columns[j]

            # Calculate features for this column pair
            features = calculate_column_affinity_features(input_df, col1, col2)

            # Compute affinity score (Heuristic based on the paper: higher ERR and close positions → higher affinity)
            # Higher emptiness reduction ratio means higher affinity
            affinity_score = np.tanh(features['emptiness_reduction_ratio'] / 10.0)

            # Adjust based on position difference (columns close to each other have higher affinity)
            # Use tanh to normalize high ERR values to [0,1] and dampen outliers — ensures affinity stays bounded
            position_factor = 1.0 / (1.0 + features['relative_position_difference'])

            # Final Affinity score
            #affinity_score = a * affinity_score + b * position_factor + intercept  # Based on trained regression model
            affinity_score = affinity_score * position_factor   # Based on a simple multiplication

            # Boost affinity if name similarity is high (extra custom boost!)
            # if features['name_similarity'] > 0:
            #     affinity_score = max(affinity_score, features['name_similarity'])

            # Set the affinity score (symmetric matrix)
            affinity_matrix.loc[col1, col2] = affinity_score
            affinity_matrix.loc[col2, col1] = affinity_score

    return affinity_matrix


def generate_pivot_prediction_table(auto_suggest_metrics, baseline_metrics, operator_name="pivot"):
    """
    Generates Table 8-style baseline results (full accuracy and rand index).

    Args:
        auto_suggest_metrics: Dictionary of metrics from Auto-Suggest.
        baseline_metrics: Dictionary of metrics from baseline methods.
        operator_name: e.g., "pivot".

    Prints:
        A formatted table with baseline comparisons, including Auto-Suggest.
        Also saves it as a CSV to 'results/metrics/{operator_name}_methods_comparison.csv'.
    """
    # All methods for comparison
    methods = ["Auto-Suggest"] + list(baseline_metrics.keys())

    # Create rows
    rows = []
    for method in methods:
        row = [method]

        # Full accuracy
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('full_accuracy', 0):.2f}")
        else:
            row.append(f"{baseline_metrics[method].get('full_accuracy', 0):.2f}")

        # Rand Index
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('rand_index', 0):.2f}")
        else:
            row.append(f"{baseline_metrics[method].get('rand_index', 0):.2f}")

        rows.append(row)

    # Headers
    headers = ["method", "full_accuracy", "rand_index"]

    # Print the table
    print(f"\nTable 8: {operator_name.capitalize()} Prediction - Baseline Results")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save the table to CSV
    os.makedirs("results/metrics", exist_ok=True)
    df = pd.DataFrame(rows, columns=headers)
    output_file = os.path.join("results/metrics", f"{operator_name}_methods_comparison.csv")
    df.to_csv(output_file, index=False)



def evaluate_pivot(test_samples: List[Dict]) -> Dict[str, float]:
    """
    Evaluates the quality of pivot column split predictions.

    Args:
        test_samples: List of test pivot samples.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Check if affinity weight model exists before proceeding
    affinity_model_path = os.path.join("models", "pivot_affinity_weights_model.pkl")
    if not os.path.exists(affinity_model_path):
        print("\nError: Required affinity weight model not found!")
        print("Please run pivot training first (it builds the affinity weight model used by unpivot).\n")
        return False

    correct_splits = 0
    total_splits = 0
    rand_index_scores = []

    # Load the previously trained linear regression model (affinity weights)
    reg_model = load_model(os.path.join("models", "pivot_affinity_weights_model.pkl"))

    # Start timer for all pivot evaluations
    start_time = time.time()

    for sample in test_samples:
        input_df = sample['input_table']
        true_index = set(sample['index_columns'])
        true_header = set(sample['header_columns'])

        # Get all dimension columns (index + header)
        # In the pivot context, dimension columns are those involved in the pivot structure,
        # typically categorical or identifier columns, and we consider all columns mentioned
        # in the ground truth (index + header) to build the affinity matrix.
        dimension_columns = list(true_index.union(true_header))

        # Skip if we have fewer than 2 dimension columns
        if len(dimension_columns) < 2:
            continue

        # Build affinity matrix
        affinity_matrix = build_affinity_matrix(input_df, dimension_columns, reg_model)

        # Solve AMPT
        pred_index, pred_header = solve_ampt(affinity_matrix)
        pred_index, pred_header = set(pred_index), set(pred_header)

        # Check if prediction matches ground truth
        # Note: The prediction could be flipped (index <-> header) and still be correct
        is_correct = (pred_index == true_index and pred_header == true_header) or \
                     (pred_index == true_header and pred_header == true_index)

        if is_correct:
            correct_splits += 1

        # Calculate Rand Index
        # - Measures how well the predicted split (index vs header) matches the true split
        # - Value 1.0 = perfect agreement, 0.0 = complete disagreement

        # Convert sets to binary matrices for all pairs of columns
        n = len(dimension_columns)
        true_same_side = np.zeros((n, n))
        pred_same_side = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                col1, col2 = dimension_columns[i], dimension_columns[j]

                true_same_side[i, j] = (col1 in true_index and col2 in true_index) or \
                                       (col1 in true_header and col2 in true_header)

                pred_same_side[i, j] = (col1 in pred_index and col2 in pred_index) or \
                                       (col1 in pred_header and col2 in pred_header)

        # Calculate Rand Index components
        a = np.sum((true_same_side == 1) & (pred_same_side == 1))  # Same side in both
        b = np.sum((true_same_side == 0) & (pred_same_side == 0))  # Different sides in both
        c = np.sum((true_same_side == 1) & (pred_same_side == 0))  # Same in true, diff in pred
        d = np.sum((true_same_side == 0) & (pred_same_side == 1))  # Diff in true, same in pred

        rand_index = (a + b) / (a + b + c + d) if (a + b + c + d) > 0 else 1.0
        rand_index_scores.append(rand_index)

        total_splits += 1

    # End timer for all samples
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal AMPT evaluation time for all pivot samples: {total_time:.2f} seconds")

    if total_splits == 0:
        print("No valid test samples for pivot evaluation")
        return { 'full_accuracy': 0.0, 'rand_index': 0.0 }

    # Final metrics
    metrics = {
        'full_accuracy': correct_splits / total_splits if total_splits > 0 else 0.0,
        'rand_index': np.mean(rand_index_scores) if rand_index_scores else 0.0
    }

    # Create full evaluation record
    eval_dict = {
        "operator": "pivot",
        "mode": "evaluation",
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': int(len(test_samples)),
        'correct_splits': int(correct_splits),
        "test_accuracy": float(metrics['full_accuracy']),
        "rand_index": float(metrics['rand_index']),
        'algorithm': 'AMPT',
        'solver': 'Stoer-Wagner'
    }

    # Converts numpy types to native Python types (important for JSON!)
    eval_dict = numpy_to_list(eval_dict)

    # Save to JSON
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

    # Calculate the metrics based on other heuristic methods
    baseline_metrics = evaluate_baselines(test_samples)

    # Generate Table 8 from the paper
    generate_pivot_prediction_table(
        auto_suggest_metrics=metrics,  # The metrics computed for Auto-Suggest
        baseline_metrics=baseline_metrics,  # Baseline metrics to compare with
    )

    return metrics


def plot_affinity_graph(affinity_matrix, index_columns, header_columns, save_path='results/figures/pivot_affinity_graph.png', show=False, title="Affinity Graph (Figure 10)"):
    """
    Plots and saves an affinity graph with clear left-right structure and full edge weights.

    Nodes: dimension columns.
    Edges: affinity scores (with weight labels).
    Red dashed line: cut between index and header columns.

    Args:
        affinity_matrix: pandas DataFrame of affinities.
        index_columns: List of index columns (left side).
        header_columns: List of header columns (right side).
        save_path: Path to save the figure (default: 'results/pivot_affinity_graph.png').
        show: If True, display the figure. Otherwise, just save it.
        title: Title for the plot.
    """
    G = nx.Graph()

    # Add nodes with colors
    for col in affinity_matrix.columns:
        color = 'lightblue' if col in index_columns else 'lightgreen'
        G.add_node(col, color=color)

    # Add edges (all edges above threshold for better structure)
    for i, col1 in enumerate(affinity_matrix.columns):
        for j, col2 in enumerate(affinity_matrix.columns):
            if i < j:
                weight = affinity_matrix.loc[col1, col2]
                if weight > 0.0:  # Show all edges with weight > 0
                    G.add_edge(col1, col2, weight=weight)

    # Explicit bipartite layout: index on left, header on right
    pos = {}
    vertical_spacing = 1.5
    for i, col in enumerate(index_columns):
        pos[col] = (-1, -i * vertical_spacing)

    for i, col in enumerate(header_columns):
        pos[col] = (1, -i * vertical_spacing)

    plt.figure(figsize=(10, 8))

    # Node colors and border
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, edgecolors='black', linewidths=2)

    # Draw edges as dashed lines with visible width
    edges = G.edges(data=True)
    weights = [d['weight'] * 2 for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8, style='dashed', edge_color='gray')

    # Draw edge labels for all edges
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Draw red cut line at x=0
    plt.axvline(0, color='red', linestyle='--', linewidth=2)

    # Title and formatting
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Index Columns'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Header Columns')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Affinity graph saved to 'results' directory\n")

    if show:
        plt.show()
    else:
        plt.close()


def predict_pivot_split(table, aggfunc='mean', model_path='models/pivot_affinity_weights_model.pkl'):
    """
    Generates pivot index and header recommendations for a new input table
    and displays the results in the console.

    This function:
    - Heuristically identifies dimension and measure columns.
    - Optionally (commented out) can use a trained GroupBy predictor for dimension columns.
    - Loads the trained affinity weights model to build an affinity matrix.
    - Solves the AMPT problem to determine index and header columns for the pivot.
    - Builds a sample pivot table using the entire input table.
    - Displays:
        - The recommended index and header columns.
        - A sample preview of the pivot table (first 3 rows).
        - Example pandas code to create the same pivot table.

    Args:
        table: pandas DataFrame of the input table.
        aggfunc: Aggregation function to use in pivot table.
        model_path: Path to the saved affinity weight model.

    Returns:
        None. Prints the recommended pivot split and a sample pandas code.
    """
    # Heuristically identify dimension and measure columns
    # dimension_columns = []
    # measure_columns = []
    #
    # for col in table.columns:
    #     if pd.api.types.is_numeric_dtype(table[col]) and table[col].nunique() > 15:
    #         measure_columns.append(col)
    #     else:
    #         dimension_columns.append(col)

    # Use trained groupby predictor to identify dimension columns
    groupby_model, feature_names = load_model('models/groupby_column_model.pkl')
    groupby_likelihoods = predict_column_groupby_scores(groupby_model, feature_names, table)
    # print(type(groupby_likelihoods))
    # print("nGroupby_likelihoods:")
    # for entry in groupby_likelihoods:
    #     print(entry)    #, type(entry)

    dimension_columns = [entry[0] for entry in groupby_likelihoods if float(entry[1]) > 0.5]
    measure_columns = [col for col in table.columns if col not in dimension_columns]

    if len(dimension_columns) < 2:
        print("Error: Need at least 2 dimension columns for pivot structure.")
        return

    # Load affinity weights model
    model = load_model(model_path)

    # Build affinity matrix
    affinity_matrix = build_affinity_matrix(table, dimension_columns, model)

    # Solve AMPT to get recommended split
    index_columns, header_columns = solve_ampt(affinity_matrix)

    print("\n=== Pivot Structure Recommendation ===")
    print(f"\nIndex columns: {index_columns}")
    print(f"Header columns: {header_columns}")

    # Pick a sample numeric column as value column
    value_column = None
    for col in measure_columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            value_column = col
            break

    if not value_column:
        print("\nNo obvious numeric value column found. Please specify manually for real pivot usage.")
        return

    # Show sample(1) pivot table
    pivot_sample = pd.pivot_table(
        table,  # or use some f the rows table.head(10)
        index=index_columns,
        columns=header_columns,
        values=value_column,
        aggfunc=aggfunc
    )

    print("\nSample pivot table preview (first 3rows):")
    print(pivot_sample.head(3))
    #print(pivot_sample.iloc[:3, :10])   # first 3 rows and 10 columns

    # Provide example pandas code
    print("\n=== Example Pandas Code ===")
    index_str = ', '.join([f"'{col}'" for col in index_columns])
    header_str = ', '.join([f"'{col}'" for col in header_columns])

    print("# Using pandas to create a pivot table:")
    print("pivot_table = pd.pivot_table(")
    print("    df,")
    print(f"    index=[{index_str}],")
    print(f"    columns=[{header_str}],")
    print(f"    values='{value_column}',")
    print(f"    aggfunc='{aggfunc}'")
    print(")")
    print()

    # Plot and save affinity graph with cut line
    plot_affinity_graph(affinity_matrix, index_columns, header_columns)
