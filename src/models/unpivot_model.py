# src/models/unpivot_model.py
#
# Implementation of the Unpivot prediction pipeline, based on Section 4.4
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks".
#
# This module includes:
# 1. CMUT algorithm implementation for unpivot (solve_cmut and calculate_cmut_objective).
# 2. Functions to process unpivot samples and build affinity matrices.
# 3. Functions to evaluate unpivot baselines and generate Table 9-style results.
# 4. A predict_unpivot function that combines groupby-based dimension identification
#    and affinity-based unpivot column grouping.
# 5. Helper functions for numeric conversion and visualization.
#
# This structure supports evaluating Auto-Suggest and baseline performance for unpivot
# and making predictions on new tables.
#

import os
import json
import time
import tabulate
import numpy as np
import pandas as pd
import networkx as nx
from tabulate import tabulate
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.utils.model_utils import load_model, numpy_to_list
from src.models.pivot_model import build_affinity_matrix, plot_affinity_graph
from src.baselines.unpivot_baselines import evaluate_baselines
# from src.models.groupby_model import predict_column_groupby_scores


# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
models_dir = os.path.join(base_dir, "models")


def process_unpivot_samples(samples: List[Dict]) -> List[Dict]:
    """
    Process Unpivot (melt) samples to extract input tables and unpivot parameters.

    This function:
    1. Validates input samples.
    2. Extracts id_vars and value_vars from parameters.
    3. Filters out invalid or incomplete samples.

    Args:
        samples: List of raw samples loaded from the data directory.

    Returns:
        List of processed samples (dicts) containing input_table, id_vars, value_vars, and sample_id.
    """
    # Inspect a sample (look how it looks like)
    # print("Input sample:")
    # for k, v in samples[0].items():
    #     print(k, v)

    processed_samples = []
    problematic_samples = 0

    #print(f"Starting to process {len(samples)} unpivot samples...")

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

            # Extract id_vars
            id_vars = []
            for key in ['id_vars', 'id']:
                if key in params:
                    val = params[key]
                    id_vars = val if isinstance(val, list) else [val]
                    break

            # Extract value_vars
            value_vars = []
            for key in ['value_vars', 'values']:
                if key in params:
                    val = params[key]
                    value_vars = val if isinstance(val, list) else [val]
                    break

            # Validate columns
            valid_columns = list(input_table.columns)
            valid_id_vars = [col for col in id_vars if col in valid_columns]
            valid_value_vars = [col for col in value_vars if col in valid_columns]

            # Remove known 'Unnamed' columns again: this is double safety net!
            noise_columns = {'Unnamed: 0', 'index'}
            valid_id_vars = [col for col in valid_id_vars if col not in noise_columns]
            valid_value_vars = [col for col in valid_value_vars if col not in noise_columns]

            # Add this valid sample to our processed samples
            if valid_id_vars and valid_value_vars:
                processed_samples.append({
                    'input_table': input_table,
                    'id_vars': valid_id_vars,
                    'value_vars': valid_value_vars,
                    'sample_id': sample.get('sample_id', f"sample_{idx}")
                })
                # print(f"Sample {idx} processed successfully")
            else:
                #print(f"Sample {idx} has no valid id_vars or value_vars. Original id_vars: {id_vars}, value_vars: {value_vars}")
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

    print(f"\nProcessed {len(processed_samples)} valid unpivot samples")  # out of {len(samples)} total
    #print(f"Encountered {problematic_samples} problematic samples")

    return processed_samples


def calculate_cmut_objective(compatibility_matrix: pd.DataFrame, unpivot_set: List[str]) -> float:
    """
    Calculates the CMUT objective function value for a given set of columns to unpivot.

    From Section 4.4 of the paper, the objective function is:

    max avg(c_i, c_j ∈ C) a(c_i, c_j) - avg(c_i ∈ C, c_j ∈ C\\C) a(c_i, c_j)

    Where:
    - The first term is the average compatibility between all pairs of columns in the unpivot set (Intra group)
    - The second term is the average compatibility between columns in the unpivot set and columns not in the set (Inter group)
    - We want to maximize internal compatibility while minimizing external compatibility

    Example:
        Consider a table with columns ["ID", "Name", "2019", "2020", "2021"] and
        we're evaluating unpivot_set=["2019", "2020", "2021"]:

        ```
        # For the unpivot set ["2019", "2020", "2021"]:
        # Intra-group compatibility: avg of (2019,2020), (2019,2021), (2020,2021) = avg(0.9, 0.9, 0.9) = 0.9
        # Inter-group compatibility: avg of (2019,ID), (2019,Name), (2020,ID), etc. = avg(0.1, 0.2, 0.1, ...) ≈ 0.15
        # Objective = 0.9 - 0.15 = 0.75
        ```

        The high objective value (0.75) suggests this is a good set of columns to unpivot.

    Args:
        compatibility_matrix: Compatibility matrix of columns.
        unpivot_set: List of columns selected for unpivot.

    Returns:
        Objective function value. Higher values suggest better column groupings for unpivot.

    Note:
        The function returns 0.0 for degenerate cases (e.g., unpivot_set with fewer than 2 columns,
        or when all columns are in the unpivot set).
    """
    all_columns = list(compatibility_matrix.columns)
    remaining_columns = [col for col in all_columns if col not in unpivot_set]

    # If there's only one column in unpivot_set or no remaining columns, objective is undefined
    if len(unpivot_set) < 2 or len(remaining_columns) == 0:
        return 0.0

    # Calculate average intra-group compatibility
    intra_compat_sum = 0.0
    intra_compat_count = 0

    for i in range(len(unpivot_set)):
        for j in range(i + 1, len(unpivot_set)):
            col1 = unpivot_set[i]
            col2 = unpivot_set[j]
            intra_compat_sum += compatibility_matrix.loc[col1, col2]
            intra_compat_count += 1

    avg_intra_compat = intra_compat_sum / intra_compat_count if intra_compat_count > 0 else 0.0

    # Calculate average inner-group compatibility
    inter_compat_sum = 0.0
    inter_compat_count = 0

    for sel_col in unpivot_set:
        for unselected_col in remaining_columns:
            inter_compat_sum += compatibility_matrix.loc[sel_col, unselected_col]
            inter_compat_count += 1

    avg_inter_compat = inter_compat_sum / inter_compat_count if inter_compat_count > 0 else 0.0

    # Calculate objective function value
    objective = avg_intra_compat - avg_inter_compat

    return objective


def solve_cmut(compatibility_matrix: pd.DataFrame) -> List[str]:
    """
    Solves the Compatibility-Maximizing Unpivot-Table (CMUT) problem using a greedy algorithm.

    As described in Section 4.4 of the paper, CMUT is NP-complete, so we use a greedy algorithm.
    The algorithm maximizes the objective function:

    max avg(c_i, c_j ∈ C) a(c_i, c_j) - avg(c_i ∈ C, c_j ∈ C\\C) a(c_i, c_j)

    where:
    - C is the set of columns to unpivot
    - a(c_i, c_j) is the compatibility between columns c_i and c_j

    The algorithm:
    1. Starts with the pair of columns with the highest compatibility
    2. Iteratively adds columns that maximize the objective function
    3. Keeps track of the best solution found so far

    Example:
        Consider a table with columns ["ID", "Name", "2019", "2020", "2021"] where
        the yearly columns should be melted.

        The objective function: Objective = average internal compatibility (Intra group) - average external compatibility (Inter group)

        In practice:
            - Intra group: columns in the same group (likely to be melted together, e.g., yearly columns like 2019, 2020)
            - Inter group: columns in different groups (less compatible, e.g., IDs with values)

        ```
        # Simplified compatibility matrix
        compat_matrix = pd.DataFrame([
            [1.0, 0.8, 0.1, 0.1, 0.1],  # ID
            [0.8, 1.0, 0.2, 0.2, 0.2],  # Name
            [0.1, 0.2, 1.0, 0.9, 0.9],  # 2019
            [0.1, 0.2, 0.9, 1.0, 0.9],  # 2020
            [0.1, 0.2, 0.9, 0.9, 1.0]   # 2021
        ], index=["ID", "Name", "2019", "2020", "2021"],
           columns=["ID", "Name", "2019", "2020", "2021"])

        value_vars = solve_cmut_greedy(compat_matrix)
        # Returns: ["2019", "2020", "2021"]
        ```

        This correctly identifies the yearly columns that should be melted.

    Args:
        compatibility_matrix: Compatibility matrix of columns.

    Returns:
        List of columns to unpivot that maximizes the objective function.

    Note:
        Since CMUT is NP-complete, this greedy approach may not find the globally
        optimal solution, but in practice it performs well.
    """
    columns = list(compatibility_matrix.columns)
    n = len(columns)

    if n < 2:
        return columns

    max_iterations = 15  # Limit total loop iterations for speed
    iteration_count = 0  # Count how many iterations have been run

    # Find the pair with the highest compatibility score
    max_score = -1
    best_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            col1 = columns[i]
            col2 = columns[j]
            score = compatibility_matrix.loc[col1, col2]

            if score > max_score:
                max_score = score
                best_pair = (col1, col2)

    # Initialize the selected columns with the best pair
    selected_columns = list(best_pair)   # selected columns represent Intra (Unpivot) columns group
    remaining_columns = [col for col in columns if col not in selected_columns]

    # Compute initial objective function value
    best_objective = calculate_cmut_objective(compatibility_matrix, selected_columns)
    best_solution = selected_columns.copy()  # Keep track of the best solution found

    # Debug print
    #print(f"Starting with columns: {selected_columns}, objective: {best_objective:.4f}")

    # Iteratively add columns that improve the objective function
    while remaining_columns and iteration_count < max_iterations:
        best_col = None
        best_new_objective = best_objective
        iteration_count += 1

        for col in remaining_columns:
            new_selected = selected_columns + [col]
            objective = calculate_cmut_objective(compatibility_matrix, new_selected)

            if objective > best_new_objective:
                best_new_objective = objective
                best_col = col

        # If adding any column doesn't improve the objective, stop
        if best_col is None:
            break

        # Add the best column to selected columns
        selected_columns.append(best_col)
        remaining_columns.remove(best_col)

        # Limit group size to avoid overly large groups (improves precision, avoids runtime blowup)
        if len(selected_columns) >= 5:  # Typically melted columns are 3-5
            break

        # Update the best solution if this one is better

        # We allow slightly worse objectives to be accepted to avoid stopping too early.
        # This flexibility helps in messy data where strict improvements might miss good groups.
        # - If we increase the threshold (e.g., closer to 1.0), CMUT becomes stricter:
        #   - Higher precision (fewer false positives, only very strong groupings are accepted)
        #   - Lower recall (misses some good groupings, stops early)
        # - If we decrease the threshold (e.g., 0.6), CMUT becomes more permissive:
        #   - Higher recall (finds more possible groupings, even borderline ones)
        #   - Lower precision (more false positives)
        # Adjusting this based on our dataset’s typical noise and your goals (precision vs recall trade-off).

        if best_new_objective > best_objective * 0.98:  # higher -> higher precision, lower -> higher recall
            best_objective = best_new_objective
            best_solution = selected_columns.copy()

        # Suppress detailed algorithm steps during training
        #print(f"Adding column: {best_col}, new objective: {best_new_objective:.4f}")

    # Debug print to see what CMUT selected
    # print(f"CMUT selected {len(best_solution)} columns to unpivot: {best_solution}")

    # Return the best solution found (selected columns)
    return best_solution


def generate_unpivot_prediction_table(auto_suggest_metrics, baseline_metrics, operator_name="unpivot"):
    """
    Generates Table 9-style baseline results (full accuracy, precision, recall, F1).

    Args:
        auto_suggest_metrics: Dictionary of metrics from Auto-Suggest.
        baseline_metrics: Dictionary of metrics from baseline methods.
        operator_name: e.g., "unpivot".

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

        # Column precision
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('precision', 0):.2f}")
        else:
            row.append(f"{baseline_metrics[method].get('precision', 0):.2f}")

        # Column recall
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('recall', 0):.2f}")
        else:
            row.append(f"{baseline_metrics[method].get('recall', 0):.2f}")

        # Column F1-score
        if method == "Auto-Suggest":
            row.append(f"{auto_suggest_metrics.get('f1_score', 0):.2f}")
        else:
            row.append(f"{baseline_metrics[method].get('f1_score', 0):.2f}")

        rows.append(row)

    # Headers
    headers = ["method", "full_accuracy", "column_precision", "column_recall", "column_f1"]

    # Print the table
    print(f"\nTable 9: {operator_name.capitalize()} Prediction - Baseline Results")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save the table to CSV
    os.makedirs("results/metrics", exist_ok=True)
    df = pd.DataFrame(rows, columns=headers)
    output_file = os.path.join("results/metrics", f"{operator_name}_methods_comparison.csv")
    df.to_csv(output_file, index=False)


def evaluate_unpivot(test_samples: List[Dict]) -> Dict[str, float]:
    """
    Evaluates the quality of unpivot (melt) column predictions.

    Args:
        test_samples: List of test unpivot samples.

    Returns:
        Dictionary of evaluation metrics.
    """
    correct_predictions = 0
    total_samples = 0
    precisions = []
    recalls = []
    f1_scores = []

    # Load the previously trained linear regression model (affinity weights)
    reg_model = load_model(os.path.join("models", "unpivot_affinity_weights_model.pkl"))

    # Start timer for all samples
    start_time = time.time()

    for sample in test_samples:
        input_df = sample['input_table']
        true_id_vars = set(sample['id_vars'])
        true_value_vars = set(sample['value_vars'])

        # Get all dimension columns (id_vars + value_vars)
        dimension_columns = list(true_id_vars.union(true_value_vars))

        # Skip if we have fewer than 2 or extremely large dimension columns
        if len(dimension_columns) < 2 or len(dimension_columns) > 30:
            continue

        # Build affinity matrix
        affinity_matrix = build_affinity_matrix(input_df, dimension_columns, reg_model)

        # Solve CMUT for unpivot prediction
        pred_value_vars = set(solve_cmut(affinity_matrix))
        pred_id_vars = set(dimension_columns) - pred_value_vars

        # Check if prediction matches ground truth
        is_correct = (pred_id_vars == true_id_vars and pred_value_vars == true_value_vars) or \
                     (pred_id_vars == true_value_vars and pred_value_vars == true_id_vars)

        if is_correct:
            correct_predictions += 1

        # Calculate precision, recall, and f1-score
        intersection = len(pred_value_vars.intersection(true_value_vars))
        precision = intersection / len(pred_value_vars) if pred_value_vars else 0
        recall = intersection / len(true_value_vars) if true_value_vars else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        total_samples += 1

    # End timer for all samples
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal CMUT evaluation time for all samples: {total_time:.2f} seconds")

    if total_samples == 0:
        print("No valid test samples for unpivot evaluation")
        return {'full_accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    # Final metrics
    metrics = {
        'full_accuracy': correct_predictions / total_samples if total_samples > 0 else 0.0,
        'precision': np.mean(precisions) if precisions else 0.0,
        'recall': np.mean(recalls) if recalls else 0.0,
        'f1_score': np.mean(f1_scores) if f1_scores else 0.0
    }

    # Create full evaluation record
    eval_dict = {
        "operator": "unpivot",
        "mode": "evaluation",
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': int(len(test_samples)),
        'correct_predictions': int(correct_predictions),
        "test_accuracy": float(metrics['full_accuracy']),
        "precision": float(metrics['precision']),
        "recall": float(metrics['recall']),
        "f1_score": float(metrics['f1_score']),
        'algorithm': 'CMUT',
        'solver': 'Greedy'
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
    generate_unpivot_prediction_table(
        auto_suggest_metrics=metrics,  # The metrics computed for Auto-Suggest
        baseline_metrics=baseline_metrics  # Baseline metrics to compare with
    )

    return metrics


def plot_affinity_graph_unpivot(affinity_matrix, id_vars, value_vars, save_path='results/figures/unpivot_affinity_graph.png', show=False, title="Affinity Graph (Unpivot)"):
    """
    Plots and saves an affinity graph for unpivot predictions (no left-right cut).

    Nodes: all dimension-like columns (id_vars + value_vars).
    Edges: affinity scores.
    """
    G = nx.Graph()

    # Add nodes
    for col in affinity_matrix.columns:
        color = 'lightblue' if col in id_vars else 'lightgreen'
        G.add_node(col, color=color)

    # Add edges
    for i, col1 in enumerate(affinity_matrix.columns):
        for j, col2 in enumerate(affinity_matrix.columns):
            if i < j:
                weight = affinity_matrix.loc[col1, col2]
                if weight > 0.0:
                    G.add_edge(col1, col2, weight=weight)

    # Use spring layout for natural grouping
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))

    # Node colors and border
    node_colors = [G.nodes[n]['color'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, edgecolors='black', linewidths=2)

    # Edges and edge weights
    edges = G.edges(data=True)
    weights = [d['weight'] * 2 for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8, style='dashed', edge_color='gray')
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Title and formatting
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.axis('off')

    # Legend
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Dimension Columns'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Collapsed Columns (value_vars)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nAffinity graph saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

import seaborn as sns
def plot_affinity_heatmap_unpivot(affinity_matrix, save_path='results/figures/unpivot_affinity_matrix_heatmap.png', show=True, title='Affinity Matrix Heatmap (Unpivot)'):
    """
    Plots and saves a heatmap of the affinity matrix for the unpivot operator.

    Args:
        affinity_matrix: pandas DataFrame of affinities.
        save_path: Path to save the heatmap (default: 'results/figures/unpivot_affinity_matrix_heatmap.png').
        show: If True, display the figure. Otherwise, just save it.
        title: Title for the heatmap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(affinity_matrix, annot=True, fmt=".2f", cmap='YlGnBu', linewidths=0.5, cbar_kws={'label': 'Affinity Score'})
    plt.title(title, fontsize=14)
    plt.xlabel('Columns')
    plt.ylabel('Columns')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\nAffinity matrix heatmap saved to: {save_path}\n")

    if show:
        plt.show()
    else:
        plt.close()


def predict_unpivot(table, model_path='models/unpivot_affinity_weights_model.pkl'):
    """
    Generates unpivot (melt) recommendations for a new input table
    and displays the results in the console.

    This function:
    - Heuristically identifies candidate dimension and measure columns.
    - Loads the trained affinity weights model to build an affinity matrix.
    - Solves the CMUT problem to determine which columns to unpivot (value_vars).
    - Displays:
        - The recommended id_vars and value_vars.
        - Example pandas code to perform the unpivot operation.

    Args:
        table: pandas DataFrame of the input table.
        model_path: Path to the saved affinity weight model.

    Returns:
        None. Prints the recommended unpivot split and a sample pandas code.
    """
    # Heuristically identify dimension and measure columns
    dimension_columns = []
    measure_columns = []

    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]) and table[col].nunique() > 15:
            measure_columns.append(col)
        else:
            dimension_columns.append(col)

    # Use trained groupby predictor to identify dimension columns
    # groupby_model, feature_names = load_model('models/groupby_column_model.pkl')
    # groupby_likelihoods = predict_column_groupby_scores(groupby_model, feature_names, table)
    # print(type(groupby_likelihoods))
    # print("nGroupby_likelihoods:")
    # for entry in groupby_likelihoods:
    #     print(entry)    #, type(entry)

    # dimension_columns = [entry[0] for entry in groupby_likelihoods if float(entry[1]) > 0.5]
    # measure_columns = [col for col in table.columns if col not in dimension_columns]

    if len(dimension_columns) < 2:
            print("Error: Need at least 2 dimension-like columns to build affinity matrix.")
            return

    # Load affinity weights model
    model = load_model(model_path)

    # Build affinity matrix
    affinity_matrix = build_affinity_matrix(table, dimension_columns, model)

    # Solve CMUT to get recommended columns to unpivot (value_vars)
    value_vars = solve_cmut(affinity_matrix)
    id_vars = [col for col in dimension_columns if col not in value_vars]

    print("\n=== Unpivot Structure Recommendation ===")
    print(f"\nID columns (id_vars): {id_vars}")
    print(f"Value columns (value_vars): {value_vars}")

    # Provide example pandas code
    print("\n=== Example Pandas Code ===")
    id_str = ', '.join([f"'{col}'" for col in id_vars])
    value_str = ', '.join([f"'{col}'" for col in value_vars])

    print("# Using pandas to unpivot (melt) the table:")
    print("melted_table = pd.melt(")
    print("    df,")
    print(f"    id_vars=[{id_str}],")
    print(f"    value_vars=[{value_str}],")
    print("    var_name='variable',")
    print("    value_name='value'")
    print(")")
    print()

    # Plot and save affinity graph (optional, just like pivot)
    plot_affinity_graph(affinity_matrix, id_vars, value_vars, save_path='results/figures/unpivot_affinity_graph.png')

    # Plot the heatmap
    # plot_affinity_heatmap_unpivot(affinity_matrix)

    # Show sample rows of melted table (uncomment to preview)
    # melted_table = pd.melt(table, id_vars=id_vars, value_vars=value_vars,
    #                        var_name='variable', value_name='value')
    # print("\nSample melted table preview (first 3 rows):")
    # print(melted_table.head(3))
