# src/models/unpivot_model.py
#
# Implementation of the Unpivot prediction model based on Section 4.4
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Implements the Compatibility-Maximizing Unpivot-Table (CMUT) algorithm
# 2. Provides functions to evaluate and predict unpivot operations
# 3. Supports prediction on new tables

import pandas as pd
import numpy as np
import os
import time
import re
from typing import List, Dict, Tuple, Any, Set, Optional

# Import from our package structure
from src.features.unpivot_features import build_compatibility_matrix


def plot_compatibility_graph(compatibility_matrix: pd.DataFrame, unpivot_columns: List[str], save_path=None,
                             show=False):
    """
    Plot a compatibility graph similar to Figure 12 in the paper.

    This visualizes the compatibility scores between columns, with nodes
    representing columns and edge weights representing compatibility scores.
    Columns selected for unpivoting are shown in one color, others in another.

    Args:
        compatibility_matrix: DataFrame containing compatibility scores
        unpivot_columns: Columns selected for unpivoting
        save_path: Path to save the figure (default: None)
        show: Whether to display the figure (default: False)
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a graph from the compatibility matrix
    G = nx.Graph()

    # Add nodes with different colors based on whether they're selected for unpivot
    for col in compatibility_matrix.columns:
        if col in unpivot_columns:
            G.add_node(col, color='salmon', selected=True)  # Selected for unpivot
        else:
            G.add_node(col, color='lightblue', selected=False)  # Kept as-is

    # Add edges with weights from compatibility matrix
    for i, col1 in enumerate(compatibility_matrix.columns):
        for j, col2 in enumerate(compatibility_matrix.columns):
            if i < j:  # Only add each edge once
                weight = compatibility_matrix.loc[col1, col2]
                if weight > 0.05:  # Only show significant edges
                    G.add_edge(col1, col2, weight=weight)

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Use spring layout with unpivot columns more clustered
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    # Draw selected nodes (columns to unpivot)
    selected_nodes = [n for n, d in G.nodes(data=True) if d.get('selected', False)]
    unselected_nodes = [n for n, d in G.nodes(data=True) if not d.get('selected', False)]

    nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_color='salmon',
                           node_size=700, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=unselected_nodes, node_color='lightblue',
                           node_size=700, alpha=0.8)

    # Draw edges with width based on compatibility score
    # Highlight edges between selected nodes
    selected_edges = [(u, v) for u, v in G.edges() if u in unpivot_columns and v in unpivot_columns]
    unselected_edges = [(u, v) for u, v in G.edges() if (u, v) not in selected_edges]

    # Draw edges between selected nodes with higher width
    selected_weights = [G[u][v]['weight'] * 4 for u, v in selected_edges]
    nx.draw_networkx_edges(G, pos, edgelist=selected_edges, width=selected_weights, alpha=0.7, edge_color='red')

    # Draw other edges
    unselected_weights = [G[u][v]['weight'] * 2 for u, v in unselected_edges]
    nx.draw_networkx_edges(G, pos, edgelist=unselected_edges, width=unselected_weights, alpha=0.4, edge_color='gray')

    # Draw edge labels for significant edges (compatibility > 0.4)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()
                   if G[u][v]['weight'] > 0.4}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')

    # Add a title
    plt.title(f"Compatibility Graph for Unpivot (CMUT)\n{len(unpivot_columns)} columns selected for unpivot",
              fontsize=14)

    # Remove axes
    plt.axis('off')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='salmon', edgecolor='black', label='Columns to Unpivot'),
        Patch(facecolor='lightblue', edgecolor='black', label='Columns to Keep')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Save if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Compatibility graph saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def calculate_cmut_objective(compatibility_matrix: pd.DataFrame, unpivot_set: List[str]) -> float:
    """
    Calculate the CMUT objective function value for a given set of columns to unpivot.

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

    # Calculate average inter-group compatibility
    inter_compat_sum = 0.0
    inter_compat_count = 0

    for sel_col in unpivot_set:
        for unsel_col in remaining_columns:
            inter_compat_sum += compatibility_matrix.loc[sel_col, unsel_col]
            inter_compat_count += 1

    avg_inter_compat = inter_compat_sum / inter_compat_count if inter_compat_count > 0 else 0.0

    # Calculate objective function value
    objective = avg_intra_compat - avg_inter_compat

    return objective


def solve_cmut_greedy(compatibility_matrix: pd.DataFrame) -> List[str]:
    """
    Solve the Compatibility-Maximizing Unpivot-Table (CMUT) problem using a greedy algorithm.

    As described in Section 4.4 of the paper, CMUT is NP-complete, so we use a greedy algorithm.
    The algorithm maximizes the objective function:

    max avg(c_i, c_j ∈ C) a(c_i, c_j) - avg(c_i ∈ C, c_j ∈ C\\C) a(c_i, c_j)

    where:
    - C is the set of columns to unpivot
    - a(c_i, c_j) is the compatibility between columns c_i and c_j

    The algorithm:
    1. Starts with the pair of columns with highest compatibility
    2. Iteratively adds columns that maximize the objective function
    3. Keeps track of the best solution found so far

    Example:
        Consider a table with columns ["ID", "Name", "2019", "2020", "2021"] where
        the yearly columns should be unpivoted.

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

        This correctly identifies the yearly columns that should be unpivoted.

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

    if best_pair is None:
        # Fallback if no good pairs are found
        return columns[:2]

    # Initialize the selected columns with the best pair
    selected_columns = list(best_pair)   # selected columns represent Intra (Unpivot) columns group
    remaining_columns = [col for col in columns if col not in selected_columns]

    # Compute initial objective function value
    best_objective = calculate_cmut_objective(compatibility_matrix, selected_columns)
    best_solution = selected_columns.copy()  # Keep track of the best solution found

    # Suppress detailed algorithm steps during training
    #print(f"Starting with columns: {selected_columns}, objective: {best_objective:.4f}")

    # Iteratively add columns that improve the objective function
    while remaining_columns:
        best_col = None
        best_new_objective = best_objective

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

        # Update best solution if this one is better
        if best_new_objective > best_objective:
            best_objective = best_new_objective
            best_solution = selected_columns.copy()

        # Suppress detailed algorithm steps during training
        #print(f"Adding column: {best_col}, new objective: {best_new_objective:.4f}")

    # Return the best solution found (selected columns)
    return best_solution


def evaluate_unpivot_prediction(test_samples: List[Dict]) -> Dict[str, float]:
    """
    Evaluate the quality of unpivot column predictions using the CMUT algorithm.

    Args:
        test_samples: List of test unpivot samples.

    Returns:
        Dictionary of evaluation metrics.
    """
    total_samples = len(test_samples)
    if total_samples == 0:
        print("No test samples to evaluate")
        return {}

    correct_predictions = 0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    # Suppress detailed output during training
    # print(f"\nEvaluating unpivot prediction on {total_samples} test samples...")

    for sample_idx, sample in enumerate(test_samples):
        input_df = sample['input_table']

        # Get the ground truth columns to unpivot
        # Check if we're using the old or new feature structure
        if 'unpivot_columns' in sample:
            true_unpivot_columns = set(sample['unpivot_columns'])
        elif 'value_vars' in sample:
            true_unpivot_columns = set(sample['value_vars'])
        else:
            # Suppress output during training
            # print(f"Warning: Sample {sample_idx} doesn't have unpivot columns information")
            continue

        # Get all columns
        all_columns = list(input_df.columns)

        # Build compatibility matrix
        compatibility_matrix = build_compatibility_matrix(input_df, all_columns)

        # Solve CMUT to get predicted columns to unpivot
        predicted_unpivot_columns = set(solve_cmut_greedy(compatibility_matrix))

        # Check for exact match
        is_exact_match = (true_unpivot_columns == predicted_unpivot_columns)
        if is_exact_match:
            correct_predictions += 1

        # Calculate precision, recall, F1
        true_positives = len(true_unpivot_columns.intersection(predicted_unpivot_columns))  # Correctly predicted unpivot columns (true and predicted)
        false_positives = len(predicted_unpivot_columns - true_unpivot_columns) # # Incorrectly predicted as unpivot (predicted but not true)
        false_negatives = len(true_unpivot_columns - predicted_unpivot_columns) # # Missed unpivot columns (true but not predicted)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

        # Suppress detailed output during training
        # print(f"Sample {sample_idx + 1}: {'✓' if is_exact_match else '✗'} " +
        #      f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        # print(f"  True: {len(true_unpivot_columns)} columns")
        # print(f"  Pred: {len(predicted_unpivot_columns)} columns")

    # Calculate averages
    avg_precision = precision_sum / total_samples
    avg_recall = recall_sum / total_samples
    avg_f1 = f1_sum / total_samples
    full_accuracy = correct_predictions / total_samples

    # Suppress output during training
    # print(f"\nOverall metrics:")
    # print(f"  Full accuracy: {full_accuracy:.4f} ({correct_predictions}/{total_samples})")
    # print(f"  Avg precision: {avg_precision:.4f}")
    # print(f"  Avg recall: {avg_recall:.4f}")
    # print(f"  Avg F1 score: {avg_f1:.4f}")

    # Create a dictionary with all relevant metrics
    metrics_dict = {
        'operator': 'unpivot',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_samples': total_samples,
        'correct_predictions': correct_predictions,
        'full_accuracy': full_accuracy,
        'column_precision': avg_precision,
        'column_recall': avg_recall,
        'column_F1': avg_f1,
        'algorithm': 'CMUT',
        'solver': 'Greedy'
    }

    # Convert to DataFrame for easy CSV export
    metrics_df = pd.DataFrame([metrics_dict])

    # Ensure results directory exists
    os.makedirs('results/metrics', exist_ok=True)

    # Also save to a combined metrics file for all operators
    combined_metrics_file = 'results/metrics/all_operators_metrics.csv'
    eval_metrics_file = 'results/metrics/unpivot_eval_metrics.csv'

    if os.path.exists(combined_metrics_file):
        existing_df = pd.read_csv(combined_metrics_file)
        # If a training row for unpivot already exists, save evaluation separately
        if 'unpivot' in existing_df['operator'].values:
            metrics_df.to_csv(eval_metrics_file, index=False)
        else:
            metrics_df.to_csv(combined_metrics_file, mode='a', header=False, index=False)
    else:
        # No combined file yet; write the first one as combined
        metrics_df.to_csv(combined_metrics_file, index=False)

    return {
        'full_accuracy': full_accuracy,
        'column_precision': avg_precision,
        'column_recall': avg_recall,
        'column_F1': avg_f1
    }


def predict_unpivot_columns(table: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Predict which columns should be unpivoted for a new table.

    Args:
        table: Input DataFrame.

    Returns:
        Tuple of (id_vars, value_vars) where id_vars are columns to keep as is
        and value_vars are columns to unpivot.
    """
    # Get all columns
    all_columns = list(table.columns)

    if len(all_columns) < 3:  # Need at least 3 columns for meaningful unpivot
        return [], all_columns

    # Build compatibility matrix
    compatibility_matrix = build_compatibility_matrix(table, all_columns)

    # Solve CMUT
    value_vars = solve_cmut_greedy(compatibility_matrix)

    # The remaining columns are id_vars
    id_vars = [col for col in all_columns if col not in value_vars]

    return id_vars, value_vars


def recommend_unpivot(table: pd.DataFrame, generate_graph: bool = True):
    """
    Generate complete unpivot table recommendations for a given table.

    This function:
    1. Builds a compatibility matrix for all columns
    2. Uses the CMUT algorithm to identify which columns to unpivot
    3. Creates a sample result for preview
    4. Optionally generates a visualization of the compatibility graph

    Args:
        table: Input table to unpivot
        generate_graph: Whether to generate and save the compatibility graph (default: True)

    Returns:
        Dictionary with unpivot recommendations
    """
    # Step 1: Build compatibility matrix for all columns
    all_columns = list(table.columns)
    print(f"\nBuilding compatibility matrix for {len(all_columns)} columns")
    compatibility_matrix = build_compatibility_matrix(table, all_columns)

    # Step 2: Use CMUT to predict which columns to unpivot
    print(f"Determining columns to unpivot using CMUT algorithm")
    value_vars = solve_cmut_greedy(compatibility_matrix)
    id_vars = [col for col in all_columns if col not in value_vars]

    # Step 3: Try to create better names for the resulting variable and value columns
    # Based on patterns in the value_vars
    if len(value_vars) > 1:
        # Find common prefix
        prefix = os.path.commonprefix(value_vars)
        # Find common suffix
        value_vars_reversed = [col[::-1] for col in value_vars]
        suffix = os.path.commonprefix(value_vars_reversed)[::-1]

        # Remove common parts from one column to see if we get a meaningful key
        if prefix and len(prefix) > 1:
            sample_key = value_vars[0][len(prefix):].strip('_')
            var_name = prefix.strip('_')
        elif suffix and len(suffix) > 1:
            sample_key = value_vars[0][:-len(suffix)].strip('_')
            var_name = suffix.strip('_')
        else:
            # Look for common patterns like "year_2020", "year_2021"
            # Extract the non-numeric part as var_name
            non_numeric_parts = []
            for col in value_vars:
                non_numeric = re.sub(r'\d+', '', col).strip('_')
                if non_numeric:
                    non_numeric_parts.append(non_numeric)

            if non_numeric_parts:
                var_name = max(set(non_numeric_parts), key=non_numeric_parts.count)
            else:
                var_name = "variable"
    else:
        var_name = "variable"

    # For the value column name, try to find what the values represent
    value_name = "value"

    # If we can get a sample of the data type, use it as a hint
    for col in value_vars[:3]:  # Check first few columns
        if col in table.columns:
            if pd.api.types.is_numeric_dtype(table[col]):
                for measure_term in ['amount', 'value', 'price', 'quantity', 'score', 'count', 'number']:
                    if measure_term.lower() in var_name.lower():
                        value_name = measure_term
                        break
                if value_name == "value":  # If no specific term found
                    value_name = "amount"  # Default to "amount" for numeric
            break

    # Step 4: Create a sample of the unpivoted result for display
    try:
        # Create a small sample for demonstration
        sample_df = table.head(10)
        result_sample = pd.melt(
            sample_df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
        result_size = f"{len(result_sample)} rows × {len(result_sample.columns)} columns"
    except Exception as e:
        print(f"Could not create sample unpivot result: {e}")
        result_sample = None
        result_size = "unknown"

    # Step 5: Generate and save the compatibility graph (similar to Figure 12 in the paper)
    if generate_graph:
        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Create figures directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)
        # Save the graph
        graph_path = f'results/figures/unpivot_compatibility_graph_{timestamp}.png'
        plot_compatibility_graph(compatibility_matrix, value_vars, save_path=graph_path)

    # Create the recommendation
    recommendation = {
        'id_vars': id_vars,
        'value_vars': value_vars,
        'var_name': var_name,
        'value_name': value_name,
        'result_sample': result_sample,
        'result_size': result_size,
        'compatibility_matrix': compatibility_matrix  # Include the compatibility matrix in the recommendation
    }

    return recommendation


def display_unpivot_recommendations(recommendation, max_rows=10):
    """
    Display unpivot recommendations in a readable format.

    Args:
        recommendation: Dictionary with unpivot recommendations
        max_rows: Maximum number of rows to display in sample (default: 10)
    """
    if not recommendation:
        print("No unpivot recommendations found.")
        return

    # Print recommendations in a readable format
    print("\n=== Unpivot Table Recommendations ===")
    print("=" * 80)

    # Basic information
    print(f"\nRecommended unpivot structure:")

    id_vars = recommendation['id_vars']
    value_vars = recommendation['value_vars']

    print(f"  Keep columns (id_vars): {', '.join(id_vars)}")
    print(f"  Unpivot columns (value_vars): {', '.join(value_vars[:10])}" +
          (f" and {len(value_vars) - 10} more..." if len(value_vars) > 10 else ""))
    print(f"  Resulting variable column name: '{recommendation['var_name']}'")
    print(f"  Resulting value column name: '{recommendation['value_name']}'")
    print(f"  Resulting unpivoted size: {recommendation['result_size']}")

    # Show sample if available
    result_sample = recommendation.get('result_sample')
    if result_sample is not None:
        print("\nSample unpivoted result preview:")

        # Limit display size
        if len(result_sample) > max_rows:
            print(f"(Showing first {max_rows} rows of {len(result_sample)} total)")
            print(result_sample.head(max_rows))
            print("...")
        else:
            print(result_sample)

    # Provide example pandas code
    print("\n=== Example Pandas Code ===")

    id_vars_str = ', '.join([f"'{col}'" for col in id_vars])

    # If there are many value_vars, truncate the display
    if len(value_vars) > 10:
        value_vars_display = value_vars[:10]
        value_vars_str = ', '.join([f"'{col}'" for col in value_vars_display])
        value_vars_str += ", ..."  # Indicate there are more
    else:
        value_vars_str = ', '.join([f"'{col}'" for col in value_vars])

    var_name = recommendation['var_name']
    value_name = recommendation['value_name']

    print("# Using pandas to unpivot (melt) the table:")
    print(f"result = pd.melt(")
    print(f"    df,")
    print(f"    id_vars=[{id_vars_str}],")
    print(f"    value_vars=[{value_vars_str}],")
    print(f"    var_name='{var_name}',")
    print(f"    value_name='{value_name}'")
    print(f")")


def predict_on_file(file_path, generate_graph=True):
    """
    Run unpivot prediction on a CSV file.

    Args:
        file_path: Path to CSV file
        generate_graph: Whether to generate and save the compatibility graph (default: True)

    Returns:
        Dictionary with unpivot recommendations
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return None

    # Load table
    try:
        table = pd.read_csv(file_path)

        print(f"\nLoaded table: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"Column names: {', '.join(table.columns[:10])}" +
              (f" and {len(table.columns) - 10} more..." if len(table.columns) > 10 else ""))

        # Show sample rows
        print("\nSample data (first 3 rows):")
        print(table.head(3))

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Generate unpivot recommendations
    recommendations = recommend_unpivot(table, generate_graph=generate_graph)

    # Display recommendations
    display_unpivot_recommendations(recommendations)

    # If graph was generated, mention it
    if generate_graph:
        print(
            "\nA compatibility graph (similar to Figure 12 in the paper) has been saved to the results/figures directory.")
        print("This graph shows the relationships between columns and which ones were selected for unpivoting.")

    return recommendations