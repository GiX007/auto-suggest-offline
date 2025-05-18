# src/models/pivot_model.py
#
# Implementation of the Pivot prediction model based on Section 4.3
# of "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks"
#
# This module:
# 1. Implements the Affinity-Maximizing Pivot-Table (AMPT) algorithm
# 2. Provides functions to evaluate and predict pivot operations
# 3. Supports prediction on new tables

import pandas as pd
import numpy as np
import networkx as nx
import os
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Import from our package structure
from src.features.pivot_features import build_affinity_matrix


def plot_affinity_graph(affinity_matrix, index_columns, header_columns, save_path=None, show=False):
    """
    Plot an affinity graph similar to Figure 10 in the paper.

    This visualizes the affinity scores between columns, with nodes
    representing columns and edge weights representing affinity scores.

    Args:
        affinity_matrix: DataFrame containing affinity scores
        index_columns: Columns predicted to be in the index (one side of the cut)
        header_columns: Columns predicted to be in the header (other side of the cut)
        save_path: Path to save the figure (default: None)
        show: Whether to display the figure (default: False)
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a graph from the affinity matrix
    G = nx.Graph()

    # Add nodes with different colors based on which side of the cut they belong to
    for col in affinity_matrix.columns:
        if col in index_columns:
            G.add_node(col, color='lightblue', side='index')
        else:
            G.add_node(col, color='lightgreen', side='header')

    # Add edges with weights from affinity matrix
    for i, col1 in enumerate(affinity_matrix.columns):
        for j, col2 in enumerate(affinity_matrix.columns):
            if i < j:  # Only add each edge once
                weight = affinity_matrix.loc[col1, col2]
                if weight > 0.05:  # Only show significant edges
                    G.add_edge(col1, col2, weight=weight)

    # Create the plot
    plt.figure(figsize=(12, 10))

    # Get position of nodes - try to separate the two sides of the cut
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    colors = [G.nodes[n]['color'] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, alpha=0.8)

    # Draw edges with width based on affinity score
    edges = G.edges(data=True)
    weights = [d['weight'] * 3 for (_, _, d) in edges]  # Scale up for visibility
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6)

    # Draw edge labels for significant edges (affinity > 0.4)
    edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in edges if d['weight'] > 0.4}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add a title showing the cut
    plt.title(f"Affinity Graph with Cut: {', '.join(index_columns)} | {', '.join(header_columns)}")

    # Remove axes
    plt.axis('off')

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Index Columns'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Header Columns')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Save if a path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Affinity graph saved to {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def solve_ampt(affinity_matrix: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Solve the Affinity-Maximizing Pivot-Table (AMPT) problem.

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
        cut_value, partition = nx.stoer_wagner(G)
        objective_value = -cut_value  # Invert cut value to interpret as max-affinity

        # Step 6: Handle degenerate partition (empty side)
        if not partition[0] or not partition[1]:
            mid = len(columns) // 2
            return columns[:mid], columns[mid:]

        return partition[0], partition[1]

    except Exception as e:
        # Step 7: If Stoer-Wagner fails (e.g., due to disconnections), fallback to greedy
        return greedy_ampt_split(affinity_matrix)


def greedy_ampt_split(affinity_matrix: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Greedy algorithm for solving the AMPT problem when Stoer-Wagner fails.

    This function uses a greedy approach to partition columns into two groups
    (index and header) for pivot tables. The algorithm:
    1. Starts with the pair of columns having highest affinity
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
        1. Find highest affinity pair: (Year, Quarter) with score 0.9
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
            if col2 in affinity_matrix.index and col1 in affinity_matrix.columns:
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
            affinity_to_p1 = sum(affinity_matrix.loc[col, p]
                                 for p in partition_1
                                 if p in affinity_matrix.index and col in affinity_matrix.columns)
            affinity_to_p2 = sum(affinity_matrix.loc[col, p]
                                 for p in partition_2
                                 if p in affinity_matrix.index and col in affinity_matrix.columns)

            # Assign to partition with higher affinity
            if affinity_to_p1 >= affinity_to_p2:
                partition_1.append(col)
            else:
                partition_2.append(col)

        return partition_1, partition_2
    else:
        # If no good pairs found, use simple split
        mid = len(columns) // 2
        return columns[:mid], columns[mid:]


def evaluate_pivot_split(test_samples: List[Dict]) -> Dict[str, float]:
    """
    Evaluate the quality of pivot column split predictions.

    Args:
        test_samples: List of test pivot samples.

    Returns:
        Dictionary of evaluation metrics.
    """
    correct_splits = 0
    total_splits = 0

    rand_index_scores = []

    for sample in test_samples:
        input_df = sample['input_table']
        true_index = set(sample['index_columns'])
        true_header = set(sample['header_columns'])

        # Get all dimension columns (index + header)
        dimension_columns = list(true_index.union(true_header))

        # Skip if we have fewer than 2 dimension columns
        if len(dimension_columns) < 2:
            continue

        # Build affinity matrix
        affinity_matrix = build_affinity_matrix(input_df, dimension_columns)

        # Solve AMPT
        pred_index, pred_header = solve_ampt(affinity_matrix)
        pred_index = set(pred_index)
        pred_header = set(pred_header)

        # Check if prediction matches ground truth
        # Note: The prediction could be flipped (index <-> header) and still be correct
        is_correct = (pred_index == true_index and pred_header == true_header) or \
                     (pred_index == true_header and pred_header == true_index)

        if is_correct:
            correct_splits += 1

        # Calculate Rand Index
        # Convert sets to binary matrices for all pairs of columns
        n = len(dimension_columns)
        true_same_side = np.zeros((n, n))
        pred_same_side = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                col1 = dimension_columns[i]
                col2 = dimension_columns[j]

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

    if total_splits == 0:
        print("No valid test samples for pivot evaluation")
        return {
            'full_accuracy': 0.0,
            'rand_index': 0.0
        }

    #print(f"Evaluated {total_splits} pivot splits")
    metrics = {
        'full_accuracy': correct_splits / total_splits if total_splits > 0 else 0.0,
        'rand_index': np.mean(rand_index_scores) if rand_index_scores else 0.0
    }

    # Create a dictionary with all relevant metrics
    metrics_dict = {
        'operator': 'pivot',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': total_splits,
        'correct_splits': correct_splits,
        'full_accuracy': metrics['full_accuracy'],
        'rand_index': metrics['rand_index'],
        'algorithm': 'AMPT',
        'solver': 'Stoer-Wagner'
    }

    # Convert to DataFrame for easy CSV export
    metrics_df = pd.DataFrame([metrics_dict])

    # Ensure results directories exists
    os.makedirs('results/metrics', exist_ok=True)

    # Save to operator-specific metrics file
    # operator_metrics_file = f'results/metrics/pivot_metrics.csv'
    # if os.path.exists(operator_metrics_file):
    #     metrics_df.to_csv(operator_metrics_file, mode='a', header=False, index=False)
    # else:
    #     metrics_df.to_csv(operator_metrics_file, index=False)

    # Save to a combined metrics file for all operators
    combined_metrics_file = 'results/metrics/all_operators_metrics.csv'
    eval_metrics_file = 'results/metrics/pivot_eval_metrics.csv'

    if os.path.exists(combined_metrics_file):
        existing_df = pd.read_csv(combined_metrics_file)
        # If a training row for pivot already exists, save evaluation separately
        if 'pivot' in existing_df['operator'].values:
            metrics_df.to_csv(eval_metrics_file, index=False)
        else:
            metrics_df.to_csv(combined_metrics_file, mode='a', header=False, index=False)
    else:
        # No combined file yet; write the first one as combined
        metrics_df.to_csv(combined_metrics_file, index=False)

    #print(f"Metrics saved to {operator_metrics_file} and {combined_metrics_file}")

    # Return the metrics
    return metrics


def identify_dimension_measure_columns(table: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify dimension and measure columns in a table.

    This uses heuristics to determine which columns are good candidates for
    dimensions (index/header) and which are good candidates for measures (values).

    Note: Although the paper suggests using the trained GroupBy model for this,
    we use simple rule-based heuristics here instead — primarily because our
    GroupBy model is trained on a small dataset and may not generalize well.

    Args:
        table: Input table.

    Returns:
        Tuple of (dimension_columns, measure_columns).
    """
    dimension_columns = []
    measure_columns = []

    for col in table.columns:
        # Categorical or string columns are likely dimensions
        if pd.api.types.is_categorical_dtype(table[col]) or pd.api.types.is_string_dtype(
                table[col]) or pd.api.types.is_object_dtype(table[col]):
            dimension_columns.append(col)
        # Numeric columns with low cardinality are likely dimensions
        elif pd.api.types.is_numeric_dtype(table[col]):
            unique_ratio = table[col].nunique() / len(table) if len(table) > 0 else 0
            if unique_ratio < 0.1:  # Low cardinality
                dimension_columns.append(col)
            else:  # High cardinality numeric columns are likely measures
                measure_columns.append(col)
        else:  # Other types are assumed to be dimensions
            dimension_columns.append(col)

    return dimension_columns, measure_columns


def predict_pivot_split(table: pd.DataFrame, dimension_columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Predict the index/header split for pivot.

    Args:
        table: Input table.
        dimension_columns: List of dimension columns.

    Returns:
        Tuple of (index_columns, header_columns).
    """
    # Build affinity matrix
    affinity_matrix = build_affinity_matrix(table, dimension_columns)

    # Solve AMPT
    index_columns, header_columns = solve_ampt(affinity_matrix)

    return index_columns, header_columns


def recommend_pivot(table: pd.DataFrame, dimension_columns: Optional[List[str]] = None,
                    value_column: Optional[str] = None, aggfunc: str = 'mean',
                    generate_graph: bool = True):
    """
    Generate complete pivot table recommendations for a given table.

    This function implements the two-step pivot recommendation process:
    1. First identifies dimension vs. measure columns (if not provided)
    2. Then predicts how to split dimension columns into index and header

    Args:
        table: Input table to pivot
        dimension_columns: Optional list of dimension columns (if already known)
        value_column: Optional value column to aggregate (if already known)
        aggfunc: Aggregation function to use ('mean', 'sum', 'count', etc.)
        generate_graph: Whether to generate and save the affinity graph (default: True)

    Returns:
        Dictionary with pivot recommendations
    """
    # Step 1: Identify dimension vs. measure columns if not provided
    if dimension_columns is None or value_column is None:
        # Use heuristics to identify likely dimension and measure columns
        dimensions, measures = identify_dimension_measure_columns(table)

        # If dimension_columns was not provided, use our identified dimensions
        if dimension_columns is None:
            dimension_columns = dimensions

        # If value_column was not provided, use the first measure we found
        if value_column is None and measures:
            value_column = measures[0]
        elif value_column is None:
            # If no obvious measure was found, use the last numeric column
            numeric_cols = [col for col in table.columns if pd.api.types.is_numeric_dtype(table[col])]
            if numeric_cols:
                value_column = numeric_cols[-1]
            else:
                # If no numeric column, just use the last column
                value_column = table.columns[-1]

    # Step 2: Build affinity matrix
    print(f"\nBuilding affinity matrix for {len(dimension_columns)} dimension columns")
    affinity_matrix = build_affinity_matrix(table, dimension_columns)

    # Step 3: Split dimension columns into index and header
    print(f"Splitting dimension columns into index and header using AMPT algorithm")
    index_columns, header_columns = solve_ampt(affinity_matrix)

    # Create a sample of the result for display
    try:
        # Try to create a small sample pivot table for demonstration
        sample_df = table.head(100)  # Limit to 100 rows for preview
        pivot_sample = pd.pivot_table(
            sample_df,
            index=index_columns,
            columns=header_columns,
            values=value_column,
            aggfunc=aggfunc
        )

        empty_cells = pivot_sample.isna().sum().sum()
        total_cells = pivot_sample.size
        empty_ratio = empty_cells / total_cells if total_cells > 0 else 0

        # If the pivot table has too many empty cells, try swapping index and header
        if empty_ratio > 0.7:  # If more than 70% cells are empty
            print("Swapping index and header to reduce empty cells")
            index_columns, header_columns = header_columns, index_columns
            pivot_sample = pd.pivot_table(
                sample_df,
                index=index_columns,
                columns=header_columns,
                values=value_column,
                aggfunc=aggfunc
            )

        pivot_size = f"{pivot_sample.shape[0]} rows × {pivot_sample.shape[1]} columns"
    except Exception as e:
        print(f"Could not create sample pivot table: {e}")
        pivot_sample = None
        pivot_size = "unknown"

    # Generate and save the affinity graph (similar to Figure 10 in the paper)
    if generate_graph:
        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # Create figures directory if it doesn't exist
        os.makedirs('results/figures', exist_ok=True)
        # Save the graph
        graph_path = f'results/figures/pivot_affinity_graph_{timestamp}.png'
        plot_affinity_graph(affinity_matrix, index_columns, header_columns, save_path=graph_path)

    # Create the recommendation
    recommendation = {
        'index_columns': index_columns,
        'header_columns': header_columns,
        'value_column': value_column,
        'aggfunc': aggfunc,
        'pivot_sample': pivot_sample,
        'pivot_size': pivot_size,
        'affinity_matrix': affinity_matrix  # Include the affinity matrix in the recommendation
    }

    return recommendation


def display_pivot_recommendations(recommendation, max_rows=5, max_cols=5):
    """
    Display pivot recommendations in a readable format.

    Args:
        recommendation: Dictionary with pivot recommendations
        max_rows: Maximum number of rows to display in sample (default: 5)
        max_cols: Maximum number of columns to display in sample (default: 5)
    """
    if not recommendation:
        print("No pivot recommendations found.")
        return

    # Print recommendations in a readable format
    print("\n=== Pivot Table Recommendations ===")
    print("=" * 80)

    # Basic information
    print(f"\nRecommended pivot structure:")
    print(f"  Row indices (left side): {', '.join(recommendation['index_columns'])}")
    print(f"  Column headers (top): {', '.join(recommendation['header_columns'])}")
    print(f"  Values to aggregate: {recommendation['value_column']}")
    print(f"  Aggregation function: {recommendation['aggfunc']}")
    print(f"  Resulting pivot size: {recommendation['pivot_size']}")

    # Show sample if available
    pivot_sample = recommendation.get('pivot_sample')
    if pivot_sample is not None:
        print("\nSample pivot table preview:")

        # Limit display size
        if pivot_sample.shape[0] > max_rows or pivot_sample.shape[1] > max_cols:
            # Create a truncated version for display
            display_sample = pivot_sample.iloc[:max_rows, :max_cols]
            print(
                f"(Showing {min(max_rows, pivot_sample.shape[0])} rows × {min(max_cols, pivot_sample.shape[1])} columns of {pivot_sample.shape[0]} × {pivot_sample.shape[1]} total)")
            print(display_sample)
            print("...")
        else:
            print(pivot_sample)

    # Provide example pandas code
    print("\n=== Example Pandas Code ===")
    index_str = ', '.join([f"'{col}'" for col in recommendation['index_columns']])
    header_str = ', '.join([f"'{col}'" for col in recommendation['header_columns']])
    value_col = recommendation['value_column']
    aggfunc = recommendation['aggfunc']

    print("# Using pandas to create a pivot table:")
    print(f"pivot_table = pd.pivot_table(")
    print(f"    df,")
    print(f"    index=[{index_str}],")
    print(f"    columns=[{header_str}],")
    print(f"    values='{value_col}',")
    print(f"    aggfunc='{aggfunc}'")
    print(f")")

    # Additional reset_index for tabular format
    print("\n# To convert back to tabular format:")
    print("pivot_table = pivot_table.reset_index()")


def predict_on_file(file_path, aggfunc='mean', generate_graph=True):
    """
    Run pivot prediction on a CSV file.

    Args:
        file_path: Path to CSV file
        aggfunc: Aggregation function to use (default: 'mean')
        generate_graph: Whether to generate and save the affinity graph (default: True)

    Returns:
        Dictionary with pivot recommendations
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        return None

    # Load table
    try:
        table = pd.read_csv(file_path)

        print(f"\nLoaded table: {table.shape[0]} rows × {table.shape[1]} columns")
        print(f"Column names: {', '.join(table.columns)}")

        # Show sample rows
        print("\nSample data (first 3 rows):")
        print(table.head(3))

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Generate pivot recommendations
    recommendations = recommend_pivot(table, aggfunc=aggfunc, generate_graph=generate_graph)

    # Display recommendations
    display_pivot_recommendations(recommendations)

    # If graph was generated, mention it
    if generate_graph:
        print(
            "\nAn affinity graph (similar to Figure 10 in the paper) has been saved to the results/figures directory.")
        print(
            "This graph shows the relationships between dimension columns and how they were split into index and header.")

    return recommendations