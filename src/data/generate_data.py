# generate_data.py
#
# This script:
# 1. Generates synthetic sequences of data preparation operations (e.g., 'merge', 'groupby', 'pivot', etc.)
#    to simulate the patterns found in real-world data science notebooks.
# 2. Creates datasets for both N-gram and RNN models and calculates statistics for all generated sequences.
# 3. Prepares training data for the final MLP model by associating each synthetic operator sequence
#    with a table (or a pair of tables) from extracted data and extracting relevant operator scores.
#
# The final dataset is saved to disk and used for training and evaluating the MLP next-operator prediction model.
#

import os
import json
import random
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict
# from collections import Counter

# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(data_dir, "generated_data")
os.makedirs(output_dir, exist_ok=True)
extracted_data_tables_dir = os.path.join(data_dir, "extracted_data")

# Defined operators as in the paper
OPERATORS = [
    'dropna', 'fillna', 'concat', 'merge', 'melt', 'pivot', 'pivot_table',
    'groupby', 'json_normalize', 'apply'
]

# Operator weights (relative frequency)
OPERATOR_WEIGHTS = {
    'groupby': 0.20,  # Very common
    'merge': 0.18,  # Very common
    'fillna': 0.12,  # Common
    'concat': 0.10,  # Common
    'dropna': 0.10,  # Common
    'pivot': 0.08,  # Less common
    'melt': 0.05,  # Less common
    'apply': 0.10,  # Common
    'pivot_table': 0.04,  # Rare
    'json_normalize': 0.03,  # Rare
}

# Define common operation transitions (A -> B means B commonly follows A)
# Higher probability indicates stronger relationship
OPERATION_TRANSITIONS = {
    'fillna': {
        'merge': 0.4,  # fillna often before joining
        'groupby': 0.3,  # fillna before aggregating
        'dropna': 0.05,  # rarely fillna then dropna
        'apply': 0.1,  # sometimes apply after fillna
        'pivot': 0.05,  # sometimes pivot after fillna
        'concat': 0.1  # sometimes concat after fillna
    },
    'dropna': {
        'groupby': 0.4,  # dropna before aggregating
        'pivot': 0.2,  # dropna before pivot
        'merge': 0.2,  # dropna before joining
        'apply': 0.1,  # sometimes apply after dropna
        'melt': 0.1  # sometimes melt after dropna
    },
    'merge': {
        'fillna': 0.2,  # fill nulls after merge
        'dropna': 0.15,  # drop nulls after merge
        'groupby': 0.4,  # common to aggregate after join
        'apply': 0.15,  # transform after merge
        'concat': 0.05,  # sometimes concat after merge
        'pivot': 0.05  # sometimes pivot after merge
    },
    'groupby': {
        'apply': 0.3,  # apply functions after grouping
        'merge': 0.3,  # merge after grouping
        'concat': 0.2,  # concat after grouping
        'pivot': 0.1,  # pivot after grouping
        'fillna': 0.1  # fillna after grouping
    },
    'concat': {
        'fillna': 0.3,  # fill nulls after concat
        'dropna': 0.2,  # drop nulls after concat
        'groupby': 0.3,  # group after concat
        'apply': 0.1,  # apply after concat
        'merge': 0.1  # merge after concat
    },
    'pivot': {
        'fillna': 0.3,  # fill nulls after pivot
        'apply': 0.3,  # apply after pivot
        'merge': 0.2,  # merge after pivot
        'concat': 0.1,  # concat after pivot
        'groupby': 0.1  # group after pivot
    },
    'melt': {
        'fillna': 0.2,  # fill nulls after melt
        'groupby': 0.4,  # group after melt
        'merge': 0.2,  # merge after melt
        'apply': 0.1,  # apply after melt
        'concat': 0.1  # concat after melt
    },
    'apply': {
        'groupby': 0.3,  # group after apply
        'fillna': 0.2,  # fill nulls after apply
        'dropna': 0.2,  # drop nulls after apply
        'merge': 0.2,  # merge after apply
        'concat': 0.1  # concat after apply
    },
    'pivot_table': {
        'fillna': 0.3,  # fill nulls after pivot_table
        'apply': 0.3,  # apply after pivot_table
        'merge': 0.2,  # merge after pivot_table
        'dropna': 0.1,  # drop nulls after pivot_table
        'concat': 0.1  # concat after pivot_table
    },
    'json_normalize': {
        'fillna': 0.3,  # fill nulls after json_normalize
        'dropna': 0.3,  # drop nulls after json_normalize
        'apply': 0.2,  # apply after json_normalize
        'merge': 0.1,  # merge after json_normalize
        'groupby': 0.1  # group after json_normalize
    }
}

# Common sequence patterns (these are complete sequences that commonly occur together)
COMMON_PATTERNS = [
    ['fillna', 'merge', 'groupby'],
    ['dropna', 'groupby', 'apply'],
    ['merge', 'fillna', 'groupby'],
    ['concat', 'fillna', 'groupby'],
    ['json_normalize', 'fillna', 'merge'],
    ['melt', 'fillna', 'groupby'],
    ['fillna', 'pivot', 'apply'],
    ['dropna', 'merge', 'fillna'],
    ['dropna', 'pivot', 'fillna'],
    ['merge', 'merge', 'groupby']
]


def generate_sequence(min_length: int = 3, max_length: int = 10) -> List[str]:
    """
    Generates a single sequence of operations.

    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of operations representing a sequence
    """
    length = random.randint(min_length, max_length)

    # 30% chance to start with a common pattern
    use_common = random.random() < 0.3
    sequence = random.choice(COMMON_PATTERNS) if use_common else []

    while len(sequence) < length:
        last_op = sequence[-1] if sequence else None

        if last_op and last_op in OPERATION_TRANSITIONS:
            transitions = OPERATION_TRANSITIONS[last_op]
            candidates, weights = zip(*transitions.items())
        else:
            candidates, weights = zip(*OPERATOR_WEIGHTS.items())

        next_op = random.choices(candidates, weights=weights)[0]
        sequence.append(next_op)

    return sequence


def generate_dataset(num_sequences: int = 1000, min_length: int = 3, max_length: int = 10) -> List[List[str]]:
    """
    Generates a dataset of operation sequences.

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of operation sequences
    """
    return [generate_sequence(min_length, max_length) for _ in range(num_sequences)]


def calculate_sequence_statistics(sequences: List[List[str]], save_dir=output_dir) -> Dict:
    """
    Calculates and saves statistics about the generated sequences.

    Args:
        sequences: List of operation sequences.
        save_dir: Directory in which the results will be saved.

    Returns:
        Dictionary with statistics.
    """
    total_sequences = len(sequences)
    avg_length = sum(len(seq) for seq in sequences) / total_sequences if total_sequences > 0 else 0
    min_length = min((len(seq) for seq in sequences), default=0)
    max_length = max((len(seq) for seq in sequences), default=0)

    # Initialize counts
    operator_counts = {op: 0 for op in OPERATORS}
    transition_counts = {op1: {op2: 0 for op2 in OPERATORS} for op1 in OPERATORS}

    # Single loop for counting
    for sequence in sequences:
        for i, op in enumerate(sequence):
            operator_counts[op] += 1
            if i < len(sequence) - 1:
                next_op = sequence[i + 1]
                transition_counts[op][next_op] += 1

    stats = {
        "total_sequences": total_sequences,
        "avg_sequence_length": avg_length,
        "min_sequence_length": min_length,
        "max_sequence_length": max_length,
        "operator_counts": operator_counts,
        "transition_counts": transition_counts
    }

    # Save to file
    os.makedirs(save_dir, exist_ok=True)
    stats_path = os.path.join(save_dir, "sequence_data_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    #print(f"Saved sequence statistics to '{stats_path}'")


def save_sequences(sequences: List[List[str]], out_dir: str):
    """
    Saves sequences and statistics to JSON files.

    Args:
        sequences: List of operation sequences.
        out_dir: Directory to save files.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save all sequences in one file
    all_sequences_path = os.path.join(out_dir, "all_sequences.json")
    with open(all_sequences_path, "w") as f:
        json.dump(sequences, f, indent=2)

    # Save sequence statistics
    stats = calculate_sequence_statistics(sequences)
    stats_path = os.path.join(output_dir, "sequence_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # print(f"Saved {len(sequences)} sequences to '{all_sequences_path}'")
    # print(f"Saved sequence statistics to '{stats_path}'")


def create_training_data(sequences: List[List[str]], out_dir: str):
    """
    Creates a unified training data JSON file for both n-gram and RNN models.
    Stores history and next as strings. Later, RNN-specific numeric conversion
    can be done during preprocessing.

    Args:
        sequences: List of operator sequences.
        out_dir: Directory to save the training data JSON file.
    """
    # Build training data for both n-gram and rnn
    training_data = []

    for sequence in sequences:
        for i in range(1, len(sequence)):
            # For each position in the sequence, record the history and the next operation
            training_data.append({
                "history": sequence[:i],
                "next": sequence[i]
            })

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Save as JSON
    output_path = os.path.join(out_dir, "sequence_data.json")
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    # Debug: print the first sample for inspection
    # if training_data:
    #     print("\nFirst training sample:", json.dumps(training_data[0], indent=2), "\n")

    print("\nSaved sequence training data to 'data' directory")

    return training_data


def generate_combined_data(sequence_data, extracted_data_path=extracted_data_tables_dir):
    """
    Generates a combined dataset (X, y) for training the final next-operator prediction model.

    For each example:
    - For merge: X is (history_sequence, left_table, right_table)
    - For others: X is (history_sequence, table)
    - y is the next operator (string label)

    This function:
    1. Takes synthetic operator sequences directly as a list of dicts.
    2. Iterates through each example, where:
        - history_sequence is a list of operator names (str) coming from 'history'.
        - next_op is the next operator to predict (label) coming from 'next'.
    3. For the 4 supported operators (merge, groupby, pivot, melt):
        - Considers up to 100 subdirectories (repo folders) to retrieve tables.
        - Loads left/right tables for merge or 'data.csv' for others.
        - Skips examples if no valid CSV file is found for that example.
    4. Appends the tuple (history, left_table, right_table) or (history, table) to X and the next operator label to y.

    Note:
        - This implementation is constrained to only include examples where 'next' is one of these 4 operators:
          ['merge', 'groupby', 'pivot', 'melt'].

    Args:
        sequence_data (list of dicts): Synthetic operator sequences.
        extracted_data_path (str): Path to the folder containing extracted operator data tables.

    Returns:
        tuple:
            - X (list of tuples): Each tuple contains the data described above.
            - y (list of str): Next operator labels corresponding to each history.
    """
    print(f"\nLoaded {len(sequence_data)} synthetic sequences\n")

    # Prepare output lists
    X = []
    y = []

    # List of supported operator directories (for table retrieval)
    supported_ops = ["merge", "groupby", "pivot", "melt"]

    # For each operator, precompute up to 100 subdirectories
    operator_subdirs = {}
    for op in supported_ops:
        op_dir = os.path.join(extracted_data_path, op)
        if os.path.exists(op_dir):
            subdirs = [d for d in os.listdir(op_dir) if os.path.isdir(os.path.join(op_dir, d))][:100]
            operator_subdirs[op] = [os.path.join(op_dir, d) for d in subdirs]
        else:
            operator_subdirs[op] = []

    # Track which subdir index to use for each operator
    operator_subdir_index = {op: 0 for op in supported_ops}

    # Iterate through each example in the sequence data
    for idx, example in enumerate(sequence_data):
        history = example["history"]
        next_op = example["next"]

        # Skip if next_op is not in supported_ops
        if next_op not in supported_ops:
            continue

        table_df = None
        subdirs = operator_subdirs[next_op]
        current_idx = operator_subdir_index[next_op]

        # If we have not exhausted the 100 repo-folders for this operator
        if current_idx < len(subdirs):
            subdir_path = subdirs[current_idx]
            operator_subdir_index[next_op] += 1  # Move to next subdir for next usage

            if next_op == "merge":
                # For merge, retrieve both left.csv and right.csv
                left_csv_path = os.path.join(subdir_path, "left.csv")
                right_csv_path = os.path.join(subdir_path, "right.csv")
                if (os.path.exists(left_csv_path) and os.path.getsize(left_csv_path) > 0 and
                        os.path.exists(right_csv_path) and os.path.getsize(right_csv_path) > 0):
                    try:
                        left_df = pd.read_csv(left_csv_path, low_memory=False)
                        right_df = pd.read_csv(right_csv_path, low_memory=False)
                        X.append((history, left_df, right_df))
                        y.append(next_op)
                    except Exception as e:
                        print(f"Could not read left/right CSV for 'merge' at {subdir_path}. Error: {e}")
            else:
                # For other operators, use data.csv
                data_csv_path = os.path.join(subdir_path, "data.csv")
                if os.path.exists(data_csv_path) and os.path.getsize(data_csv_path) > 0:
                    try:
                        table_df = pd.read_csv(data_csv_path, low_memory=False)
                        X.append((history, table_df))
                        y.append(next_op)
                    except Exception as e:
                        print(f"Could not read data.csv for '{next_op}' at {data_csv_path}. Error: {e}")

    # Debug print for some statistics
    # label_counts = Counter(y)
    # print("Dataset statistics by label:")
    # for op in supported_ops:
    #     count = label_counts.get(op, 0)
    #     print(f"  {op}: {count} samples")

    # Debug print to see how a sample looks like
    # print("\nSample output (first example):")
    # print(f"X[0] history: {X[0][0]}")
    # print(f"X[0] table shape: {X[0][1].shape if X[0][1] is not None else 'No table'}")
    # print(f"y[0]: {y[0]}")

    print(f"Generated combined dataset (of size {len(X)}) preparation completed!\n")

    # Save X and y as a binary pickle file
    X_y_output_path = os.path.join(output_dir, "combined_data.pkl")
    with open(X_y_output_path, "wb") as f:
        pickle.dump((X, y), f)

    print(f"Saved combined dataset to 'data' directory")


def main():

    # Set random seed
    random.seed(42)
    np.random.seed(42)
    num_sequences = 1000  # Total sequences to generate

    print(f"\nGenerating {num_sequences} operation sequences...")
    sequences = generate_dataset()

    # Calculate and save sequence statistics
    calculate_sequence_statistics(sequences)

    print("\nCreating training data for sequence models...")
    train_seq = create_training_data(sequences, output_dir)

    # Generate combined dataset (X, y) using generate_combined_data
    print("\nGenerating combined dataset for final model...")
    generate_combined_data(train_seq)

    print("\nProcess of generating artificial operation sequences and combined dataset successfully completed!\n")


if __name__ == "__main__":
    main()
