"""
generate_op_sequences.py

This script generates synthetic sequences of data preparation operations (such as 'merge',
'groupby', 'pivot', etc.) to simulate the patterns found in real-world data science notebooks.
These sequences are used to train and evaluate a next-operator prediction model.

The script creates three types of output data:
1. Raw operation sequences
2. N-gram specific training data
3. RNN specific training data

N-gram models and RNN models process sequences differently and thus require different data formats:

N-gram Models:
- Use a fixed context window (e.g., last 2-3 operations to predict the next)
- Work directly with categorical string data
- Process sequences as independent history-next pairs
- Simple frequency-based approach, often using count matrices
- Data format example: {"history": ["fillna", "merge"], "next": "groupby"}

RNN Models:
- Handle variable-length input sequences through padding
- Require numerical inputs (operations encoded as integers)
- Need a vocabulary mapping from operations to indices
- Organize data as tensors for batched training
- Often use embeddings and softmax output layers
- Data format example: {"history": [2, 5], "next": 3, "history_text": ["fillna", "merge"], "next_text": "groupby"}

By creating separate formats, we optimize the data structure for each model type,
making implementation more straightforward and training more efficient.

"""

import os
import json
import random
import numpy as np
from typing import List, Dict, Tuple
import itertools

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
    Generate a single sequence of operations.

    Args:
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of operations representing a sequence
    """
    # Determine sequence length
    length = random.randint(min_length, max_length)
    sequence = []

    # 30% chance to start with a common pattern, but only if min_length = 3 to ensure short patterns fit.
    use_common = random.random() < 0.3 and min_length <= 3
    if use_common:
        sequence = random.choice(COMMON_PATTERNS).copy()

    while len(sequence) < length:
        last_op = sequence[-1] if sequence else None

        if last_op and last_op in OPERATION_TRANSITIONS:
            # Get the weighted frequences of next operations
            transitions = OPERATION_TRANSITIONS[last_op]
            candidates, weights = zip(*transitions.items())
        else:
            candidates, weights = zip(*OPERATOR_WEIGHTS.items())

        # Normalize weights to probabilities and sample the next operation based on transition likelihoods,
        # then append it to the growing sequence.
        weights = [w / sum(weights) for w in weights]
        next_op = random.choices(candidates, weights=weights)[0]
        sequence.append(next_op)

    return sequence


def generate_dataset(num_sequences: int, min_length: int = 3, max_length: int = 10) -> List[List[str]]:
    """
    Generate a dataset of operation sequences.

    Args:
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length

    Returns:
        List of operation sequences
    """
    sequences = []
    for _ in range(num_sequences):
        sequence = generate_sequence(min_length, max_length)
        sequences.append(sequence)
    return sequences


def calculate_sequence_statistics(sequences: List[List[str]]) -> Dict:
    """
    Calculate statistics about the generated sequences.

    Args:
        sequences: List of operation sequences

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_sequences": len(sequences),
        "avg_sequence_length": sum(len(seq) for seq in sequences) / len(sequences),
        "min_sequence_length": min(len(seq) for seq in sequences),
        "max_sequence_length": max(len(seq) for seq in sequences),
        "operator_counts": {op: 0 for op in OPERATORS},
        "transition_counts": {op1: {op2: 0 for op2 in OPERATORS} for op1 in OPERATORS}
    }

    # Count operators
    for sequence in sequences:
        for op in sequence:
            stats["operator_counts"][op] = stats["operator_counts"].get(op, 0) + 1

    # Count transitions
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            op1 = sequence[i]
            op2 = sequence[i + 1]
            stats["transition_counts"][op1][op2] = stats["transition_counts"][op1].get(op2, 0) + 1

    return stats


def save_sequences(sequences: List[List[str]], output_dir: str):
    """
    Save sequences to files.

    Args:
        sequences: List of operation sequences
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save sequences as individual files
    # for i, sequence in enumerate(sequences):
    #     # Create op_seq.json files
    #     seq_dir = os.path.join(output_dir, f"sequence_{i + 1}")
    #     os.makedirs(seq_dir, exist_ok=True)
    #
    #     with open(os.path.join(seq_dir, "op_seq.json"), "w") as f:
    #         json.dump(sequence, f, indent=2)

    # Save all sequences in a single file
    with open(os.path.join(output_dir, f"all_sequences.json"), "w") as f:
        json.dump(sequences, f, indent=2)

    # Save statistics
    stats = calculate_sequence_statistics(sequences)
    with open(os.path.join(output_dir, f"sequence_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)


def create_n_gram_training_data(sequences: List[List[str]], output_dir: str):
    """
    Create training data specifically formatted for n-gram models.

    Args:
        sequences: List of operation sequences
        output_dir: Directory to save files
    """
    n_gram_data = []

    for sequence in sequences:
        for i in range(1, len(sequence)):
            # For each position in the sequence, record the history and the next operation
            history = sequence[:i]
            next_op = sequence[i]
            n_gram_data.append({
                "history": history,
                "next": next_op
            })

    # Save n-gram training data
    with open(os.path.join(output_dir, "n_gram_training_data.json"), "w") as f:
        json.dump(n_gram_data, f, indent=2)


def create_rnn_training_data(sequences: List[List[str]], output_dir: str):
    """
    Create training data specifically formatted for RNN models.
    This includes:
    - A vocabulary mapping of operations to indices
    - Padded input sequences
    - One-hot encoded target operations

    Args:
        sequences: List of operation sequences
        output_dir: Directory to save files
    """
    # Create a vocabulary of operations
    all_operators = sorted(list(set(op for seq in sequences for op in seq)))
    vocab = {op: i for i, op in enumerate(all_operators)}

    # Save vocabulary
    # with open(os.path.join(output_dir, "operator_vocabulary.json"), "w") as f:
    #     json.dump(vocab, f, indent=2)

    # Create training data examples
    training_data = []

    for sequence in sequences:
        for i in range(1, len(sequence)):
            history = sequence[:i]
            next_op = sequence[i]

            # Convert history to indices
            history_indices = [vocab[op] for op in history]
            next_op_index = vocab[next_op]

            training_data.append({
                "history": history_indices,
                "next": next_op_index,
                "history_text": history,
                "next_text": next_op
            })

    # Save training data
    with open(os.path.join(output_dir, "rnn_training_data.json"), "w") as f:
        json.dump(training_data, f, indent=2)


def main():
    # Set parameters directly
    output_dir = r"C:\Users\giorg\Auto_Suggest\data\generated_sequences"
    num_sequences = 1000  # Total sequences to generate
    min_length = 3
    max_length = 10
    seed = 42

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    print(f"\nGenerating {num_sequences} operation sequences...")
    sequences = generate_dataset(num_sequences, min_length, max_length)

    print(f"Saving sequences to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    save_sequences(sequences, output_dir)

    print("\nCreating N-gram training data...")
    create_n_gram_training_data(sequences, output_dir)

    print("Creating RNN training data...")
    create_rnn_training_data(sequences, output_dir)

    print("\nProcess of generating artificial operation sequences successfully completed!\n")


if __name__ == "__main__":
    main()