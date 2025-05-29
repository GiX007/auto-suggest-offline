# ngram_rnn_models.py
#
# This module implements two types of models for predicting the next data preparation operator
# in a notebook pipeline, inspired by the Auto-Suggest system (SIGMOD 2020):
#
# 1. N-gram Model:
#    - A statistical language model that learns operator transition probabilities based on local history.
#    - Predicts the next operator based on recent N previous ones.
#    - Includes functions to train, evaluate, save, and load n-gram models.
#
# 2. RNN (Recurrent Neural Network) Model:
#    - A neural sequence model (LSTM-based) that learns complex sequential dependencies between operators.
#    - Encodes history into embeddings and predicts the next operator via a softmax classifier.
#    - Includes functions to train, evaluate, save, and load RNN models using TensorFlow/Keras.
#

import os

# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
data_dir = os.path.join(base_dir, 'data')
model_dir = os.path.join(base_dir, 'models')
data_path = os.path.join(data_dir, 'generated_sequences', 'sequence_data.json')

# Disable oneDNN optimizations and suppress TensorFlow info messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN ops (removes that custom ops message)

# Prevent TensorFlow from printing hardware optimization and startup info (e.g., CPU instructions, oneDNN notices)
# Levels: '0' = all logs, '1' = no INFO, '2' = no INFO or WARNING, '3' = only ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import tensorflow as tf
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import argparse


# Create necessary directories for saving models, metrics, and figures
os.makedirs(r"C:\Users\giorg\Auto_Suggest\models", exist_ok=True)
os.makedirs(r"C:\Users\giorg\Auto_Suggest\results\metrics", exist_ok=True)
os.makedirs(r"C:\Users\giorg\Auto_Suggest\results\figures", exist_ok=True)

# ------------------------------
# N-gram model functions
# ------------------------------

def train_ngram_model(sequences, n=3):
    """
    Trains an n-gram model on given sequences of operators.

    For every sequence, this model counts how often each next operator follows
    a given history (context) of up to `n` previous operators.

    Example:
        '''
        sequences = [
                 ["fillna", "merge", "groupby"],
                 ["dropna", "merge", "apply"] ]

        >>model = train_ngram_model(sequences, n=2)
        >>print(model['counts'][('fillna',)])
        Counter({'merge': 1})
        >>print(model['counts'][('merge',)])
        Counter({'groupby': 1, 'apply': 1})
        '''

    Args:
        sequences: List of operator sequences (each sequence is a list of operators).
        n: The maximum n-gram size (default: 3).

    Returns:
        Dictionary containing:
            - 'counts': default dict mapping history tuples to Counter of next operators.
            - 'vocabulary': Set of all unique operators seen.
            - 'n': The n-gram size used for the model.
    """
    print(f"\nTraining N-gram model (n={n})...")
    counts = defaultdict(Counter)  # Mapping of history -> Counter of next ops
    vocabulary = set()  # All unique operations seen
    for sequence in sequences:
        vocabulary.update(sequence)  # Add all operators in this sequence to the vocabulary

        for i in range(1, len(sequence)):
            # Look at each position in the sequence, starting from the second item

            for j in range(1, min(i + 1, n + 1)):
                # For each position i, look back up to 'n' previous operations (or as many as available)
                # This builds all history slices of size 1 to n before i
                history = tuple(sequence[i - j:i])  # Get a slice of the previous j operators
                next_op = sequence[i]  # This is the operator that follows the history

                counts[history][next_op] += 1  # Count how often next_op follows this history

    print(f"\nN-gram model trained on {len(sequences)} sequences with {len(vocabulary)} unique operators")
    return {'counts': counts, 'vocabulary': vocabulary, 'n': n}

def predict_with_ngram(model, history, top_k=2):
    """
    Predicts the next operator(s) using the trained n-gram model.

    This function takes the current sequence history and looks for the longest matching context
    in the learned n-gram model (from the longest n-gram down to unigrams). It then retrieves the
    learned probabilities of possible next operators that follow this context and returns the
    top-k most likely next operators along with their probabilities.

    For example, if the training data often had "groupby" or "apply" following the history
    ['fillna', 'merge'], the model might return:
      [('groupby', 0.4), ('apply', 0.2)]

    Args:
        model: The trained n-gram model (output of train_ngram_model).
        history: List of recent operators (context) to predict from.
        top_k: Number of top next operators to return (default: 2).

    Returns:
        List of tuples (operator, probability), representing the top-k most likely next operators
        and their probabilities.
    """
    counts = model['counts']
    vocabulary = model['vocabulary']
    history = list(history)  # Ensure it's a list

    # Try to match the longest history to shortest
    for j in range(len(history), 0, -1):
        context = tuple(history[-j:])  # Use the last j items from history as context

        if context in counts and len(counts[context]) > 0:
            # If we have seen this exact context before
            counter = counts[context]  # Get counts of next operators after this context
            total = sum(counter.values())  # Total occurrences

            # Convert counts to probabilities (sorted most common first)
            probs = [(op, count / total) for op, count in counter.most_common()]

            return probs[:top_k]  # Return top-k most likely next operators

    # If no matching context found, fall back to unigram model (no history)
    if () in counts:
        counter = counts[()]
        total = sum(counter.values())
        probs = [(op, count / total) for op, count in counter.most_common()]
        return probs[:top_k]

    # If all else fails, return uniform probabilities over the vocabulary
    uniform_prob = 1.0 / len(vocabulary) if vocabulary else 0.0
    return [(op, uniform_prob) for op in vocabulary][:top_k]

def evaluate_ngram_model(model, test_sequences, top_k=2):
    """
    Evaluates the n-gram model on test data using precision@k, recall@k, and F1@k metrics.

    Each prediction step has exactly one true label, so precision@k and recall@k are numerically equal.
    We report both for consistency with standard evaluation formats, like Table 11 in the Auto-Suggest paper.

    Example:
        '''
        >>metrics = evaluate_ngram_model(model, test_sequences, top_k=2)
        >>print(metrics)
        {'precision@1': 0.72, 'recall@1': 0.72, 'f1@1': 0.72,
         'precision@2': 0.89, 'recall@2': 0.89, 'f1@2': 0.89}
         '''

    Args:
        model: The trained n-gram model (output of train_ngram_model).
        test_sequences: List of operator sequences to evaluate (each sequence is a list of operators).
        top_k: Maximum k value to calculate metrics for (default: 2).

    Returns:
        Dictionary with precision@k, recall@k, and F1@k metrics for all k up to the provided top_k.

    """
    # Create a list of k-values up to the provided top_k
    k_values = list(range(1, top_k + 1))
    print("\nEvaluating N-gram model...")
    metrics = {}

    for k in k_values:
        correct_at_k = 0
        total_examples = 0

        for sequence in test_sequences:
            for i in range(1, len(sequence)):
                history = sequence[:i]
                true_next = sequence[i]

                predictions = predict_with_ngram(model, history, top_k=k)
                pred_ops = [op for op, _ in predictions]

                if true_next in pred_ops:
                    correct_at_k += 1
                total_examples += 1

        # Calculate precision and recall
        precision = correct_at_k / total_examples if total_examples > 0 else 0.0
        recall = precision  # One correct label per step makes precision == recall
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'precision@{k}'] = round(precision, 2)
        metrics[f'recall@{k}'] = round(recall, 2)
        metrics[f'f1_score@{k}'] = round(f1, 2)

        # Debug print for detailed counts
        # print(f"Top-{k}: Correct predictions: {correct_at_k}/{total_examples} — Precision: {precision:.2f}")

    print("\nN-gram Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
    print()


def save_ngram_model(model, file_path):
    """
    Saves the n-gram model to a file in JSON format.

    Args:
        model: The trained n-gram model (output of train_ngram_model).
        file_path: Path to save the model JSON file.

    Returns:
        None
    """
    model_data = {
        'n': model['n'],
        'counts': {str(k): dict(v) for k, v in model['counts'].items()},
        'vocabulary': list(model['vocabulary'])
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"\nN-gram model saved to 'models' directory\n")

def load_ngram_model(file_path):
    """
    Loads an n-gram model from a JSON file.

    Args:
        file_path: Path to the saved model JSON file.

    Returns:
        The loaded n-gram model (dictionary with 'n', 'counts', and 'vocabulary').
    """
    with open(file_path, 'r') as f:
        model_data = json.load(f)

    counts = defaultdict(Counter)
    for k, v in model_data['counts'].items():
        if k == '()':
            key = ()
        else:
            # Convert the string representation of the tuple back to a tuple of strings
            content = k[1:-1]
            key = tuple(item.strip("'\" ") for item in content.split(',')) if content else ()
        counts[key] = Counter(v)
    model = {
        'n': model_data['n'],
        'counts': counts,
        'vocabulary': set(model_data['vocabulary'])
    }
    print(f"\nN-gram model loaded from {file_path}")
    return model

# ------------------------------
# RNN model functions
# ------------------------------

def train_rnn_model(training_data, embedding_dim=32, lstm_units=64, max_sequence_length=10, epochs=30):
    """
    Trains an RNN model on given operator history/next pairs.

    Args:
        training_data: List of dicts with 'history' and 'next' keys (strings).
        embedding_dim: Dimension of the embedding layer (default: 32).
        lstm_units: Number of units in the LSTM layer (default: 64).
        max_sequence_length: Maximum length of input sequences (default: 10).
        epochs: Number of training epochs (default: 30).

    Returns:
        Dictionary containing:
            - 'keras_model': The trained Keras RNN model.
            - 'label_encoder': LabelEncoder for operator encoding.
            - 'embedding_dim': The embedding dimension used.
            - 'lstm_units': The number of LSTM units.
            - 'max_sequence_length': The maximum sequence length.
            - 'vocabulary_size': The size of the operator vocabulary.
    """
    print("\nTraining RNN model...")

    # Build vocabulary from entire dataset
    all_ops = set(op for d in training_data for op in d["history"] + [d["next"]])
    all_operators = sorted(list(all_ops))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_operators)
    vocabulary_size = len(label_encoder.classes_)

    # Prepare input sequences and targets
    X = [d["history"] for d in training_data]
    y = [d["next"] for d in training_data]

    X_encoded = [label_encoder.transform(seq) for seq in X]
    y_encoded = label_encoder.transform(y)
    X_padded = pad_sequences(X_encoded, maxlen=max_sequence_length, padding='pre')
    y_onehot = to_categorical(y_encoded, num_classes=vocabulary_size)

    # Define RNN model
    model = Sequential([
        Input(shape=(max_sequence_length,)),
        Embedding(vocabulary_size, embedding_dim, mask_zero=True),
        LSTM(lstm_units),
        Dropout(0.2),
        Dense(vocabulary_size, activation='softmax')
    ])

    # noinspection SpellCheckingInspection
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_padded, y_onehot, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

    rnn_model = {
        'keras_model': model,
        'label_encoder': label_encoder,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'max_sequence_length': max_sequence_length,
        'vocabulary_size': vocabulary_size
    }
    print(f"\nRNN model trained on {len(training_data)} samples with {vocabulary_size} unique operators")
    return rnn_model


def predict_with_rnn(model, history, top_k=2):
    """
    Predicts the next operator(s) using the trained RNN model given a sequence history.

    Args:
        model: The trained RNN model (output of train_rnn_model).
        history: List of recent operators (context) to predict from.
        top_k: Number of top next operators to return (default: 2).

    Returns:
        List of tuples (operator, probability) representing the top-k most likely next operators.
    """
    keras_model = model['keras_model']
    label_encoder = model['label_encoder']
    max_sequence_length = model['max_sequence_length']

    # Encode the history
    try:
        history_encoded = label_encoder.transform(history)
    except ValueError:
        # Fallback for unknown operators: encode known, use 0 for unknown
        history_encoded = [label_encoder.transform([op])[0] if op in label_encoder.classes_ else 0 for op in history]

    # Pad to fixed length
    history_padded = pad_sequences([history_encoded], maxlen=max_sequence_length, padding='pre')

    # Predict probabilities
    predictions = keras_model.predict(history_padded, verbose=0)[0]

    # Get top-k indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_operators = label_encoder.inverse_transform(top_indices)

    return list(zip(top_operators, top_probs))


def evaluate_rnn_model(model, test_data, top_k=2):
    """
    Evaluates the RNN model using precision@k, recall@k, and F1@k.

    In this top-k next-operator prediction task, we have exactly one true label per prediction.
    Therefore, precision@k and recall@k are numerically equal:

    Metric     | What it asks                                        | How computed (1 true label)
    -----------|------------------------------------------------------|------------------------------
    Precision@k| Of the k predictions made, was the true one included? | correct_at_k / total_predictions
    Recall@k   | Of the relevant (true) labels, how many were retrieved? | correct_at_k / total_true_labels = same here
    F1@k       | Harmonic mean of precision and recall                 | (2 * precision * recall) / (precision + recall)

    Args:
        model: The trained RNN model (output of train_rnn_model).
        test_data: List of {'history': [...], 'next': ...} dicts.
        top_k: Maximum k value to calculate metrics for (default: 2).

    Returns:
        Dictionary of evaluation metrics (e.g., {'precision@1': 0.80, 'precision@2': 0.90}).
    """
    k_values = list(range(1, top_k + 1))
    print("\nEvaluating RNN model...")
    metrics = {}

    for k in k_values:
        correct_at_k = 0
        total_examples = 0

        for sample in test_data:
            history = sample["history"]
            true_next = sample["next"]

            predictions = predict_with_rnn(model, history, top_k=k)
            pred_ops = [op for op, _ in predictions]

            if true_next in pred_ops:
                correct_at_k += 1
            total_examples += 1

        precision = correct_at_k / total_examples if total_examples > 0 else 0.0
        recall = precision
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'precision@{k}'] = round(precision, 2)
        metrics[f'recall@{k}'] = round(recall, 2)
        metrics[f'f1_score@{k}'] = round(f1, 2)

        # Optional: log for clarity
        print(f"Top-{k}: Correct: {correct_at_k}/{total_examples} — Precision: {precision:.2f}")

    print("\nRNN Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
    print()


def save_rnn_model(model, model_path: str) -> None:
    """
    Saves the RNN model architecture, weights, and metadata.

    Args:
        model: The trained RNN model (output of train_rnn_model).
        model_path: Base path to save the model architecture (.keras) and parameters (.json).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the Keras model
    model['keras_model'].save(f"{model_path}.keras")

    # Save metadata
    params = {
        'embedding_dim': model['embedding_dim'],
        'lstm_units': model['lstm_units'],
        'max_sequence_length': model['max_sequence_length'],
        'vocabulary_size': model['vocabulary_size'],
        'classes': model['label_encoder'].classes_.tolist()
    }
    with open(f"{model_path}_params.json", 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\nRNN model and its parameters saved to 'models' directory\n")


def load_rnn_model(model_path: str) -> dict:
    """
    Loads the RNN model architecture, weights, and metadata from saved files.

    Args:
        model_path: Base path to load the model architecture (.keras) and parameters (.json).

    Returns:
        The loaded RNN model as a dictionary with:
            - 'keras_model'
            - 'label_encoder'
            - 'embedding_dim'
            - 'lstm_units'
            - 'max_sequence_length'
            - 'vocabulary_size'
    """
    with open(f"{model_path}_params.json", 'r') as f:
        params = json.load(f)

    keras_model = tf.keras.models.load_model(f"{model_path}.keras")

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(params['classes'])

    model = {
        'keras_model': keras_model,
        'label_encoder': label_encoder,
        'embedding_dim': params['embedding_dim'],
        'lstm_units': params['lstm_units'],
        'max_sequence_length': params['max_sequence_length'],
        'vocabulary_size': params['vocabulary_size']
    }

    print(f"\nRNN model loaded from {os.path.dirname(model_path)}")
    return model


# --------------
# Command Line Interface
# --------------

def main():

    parser = argparse.ArgumentParser(description="Train, evaluate, or predict with N-gram or RNN models.")
    parser.add_argument('--model', choices=['ngram', 'rnn'], required=True, help="Model type to use")
    parser.add_argument('mode', choices=['train', 'eval', 'predict'], help="Mode to run")
    args = parser.parse_args()

    if args.model == 'ngram':
        with open(data_path, 'r') as f:
            records = json.load(f)
        model_path = os.path.join(model_dir, 'ngram_model.json')

        # Reconstruct full operator sequences for N-gram
        sequences = [r['history'] + [r['next']] for r in records]
        train_seqs, test_seqs = train_test_split(sequences, test_size=0.2, random_state=42)

        if args.mode == 'train':
            print(f"\nLoaded {len(train_seqs)}/{len(sequences)} samples for training...")
            model = train_ngram_model(train_seqs)
            save_ngram_model(model, model_path)

        elif args.mode == 'eval':
            print(f"\nLoaded {len(test_seqs)}/{len(sequences)} samples for evaluation...")
            model = load_ngram_model(model_path)
            evaluate_ngram_model(model, test_seqs)

        elif args.mode == 'predict':
            model = load_ngram_model(model_path)
            example = ['json_normalize', 'fillna']
            predictions = predict_with_ngram(model, example)
            rounded_predictions = [(op, round(prob, 2)) for op, prob in predictions]
            print(f"\nPredictions for {example}: {rounded_predictions}\n")


    elif args.model == 'rnn':
        with open(data_path, 'r') as f:
            records = json.load(f)
        model_path = os.path.join(model_dir, 'rnn_model')

        # Use records directly as training samples
        train_data, test_data = train_test_split(records, test_size=0.2, random_state=42)

        if args.mode == 'train':
            print(f"\nLoaded {len(train_data)}/{len(records)} samples for training...")
            model = train_rnn_model(train_data)
            save_rnn_model(model, model_path)

        elif args.mode == 'eval':
            # Note: Evaluation is slow because we run model.predict() separately for each history step of a test sequence,
            # unlike training which is batched and vectorized for speed.
            print(f"\nLoaded {len(test_data)}/{len(records)} samples for evaluation...")
            model = load_rnn_model(model_path)
            evaluate_rnn_model(model, test_data)


        elif args.mode == 'predict':
            model = load_rnn_model(model_path)
            example = ['json_normalize', 'fillna']  # Same example as above
            predictions = predict_with_rnn(model, example)
            rounded_predictions = [(op, round(prob, 2)) for op, prob in predictions]
            print(f"\nPredictions for {example}: {rounded_predictions}\n")

if __name__ == '__main__':
    main()
