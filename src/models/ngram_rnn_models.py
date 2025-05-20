"""
ngram_rnn_models.py

This module implements two types of models for predicting the next data preparation operator
in a notebook pipeline, inspired by the Auto-Suggest system (SIGMOD 2020):

1. N-gram Model:
   - A statistical language model that learns operator transition probabilities based on local history.
   - Predicts the next operator based on recent N previous ones.
   - Includes functions to train, evaluate, save, and load n-gram models.

2. RNN (Recurrent Neural Network) Model:
   - A neural sequence model (LSTM-based) that learns complex sequential dependencies between operators.
   - Encodes history into embeddings and predicts the next operator via a softmax classifier.
   - Includes functions to train, evaluate, save, and load RNN models using TensorFlow/Keras.

"""

import os
# Disable oneDNN optimizations and suppress TensorFlow info messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN ops (removes that custom ops message)

# Prevent TensorFlow from printing hardware optimization and startup info (e.g., CPU instructions, oneDNN notices)
# Levels: '0' = all logs, '1' = no INFO, '2' = no INFO or WARNING, '3' = only ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, LSTM, Dropout, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import argparse
import matplotlib.pyplot as plt
import time
from src.utils.model_utils import save_model, load_model


# Create necessary directories for saving models, metrics, and figures
os.makedirs(r"C:\Users\giorg\Auto_Suggest\models", exist_ok=True)
os.makedirs(r"C:\Users\giorg\Auto_Suggest\results\metrics", exist_ok=True)
os.makedirs(r"C:\Users\giorg\Auto_Suggest\results\figures", exist_ok=True)

# ------------------------------
# N-gram model functions
# ------------------------------

def train_ngram_model(sequences, n=3):
    """Train an n-gram model on given sequences of operators."""
    print("\nTraining N-gram model (n=3)...")
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
    """Predict the next operator using the n-gram model."""
    counts = model['counts']
    vocabulary = model['vocabulary']
    if not isinstance(history, list):
        history = list(history)

    # Try to match longest history to shortest
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

    else:
        # As a last resort, return a uniform distribution over the vocabulary
        uniform_prob = 1.0 / len(vocabulary) if vocabulary else 0.0
        return [(op, uniform_prob) for op in vocabulary][:top_k]

def evaluate_ngram_model(model, test_sequences, k_values=[1, 2]):
    """
    Evaluate n-gram model on test data using precision@k, recall@k, and F1@k.

    Each prediction step has one correct label, so precision@k and recall@k are numerically equal.
    We report both for consistency with standard evaluation formats like Table 11 in the Auto-Suggest paper.
    """
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
        #recall = precision  # One correct label per step makes precision == recall
        #f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'precision@{k}'] = round(precision, 2)
        #metrics[f'recall@{k}'] = round(recall, 2)
        #metrics[f'f1@{k}'] = f1
    print(f"\nN-gram evaluation results: {metrics}\n")
    return metrics

def save_ngram_model(model, file_path):
    """Save the n-gram model to a file as JSON."""
    model_data = {
        'n': model['n'],
        'counts': {str(k): dict(v) for k, v in model['counts'].items()},
        'vocabulary': list(model['vocabulary'])
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"\nN-gram model saved to {file_path}\n")

def load_ngram_model(file_path):
    """Load an n-gram model from a JSON file."""
    with open(file_path, 'r') as f:
        model_data = json.load(f)
    counts = defaultdict(Counter)
    for k, v in model_data['counts'].items():
        if k == '()':
            key = ()
        else:
            content = k[1:-1]
            key = tuple(item.strip("'\" ") for item in content.split(',')) if content else ()
        counts[key] = Counter(v)
    model = {'n': model_data['n'], 'counts': counts, 'vocabulary': set(model_data['vocabulary'])}
    print(f"\nN-gram model loaded from {file_path}")
    return model

# ------------------------------
# RNN model functions
# ------------------------------

def train_rnn_model(sequences, embedding_dim=32, lstm_units=64, max_sequence_length=10, epochs=30):
    """Train an RNN model on given operator sequences."""
    print("\nTraining RNN model...")
    all_operators = sorted(list(set(op for seq in sequences for op in seq)))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_operators)
    vocabulary_size = len(label_encoder.classes_)

    # Prepare training data (input sequences and targets)
    X, y = [], []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            X.append(sequence[:i])
            y.append(sequence[i])

    X_encoded = [label_encoder.transform(seq) for seq in X]
    y_encoded = label_encoder.transform(y)
    X_padded = pad_sequences(X_encoded, maxlen=max_sequence_length, padding='pre')
    y_onehot = to_categorical(y_encoded, num_classes=vocabulary_size)

    # Define RNN model architecture
    model = Sequential([
        Input(shape=(max_sequence_length,)),
        Embedding(vocabulary_size, embedding_dim, mask_zero=True),
        LSTM(lstm_units),
        Dropout(0.2),
        Dense(vocabulary_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(X_padded, y_onehot, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)

    rnn_model = {
        'keras_model': model,
        'label_encoder': label_encoder,
        'embedding_dim': embedding_dim,
        'lstm_units': lstm_units,
        'max_sequence_length': max_sequence_length,
        'vocabulary_size': vocabulary_size
    }
    print(f"\nRNN model trained on {len(sequences)} sequences with {vocabulary_size} unique operators")
    return rnn_model

def predict_with_rnn(model, history, top_k=2, return_raw=False):
    """Predict next operator using RNN model given a sequence history."""
    keras_model = model['keras_model']
    label_encoder = model['label_encoder']
    max_sequence_length = model['max_sequence_length']

    try:
        history_encoded = label_encoder.transform(history)
    except ValueError:
        history_encoded = [label_encoder.transform([op])[0] if op in label_encoder.classes_ else 0 for op in history]

    history_padded = pad_sequences([history_encoded], maxlen=max_sequence_length, padding='pre')
    predictions = keras_model.predict(history_padded, verbose=0)[0]

    if return_raw:
        return predictions

    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_operators = label_encoder.inverse_transform(top_indices)
    return list(zip(top_operators, top_probs))

def evaluate_rnn_model(model, test_sequences, k_values=[1, 2]):
    """
    Evaluate RNN model using precision@k, recall@k, and F1@k.

    In this top-k next-operator prediction task, we have exactly one true label per prediction.
    Therefore, precision@k and recall@k are numerically equal:

    Metric     | What it asks                                        | How computed (1 true label)
    -----------|------------------------------------------------------|------------------------------
    Precision@k| Of the k predictions made, was the true one included? | correct_at_k / total_predictions
    Recall@k   | Of the relevant (true) labels, how many were retrieved? | correct_at_k / total_true_labels = same here

    We still report both to match evaluation conventions and Table 11 in the Auto-Suggest paper.
    """
    print("\nEvaluating RNN model...")
    metrics = {}

    for k in k_values:
        correct_at_k = 0
        total_examples = 0

        for sequence in test_sequences:
            for i in range(1, len(sequence)):
                history = sequence[:i]
                true_next = sequence[i]

                predictions = predict_with_rnn(model, history, top_k=k)
                pred_ops = [op for op, _ in predictions]

                if true_next in pred_ops:
                    correct_at_k += 1
                total_examples += 1

        # Calculate precision and recall (recall is the same as precision)
        precision = correct_at_k / total_examples if total_examples > 0 else 0.0
        #recall = correct_at_k / total_examples if total_examples > 0 else 0.0

        metrics[f'precision@{k}'] = round(precision, 2)
        #metrics[f'recall@{k}'] = round(recall, 2)
        # metrics[f'f1@{k}'] = 2 * precision * recall / (precision + recall)

    print(f"\nRNN evaluation results: {metrics}\n")
    return metrics

def save_rnn_model(model, model_path):
    """Save RNN model architecture, weights, and metadata."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model['keras_model'].save(f"{model_path}.keras")
    params = {
        'embedding_dim': model['embedding_dim'],
        'lstm_units': model['lstm_units'],
        'max_sequence_length': model['max_sequence_length'],
        'vocabulary_size': model['vocabulary_size'],
        'classes': model['label_encoder'].classes_.tolist()
    }
    with open(f"{model_path}_params.json", 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\nRNN model and its parameters saved to {os.path.dirname(model_path)}\n")

def load_rnn_model(model_path):
    """Load RNN model from saved files."""
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

    model_dir = r"C:\Users\giorg\Auto_Suggest\models"
    data_dir = r"C:\Users\giorg\Auto_Suggest\data\generated_sequences"

    if args.model == 'ngram':
        data_path = os.path.join(data_dir, 'n_gram_training_data.json')
        with open(data_path, 'r') as f:
            records = json.load(f)
        model_path = os.path.join(model_dir, 'ngram_model.json')

        # Build sequences from records and split into train and test sets
        sequences = [r['history'] + [r['next']] for r in records]
        train_seqs, test_seqs = train_test_split(sequences, test_size=0.2, random_state=42)

        if args.mode == 'train':
            print(f"Loaded {len(train_seqs)}/{len(sequences)} samples for training...")
            model = train_ngram_model(train_seqs)   # n=3 by default
            save_ngram_model(model, model_path)

        elif args.mode == 'eval':
            print(f"Loaded {len(test_seqs)}/{len(sequences)} samples for evaluation...")
            model = load_ngram_model(model_path)
            metrics = evaluate_ngram_model(model, test_seqs, k_values=[1, 2])

        elif args.mode == 'predict':
            model = load_ngram_model(model_path)
            example = ['json_normalize', 'fillna']
            predictions = predict_with_ngram(model, example)    # top_k=2 by default
            print(f"\nPredictions for {example}: {predictions}\n")

    elif args.model == 'rnn':
        data_path = os.path.join(data_dir, 'rnn_training_data.json')
        with open(data_path, 'r') as f:
            records = json.load(f)
        model_path = os.path.join(model_dir, 'rnn_model')

        # Build sequences from records and split into train and test sets
        sequences = [r['history_text'] + [r['next_text']] for r in records]
        rnn_train_seqs, rnn_test_seqs = train_test_split(sequences, test_size=0.2, random_state=42)

        if args.mode == 'train':
            print(f"Loaded {len(rnn_train_seqs)}/{len(sequences)} samples for training...")
            model = train_rnn_model(rnn_train_seqs)
            save_rnn_model(model, model_path)

        elif args.mode == 'eval':
            # Note: Evaluation is slow because we run model.predict() separately for each history step of a test sequence,
            # unlike training which is batched and vectorized for speed.
            print(f"\nLoaded {len(rnn_test_seqs)}/{len(sequences)} samples for evaluation...")
            model = load_rnn_model(model_path)
            metrics = evaluate_rnn_model(model, rnn_test_seqs, k_values=[1, 2])

        elif args.mode == 'predict':
            model = load_rnn_model(model_path)
            example = ['json_normalize', 'fillna']  # Same example as above!
            predictions = predict_with_rnn(model, example)  # top_k=2 by default
            print(f"\nPredictions for {example}: {predictions}\n")

if __name__ == '__main__':
    main()

