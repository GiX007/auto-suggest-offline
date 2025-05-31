# next_operation_predictor.py
#
# This script:
# 1. Loads trained models and data to train and evaluate next-operator prediction model.
# 2. Evaluates precision and recall for next-operator predictions (Table 11-like evaluation).
# 3. Provides functionality to predict the next operator for new samples.
#
# The evaluation is based on generated combined features, including:
# - RNN-based next-operator prediction probabilities.
# - Scores from operator-specific models: GroupBy, Join, Pivot, and Unpivot.
# - Combined numeric features for final MLP predictions.
#

import os
import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import classification_report
from src.utils.model_utils import load_model, numpy_to_list
from src.models.join_type_model import predict_join_type
from src.models.join_col_model import predict_join_columns
from src.models.groupby_model import predict_column_groupby_scores
from src.models.ngram_rnn_models import load_rnn_model, load_ngram_model, predict_with_ngram, predict_with_rnn


# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
data_dir = os.path.join(base_dir, "data")
test_data = os.path.join(data_dir, "test_data")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results", "metrics")
generated_data_path = os.path.join(data_dir, "generated_data", "combined_data.pkl")
seq_data_stats_path = os.path.join(data_dir, "generated_data", "sequence_data_statistics.json")
test_data_path = os.path.join(test_data, "next_op_test_data.pkl")
table_output_path = os.path.join(results_dir, "next_operation_prediction_methods_comparison.csv")

# Load all pretrained models
rnn_model_path = os.path.join(models_dir, "rnn_model")
rnn_model = load_rnn_model(rnn_model_path)

join_col_model_path = os.path.join(models_dir, "join_column_model.pkl")
join_col_model_data = load_model(join_col_model_path)
join_col_model, join_col_feature_names = join_col_model_data    # feature_names = list of column-level features used during training

join_type_model_path = os.path.join(models_dir, "join_type_model.pkl")
join_type_model_data = load_model(join_type_model_path)
join_type_model_obj, join_type_feature_names, join_type_label_encoder = join_type_model_data

groupby_model_path = os.path.join(models_dir, "groupby_column_model.pkl")
groupby_model_data = load_model(groupby_model_path)
groupby_model, groupby_feature_names = groupby_model_data  # feature_names = list of column-level features used during training

pivot_affinity_weights_model_path = os.path.join(models_dir, "pivot_affinity_weights_model.pkl")
pivot_affinity_weights_model = load_model(pivot_affinity_weights_model_path)

unpivot_affinity_weights_model_path = os.path.join(models_dir, "unpivot_affinity_weights_model.pkl")
unpivot_affinity_weights_model = load_model(unpivot_affinity_weights_model_path)


def single_operator_scores(tables,
                           join_col_model, join_col_feature_names,
                           join_type_model_obj, join_type_feature_names, join_type_label_encoder,
                           groupby_model, groupby_feature_names,
                           pivot_affinity_weights_model,
                           unpivot_affinity_weights_model):
    """
    Runs pre-trained models to compute operator scores and relevant join/pivot/unpivot recommendations.

    Args:
        tables: Either:
            - (df, ) for single-table operators (groupby, pivot, melt)
            - (left_df, right_df) for join
        join_col_model: Trained join column prediction model
        join_col_feature_names: Feature names for join column model
        join_type_model_obj: Trained join type prediction model
        join_type_feature_names: Feature names for join type model
        join_type_label_encoder: Label encoder for join type classes
        groupby_model: Trained groupby column prediction model
        groupby_feature_names: Feature names for groupby column model
        pivot_affinity_weights_model: Trained affinity weights model for pivot (AMPT)
        unpivot_affinity_weights_model: Trained affinity weights model for melt (CMUT)

    Returns:
        dict: Key operator scores and relevant keys.

    Note:
        - The pivot score is computed as: (number of recommended index columns + number of header columns) / total number of columns.
        - The unpivot score is computed as: number of recommended value columns / total number of columns.
        - These scores are heuristic measures of how much of the table is involved in pivot/unpivot structures.
    """
    from src.models.unpivot_model import solve_cmut
    from src.models.pivot_model import build_affinity_matrix, solve_ampt

    # Initialize join-related keys with default values to ensure every example has consistent fields.
    # This avoids KeyError in non-join cases where join keys are not predicted.
    op_scores = {
        "join_column_score": 0.0,
        "predicted_left_key": [],
        "predicted_right_key": [],
        "join_type_score": 0.0,
        "predicted_join_type": "unknown"
    }

    # Joins (predict join column keys and join type)
    if len(tables) == 2:
        left_df, right_df = tables

        # Drop any 'Unnamed' and 'Index' columns
        left_df = left_df.loc[:, ~left_df.columns.str.startswith("Unnamed:")]
        right_df = right_df.loc[:, ~right_df.columns.str.startswith("Unnamed:")]
        left_df = left_df.loc[:, left_df.columns != 'index']
        right_df = right_df.loc[:, right_df.columns != 'index']

        join_preds = predict_join_columns(join_col_model, join_col_feature_names, left_df, right_df, top_k=1, verbose=False)    # Returns a list of preds of top_k elements. if top_k_k=2:[  (['left_col1'], ['right_col1'], 0.98), ['left_col2'], ['right_col2'], 0.75)  ]
        # op_scores["join_column_top_k"] = join_preds

        if join_preds:
            left_join_keys, right_join_keys, join_column_score = join_preds[0]
        else:
            left_join_keys, right_join_keys, join_column_score = [], [], 0.0

        op_scores["join_column_score"] = join_column_score
        op_scores["predicted_left_key"] = left_join_keys
        op_scores["predicted_right_key"] = right_join_keys

        if join_preds:
            join_type_result = predict_join_type(join_type_model_obj, join_type_feature_names,
                                                 join_type_label_encoder, left_df, right_df,
                                                 left_join_keys, right_join_keys)
            join_type_score = join_type_result["confidence"]
            predicted_join_type = join_type_result["predicted_join_type"]
        else:
            join_type_score = 0.0
            predicted_join_type = "unknown"

        op_scores["join_type_score"] = join_type_score
        op_scores["predicted_join_type"] = predicted_join_type

    # Groupby
    df = tables[0]  # (one table here and for the rest)

    # Drop any 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    groupby_preds = predict_column_groupby_scores(groupby_model, groupby_feature_names, df)
    # groupby_preds = groupby_preds[:top_k]   # Get the top_k predictions
    groupby_scores_dict = {col: float(score) for col, score in groupby_preds}
    op_scores["groupby_column_score"] = max(groupby_scores_dict.values()) if groupby_scores_dict else 0.0
    op_scores["groupby_columns_and_scores"] = groupby_scores_dict

    # Pivot
    groupby_likelihoods = predict_column_groupby_scores(groupby_model, groupby_feature_names, df)
    dimension_columns = [entry[0] for entry in groupby_likelihoods if float(entry[1]) > 0.5]

    if len(dimension_columns) < 2:
        op_scores["pivot_score"] = 0.0
        op_scores["pivot_index_columns"] = []
        op_scores["pivot_header_columns"] = []
    else:
        affinity_matrix = build_affinity_matrix(df, dimension_columns, pivot_affinity_weights_model)
        index_columns, header_columns = solve_ampt(affinity_matrix)
        pivot_score = (len(index_columns) + len(header_columns)) / len(df.columns)
        op_scores["pivot_score"] = pivot_score
        op_scores["pivot_index_columns"] = index_columns
        op_scores["pivot_header_columns"] = header_columns

    # Unpivot (melt)
    dimension_columns = []
    measure_columns = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 15:
            measure_columns.append(col)
        else:
            dimension_columns.append(col)

    if len(dimension_columns) < 2:
        op_scores["unpivot_score"] = 0.0
        op_scores["unpivot_id_vars"] = []
        op_scores["unpivot_value_vars"] = []
    else:
        affinity_matrix = build_affinity_matrix(df, dimension_columns, unpivot_affinity_weights_model)
        value_vars = solve_cmut(affinity_matrix)
        id_vars = [col for col in dimension_columns if col not in value_vars]
        unpivot_score = len(value_vars) / len(df.columns)
        op_scores["unpivot_score"] = unpivot_score
        op_scores["unpivot_id_vars"] = id_vars
        op_scores["unpivot_value_vars"] = value_vars

    # Debug print of a sample of the returned dictionary
    # print("\nOperator scores for the first example:")
    # for k, v in op_scores.items():
    #     print(f"  {k}: {v}")

    # Isolate the 5 main scores as a feature vector
    feature_vector = [
        op_scores["join_column_score"],
        op_scores["join_type_score"],
        op_scores["groupby_column_score"],
        op_scores["pivot_score"],
        op_scores["unpivot_score"]
    ]

    # Ensure we do not have any NaN values by replacing them with zero
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)

    # Round all probabilities to 2 decimal places
    feature_vector = [round(score, 2) for score in feature_vector]
    # print("\nFeature vector for MLP:")
    # print(feature_vector)
    # print()

    return feature_vector, op_scores


def train_final_model():
    """
    Trains the final MLP model by combining RNN sequence predictions and operator scores
    (5 operator scores + 1 RNN score → 6-dimensional feature vector).

    Loads data from generated_data_dir, trains and saves final MLP to models_dir.
    """
    # Load generated data
    with open(generated_data_path, "rb") as f:
        X_y = pickle.load(f)
    X_data, y_data = X_y
    print(f"Loaded combined data: {len(X_data)} examples")

    # Remember our data like X_data = [(history_sequence, table), ...] and y_data = [next_op, ...]
    # See how the first item looks like
    # print("\nFirst item of X_data:", X_data[0])
    # print("Type of first item:", type(X_data[0]))

    # Build numeric features for final MLP model (6-dimensional: 5 operator scores + 1 RNN probability)
    final_features = []
    final_labels = []
    original_sequences = []  # To keep operator sequences for RNN/N-gram evaluation

    print("\nBuilding final MLP model input vectors by combining operator scores and RNN probabilities...")
    for i, data in enumerate(X_data):
        sequence = data[0]
        if len(data) == 2:
            # Single-table operators
            tables = (data[1],)
        elif len(data) == 3:
            # Join operator (2 tables)
            tables = (data[1], data[2])
        else:
            raise ValueError(f"Unexpected data format at index {i}")

        # 1. Compute operator scores
        single_operator_vector, _ = single_operator_scores(
            tables,
            join_col_model, join_col_feature_names,
            join_type_model_obj, join_type_feature_names, join_type_label_encoder,
            groupby_model, groupby_feature_names,
            pivot_affinity_weights_model, unpivot_affinity_weights_model
        )

        # 2. Compute RNN-based sequence score
        rnn_prob = predict_with_rnn(rnn_model, sequence)
        # print(f"rnn_prob: {rnn_prob} (type: {type(rnn_prob)})")
        rnn_prob = rnn_prob[0][1]  # Grab the 0th tuple, then the probability
        rnn_prob = round(rnn_prob, 2)

        # 3. Build final 6-dimensional feature vector
        feature_vector = single_operator_vector + [rnn_prob]

        final_features.append(feature_vector)
        final_labels.append(y_data[i])
        original_sequences.append(sequence)

        # Debug print for first few examples
        # if i < 5:
        #     print(f"\nExample {i+1}:")
        #     print("  Sequence:", sequence)
        #     print("  Feature vector:", feature_vector)
        #     print("  Label:", y_data[i])

    # Convert to arrays
    X_final = np.array(final_features)
    y_final = np.array(final_labels)

    # Get the indices once
    indices = np.arange(len(X_final))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

    # Split numeric features
    X_train, X_test = X_final[train_indices], X_final[test_indices]
    y_train, y_test = y_final[train_indices], y_final[test_indices]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the fitted scaler to models_dir for later predictions (like in eval or predict mode)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    with open(os.path.join(models_dir, "final_mlp_label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

    # Create final test samples with operator sequences + tables
    test_data_with_tables = [X_data[i] for i in test_indices]

    # Save them for later evaluations (like Single-Operators)
    with open(os.path.join(test_data, "next_op_test_data_with_tables.pkl"), "wb") as f:
        pickle.dump(test_data_with_tables, f)

    # Save numeric train/test splits (MLP)
    with open(os.path.join(test_data, "next_op_train_data.pkl"), "wb") as f:
        pickle.dump((X_train, y_train_encoded), f)
    with open(os.path.join(test_data, "next_op_test_data.pkl"), "wb") as f:
        pickle.dump((X_test, y_test_encoded), f)

    # Split operator sequences using the **same indices**
    X_train_seq = [original_sequences[i] for i in train_indices]
    X_test_seq = [original_sequences[i] for i in test_indices]
    y_train_seq = [y_final[i] for i in train_indices]
    y_test_seq = [y_final[i] for i in test_indices]

    # Save operator sequence samples (N-gram/RNN)
    train_samples = [{"history": history, "next": next_op} for history, next_op in zip(X_train_seq, y_train_seq)]
    test_samples = [{"history": history, "next": next_op} for history, next_op in zip(X_test_seq, y_test_seq)]

    with open(os.path.join(test_data, "next_op_train_samples.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(test_data, "next_op_test_samples.pkl"), "wb") as f:
        pickle.dump(test_samples, f)
    # print("\nNumeric features, operator sequence samples and tables saved successfully!")


    # Train final MLP
    print("\nTraining the final model...")
    start_time = time.time()

    param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (64,), (128,)],
        'learning_rate_init': [0.0001, 0.001, 0.01],
        'alpha': [0.0001, 0.001, 0.01],
        'solver': ['adam', 'lbfgs']
    }

    # Grid Search with no early stopping
    mlp = MLPClassifier(max_iter=500, random_state=42)
    grid_search = GridSearchCV(mlp,
                               param_grid,
                               cv=3,
                               scoring='accuracy',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Extract the best trained model
    mlp = grid_search.best_estimator_
    # best_params = grid_search.best_params_
    # print(f"Best hyperparameters: {best_params}")

    # And retrain with the best configuration found
    # mlp = MLPClassifier(**best_params, max_iter=500,
    #                     early_stopping=True,    # enable early stopping for final training
    #                     n_iter_no_change=10,
    #                     validation_fraction=0.1,
    #                     random_state=42)
    # mlp.fit(X_train, y_train)

    # Final chosen model
    # mlp = MLPClassifier(
    #     hidden_layer_sizes=(128,),
    #     alpha=0.0001,
    #     learning_rate_init=0.01,
    #     max_iter=500,
    #     solver='adam',
    #     random_state=42,
    #     early_stopping=True,
    #     n_iter_no_change=10,
    #     validation_fraction=0.1
    # )
    # mlp.fit(X_train, y_train)

    end_time = time.time()
    total_training_time = round(end_time - start_time, 2)

    print(f"\nModel training completed in {total_training_time} seconds")
    # print(f"Best hyperparameters: {grid_search.best_params_}")
    # print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    print(f"Trained model: MLPClassifier with {mlp.hidden_layer_sizes} hidden layers and {mlp.max_iter} iterations")


    # Calculate macro-averaged metrics (multi-class) for both training and test sets (checking for overfitting or underfitting)
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    print("\nTraining vs Test Metrics:")
    print(f"Accuracy: Train = {train_accuracy:.4f}, Test = {test_accuracy:.4f}")
    print(f"Precision (macro): Train = {train_precision:.4f}, Test = {test_precision:.4f}")
    print(f"Recall (macro): Train = {train_recall:.4f}, Test = {test_recall:.4f}")
    print(f"F1-score (macro): Train = {train_f1:.4f}, Test = {test_f1:.4f}")
    # print("\n[INFO] Classification Report:")
    # print(classification_report(y_test, y_test_pred))

    # Prepare metrics dictionary
    metrics_dict = {
        'operator': 'next_operation_predictor',  # final task
        'mode': 'training/evaluation',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'samples': int(len(y_train) + len(y_test)),
        'train_examples': int(len(y_train)),
        'test_examples': int(len(y_test)),
        'model_type': 'MLPClassifier',
        'hidden_layers': mlp.hidden_layer_sizes,
        'max_iterations': int(mlp.max_iter),
        'training_time': float(total_training_time),
        'num_features': int(X_final.shape[1]),
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_precision_macro': float(train_precision),
        'test_precision_macro': float(test_precision),
        'train_recall_macro': float(train_recall),
        'test_recall_macro': float(test_recall),
        'train_f1_macro': float(train_f1),
        'test_f1_macro': float(test_f1)
    }

    # All metrics to a JSON file
    metrics_path = 'results/metrics/all_operators_metrics.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Converts numpy types to native Python types (important for JSON!)
    metrics_dict = numpy_to_list(metrics_dict)

    # Append new metrics to existing list or create new
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(metrics_dict)

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)


    # Save the final model
    final_model_path = os.path.join(models_dir, "final_mlp_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(mlp, f)
    print(f"\nFinal model saved to 'models' directory")


    print("\nFinal model trained successfully!\n")


def calc_prec_recall(y_true, y_pred_top_k, top_k):
    """
    Calculates precision@k and recall@k for next-operator prediction for all k in [1, 2, ..., top_k].

    This function is consistent with the definitions in Table 11 of the
    Auto-Suggest paper. Here’s how it works:

    - For k=1:
        Each example has exactly one prediction and one true label.
        Precision@1 = Recall@1 = fraction of examples where prediction == true label.

    - For k=2 (or higher):
        Precision@k and Recall@k are not the same!

        - Precision@k = average fraction of top-k predictions that are correct.
          (total number of correct predictions across all examples) divided by (number of examples).
          Essentially: how many of the k guesses were correct.

        - Recall@k = fraction of examples where the true label is in the top-k predictions.
          (number of examples where true label is present in top-k predictions) divided by (number of examples).
          Essentially: whether at least one of the guesses was correct.

    For single-label classification, this general approach matches what’s shown in Table 11.

    Args:
        y_true : Ground-truth labels, np.ndarray of shape (n_samples,).
        y_pred_top_k : Top-k predictions for each example, sorted in descending confidence order, np.ndarray of shape (n_samples, k)
        top_k : The maximum k value for which to compute precision and recall (e.g., 3 for k=1,2,3).

    Returns:
        precision_at_k : Average fraction of correct predictions in top-k predictions.
        recall_at_k : Fraction of examples where the true label is present in the top-k predictions.
    """
    metrics = {}
    n_samples = len(y_true)

    k_values = list(range(1, top_k + 1))

    for k in k_values:
        preds_k = y_pred_top_k[:, :k]
        # Recall@k: fraction of examples where true label is in top-k
        hits = np.any(preds_k == y_true[:, None], axis=1)
        recall_at_k = np.mean(hits)

        # Precision@k: correct predictions / (n_samples)    # (k * n_samples)
        correct_preds = (preds_k == y_true[:, None]).sum()
        precision_at_k = correct_preds / n_samples

        # Round to 2 decimals
        precision_at_k = round(precision_at_k, 2)
        recall_at_k = round(recall_at_k, 2)

        metrics[k] = (precision_at_k, recall_at_k)

    return metrics


def eval_final_model(top_k=2):
    """
    Evaluation function for 'eval' mode that produces Table 10
    (distribution of operators in data flows) as described in the paper.
    """
    # 1. Display Table 10

    # Load the JSON file with sequence data statistics
    with open(seq_data_stats_path, 'r') as f:
        data_stats = json.load(f)

    # Extract operator counts and calculate the total number of operator invocations
    operator_counts = data_stats['operator_counts']
    total_operator_count = sum(operator_counts.values())

    # Calculate percentage for each operator
    operator_distribution = []
    for operator, count in operator_counts.items():
        percentage = (count / total_operator_count) * 100
        operator_distribution.append({
            "operator": operator,
            "count": count,
            "percentage %": round(percentage, 2)
        })

    # Create a DataFrame similar to Table 10
    df_operator_dist = pd.DataFrame(operator_distribution)
    df_operator_dist.sort_values(by="count", ascending=False, inplace=True)
    df_operator_dist.reset_index(drop=True, inplace=True)

    # Display Table 10
    print("Table 10: Operator Distribution in Data Flows")
    print(tabulate(df_operator_dist, headers="keys", tablefmt="grid", showindex=False))

    # 2. Display Table 11

    # Load test data (for MLP/Random baseline)
    with open(test_data_path, "rb") as f:
        X_test, y_test = pickle.load(f)

    # Load test samples (operator sequences)
    with open(os.path.join(test_data, "next_op_test_samples.pkl"), "rb") as f:
        test_samples = pickle.load(f)

    # Load test data with tables (for Single-Operators)
    with open(os.path.join(test_data, "next_op_test_data_with_tables.pkl"), "rb") as f:
        test_data_with_tables = pickle.load(f)
    y_test_ops = [sample["next"] for sample in test_samples]

    # Load N-gram model
    with open(os.path.join(models_dir, "ngram_model.json"), "r") as f:
        ngram_model = json.load(f)
    n, counts = ngram_model["n"], ngram_model["counts"]

    # Load final MLP model
    with open(os.path.join(models_dir, "final_mlp_model.pkl"), "rb") as f:
        final_mlp_model = pickle.load(f)
    with open(os.path.join(models_dir, "final_mlp_label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    methods = ["Auto-Suggest", "RNN", "N-gram", "Single-Operators", "Random"]
    results = []

    for method in methods:
        top_preds = []
        y_true_ops = []

        if method == "Auto-Suggest":
            # MLP predictions (already numeric y_test)
            y_prob = final_mlp_model.predict_proba(X_test)
            top_preds = np.argsort(y_prob, axis=1)[:, -top_k:]
            top_preds = np.fliplr(top_preds)
            y_true_ops = y_test

        elif method == "RNN":
            for sample in test_samples:
                history, true_next = sample["history"], sample["next"]
                predictions = predict_with_rnn(rnn_model, history, top_k=top_k)
                top_preds.append([op for op, _ in predictions])
                y_true_ops.append(true_next)

        elif method == "N-gram":
            for sample in test_samples:
                history, true_next = sample["history"], sample["next"]
                context = tuple(history[-(n - 1):])
                candidates = {}
                while len(context) > 0:
                    context_key = str(context)
                    if context_key in counts:
                        candidates = counts[context_key]
                        break
                    context = context[1:]
                if not candidates and "()" in counts:
                    candidates = counts["()"]
                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                top_ops = [op for op, _ in sorted_candidates[:top_k]]
                top_preds.append(top_ops)
                y_true_ops.append(true_next)

        elif method == "Single-Operators":
            for idx, data in enumerate(test_data_with_tables):
                sequence = data[0]
                true_next = y_test_ops[idx]
                if len(data) == 2:
                    tables = (data[1],)
                elif len(data) == 3:
                    tables = (data[1], data[2])
                else:
                    raise ValueError("Unexpected data format")

                single_operator_vector, _ = single_operator_scores(
                    tables,
                    join_col_model, join_col_feature_names,
                    join_type_model_obj, join_type_feature_names, join_type_label_encoder,
                    groupby_model, groupby_feature_names,
                    pivot_affinity_weights_model, unpivot_affinity_weights_model
                )

                # Construct 4-operator prediction scores
                join_score = single_operator_vector[0] # + single_operator_vector[1]
                groupby_score = single_operator_vector[2]
                pivot_score = single_operator_vector[3]
                unpivot_score = single_operator_vector[4]

                operator_names = ["join", "groupby", "pivot", "unpivot"]
                operator_scores = [join_score, groupby_score, pivot_score, unpivot_score]

                # Sort and get top-k predictions
                top_indices = np.argsort(operator_scores)[-top_k:][::-1]
                top_ops = [operator_names[idx] for idx in top_indices]

                top_preds.append(top_ops)
                y_true_ops.append(true_next)


        elif method == "Random":
            operator_names = ["join", "groupby", "pivot", "unpivot"]
            for true_next in y_test_ops:
                top_ops = list(np.random.choice(operator_names, size=top_k, replace=False))
                top_preds.append(top_ops)
                y_true_ops.append(true_next)

        # Debug print
        # print(f"\nMethod: {method}")

        # Decode true label if numeric
        # if isinstance(y_true_ops[0], (int, np.integer)):
        #     true_label_decoded = label_encoder.classes_[y_true_ops[0]]
        # else:
        #     true_label_decoded = y_true_ops[0]
        # print("True label:", true_label_decoded)

        # Decode top-k predictions if numeric
        # if isinstance(top_preds[0][0], (int, np.integer)):
        #     decoded_top_preds = label_encoder.inverse_transform(top_preds[0])
        #     top_1 = decoded_top_preds[:1].tolist()
        #     top_2 = decoded_top_preds[:2].tolist()
        # else:
        #     top_1 = top_preds[0][:1]
        #     top_2 = top_preds[0][:2]
        #
        # print("Top-1 example:", top_1)
        # print("Top-2 example:", top_2)

        # Calculate precision/recall metrics ensuring we have numpy arrays
        if not isinstance(y_true_ops, np.ndarray):
            y_true_ops = np.array(y_true_ops)
        if not isinstance(top_preds, np.ndarray):
            top_preds = np.array(top_preds)
        metrics = calc_prec_recall(y_true_ops, top_preds, top_k=top_k)

        # Build row for Table 11
        row = [method]
        for k in range(1, top_k + 1):
            p, r = metrics[k]
            row.extend([f"{p:.3f}", f"{r:.3f}"])
        results.append(row)

    # Display Table 11
    # headers = ["method"] + [f"prec@{k}" for k in range(1, top_k + 1)] + [f"recall@{k}" for k in range(1, top_k + 1)]

    # Simpler headers: just method, precision@1, recall@1
    simple_headers = ["method", "precision", "recall"]

    print("\nTable 11: Precision and Recall for Next Operator Prediction")
    # print(tabulate(results, headers=headers, tablefmt="grid"))

    simple_results = [[row[0], row[1], row[-2]] for row in results]
    print(tabulate(simple_results, headers=simple_headers, tablefmt="grid"))

    # Save the table as CSV
    # df_results = pd.DataFrame(results, columns=headers)
    df_results = pd.DataFrame(simple_results, columns=simple_headers)
    df_results.to_csv(table_output_path, index=False)
    print(f"\nMetrics have been saved to the 'results' directory\n")

"""
Explanation of results with illustrative examples
---------------------------------------

In next-operator prediction, we only have **one true next operator label** per test sample.
Therefore:
- precision@k and recall@k are calculated by checking if this one label is in the top-k predictions.
- If top-1 already contains the true label, top-2 doesn't add any new correct prediction.
- If top-1 doesn't contain the true label, top-2 also won't improve unless it contains the true label.

Example for next-operator prediction:
--------------------------------------
Imagine 3 test samples:
| Sample | True label | Top-1 prediction | Top-2 predictions          |
|--------|------------|------------------|----------------------------|
| 1      | melt       | pivot            | pivot, groupby             |
| 2      | groupby    | groupby          | groupby, pivot             |
| 3      | join       | groupby          | groupby, pivot             |

- Sample 1: top-1 and top-2 do not contain 'melt' → 0
- Sample 2: top-1 and top-2 contain 'groupby' → 1
- Sample 3: top-1 and top-2 do not contain 'join' → 0

Precision@1 = (0 + 1 + 0) / 3 = 0.33  
Recall@1    = (0 + 1 + 0) / 3 = 0.33  
Precision@2 = (0 + 1 + 0) / 3 = 0.33  
Recall@2    = (0 + 1 + 0) / 3 = 0.33

Thus, **precision@1 == recall@1 == precision@2 == recall@2**.

Example for join column / groupby column prediction:
------------------------------------------------------
Imagine 3 test samples with **multiple true columns**:
| Sample | True join columns | Top-1 predictions  | Top-2 predictions         |
|--------|--------------------|---------------------|---------------------------|
| 1      | [A, B]             | [C]                | [C, B]                    |
| 2      | [X]                | [X]                | [X, Y]                    |
| 3      | [P, Q]             | [R]                | [P, R]                    |

- Sample 1:
    - top-1: no correct → precision@1=0, recall@1=0
    - top-2: 1 correct (B) / 2 → precision@2=0.5, recall@2=0.5 (1 correct of 2 true)
- Sample 2:
    - top-1: 1 correct (X) / 1 → precision@1=1, recall@1=1
    - top-2: 1 correct (X) / 2 → precision@2=0.5, recall@2=1 (1 correct of 1 true)
- Sample 3:
    - top-1: no correct → precision@1=0, recall@1=0
    - top-2: 1 correct (P) / 2 → precision@2=0.5, recall@2=0.5 (1 correct of 2 true)

Average metrics:
- precision@1 = (0 + 1 + 0) / 3 = 0.33
- precision@2 = (0.5 + 0.5 + 0.5) / 3 = 0.5
- recall@1    = (0 + 1 + 0) / 3 = 0.33
- recall@2    = (0.5 + 1 + 0.5) / 3 = 0.67

This shows **recall@2 > recall@1** because adding more predictions can help find more correct columns.

Summary:
---------
This difference explains why in the paper, for join column and groupby column predictions, precision and recall @k increase as k increases — because there are multiple correct labels.

In the paper’s Table 11, they also get different precision and recall values for k=1 and k=2 because they treat next-operator prediction as a **multi-label task** — multiple plausible next operators per test sample.
For each test sample, they consider several potential next operators as correct (like groupby or pivot), not just one. This means that top-2 predictions can include additional true operators beyond the top-1 prediction, leading to higher recall and precision for k=2.

In contrast, in our current implementation, we only consider **one true next operator label per sample**. Therefore, precision and recall @k do not improve with k: once the top-1 prediction is correct, adding more predictions does not increase the number of true positives.  
As a result, for single-label next-operator prediction, precision@1 == recall@1 == precision@2 == recall@2 for each method.

To get similar results as in Table 11 of the paper, we would need to transform our implementation to a **multi-operator prediction task**:
- Annotate the dataset with **multiple true next operators** for each test sample (e.g., "groupby" and "pivot" can both be reasonable next operations).
- Adjust our evaluation logic and metric calculations to account for multiple true next-operators per test sample.

This multi-label approach would let precision and recall improve with larger k values, reflecting the richer set of plausible next-operator predictions, just as reported in the paper.
"""


def predict_next_operation(final_mlp_model, label_encoder, history_sequence, tables, join_col_model, join_col_feature_names, join_type_model_obj, join_type_feature_names, join_type_label_encoder,
                           groupby_model, groupby_feature_names, pivot_affinity_weights_model, unpivot_affinity_weights_model, rnn_model, scaler):
    """
    Predicts the next operator given the history sequence and table snapshot.

    Args:
        final_mlp_model: Trained final MLP model.
        label_encoder: Fitted label encoder for operator names.
        history_sequence: List of past operators.
        tables: Current table snapshot(s): (df,) or (left_df, right_df).
        ... (all operator-specific models and features)
        scaler: The scaler used to standardize the numeric feature vectors.

    Returns:
        The predicted next operator (string).
    """
    # 1. Compute operator-specific scores (5 operator scores)
    single_operator_vector, _ = single_operator_scores(
        tables,
        join_col_model, join_col_feature_names,
        join_type_model_obj, join_type_feature_names, join_type_label_encoder,
        groupby_model, groupby_feature_names,
        pivot_affinity_weights_model, unpivot_affinity_weights_model
    )

    # 2. Compute RNN-based sequence score
    rnn_prob = predict_with_rnn(rnn_model, history_sequence)
    rnn_prob = rnn_prob[0][1]  # Extract probability
    rnn_prob = round(rnn_prob, 2)

    # 3. Build final 6-dimensional feature vector
    feature_vector = single_operator_vector + [rnn_prob]

    # Reshape feature vector to 2D array with shape (1, num_features)
    # because scikit-learn models expect 2D input even for a single sample.
    feature_vector = np.array(feature_vector).reshape(1, -1)

    # 4. Standardize using the scaler
    feature_vector_scaled = scaler.transform(feature_vector)

    # Debug print to check feature vector
    print("Final feature vector (MLP input):", feature_vector_scaled)

    # 5. Predict probabilities for all possible next operators
    y_prob = final_mlp_model.predict_proba(feature_vector_scaled)[0]
    top_index = np.argmax(y_prob)

    # 6. Convert numeric prediction back to operator name
    predicted_operator = label_encoder.inverse_transform([top_index])[0]

    # Print all probabilities sorted from highest to lowest
    operator_names = label_encoder.classes_
    probs_sorted = sorted(zip(operator_names, y_prob), key=lambda x: x[1], reverse=True)
    print("\nAll predicted probabilities:")
    for op, prob in zip(operator_names, y_prob):
        print(f"{op}: {prob:.4f}")

    return predicted_operator


# --------------
# Command Line Interface
# --------------

def main():
    """
       Main function to run single-operator predictions with command line args.
    """
    parser = argparse.ArgumentParser(description='Auto-Suggest: Next Operation Prediction')
    parser.add_argument('--mode', choices=['train', 'eval', 'predict', 'all'], default='train', help='Mode of operation (default: train)')
    parser.add_argument("--input_file", type=str, default=None, help="Path to CSV file for prediction mode")
    parser.add_argument("--history", type=str, default=None, help="Comma-separated list of previous operators (e.g., 'groupby,pivot')")
    args = parser.parse_args()

    mode = args.mode

    if mode == 'all':
        print("\n" + "=" * 80)
        print("Next Operation Prediction")
        print("=" * 80)
        print()

        # Train
        train_final_model()

        # Evaluate
        eval_final_model()

        # Predict - we’ll just use a dummy file and dummy history for demonstration
        input_file = 'data/test_data/unpivot_product_sales.csv'
        history_sequence = ['dropna', 'merge', 'pivot']  # Example previous ops

        with open(os.path.join(models_dir, "final_mlp_model.pkl"), "rb") as f:
            final_mlp_model = pickle.load(f)
        with open(os.path.join(models_dir, "final_mlp_label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)

        df = pd.read_csv(input_file)
        tables = (df,)

        print("Loaded table shape:", df.shape)
        print("Top 3 rows:\n", df.head(3))

        predicted_op = predict_next_operation(
            final_mlp_model, label_encoder,
            history_sequence, tables,
            join_col_model, join_col_feature_names,
            join_type_model_obj, join_type_feature_names, join_type_label_encoder,
            groupby_model, groupby_feature_names,
            pivot_affinity_weights_model, unpivot_affinity_weights_model,
            rnn_model, scaler
        )

        print(f"\nPredicted next operator: {predicted_op}\n")

    else:
        # Existing logic for a single mode


        # Update the header based on the mode
        if mode == 'eval':
            print("\n" + "=" * 80)
            print("Evaluation Mode: Next Operation Prediction")
            print("=" * 80)
            print()
        else:
            print("\n" + "=" * 80)
            print(f"{mode.capitalize()} Mode: Next Operation Prediction")
            print("=" * 80)
            print()

        # TRAIN mode
        if mode == 'train':
            train_final_model()

        # EVAL mode
        elif mode == 'eval':
            eval_final_model()

        # PREDICT mode
        elif mode == 'predict':
            print("Predicting next operation for a sample...")

            # Load final MLP model and label encoder
            with open(os.path.join(models_dir, "final_mlp_model.pkl"), "rb") as f:
                final_mlp_model = pickle.load(f)
            with open(os.path.join(models_dir, "final_mlp_label_encoder.pkl"), "rb") as f:
                label_encoder = pickle.load(f)

            # Load scaler just like during training
            with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
                scaler = pickle.load(f)

            # Load the CSV table
            # csv_path = os.path.join(test_data, "dummy_table.csv")
            # df = pd.read_csv(csv_path)
            df = pd.read_csv(args.input_file)
            tables=(df,)    # or (left_df, right_df) for joins

            # Print shape and top 3 rows
            print("Loaded table shape:", df.shape)
            print("Top 3 rows:\n", df.head(3))

            # Parse history string into a list
            history_sequence = [op.strip() for op in args.history.split(",")]   # ["groupby", "pivot"]

            # Inputs: tables, history_sequence

            #Predict
            predicted_op = predict_next_operation(
                final_mlp_model, label_encoder,
                history_sequence, tables,
                join_col_model, join_col_feature_names,
                join_type_model_obj, join_type_feature_names, join_type_label_encoder,
                groupby_model, groupby_feature_names,
                pivot_affinity_weights_model, unpivot_affinity_weights_model,
                rnn_model, scaler
            )

            print(f"\nPredicted next operator: {predicted_op}\n")


if __name__ == "__main__":
    main()
