# src/main.py
#
# Main driver script for Auto-Suggest implementation
# Based on the "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks" paper
#
# This module:
# 1. Provides functions for training and evaluating each operator
# 2. Sets up command-line interface for the project
# 3. Coordinates the overall workflow
# 4. Creates necessary package structure
#

import os
import pickle
import argparse
import traceback
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from src.data.sample_loader import load_operator_samples

# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
test_data = os.path.join(base_dir, "data", "test_data")


def run_join_prediction(data_dir, models_dir, mode='train', left_file=None, right_file=None, top_k=2):
    """
    Executes the Join Prediction Pipeline

    This function provides a complete pipeline for the join prediction task, including:
    1. Training models for join column and join type prediction
    2. Evaluating performance on test data
    3. Making predictions on new 'unseen' tables

    The function has three modes:
    - 'train': Train join column and join type models using samples from data_dir
    - 'eval': Evaluate trained models on test samples and compute metrics
    - 'predict': Apply trained models to make predictions on new tables

    Args:
        data_dir: Directory containing training samples
        models_dir: Directory to save/load models
        mode: Operation mode ('train', 'eval', or 'predict')
        left_file: Path to left table CSV for prediction (only in predict mode)
        right_file: Path to right table CSV for prediction (only in predict mode)
        top_k: Number of top recommendations to display in prediction mode

    Returns:
        Boolean indicating success or failure of the operation

    Note:
        For all modes, this function runs for both the join column and join type models.
    """
    # Import local necessary modules
    from src.utils.model_utils import save_model, load_model, recommend_joins, display_join_recommendations
    from src.models.join_col_model import process_join_samples, prepare_join_data, train_join_column_model, \
        evaluate_join_column_model
    from src.models.join_type_model import prepare_join_type_training_data, train_join_type_model, \
        evaluate_join_type_model

    # Update the header based on the mode
    if mode == 'eval':
        print("\n" + "=" * 80)
        print("Predict Single Operators -  Join (Section 6.5)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("Join Prediction (Section 4.1)")
        print("=" * 80)

    # Training mode
    if mode == 'train':
        print("\nTraining join prediction models...")

        # Load join samples
        join_dir = os.path.join(data_dir, "merge")
        join_samples = load_operator_samples(join_dir, 'join')  # returns a list of dictionaries

        # Check if any samples were loaded
        if not join_samples:
            print(f"Error: No join samples found in {join_dir}. Check your data directory.")
            return False

        # Process samples
        processed_join_samples = process_join_samples(join_samples) # returns a list of dictionaries

        # Check if we have any valid samples
        if len(processed_join_samples) == 0:
            print("Error: No valid join samples found after processing.")
            return False

        # First split to get test samples (take into account imbalanced join type samples)

        # Stratify the train-test split based on join type labels to ensure all join types (e.g., inner, left, right, outer)
        # are represented proportionally in both the training and test sets. This prevents label imbalance issues during evaluation.
        join_type_labels = [s['join_type'] for s in processed_join_samples]
        label_counts = Counter(join_type_labels)    # Count how many samples per join type

        # (We have just one sample of 'right' join type)
        # counts = Counter([s.get('join_type', 'unknown') for s in processed_join_samples])
        # print("Join type counts:", counts)

        # Filter out join types with only 1 sample (singleton classes)
        singleton_classes = [cls for cls, count in label_counts.items() if count < 2]
        if singleton_classes:
            print("Warning: Excluding join types with only 1 sample (singleton classes) from join models training:")
            print(f"  Excluded classes: {singleton_classes}\n")

        # Remove samples with these singleton join types
        filtered_samples = [s for s in processed_join_samples if s['join_type'] not in singleton_classes]

        # Confirm filtered dataset size
        # print(f"Filtered dataset size after removing singleton join types: {len(filtered_samples)}")

        # Updated join type labels
        join_type_labels = [s['join_type'] for s in filtered_samples]

        tmp_samples, test_samples = train_test_split(filtered_samples, test_size=0.1, random_state=42, stratify=join_type_labels)

        # Second split to get train and val samples
        join_type_labels_train_val = [s['join_type'] for s in tmp_samples]

        # Second split: 90/10 = ~11% for val (~10 examples)
        train_samples, val_samples = train_test_split(tmp_samples, test_size=0.1111, random_state=42, stratify=join_type_labels_train_val)   # 10 / 90 ≈ 0.1111

        print(f"Split {len(processed_join_samples)} samples into {len(train_samples)} train, {len(val_samples)} validation and {len(test_samples)} test samples")
        #print(test_samples[0]) # Has also the same form as processed_samples

        # Save train, val and test data for later use
        with open(os.path.join(test_data, "join_train_samples.pkl"), "wb") as f:
            pickle.dump(train_samples, f)

        with open(os.path.join(test_data, "join_val_samples.pkl"), "wb") as f:
            pickle.dump(val_samples, f)

        with open(os.path.join(test_data, "join_test_samples.pkl"), "wb") as f:
            pickle.dump(test_samples, f)
        #print(f"Saved join train, validation and test samples to {test_data}")


        # 1. Train Join Column Prediction Model
        print("\n--- Training Join Column Prediction Model ---")

        # Prepare training data (extract numeric features and labels from train_samples)
        X_join_column_train, feature_names, y_join_column_train = prepare_join_data(train_samples)
        X_join_column_val, _, y_join_column_val = prepare_join_data(val_samples)

        # Train model
        join_col_model, col_feature_names = train_join_column_model(X_join_column_train, y_join_column_train, X_join_column_val, y_join_column_val, feature_names)

        # Check if model training was successful
        if join_col_model is None:
            print("Error: Join column model training failed.")
            return False

        # Save model, ensuring we're using the correct models_dir
        model_path = os.path.join(models_dir, "join_column_model.pkl")
        save_model((join_col_model, col_feature_names), model_path)

        # 2. Train Join Type Prediction Model
        print("\n--- Training Join Type Prediction Model ---")

        # Prepare join type training and validation data
        X_join_type_train, feature_names_type, y_join_type_train = prepare_join_type_training_data(train_samples)
        X_join_type_val, _, y_join_type_val = prepare_join_type_training_data(val_samples)

        # Train join type model
        type_model, type_feature_names, type_label_encoder = train_join_type_model(X_join_type_train, y_join_type_train, X_join_type_val, y_join_type_val, feature_names_type)

        # Check if model training was successful
        if type_model is None:
            print("Error: Join type model training failed.")
            return False

        # Save join type model - ensure we're using the correct models_dir
        model_path = os.path.join(models_dir, "join_type_model.pkl")
        save_model((type_model, type_feature_names, type_label_encoder), model_path)

        print("\nJoin models trained successfully!\n")
        return True

    # Evaluation mode
    elif mode == 'eval':
        print("\nEvaluating join prediction models...")
        print(f"Using top k = {top_k} predictions for evaluation")

        # Load test data
        test_path = os.path.join(test_data, "join_test_samples.pkl")

        if not os.path.exists(test_path):
            print("Error: Test data not found. Run training first.")
            return False

        with open(test_path, "rb") as f:
            test_samples = pickle.load(f)
        #print(test_samples[0])    # Check if it is the same as defined above before saving and loading

        print(f"Using {len(test_samples)} test samples for evaluation\n")

        relative_path = os.path.relpath(test_data, base_dir)
        print(f"Test data loaded from {os.path.join(relative_path, 'join_test_samples.pkl')}")

        # Load models
        try:
            # Load join column model
            column_model_path = os.path.join(models_dir, "join_column_model.pkl")
            column_model_data = load_model(column_model_path)
            col_model, col_feature_names = column_model_data

            # Load join type model
            type_model_path = os.path.join(models_dir, "join_type_model.pkl")
            type_model_data = load_model(type_model_path)
            type_model, type_feature_names, type_label_encoder = type_model_data
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you have trained the models first.")
            return False

        print("\n--- Evaluating Join Models on Test Set ---")

        # For join column model, we need to generate candidate join pairs and produce feature/label matrix.
        # This happens inside the evaluate_join_column_model function through the predict_join_columns function.
        # For join type model, no need to generate candidates — we directly use test_samples as they are, treating each sample as a multi-class classification example.

        # 1. Evaluate join column model with specified k values
        col_metrics = evaluate_join_column_model(col_model, col_feature_names, test_samples, top_k)

        print("\nJoin Column Evaluation Results:")
        for metric, value in col_metrics.items():
            print(f"  {metric}: {value:.2f}")


        # 2. Evaluate join type model
        type_metrics = evaluate_join_type_model(type_model, type_feature_names, type_label_encoder, test_samples)

        print("\nJoin Type Evaluation Results:")
        for metric, value in type_metrics.items():
            # Safely print evaluation metrics: format floats to 2 decimal places, print others (e.g., strings) as-is
            if isinstance(value, float):
                print(f"  {metric}: {value:.2f}")
            else:
                print(f"  {metric}: {value}")

        print("\nMetrics and figures have been saved to the 'results' directory")

        print("\nJoin models evaluated successfully!\n")
        return True

    # Predict mode - for new 'unseen' tables
    elif mode == 'predict':
        print("\nPredicting join columns and types for new tables...")

        # Check if files are provided
        if not left_file or not right_file:
            print("Error: Both left and right tables must be provided for prediction.")
            return False

        # Check if files exist
        for file, name in [(left_file, "Left"), (right_file, "Right")]:
            if not os.path.exists(file):
                print(f"Error: {name} file '{file}' not found.")
                return False

        # Run prediction using join pipeline
        try:
            # Read tables with pandas
            left_table = pd.read_csv(left_file)
            right_table = pd.read_csv(right_file)

            print(f"\nLoaded tables:")
            print(f"Left table: {left_table.shape[0]} rows × {left_table.shape[1]} columns")
            print(f"Right table: {right_table.shape[0]} rows × {right_table.shape[1]} columns\n")

            # Show sample rows
            # print("\nLeft table (first 3 rows):")
            # print(left_table.head(3))
            # print("\nRight table (first 3 rows):")
            # print(right_table.head(3))

            # Load models - ensure we're using the correct models_dir
            column_model_path = os.path.join(models_dir, "join_column_model.pkl")
            column_model_data = load_model(column_model_path)
            col_model, col_feature_names = column_model_data

            type_model_path = os.path.join(models_dir, "join_type_model.pkl")
            type_model_data = load_model(type_model_path)
            type_model, type_feature_names, type_label_encoder = type_model_data

            # Create a models dictionary
            models = {
                'col_model': col_model,
                'col_feature_names': col_feature_names,
                'type_model': type_model,
                'type_feature_names': type_feature_names,
                'type_label_encoder': type_label_encoder
            }

            # Generate join recommendations
            recommendations = recommend_joins(models, left_table, right_table, top_k)

            # Display recommendations
            display_join_recommendations(recommendations, top_k)

            return True

        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return False

    else:
        print(f"Error: Invalid mode '{mode}'. Use 'train', 'eval', or 'predict'.")
        return False


def run_groupby_prediction(data_dir, models_dir, mode='train', input_file=None, top_k=2):
    """
    Executes the Groupby Prediction Pipeline

    This function provides a complete pipeline for the GroupBy prediction task, including:
    1. Training a model for GroupBy column prediction
    2. Evaluating the model on test data
    3. Making predictions on new, unseen tables

    Modes:
    - 'train': Train a GroupBy column prediction model using samples from data_dir
    - 'eval': Evaluate the trained model on test samples and compute metrics
    - 'predict': Apply the trained model to make predictions on new 'unseen' tables

    Args:
        data_dir: Directory containing training samples
        models_dir: Directory to save trained models
        mode: Operation mode ('train', 'eval', or 'predict')
        input_file: Path to input table CSV for prediction (only in predict mode)
        top_k: Number of top recommendations to display in prediction mode

    Returns:
        Boolean indicating success or failure of the operation
    """
    # Import local necessary modules
    from src.utils.model_utils import save_model, load_model
    from src.models.groupby_model import (process_groupby_samples, prepare_groupby_data, train_groupby_model,
                                          predict_column_groupby_scores, evaluate_groupby_model,
                                          display_groupby_recommendations)

    # Update the header based on the mode
    if mode == 'eval':
        print("\n" + "=" * 80)
        print("Predict Single Operators -  GroupBy (Section 6.5)")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("GroupBy Prediction (Section 4.2)")
        print("=" * 80)

    # Training mode
    if mode == 'train':
        #print("\nTraining groupby model...")

        # Load groupby samples
        #print("\nLoading groupby samples...")
        groupby_dir = os.path.join(data_dir, "groupby")
        groupby_samples = load_operator_samples(groupby_dir, 'groupby')

        # Check if any samples were loaded
        if not groupby_samples:
            print(f"Error: No groupby samples found in {groupby_dir}. Check your data directory.")
            return False

        # Process samples
        processed_groupby_samples = process_groupby_samples(groupby_samples)

        # Check if we have any valid samples
        if len(processed_groupby_samples) == 0:
            print("Error: No valid groupby samples found after processing.")
            return False

        # Split into train, val test sets (no stratification typically for groupby unless we have a class label)
        tmp_samples, test_samples = train_test_split(processed_groupby_samples, test_size=0.1, random_state=42)
        train_samples, val_samples = train_test_split(tmp_samples, test_size=0.1111, random_state=42)

        print(f"Split {len(processed_groupby_samples)} samples into {len(train_samples)} train, {len(val_samples)} validation, and {len(test_samples)} test samples")
        #print(train_samples[0])
        # print(test_samples[0]) # Has also the same form as processed_samples

        # Debug print: summarize the data splits
        # def summarize_split(samples, name):
        #     print(f"\n{name} samples: {len(samples)}")
        #     col_counts = {}
        #     for s in samples:
        #         for col in s['input_table'].columns:
        #             col_counts[col] = col_counts.get(col, 0) + 1
        #     top_cols = sorted(col_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        #     print(f"Top 10 columns in {name}: {top_cols}")
        #
        # summarize_split(train_samples, "Train")
        # summarize_split(val_samples, "Validation")
        # summarize_split(test_samples, "Test")

        # Save train, val, and test data
        with open(os.path.join(test_data, "groupby_train_data.pkl"), "wb") as f:
            pickle.dump(train_samples, f)

        with open(os.path.join(test_data, "groupby_val_data.pkl"), "wb") as f:
            pickle.dump(val_samples, f)

        with open(os.path.join(test_data, "groupby_test_data.pkl"), "wb") as f:
            pickle.dump(test_samples, f)
        # print(f"Saved groupby train, validation and test samples to {test_data}")

        # Train Groupby Prediction Model
        print("\n--- Training Groupby Prediction Model ---")

        # Prepare training data (extract features and labels from train_samples)
        train_groupby_features, feature_cols_groupby, train_groupby_labels = prepare_groupby_data(train_samples)
        val_groupby_features, _, val_groupby_labels = prepare_groupby_data(val_samples)

        # Train model
        groupby_model, groupby_feature_names = train_groupby_model(train_groupby_features, train_groupby_labels, val_groupby_features, val_groupby_labels, feature_cols_groupby)

        # Check if model training was successful
        if groupby_model is None:
            print("Error: Groupby model training failed.")
            return False

        # Save model
        model_path = os.path.join(models_dir, "groupby_column_model.pkl")
        save_model((groupby_model, groupby_feature_names), model_path)

        print("\nGroupBy model trained successfully!\n")
        return True

    # Evaluation mode
    elif mode == 'eval':
        print("\nEvaluating groupby prediction model...")
        print(f"Using top k = {top_k} predictions for evaluation")

        # Load test data
        test_path = os.path.join(test_data, "groupby_test_data.pkl")

        if not os.path.exists(test_path):
            print("Error: Test data not found. Run training first.")
            return False

        with open(test_path, "rb") as f:
            test_samples = pickle.load(f)
        #print(test_samples[0])    # Check if it is the same as defined above before saving and loading

        # Remember again: No need to apply `prepare_groupby_training_data()` here.
        # We directly pass `test_samples` to evaluation functions, which internally extract features to generate predictions.
        print(f"Using {len(test_samples)} test samples for evaluation\n")

        relative_path = os.path.relpath(test_data, base_dir)
        print(f"Test data loaded from {os.path.join(relative_path, 'groupby_test_data.pkl')}")

        # Load groupby model
        try:
            model_path = os.path.join(models_dir, "groupby_column_model.pkl")
            groupby_model_data = load_model(model_path)
            groupby_model, groupby_feature_names = groupby_model_data
            #print("Loaded feature names from model:", groupby_feature_names)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the model first.")
            return False

        print("\n--- Evaluating GroupBy Model on Test Set ---")

        # Evaluate groupby model
        metrics = evaluate_groupby_model(groupby_model, groupby_feature_names, test_samples, top_k)

        print("\nGroupby Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.2f}")

        print("\nMetrics and figures have been saved to 'results' directory")

        print("\nGroupBy model evaluated successfully!\n")
        return True

    # Predict mode - for new 'unseen' tables
    elif mode == 'predict':
        print("\nPredicting groupby columns for a new table...")

        # Check if input file is provided
        if input_file is None:
            print("Error: Input table file must be provided for prediction.")
            return False

        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Error: {input_file} file not found.")
            return False

        # Run prediction using groupby pipeline
        try:
            # Read the table
            input_table = pd.read_csv(input_file)
            print(f"\nLoaded table: {input_table.shape[0]} rows x {input_table.shape[1]} columns")
            print(f"\nTable's columns: {input_table.columns}\n")

            # Show sample rows
            # print("\nTable's first 3 rows:")
            # print(input_table.head(3))

            # Load the model
            groupby_model_path = os.path.join(models_dir, "groupby_column_model.pkl")
            groupby_model_data = load_model(groupby_model_path)
            groupby_model, groupby_feature_names = groupby_model_data

            # Generate Groupby recommendations
            # = predict_groupby_column_pairs(groupby_model, groupby_feature_names, input_table, top_k)
            recommendations = predict_column_groupby_scores(groupby_model, groupby_feature_names, input_table)

            # Display recommendations
            display_groupby_recommendations(recommendations, top_k, input_table)

            return True

        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return False

    else:
        print(f"Error: Invalid mode '{mode}'. Use 'train', 'eval', or 'predict'.")
        return False


def run_pivot_prediction(data_dir, models_dir, mode='train', input_file=None, aggfunc='mean'):
    """
    Executes the Pivot Prediction Pipeline

    Args:
        data_dir: Directory containing training samples
        models_dir: Directory to save affinity regression trained model
        mode: Operation mode ('train', 'eval', or 'predict')
        input_file: Path to input table CSV for prediction (only in predict mode)
        aggfunc: Aggregation function to use for pivot (only in predict mode)
    """
    # Import necessary modules
    from src.models.pivot_model import process_pivot_samples, train_affinity_weights, evaluate_pivot, predict_pivot_split

    # Update the header based on the mode
    if mode == 'eval':
        print("\n" + "=" * 80)
        print("Predict Single Operators - Pivot (Section 6.5)")
        print("=" * 80)
        print()
    else:
        print("\n" + "=" * 80)
        print("Pivot Prediction (Section 4.3)")
        print("=" * 80)
        print()

    if mode == 'train':
        print("Note: Pivot operator has no training phase for end-to-end pivot, but trains the affinity weights model.")

        # Load pivot samples
        pivot_dir = os.path.join(data_dir, "pivot")
        pivot_samples = load_operator_samples(pivot_dir, 'pivot')

        if not pivot_samples:
            print(f"Error: No pivot samples found in {pivot_dir}. Check your data directory.")
            return False

        # Process samples
        processed_pivot_samples = process_pivot_samples(pivot_samples)

        if len(processed_pivot_samples) == 0:
            print("Error: No valid pivot samples found after processing.")
            return False

        # Train and save affinity weights (a, b, intercept)
        train_affinity_weights(processed_pivot_samples, models_dir)

        return True

    # Evaluation mode
    elif mode == 'eval':
        print("Running pivot prediction model...")

        # Load pivot samples
        # print("\nLoading pivot samples...")
        pivot_dir = os.path.join(data_dir, "pivot")
        pivot_samples = load_operator_samples(pivot_dir, 'pivot')

        # Check if any samples were loaded
        if not pivot_samples:
            print(f"Error: No pivot samples found in {pivot_dir}. Check your data directory.")
            return False

        # Process samples
        processed_pivot_samples = process_pivot_samples(pivot_samples)

        # Check if we have any valid samples
        if len(processed_pivot_samples) == 0:
            print("Error: No valid pivot samples found after processing.")
            return False

        print(f"\nUsing {len(processed_pivot_samples)} test samples for evaluation")

        # Run (Evaluate) pivot model
        print("\n--- Evaluating Pivot Model ---\n")
        metrics = evaluate_pivot(processed_pivot_samples)

        # Print the metrics first, before Table 8
        print("\nPivot model evaluation results:")
        print(f"  Full accuracy: {metrics['full_accuracy']:.4f}")
        print(f"  Rand Index: {metrics['rand_index']:.4f}")

        print("\nMetrics have been saved to the 'results' directory")

        print("\nPivot model evaluated successfully!\n")

        return True

    # Predict mode
    elif mode == 'predict':
        print("Predicting pivot structure for new table...")

        # Check if input file is provided
        if input_file is None:
            print("Error: Input table file must be provided for prediction.")
            return False

        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            return False

        # Run prediction using pivot pipeline
        try:
            # Read the table
            table = pd.read_csv(input_file)
            print(f"\nLoaded table: {table.shape[0]} rows × {table.shape[1]} columns")
            print(f"\nColumn names: {', '.join(table.columns)}\n")

            # Show sample rows
            # print("Sample data (first 3 rows):")
            # print(table.head(3))

            # Generate Pivot recommendations
            predict_pivot_split(table, aggfunc=aggfunc)

            return True

        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return False

    else:
        print(f"Error: Invalid mode '{mode}'. Use 'train', 'eval', or 'predict'.")
        return False

def run_unpivot_prediction(data_dir, models_dir, mode='train', input_file=None):
    """
    Executes the Unpivot Prediction Pipeline

    Args:
        data_dir: Directory containing training samples
        mode: Operation mode ('train', 'eval', or 'predict')
        input_file: Path to input table CSV for prediction (only in predict mode)
    """
    # Import necessary modules
    from src.models.unpivot_model import process_unpivot_samples, evaluate_unpivot, predict_unpivot
    from src.models.pivot_model import train_affinity_weights

    # Update the header based on the mode
    if mode == 'eval':
        print("\n" + "=" * 80)
        print("Predict Single Operators - Unpivot (Section 6.5)")
        print("=" * 80)
        print()
    else:
        print("\n" + "=" * 80)
        print("Unpivot Prediction (Section 4.4)")
        print("=" * 80)
        print()

    if mode == 'train':
        print("Note: Unpivot operator has no training phase for end-to-end unpivot, but trains the affinity weights model.")

        # Load unpivot samples
        unpivot_dir = os.path.join(data_dir, "melt")
        unpivot_samples = load_operator_samples(unpivot_dir, 'unpivot')

        if not unpivot_samples:
            print(f"Error: No unpivot samples found in {unpivot_dir}. Check your data directory.")
            return False

        # Process samples
        processed_unpivot_samples = process_unpivot_samples(unpivot_samples)

        if len(processed_unpivot_samples) == 0:
            print("Error: No valid unpivot samples found after processing.")
            return False

        # Train and save affinity weights (a, b, intercept)
        train_affinity_weights(processed_unpivot_samples, models_dir)

        return True

    # Evaluation mode
    elif mode == 'eval':
        print("Running unpivot prediction model...")

        # Load unpivot samples
        unpivot_dir = os.path.join(data_dir, "melt")
        unpivot_samples = load_operator_samples(unpivot_dir, 'unpivot')

        # Check if any samples were loaded
        if not unpivot_samples:
            print(f"Error: No unpivot samples found in {unpivot_dir}. Check your data directory.")
            return False

        # Process samples
        processed_unpivot_samples = process_unpivot_samples(unpivot_samples)

        # Check if we have any valid samples
        if len(processed_unpivot_samples) == 0:
            print("Error: No valid unpivot samples found after processing.")
            return False

        print(f"\nUsing {len(processed_unpivot_samples)} test samples for evaluation")

        # Run (Evaluate) unpivot model
        print("\n--- Evaluating Unpivot Model ---\n")
        metrics = evaluate_unpivot(processed_unpivot_samples)

        # Print the metrics first, before Table 8
        print("\nUnpivot model evaluation results:")
        print(f"  Full accuracy: {metrics['full_accuracy']:.4f}")
        print(f"  Column precision: {metrics['precision']:.4f}")
        print(f"  Column recall: {metrics['recall']:.4f}")
        print(f"  Column F1-score: {metrics['f1_score']:.4f}")

        print("\nMetrics have been saved to the 'results' directory")

        print("\nUnpivot model evaluated successfully!\n")

        return True

    # Predict mode
    elif mode == 'predict':
        print("Predicting unpivot structure for new table...")

        # Check if input file is provided
        if input_file is None:
            print("Error: Input table file must be provided for prediction.")
            return False

        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            return False

        # Run prediction using unpivot pipeline
        try:
            # Read the table
            table = pd.read_csv(input_file)
            print(f"\nLoaded table: {table.shape[0]} rows × {table.shape[1]} columns")
            print(f"\nColumn names: {', '.join(table.columns)}\n")

            # Show sample rows
            # print("Sample data (first 3 rows):")
            # print(table.head(3))

            # Generate Unpivot recommendations
            predict_unpivot(table)

            return True

        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            return False

    else:
        print(f"Error: Invalid mode '{mode}'. Use 'train', 'eval', or 'predict'.")
        return False


def create_init_files():
    """
    Create necessary __init__.py files for proper imports.

    This function ensures that each directory in the project structure has an
    __init__.py file, which serves several critical purposes:

    1. Package Recognition: The presence of __init__.py tells Python that a
       directory should be treated as a package or subpackage, which is essential
       for imports using dot notation (e.g., 'from src.data import sample_loader').

    2. Import Resolution: When modules within the project need to reference each
       other (like models importing feature extraction functions), the Python
       import system needs these files to properly resolve the imports.

    3. Relative Imports: These files enable relative imports between modules
       (e.g., 'from src.data.features import join_features' from within a models module).

    Without these files, imports between modules in different directories will fail
    with errors like 'ModuleNotFoundError: No module named src.data', even if those
    directories and modules exist in your project structure.

    While modern Python (3.3+) does support namespace packages without __init__.py files,
    using them is the standard practice for structured projects and ensures compatibility
    across different Python versions and execution environments. They are small, but
    necessary for a clean, working, and shareable Python project.
    """
    # Note: We only create __init__.py in package directories, not in the root directory
    dirs_needing_init = [
        "src/data",
        "src/models",
        "src/utils",
        "src/baselines"  # Added baselines directory
    ]

    # Create src/__init__.py separately to avoid creating one in the root
    src_init = "src/__init__.py"
    if not os.path.exists(src_init):
        # Ensure src directory exists
        os.makedirs("src", exist_ok=True)
        # print(f"Creating {src_init}")
        with open(src_init, 'w') as f:
            f.write("# Auto-generated __init__.py file\n")
            f.write("# This file marks this directory as a Python package\n")
            f.write("# It enables imports between modules in different directories\n")
            f.write("# For example, it allows 'from src.data import sample_loader' to work\n")

    # Create __init__.py in each package directory
    for dir_path in dirs_needing_init:
        # Ensure directory exists
        os.makedirs(dir_path, exist_ok=True)

        init_file = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_file):
            # print(f"Creating {init_file}")
            with open(init_file, 'w') as f:
                f.write("# Auto-generated __init__.py file\n")
                f.write("# This file marks this directory as a Python package\n")
                f.write("# It enables imports between modules in different directories\n")
                f.write("# For example, it allows 'from src.data import sample_loader' to work\n")

# --------------
# Command Line Interface
# --------------

def main():
    """
    Main function to run single-operator predictions with command line args.
    """
    parser = argparse.ArgumentParser(description='Auto-Suggest: Learning-to-Recommend Data Preparation Steps')

    parser.add_argument('--data_dir', default='data/extracted_data', help='Directory containing the extracted data samples') # e.g. python -m src.main --results_dir output_results/
    parser.add_argument('--models_dir', default='models', help='Directory to save trained models')    # e.g. python -m src.main --models_dir saved_models/
    parser.add_argument('--results_dir', default='results', help='Directory for results and figures')   # e.g. python -m src.main --results_dir output_results/
    parser.add_argument('--operator', choices=['join', 'groupby', 'pivot', 'unpivot', 'all'], default='all', help='Which operator to run (default: all)')
    parser.add_argument('--mode', choices=['train', 'eval', 'predict', 'all'], # python -m src.main --operator all --mode all: will automatically run everything in one go!
                        default='train', help='Mode of operation (default: train)') # python -m src.main --operator join --mode train
    parser.add_argument('--left_file', type=str, help='Path to left table CSV for join prediction')
    parser.add_argument('--right_file', type=str, help='Path to right table CSV for join prediction')   # python -m src.main --operator join --mode predict --left_file data/test_data/join_customers.csv --right_file data/test_data/join_orders.csv
    parser.add_argument('--input_file', type=str, help='Path to input table CSV for other operators')   # e.g. python -m src.main --operator groupby --mode predict --top_k 3
    parser.add_argument('--top_k', type=int, default=2, help='Number of top recommendations to display in prediction mode') # e.g. python -m src.main --operator groupby --mode predict --top_k 3
    parser.add_argument('--aggfunc', type=str, default='mean', help='Aggregation function for pivot (default: mean)')  # e.g. python -m src.main --operator pivot --mode predict --input_file data/pivot/sample.csv --aggfunc sum
    args = parser.parse_args()

    # Create necessary directories
    models_dir = args.models_dir
    results_dir = args.results_dir
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Create __init__.py files if needed
    create_init_files()

    # Parse top_k values
    top_k = args.top_k
    results = {}

    if args.operator in ['join', 'all']:
        if args.mode == 'all':
            # Run train, eval, predict for join
            run_join_prediction(args.data_dir, models_dir, mode='train', left_file=args.left_file, right_file=args.right_file, top_k=top_k)
            run_join_prediction(args.data_dir, models_dir, mode='eval', left_file=args.left_file, right_file=args.right_file, top_k=top_k)
            run_join_prediction(args.data_dir, models_dir, mode='predict', left_file='data/test_data/join_customers.csv', right_file='data/test_data/join_orders.csv', top_k=top_k)
        else:
            # Run join with the specified mode
            success = run_join_prediction(args.data_dir, models_dir, mode=args.mode, left_file=args.left_file, right_file=args.right_file, top_k=top_k)
            if not success and args.operator == 'join':
                print("Join prediction failed. Exiting.")
                return

    if args.operator in ['groupby', 'all']:
        if args.mode == 'all':
            # Run train, eval, predict for groupby
            run_groupby_prediction(args.data_dir, models_dir, mode='train', input_file=args.input_file, top_k=top_k)
            run_groupby_prediction(args.data_dir, models_dir, mode='eval', input_file=args.input_file, top_k=top_k)
            run_groupby_prediction(args.data_dir, models_dir, mode='predict', input_file='data/test_data/groupby_sales_data.csv', top_k=top_k)

        else:
            # Run groupby with the specified mode
            success = run_groupby_prediction(args.data_dir, models_dir, mode=args.mode, input_file=args.input_file, top_k=top_k)
            if not success and args.operator == 'groupby':
                print("GroupBy prediction failed. Exiting.")
                return

    if args.operator in ['pivot', 'all']:
        if args.mode == 'all':
            try:
                # Run train, eval, predict for pivot
                run_pivot_prediction(args.data_dir, models_dir, mode='train', input_file=args.input_file, aggfunc=args.aggfunc)
                run_pivot_prediction(args.data_dir, models_dir, mode='eval', input_file=args.input_file, aggfunc=args.aggfunc)
                run_pivot_prediction(args.data_dir, models_dir, mode='predict', input_file='data/test_data/pivot_financial_data.csv', aggfunc=args.aggfunc)
            except Exception as e:
                print(f"Error during Pivot prediction (all modes): {e}")
                traceback.print_exc()
                if args.operator == 'pivot':
                    return
        else:
            try:
                # Run pivot with the specified mode
                result = run_pivot_prediction(args.data_dir, models_dir, mode=args.mode, input_file=args.input_file, aggfunc=args.aggfunc)

                if result is None and args.operator == 'pivot':
                    print("Pivot prediction failed. Exiting.")
                    return

                if args.mode == 'eval':
                    results['pivot'] = result

            except Exception as e:
                print(f"Error during Pivot prediction: {e}")
                traceback.print_exc()
                if args.operator == 'pivot':
                    return

    if args.operator in ['unpivot', 'all']:
        if args.mode == 'all':
            try:
                # Run train, eval, predict for unpivot
                run_unpivot_prediction(args.data_dir, models_dir, mode='train', input_file=args.input_file)
                run_unpivot_prediction(args.data_dir, models_dir, mode='eval', input_file=args.input_file)
                run_unpivot_prediction(args.data_dir, models_dir, mode='predict', input_file='data/test_data/unpivot_product_sales.csv')
            except Exception as e:
                print(f"Error during Unpivot prediction (all modes): {e}")
                traceback.print_exc()
                if args.operator == 'unpivot':
                    return
        else:
            try:
                # Run unpivot with the specified mode
                result = run_unpivot_prediction(args.data_dir, models_dir, mode=args.mode, input_file=args.input_file)

                if result is None and args.operator == 'unpivot':
                    print("Unpivot prediction failed. Exiting.")
                    return

                if args.mode == 'eval':
                    results['unpivot'] = result

            except Exception as e:
                print(f"Error during Unpivot prediction: {e}")
                traceback.print_exc()
                if args.operator == 'unpivot':
                    return


if __name__ == "__main__" or __name__ == "src.main":
    main()
