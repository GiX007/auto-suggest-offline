# src/utils/join_pipeline.py
#
# Join prediction pipeline that combines join column and join type prediction
# Based on the "Auto-Suggest: Learning-to-Recommend Data Preparation Steps Using Data Science Notebooks" paper
#
# This module:
# 1. Provides functions for integrating join column and join type predictions
# 2. Supports evaluation of join models on test samples
# 3. Generates complete join recommendations for new tables
# 4. Displays results in a readable format
#

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import time

# Import from our package structure
from src.models.join_col_model import predict_join_columns
from src.models.join_type_model import predict_join_type, extract_join_type_features
from src.utils.model_utils import load_model
from src.utils.evaluation import evaluate_join_column_model, evaluate_join_type_model


def load_join_models(model_dir):
    """
    Load trained join models from disk.

    Args:
        model_dir: Directory containing models

    Returns:
        Dictionary containing loaded models and related data
    """
    models = {}

    try:
        # Load join column model
        column_model_path = os.path.join(model_dir, "join_column_model.pkl")
        column_model_data = load_model(column_model_path)
        models['col_model'] = column_model_data[0]
        models['col_feature_names'] = column_model_data[1]

        # Load join type model
        type_model_path = os.path.join(model_dir, "join_type_model.pkl")
        type_model_data = load_model(type_model_path)
        models['type_model'] = type_model_data[0]
        models['type_feature_names'] = type_model_data[1]
        models['type_label_encoder'] = type_model_data[2]

        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure you have trained the models first.")
        return None


def recommend_joins(models, left_table, right_table, top_k=2):
    """
    Generate complete join recommendations for two tables.

    This function implements the two-step join recommendation process:
    1. First predicts which columns to join
    2. Then predicts what type of join to use for those columns

    Args:
        models: Dictionary of loaded models and related data
        left_table: Left table for join
        right_table: Right table for join
        top_k: Number of top recommendations to return

    Returns:
        List of recommendations, each containing join columns and join type
    """
    # Step 1: Predict Join Columns
    print("\n=== Step 1: Predicting Join Columns ===")
    # print("Generating join candidates between tables...")
    join_column_candidates = predict_join_columns(
        models['col_model'], models['col_feature_names'],
        left_table, right_table, top_k
    )

    if not join_column_candidates:
        print("No viable join candidates found between these tables.")
        return []

    # Limit to top_k candidates
    join_column_candidates = join_column_candidates[:top_k]

    # Step 2: For each top column candidate, predict join type
    print("\n=== Step 2: Predicting Join Types ===")
    print("\nJoin Type Predictions:")
    # print(f"Determining join types for top {top_k} column candidates...")

    recommendations = []

    for i, (left_cols, right_cols, column_score) in enumerate(join_column_candidates):
        # Predict join type for this candidate
        join_type_result = predict_join_type(
            models['type_model'],
            models['type_feature_names'],
            models['type_label_encoder'],
            left_table, right_table,
            left_cols, right_cols
        )

        predicted_type = join_type_result['predicted_join_type']
        print(f"Processing candidate {i + 1}: {', '.join(left_cols)} ↔ {', '.join(right_cols)} → {predicted_type} join")

        # Combine the column and type predictions into a recommendation
        recommendation = {
            'rank': i + 1,
            'left_join_columns': left_cols,
            'right_join_columns': right_cols,
            'column_confidence': column_score,
            'join_type': predicted_type,
            'join_type_confidence': join_type_result['confidence'],
            'alternative_join_types': join_type_result['alternatives']
        }

        recommendations.append(recommendation)

    print(f"\nGenerated {len(recommendations)} complete join recommendations")
    return recommendations


def display_join_recommendations(recommendations, top_k=2):
    """
    Display join recommendations in a readable format.

    Args:
        recommendations: List of join recommendations
        top_k: Number of top recommendations to display (default: 2)
    """
    if not recommendations:
        print("No recommendations found.")
        return

    # Print recommendations in a readable format
    print("\n=== Complete Join Recommendations ===")
    print("=" * 80)

    # Only display top_k recommendations
    for rec in recommendations[:top_k]:
        left_cols = ", ".join(rec['left_join_columns'])
        right_cols = ", ".join(rec['right_join_columns'])

        print(f"\nRecommendation {rec['rank']}: Join using")
        print(f"  Left columns: {left_cols}")
        print(f"  Right columns: {right_cols}")
        print(f"  Column confidence: {rec['column_confidence']:.3f}")
        print(f"  Recommended join type: {rec['join_type']} (confidence: {rec['join_type_confidence']:.3f})")

        if rec['alternative_join_types']:
            print(f"  Alternative join types: {', '.join(rec['alternative_join_types'])}")

        print("-" * 40)

    # Provide example pandas code for the top recommendation
    if recommendations:
        top_rec = recommendations[0]
        left_cols_str = ", ".join([f"'{col}'" for col in top_rec['left_join_columns']])
        right_cols_str = ", ".join([f"'{col}'" for col in top_rec['right_join_columns']])

        print("\n=== Example Pandas Code for Top Recommendation ===")

        if len(top_rec['left_join_columns']) == 1 and len(top_rec['right_join_columns']) == 1:
            # Single column join
            print(f"result = pd.merge(left_table, right_table,")
            print(f"                  left_on='{top_rec['left_join_columns'][0]}',")
            print(f"                  right_on='{top_rec['right_join_columns'][0]}',")
            print(f"                  how='{top_rec['join_type']}')\n")
        else:
            # Multi-column join
            print(f"result = pd.merge(left_table, right_table,")
            print(f"                  left_on=[{left_cols_str}],")
            print(f"                  right_on=[{right_cols_str}],")
            print(f"                  how='{top_rec['join_type']}')\ncl")


def predict_on_files(left_file, right_file, model_dir='models', top_k=3):
    """
    Run join prediction on two CSV files.

    This function:
    1. Loads two CSV files
    2. Applies the two-step join recommendation process
    3. Displays the recommendations

    Args:
        left_file: Path to left table CSV
        right_file: Path to right table CSV
        model_dir: Directory containing trained models
        top_k: Number of top recommendations to return

    Returns:
        List of join recommendations
    """
    # Check if files exist
    if not os.path.exists(left_file):
        print(f"Error: Left file '{left_file}' not found")
        return None

    if not os.path.exists(right_file):
        print(f"Error: Right file '{right_file}' not found")
        return None

    # Load tables
    try:
        left_table = pd.read_csv(left_file)
        right_table = pd.read_csv(right_file)

        print(f"\nLoaded tables:")
        print(f"Left table: {left_table.shape[0]} rows × {left_table.shape[1]} columns")
        print(f"Right table: {right_table.shape[0]} rows × {right_table.shape[1]} columns")

        # Show sample rows
        print("\nLeft table (first 3 rows):")
        print(left_table.head(3))
        print("\nRight table (first 3 rows):")
        print(right_table.head(3))

    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None

    # Load models
    models = load_join_models(model_dir)
    if not models:
        return None

    # Generate join recommendations using the two-step process
    recommendations = recommend_joins(models, left_table, right_table, top_k)

    # Display recommendations
    display_join_recommendations(recommendations)

    return recommendations