# src/data/sample_loader.py
#
# This module is responsible for loading structured operator samples from disk
# for training and evaluation in the Auto-Suggest pipeline.
#
# 1. load_sample(sample_dir, operator_type)
#    - Loads a single operator sample from a directory.
#    - For 'join': loads left.csv and right.csv as DataFrames.
#    - For 'groupby', 'pivot', 'unpivot': loads data.csv as a single input table.
#    - Also loads param.json if available.
#    - Returns a dictionary representing the sample.
#
# 2. load_operator_samples(operator_dir, operator_type)
#    - Loads all valid samples for a given operator type from the specified directory.
#    - Iterates over subdirectories, calls load_sample on each.
#    - Returns a list of samples, where each sample is a dictionary containing
#      mainly data tables and parameters (and metadata) needed for training or evaluation.
#

import os
import json
import pandas as pd
from typing import List, Dict


def load_sample(sample_dir: str, operator_type: str) -> Dict:
    """
    Loads a single sample from the directory.

    Args:
        sample_dir: Path to the sample directory
        operator_type: Type of operator (join, groupby, pivot, unpivot)

    Returns:
        Dictionary containing the loaded sample data
    """
    sample = {'sample_id': os.path.basename(sample_dir)}

    # Load parameters
    param_path = os.path.join(sample_dir, 'param.json')
    if os.path.exists(param_path):
        with open(param_path, 'r', encoding='utf-8') as f:
            try:
                sample['params'] = json.load(f)
            except json.JSONDecodeError:
                print(f"Error parsing param.json in {sample_dir}")
                sample['params'] = {}

    # Load data based on operator type
    if operator_type == 'join':
        # For join, load left and right tables
        left_path = os.path.join(sample_dir, 'left.csv')
        right_path = os.path.join(sample_dir, 'right.csv')

        if os.path.exists(left_path) and os.path.exists(right_path):
            try:
                sample['left_table'] = pd.read_csv(left_path, encoding='utf-8', low_memory=False)
                sample['right_table'] = pd.read_csv(right_path, encoding='utf-8', low_memory=False)
            except Exception as e:
                print(f"Error loading CSV files in {sample_dir}: {e}")
    else:
        # For groupby, pivot, unpivot, load data.csv
        data_path = os.path.join(sample_dir, 'data.csv')
        if os.path.exists(data_path):
            try:
                sample['input_table'] = pd.read_csv(data_path, encoding='utf-8')
            except Exception as e:
                print(f"Error loading data.csv in {sample_dir}: {e}")

    return sample


def load_operator_samples(operator_dir: str, operator_type: str) -> List[Dict]:
    """
    Loads all samples for a given operator from the directory.

    Args:
        operator_dir: Path to the operator directory
        operator_type: Type of operator (join, groupby, pivot, unpivot)

    Returns:
        List of dictionaries containing the loaded samples
    """
    samples = []

    # List all subdirectories in the operator directory
    if not os.path.exists(operator_dir):
        print(f"Directory {operator_dir} does not exist")
        return samples

    sample_dirs = [os.path.join(operator_dir, d) for d in os.listdir(operator_dir)
                   if os.path.isdir(os.path.join(operator_dir, d))]
    # print(f"Loading {len(sample_dirs)} samples from {operator_dir}...")

    for sample_dir in sample_dirs:
        try:
            sample = load_sample(sample_dir, operator_type)
            if sample:
                samples.append(sample)
        except Exception as e:
            print(f"Error loading sample from {sample_dir}: {e}")
    # print(f"Successfully loaded {len(samples)} (notebook) samples for {operator_type}")

    return samples
