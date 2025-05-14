"""
prepare_replay_notebooks.py

A repository like `Titanic-ML-Project` may contain:
- 3 notebooks: `eda.ipynb`, `feature_engineering.ipynb`, `model_train.ipynb`
- 2 datasets: `train.csv`, `test.csv`

This script prepares filtered notebooks for isolated replay.

For each notebook under:
  filtered_github_repos/<operator>/<repo>/...

It creates a folder under:
  prepared_repos/<operator>/<repo>__<notebook_name>/

Each such folder contains:
  - Exactly one notebook
  - Exactly one dataset file (e.g., .csv, .xls) that the notebook actually uses

How datasets are matched:
We parse the code cells of each notebook and check if any known dataset filenames
(e.g., 'train.csv', 'data.json') are referenced. The first referenced dataset is selected.
If none are matched explicitly, we fall back to the first dataset found in the folder.

This ensures that the dataset paired with each notebook is likely the one it's actually using,
making replaying cleaner and more reliable.

(In merge cases, a notebook may use multiple datasets, so this script parses the notebook code
to identify and copy all datasets that are actually referenced.)

"""

import os
import shutil
import nbformat
from collections import defaultdict

# Set input and output directories
FILTERED_DIR = r"C:\Users\giorg\Auto_Suggest\data\filtered_github_repos"
PREPARED_DIR = r"C:\Users\giorg\Auto_Suggest\data\prepared_replay_notebooks"

# Acceptable dataset file extensions
DATA_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".json", ".txt"}

operator_notebook_counts = defaultdict(int)
print("\nPreparing replay folders...\n")

# Walk through each operator directory (groupby, melt, merge, pivot)
for operator in os.listdir(FILTERED_DIR):
    operator_path = os.path.join(FILTERED_DIR, operator)
    if not os.path.isdir(operator_path):
        continue

    # Recursively walk through all folders in this operator group
    for root, _, files in os.walk(operator_path):
        notebooks = [f for f in files if f.endswith(".ipynb")]
        datasets = [f for f in files if os.path.splitext(f)[1].lower() in DATA_EXTENSIONS]

        if not notebooks or not datasets:
            continue  # Skip folders that don't have both a notebook and at least one dataset

        for nb in notebooks:
            nb_path = os.path.join(root, nb)

            # Try to select the dataset(s) actually used in this notebook
            selected_datasets = []
            try:
                nb_obj = nbformat.read(nb_path, as_version=4)
                code_cells = [cell.source for cell in nb_obj.cells if cell.cell_type == "code"]
                flat_code = "\n".join(code_cells)

                # Match datasets that are explicitly referenced and exist
                for dataset in datasets:
                    if dataset in flat_code:
                        selected_datasets.append(dataset)

                # Check if notebook references unsupported/missing files (e.g. .csv.gz)
                all_data_mentions = [f for f in os.listdir(root) if f in flat_code]
                unresolved = [f for f in all_data_mentions if f not in datasets]
                if unresolved:
                    print(f"Skipping notebook {nb} due to unrecognized or unsupported dataset(s): {unresolved}")
                    continue  # Skip notebook entirely

                if not selected_datasets:
                    selected_datasets = [datasets[0]]  # fallback
            except Exception as e:
                print(f"Could not parse notebook {nb_path}: {e}")
                continue  # skip broken notebook

            # Compute flat folder path using full relative structure
            relative_to_operator = os.path.relpath(root, os.path.join(FILTERED_DIR, operator))
            repo_folder_path = relative_to_operator.replace(os.sep, "_")

            nb_name = os.path.splitext(nb)[0]
            target_folder_name = f"{repo_folder_path}__{nb_name}"
            target_folder = os.path.join(PREPARED_DIR, operator, target_folder_name)

            os.makedirs(target_folder, exist_ok=True)

            # Copy notebook
            try:
                shutil.copy2(nb_path, os.path.join(target_folder, nb))
            except Exception as e:
                print(f"Failed to copy notebook to {target_folder}: {e}")
                continue

            # Copy all matched dataset files
            for dataset in selected_datasets:
                ds_path = os.path.join(root, dataset)
                try:
                    shutil.copy2(ds_path, os.path.join(target_folder, dataset))
                except Exception as e:
                    print(f"Failed to copy dataset {dataset} to {target_folder}: {e}")

            print(f"Created: {target_folder}")
            operator_notebook_counts[operator] += 1

print("\nSummary of Prepared Replay Notebooks per Operator:")
for operator, count in sorted(operator_notebook_counts.items()):
    print(f"{operator}: {count}")
