"""
process_random_repos.py

This script performs two main tasks as part of the Auto-Suggest offline pipeline:
1. Filters previously cloned GitHub repositories to extract Jupyter notebooks (.ipynb) that use specific pandas operations
   (groupby, pivot, melt, merge).
2. Attempts to get the datasets used inside those notebooks by:
   - Searching the repository itself (local file paths)
   - Downloading from URLs if present
   - Downloading from Kaggle if dataset references are unresolved

Output:
- Filtered notebooks and their datasets (if resolved) are saved in:
  filtered_github_repos/<operator>/<repo>/
  where:
    - <operator> is one of: groupby, pivot, melt, merge
    - <repo> contains at least one notebook using the target operator
    - Each repo folder includes:
        - The matched notebook (.ipynb)
        - The resolved dataset file (local, downloaded, or from Kaggle)

"""

import os
import nbformat
import shutil
import re
import requests
from collections import defaultdict
from kaggle.api.kaggle_api_extended import KaggleApi

# Define paths
BASE_DIR = r"C:\Users\giorg\Auto_Suggest\data\random_github_repos"
FILTERED_DIR = r"C:\Users\giorg\Auto_Suggest\data\filtered_github_repos"
KAGGLE_TOKEN_PATH = r"C:\Users\giorg\.kaggle\kaggle.json"


# Step 1: Filter notebooks by pandas operations ---
target_ops = ["groupby", "pivot", "pivot_table", "melt", "merge"]
matched_notebooks = []

print("\nFiltering notebooks with target pandas operations...\n")

# Iterate over all notebooks of all repos
for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if '.ipynb_checkpoints' in root:
            continue  # Skip checkpoint autosave directories

        if file.endswith(".ipynb"):
            full_path = os.path.join(root, file)  # Get the notebook (.ipynb)

            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)  # Notebook as Python object

                content = " ".join(
                    cell.get("source", "")
                    for cell in nb.cells
                    if cell.cell_type in {"code", "markdown"}
                ).lower()

                # Scan the current notebook
                if any(op in content for op in target_ops):
                    rel_path = os.path.relpath(full_path, BASE_DIR)
                    dest_path = os.path.join(FILTERED_DIR, rel_path)

                    matched_notebooks.append(full_path)

            except Exception as e:
                print(f"Failed to process {full_path}: {e}")

print(f"\nTotal notebooks with target operations: {len(matched_notebooks)}")

# Step 2: Extract dataset references from read_csv / read_table / read_html calls
read_csv_pattern = re.compile(r"""(?:read_csv|read_table|read_html)\((.*?)\)""")  # Added read_html

notebook_infos = []

print("\nExtracting dataset references from filtered notebooks...")

for nb_path in matched_notebooks:
    repo_name = os.path.basename(os.path.dirname(nb_path))
    file = os.path.basename(nb_path)

    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue

            matches = read_csv_pattern.findall(cell.source)
            for match in matches:
                path_match = re.search(r"['\"](.*?)['\"]", match)
                if path_match:
                    dataset_ref = path_match.group(1)

                    notebook_infos.append({
                        "repo": repo_name,
                        "notebook": file,
                        "dataset_ref": dataset_ref,
                        "full_path": nb_path,
                        "cell": cell.source.strip()
                    })

    except Exception as e:
        print(f"Could not parse notebook {file}: {e}")

# Step 3: Attempt to resolve dataset references
resolved_datasets = []

print("\nResolving datasets...")

for info in notebook_infos:
    dataset_path = info['dataset_ref']
    repo_dir = os.path.dirname(info['full_path'])

    # Case 1: Local file exists
    full_file_path = os.path.join(repo_dir, dataset_path)
    if os.path.exists(full_file_path):
        category = "local_file"
        resolved_path = full_file_path
        print(f"[{info['notebook']}] Found local dataset: {dataset_path}")

    # Case 2: URL-based dataset
    elif dataset_path.startswith("http"):
        category = "url"
        try:
            response = requests.get(dataset_path)
            if response.status_code == 200:
                local_filename = os.path.join(repo_dir, os.path.basename(dataset_path))
                with open(local_filename, 'wb') as f:
                    f.write(response.content)
                resolved_path = local_filename
                print(f"[{info['notebook']}] Downloaded from URL: {dataset_path}")
            else:
                resolved_path = None
                print(f"[{info['notebook']}] Failed to download URL: {dataset_path}")
        except Exception as e:
            print(f"[{info['notebook']}] Exception downloading URL: {e}")
            resolved_path = None

    # Case 3: Could not resolve
    else:
        category = "unresolved_filename"
        resolved_path = None
        print(f"[{info['notebook']}] Could not resolve dataset: {dataset_path}")

    # Only copy notebook and dataset if we successfully resolved a dataset file (exclude Case 3)
    if resolved_path and os.path.isfile(resolved_path):
        # Extract operator and repo for destination path
        relative_parts = os.path.relpath(info["full_path"], BASE_DIR).split(os.sep)
        if len(relative_parts) < 2:
            continue  # Invalid path structure

        operator = relative_parts[0]
        repo_subpath = os.path.join(*relative_parts[1:])
        dest_nb_path = os.path.join(FILTERED_DIR, operator, repo_subpath)

        # Create target folder if needed
        os.makedirs(os.path.dirname(dest_nb_path), exist_ok=True)

        # Copy notebook
        if not os.path.exists(dest_nb_path):
            shutil.copy2(info["full_path"], dest_nb_path)
            print(f"Copied notebook to: {dest_nb_path}")

        # Copy dataset
        dest_data_path = os.path.join(os.path.dirname(dest_nb_path), os.path.basename(resolved_path))
        if not os.path.exists(dest_data_path):
            shutil.copy2(resolved_path, dest_data_path)
            print(f"Copied dataset to: {dest_data_path}")

        # Save to result list
        resolved_datasets.append({
            **info,
            "resolved_path": resolved_path,
            "category": category
        })

    else:
        print(f"Skipped notebook: No resolved dataset for {info['notebook']}")

# Step 4: Try to resolve unresolved datasets using Kaggle
if os.path.exists(KAGGLE_TOKEN_PATH):
    print("\nTrying to resolve unresolved datasets using Kaggle...\n")

    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(KAGGLE_TOKEN_PATH)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        api = None

    if api:
        for info in resolved_datasets:
            if info["category"] != "unresolved_filename":
                continue

            dataset_name = info["dataset_ref"]
            repo_dir = os.path.dirname(info["full_path"])

            print(f"\nSearching Kaggle for: {dataset_name}")
            try:
                cleaned_search_term = os.path.basename(dataset_name.replace("\\", "/"))
                results = api.dataset_list(search=cleaned_search_term)
                if not results:
                    print(f"No match found on Kaggle")
                    continue

                dataset_slug = results[0].ref
                print(f"Downloading: {dataset_slug}")
                api.dataset_download_files(dataset_slug, path=repo_dir, unzip=True)

                info["resolved_path"] = os.path.join(repo_dir, dataset_name)
                info["category"] = "kaggle_download"
                print(f"Downloaded from Kaggle: {dataset_name}")

            except Exception as e:
                print(f"Kaggle download failed: {e}")

else:
    print("Kaggle token not found — skipping Kaggle fallback.")

# Summary — count filtered repos per operator
print("\nSummary: Filtered Repos per Operator:")

repo_by_operator = defaultdict(set)

for root, _, files in os.walk(FILTERED_DIR):
    for file in files:
        if file.endswith(".ipynb"):
            rel_path = os.path.relpath(root, FILTERED_DIR)
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                operator = parts[0]
                repo = parts[1]
                repo_by_operator[operator].add(repo)

for operator in ["groupby", "pivot", "melt", "merge"]:
    count = len(repo_by_operator[operator])
    print(f"{operator}: {count}")
