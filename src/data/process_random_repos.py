# process_random_repos.py
#
# This script performs two main tasks as part of the Auto-Suggest offline pipeline:
# 1. Filters previously cloned GitHub repositories to extract Jupyter notebooks (.ipynb) that use specific pandas operations
#    (groupby, pivot, melt, merge).
# 2. Attempts to get the datasets used inside those notebooks by:
#    - Searching the repository itself (local file paths)
#    - Downloading from URLs if present
#    - Downloading from Kaggle if dataset references are unresolved
# 3. Prepares isolated, replay-ready folders for each filtered notebook, by:
#    - Creating a separate folder for each notebook
#    - Copying only the notebook and the dataset(s) it actually uses
#
# Outputs:
# - Filtered notebooks and their datasets (if resolved) are saved in:
#   filtered_github_repos/<operator>/<repo>/
#   where:
#     - <operator> is one of: groupby, pivot, melt, merge
#     - <repo> contains at least one notebook using the target operator
#     - Each repo folder includes:
#         - The matched notebook (.ipynb)
#         - The resolved dataset file (local, downloaded, or from Kaggle)
#
# - Prepared replay-ready folders for each notebook in:
#   prepared_replay_notebooks/<operator>/<repo>__<notebook_name>/
#   where:
#     - Each folder contains:
#         - Exactly one notebook
#         - Exactly the dataset file(s) that notebook uses
#   This makes replaying and tracking operator usage easy and isolated.
#

import os
import re
import shutil
import requests
import nbformat
from collections import defaultdict
from kaggle.api.kaggle_api_extended import KaggleApi

# Define Paths
base_dir = r"C:\Users\giorg\Auto_Suggest"
repos_base_dir = os.path.join(base_dir, "data", "random_github_repos")
filtered_dir = os.path.join(base_dir, "data", "filtered_github_repos")
KAGGLE_TOKEN_PATH = r"C:\Users\giorg\.kaggle\kaggle.json"

target_ops = ["groupby", "pivot", "pivot_table", "melt", "merge"]


def filter_notebooks_with_target_ops() -> list:
    """
    Filters notebooks that contain pandas operator usage.

    Returns:
        list: Full paths of notebooks using target operators.
    """
    matched_notebooks = []

    print("\nFiltering notebooks with target pandas operations...\n")

    # Iterate over all notebooks of all repos
    for root, _, files in os.walk(repos_base_dir):
        for file in files:
            if file.endswith(".ipynb") and '.ipynb_checkpoints' not in root:
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
                        matched_notebooks.append(full_path)

                except Exception as e:
                    print(f"Failed to process {full_path}: {e}")

    print(f"\nTotal notebooks with target operations: {len(matched_notebooks)}")

    return matched_notebooks


def extract_dataset_references(notebooks: list) -> list:
    """
    Extracts dataset file references from read_csv/read_table/read_html calls.

    Args:
        notebooks (list): List of notebook file paths.

    Returns:
        list of dict: Extracted dataset references.
    """
    # Regex to find dataset reading functions
    read_csv_pattern = re.compile(r"""(?:read_csv|read_table|read_html)\((.*?)\)""")  # Added read_html
    notebook_infos = []

    print("\nExtracting dataset references from filtered notebooks...")

    for nb_path in notebooks:
        repo_name = os.path.basename(os.path.dirname(nb_path))
        file = os.path.basename(nb_path)

        try:
            # Read notebook
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Search for dataset references in code cells
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

    return notebook_infos


def resolve_dataset(info: dict) -> dict:
    """
    Attempts to resolve dataset reference: local file or URL download.

    Args:
        info (dict): Notebook info.

    Returns:
        dict: Updated notebook info with resolved dataset path and category.
    """
    dataset_path = info["dataset_ref"]
    repo_dir = os.path.dirname(info["full_path"])
    resolved_path = None
    category = "unresolved"

    # Try local file resolution
    full_file_path = os.path.join(repo_dir, dataset_path)
    if os.path.exists(full_file_path):
        category = "local_file"
        resolved_path = full_file_path
        print(f"[{info['notebook']}] Found local dataset: {dataset_path}")

    # Try URL-based dataset
    elif dataset_path.startswith("http"):
        category = "url"
        try:
            response = requests.get(dataset_path, timeout=30)
            if response.status_code == 200:
                local_filename = os.path.join(repo_dir, os.path.basename(dataset_path))
                with open(local_filename, 'wb') as f:
                    f.write(response.content)
                resolved_path = local_filename
                print(f"[{info['notebook']}] Downloaded from URL: {dataset_path}")
            else:
                print(f"[{info['notebook']}] Failed to download URL: {dataset_path}")
        except Exception as e:
            print(f"[{info['notebook']}] Exception downloading URL: {e}")

    info["resolved_path"] = resolved_path
    info["category"] = category
    return info


def copy_notebook_and_dataset(info: dict):
    """
    Copies the notebook and dataset (if resolved) to FILTERED_DIR.

    Args:
        info (dict): Notebook info with resolved dataset path.
    """
    if not info.get("resolved_path") or not os.path.isfile(info["resolved_path"]):
        print(f"Skipped notebook: No resolved dataset for {info['notebook']}")
        return

    # Determine destination paths
    relative_parts = os.path.relpath(info["full_path"], repos_base_dir).split(os.sep)
    if len(relative_parts) < 2:
        return

    operator = relative_parts[0]
    repo_subpath = os.path.join(*relative_parts[1:])
    dest_nb_path = os.path.join(filtered_dir, operator, repo_subpath)

    os.makedirs(os.path.dirname(dest_nb_path), exist_ok=True)

    # Copy notebook
    if not os.path.exists(dest_nb_path):
        shutil.copy2(info["full_path"], dest_nb_path)
        print(f"Copied notebook to: {dest_nb_path}")

    # Copy dataset
    dest_data_path = os.path.join(os.path.dirname(dest_nb_path), os.path.basename(info["resolved_path"]))
    if not os.path.exists(dest_data_path):
        shutil.copy2(info["resolved_path"], dest_data_path)
        print(f"Copied dataset to: {dest_data_path}")


def resolve_with_kaggle(unresolved_infos: list):
    """
    Attempts to resolve unresolved dataset references using Kaggle API.

    Args:
        unresolved_infos (list): List of notebook info dicts.
    """
    if not os.path.exists(KAGGLE_TOKEN_PATH):
        print("\nKaggle token not found — skipping Kaggle fallback.")
        return

    os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(KAGGLE_TOKEN_PATH)
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        return

    print("\nTrying to resolve unresolved datasets using Kaggle...\n")
    for info in unresolved_infos:
        dataset_name = os.path.basename(info["dataset_ref"].replace("\\", "/")).strip()

        # Skip empty dataset names (e.g., pd.read_csv() with no argument) to avoid random fallback downloads!
        if not dataset_name:
            print(f"Skipping empty dataset reference in notebook: {info['notebook']}")
            continue

        repo_dir = os.path.dirname(info["full_path"])
        print(f"Searching Kaggle for: {dataset_name}")
        try:
            results = api.dataset_list(search=dataset_name)
            if not results:
                print("No match found on Kaggle.")
                continue

            dataset_slug = results[0].ref
            print(f"Downloading: {dataset_slug}")
            api.dataset_download_files(dataset_slug, path=repo_dir, unzip=True)
            info["resolved_path"] = os.path.join(repo_dir, dataset_name)
            info["category"] = "kaggle_download"
            print(f"Downloaded from Kaggle: {dataset_name}")
        except Exception as e:
            print(f"Kaggle download failed: {e}")


def prepare_replay_notebooks():
    """
    Prepares filtered notebooks for isolated replay.

    For each notebook under:
      filtered_github_repos/<operator>/<repo>/...

    It creates a folder under:
      prepared_replay_notebooks/<operator>/<repo>__<notebook_name>/

    Each such folder contains:
      - Exactly one notebook
      - Dataset file(s) actually used by the notebook

    How datasets are matched:
    We parse the code cells of each notebook and check if any known dataset filenames
    (e.g., 'train.csv', 'data.json') are referenced. The first referenced dataset is selected.
    If none are matched explicitly, we fall back to the first dataset found in the folder.
    In 'merge' cases, a notebook may use multiple datasets, so this script identifies
    and copies all datasets that are actually referenced.
    """
    prepared_dir = os.path.join(base_dir, "data", "prepared_replay_notebooks")
    data_extensions = {".csv", ".tsv", ".xls", ".xlsx", ".json", ".txt"}
    operator_notebook_counts = defaultdict(int)

    print("\nPreparing replay folders...\n")

    # Walk through each operator directory (groupby, melt, merge, pivot)
    for operator in os.listdir(filtered_dir):
        operator_path = os.path.join(filtered_dir, operator)
        if not os.path.isdir(operator_path):
            continue

        # Recursively walk through all folders in this operator group
        for root, _, files in os.walk(operator_path):
            notebooks = [f for f in files if f.endswith(".ipynb")]
            datasets = [f for f in files if os.path.splitext(f)[1].lower() in data_extensions]

            if not notebooks or not datasets:
                continue  # Skip folders that don't have both a notebook and at least one dataset

            for nb in notebooks:
                nb_path = os.path.join(root, nb)
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

                    # Special case: merge may involve multiple datasets
                    if operator == "merge":
                        if not selected_datasets:
                            selected_datasets = datasets[:2]  # fallback to first two
                    else:
                        if not selected_datasets:
                            selected_datasets = [datasets[0]]  # fallback

                except Exception as e:
                    print(f"Could not parse notebook {nb_path}: {e}")
                    continue  # skip broken notebook

                # Compute target folder path: <operator>/<repo>__<notebook_name>
                relative_to_operator = os.path.relpath(root, os.path.join(filtered_dir, operator))
                repo_folder_path = relative_to_operator.replace(os.sep, "_")
                nb_name = os.path.splitext(nb)[0]
                target_folder_name = f"{repo_folder_path}__{nb_name}"
                target_folder = os.path.join(prepared_dir, operator, target_folder_name)

                os.makedirs(target_folder, exist_ok=True)

                # Copy notebook
                try:
                    shutil.copy2(nb_path, os.path.join(target_folder, nb))
                except Exception as e:
                    print(f"Failed to copy notebook to {target_folder}: {e}")
                    continue

                # Copy dataset(s)
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
    print()



def summarize_filtered_repos():
    """
    Prints a summary of the number of filtered repositories per operator.
    """
    print("\nSummary: Filtered Repos per Operator:")
    repo_by_operator = defaultdict(set)

    for root, _, files in os.walk(filtered_dir):
        for file in files:
            if file.endswith(".ipynb"):
                rel_path = os.path.relpath(root, filtered_dir)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    operator, repo = parts[0], parts[1]
                    repo_by_operator[operator].add(repo)

    for operator in ["groupby", "pivot", "melt", "merge"]:
        count = len(repo_by_operator[operator])
        print(f"{operator}: {count}")


def main():
    # Step 1: Filter notebooks by pandas operations ---
    notebooks = filter_notebooks_with_target_ops()

    # Step 2: Extract dataset references from read_csv / read_table / read_html calls
    notebook_infos = extract_dataset_references(notebooks)

    # Step 3: Attempt to resolve datasets
    resolved_infos = [resolve_dataset(info) for info in notebook_infos]

    # Copy notebooks and datasets
    for info in resolved_infos:
        copy_notebook_and_dataset(info)

    # Step 4: Try to resolve unresolved datasets using Kaggle (Kaggle fallback only for unresolved datasets)
    unresolved_infos = [info for info in resolved_infos if info["category"] == "unresolved"]
    if unresolved_infos:
        resolve_with_kaggle(unresolved_infos)

    # Summary of filtered repos
    summarize_filtered_repos()

    # Final step: Prepare replay-ready notebooks
    prepare_replay_notebooks()


if __name__ == "__main__":
    main()
