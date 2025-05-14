"""
github_crawler.py

This script performs two main tasks:
1. Searches GitHub for Jupyter notebooks (.ipynb) that contain 'pandas' code using the GitHub Code Search API.
2. Clones the unique repositories that contain those notebooks into a local directory.

"""

import os
import time
import requests
import subprocess

# GitHub personal access token (recommended to avoid rate limits)
GITHUB_TOKEN = ''  # Personal GitHub Token needed
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

# Define base dir
BASE_DIR = r"C:\Users\giorg\Auto_Suggest\data\random_github_repos"

# GitHub search query: find .ipynb files with 'pandas'
# query = 'pandas+in:file+extension:ipynb'  # For general pandas search

# Operator-specific queries (preferred to ensure relevance)
operators = ['groupby', 'pivot', 'melt', 'merge']

# Define GitHub search parameters (1 pages × 20 results per page = 20 results)
# Expected final repos: from 30 search results per operator (not repos), expect at least 5 well-structured, replayable notebooks after filtering for operator usage, runtime errors, and missing dependencies or data
per_page = 20 # Max per page is 100
max_pages = 1  # Increase this for better coverage per operator

def search_notebooks_for_operator(operator):
    """Search GitHub and return a list of tuples (repo_full_name, file_path, html_url) for a give n operator."""
    results = []
    print(f"\nSearching GitHub for notebooks using operator '{operator}'...")

    # Query specified by the current operator
        # if operator == 'pivot':
    #     query = 'pivot+OR+pivot_table+in:file+extension:ipynb'  # Adjust query to include both 'pivot' and 'pivot_table'
    # else:
    #     query = f'{operator}+in:file+extension:ipynb'

    # (To improve relevance, we search for operator in pairs instead of individual ones. This increases chances of finding meaningful operator sequences like in the Auto-Suggest paper)
    if operator == 'pivot':
        query = 'pivot+OR+pivot_table+import+pandas+in:file+extension:ipynb'
    elif operator == 'merge':
        query = 'merge+groupby+fillna+import+pandas+in:file+extension:ipynb'
    elif operator == 'melt':
        query = 'melt+groupby+import+pandas+in:file+extension:ipynb'
    else:  # default: groupby
        query = 'groupby+merge+fillna+import+pandas+in:file+extension:ipynb'


    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/search/code?q={query}&per_page={per_page}&page={page}"
        response = requests.get(url, headers=HEADERS)   # GitHub API Call

        # if HTTP status code returned by the GitHub server = 200, everything is OK!
        if response.status_code != 200: # Stop paging if the API response failed (e.g., rate limit, auth error)
            print(f"Error {response.status_code}: {response.text}")
            break

        # Get the contents of the current page
        items = response.json().get("items", [])

        # Iterate over the results and retrieve repo's name, path and url
        for item in items:
            repo_full_name = item["repository"]["full_name"]
            file_path = item["path"]
            html_url = item["html_url"]
            results.append((repo_full_name, file_path, html_url))

        # To avoid hitting rate limits, sleep briefly between pages
        time.sleep(0.5)

    print(f"\nTotal notebooks found for '{operator}': {len(results)}")
    return results

def clone_repositories(results, operator):
    """Clone unique repositories from the search results into a subfolder named after the operator."""
    unique_repos = sorted(set(entry[0] for entry in results))   # Deduplicate the list of results (select unique repos)
    dest_dir = os.path.join(BASE_DIR, operator)     # Create a dir of repos for each operator
    os.makedirs(dest_dir, exist_ok=True)

    print(f"\nCloning {len(unique_repos)} unique repositories into {dest_dir}...\n")
    for i, repo_full_name in enumerate(unique_repos):
        repo_url = f"https://github.com/{repo_full_name}.git"
        repo_name = repo_full_name.split("/")[-1]
        target_path = os.path.join(dest_dir, repo_name)

        if not os.path.exists(target_path):
            print(f"{i+1}. Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url, target_path])    # Clone repo by operator
        else:
            print(f"Already exists: {repo_name}")

if __name__ == "__main__":
    overall_start = time.time()

    for operator in operators:
        print(f"\nStarting '{operator}' ")
        start_time = time.time()

        notebook_results = search_notebooks_for_operator(operator)
        clone_repositories(notebook_results, operator)

        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        print(f"\nFinished '{operator}' in {elapsed:.2f} minutes")

    total_elapsed = (time.time() - overall_start) / 60
    print(f"\nTotal time: {total_elapsed:.2f} minutes\n")

    # Print total cloned repos per operator (this is in case we try to download large amounts of repos, e.g. 900 per operator: we won't get all of them!)
    print("Summary of Downloaded Repos per Operator:")
    for operator in operators:
        operator_dir = os.path.join(BASE_DIR, operator)
        try:
            repo_count = sum(1 for entry in os.scandir(operator_dir) if entry.is_dir())
            print(f"{operator}: {repo_count}")
        except Exception as e:
            print(f"{operator}: [error] {e}")

