"""
github_data_collection.py

Runs the Auto-Suggest offline pipeline for getting data:
1. Crawl GitHub for notebooks using pandas operators
2. Filter repositories for valid notebooks and datasets
3. Prepare replay-ready notebook folders (1 notebook + N datasets)
4. Replay notebooks to extract operator traces, parameters, and DAGs
"""

import subprocess

# === Step 1: Crawl GitHub for notebooks (per operator)
print("\nStep 1: Crawling GitHub notebooks...\n")
subprocess.run(["python", "-m", "src.data.github_crawler"])

# === Step 2: Filter notebooks and datasets (resolve local/URL/Kaggle)
print("\nStep 2: Filtering notebooks and resolving datasets...\n")
subprocess.run(["python", "-m", "src.data.process_random_repos"])

# === Step 3: Prepare isolated folders for replay
print("\nStep 3: Preparing replay notebooks...\n")
subprocess.run(["python", "-m", "src.data.prepare_replay_notebooks"])

# === Step 4: Replay notebooks and extract operators metadata
print("\nStep 4: Replaying notebooks and saving outputs...\n")
subprocess.run(["python", "-m", "src.data.replay_notebooks"])

print("\nAll steps have completed successfully.\n")
