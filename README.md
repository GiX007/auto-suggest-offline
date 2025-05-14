# auto-suggest-offline
Offline component of the Auto-Suggest system for learning pandas operator pipelines from real-world Jupyter notebooks.

## Contents

- `src/` – Core source code
  - `data/` – Scripts for notebook collection, processing, and replay
    - `github_crawler.py` – Crawls GitHub for notebooks using pandas operators
    - `process_random_repos.py` – Filters notebooks and resolves local/remote datasets
    - `prepare_replay_notebooks.py` – Creates replay folders (one notebook + dataset(s))
    - `replay_notebooks.py` – Executes notebooks and extracts operator metadata
    - `github_data_collection.py` – Runs the full **data manipulation pipeline**: crawl → filter → prepare → replay

- `tutorials/` – Notebook-based walkthroughs and tests
  - `download_and_replay_notebooks.ipynb` – Demonstrates downloading, filtering, and replaying a few example notebooks step by step (cell by cell)
