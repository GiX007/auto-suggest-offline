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
    - `list_archive_contents.py` – Inspects .tgz archives from the Auto-Suggest dataset without extracting them and reports file counts
    - `extract_archives.py` – Extracts a small, unique subset of high-quality operator samples (e.g., 30–100) from the full archive

  - `tutorials/` – Notebook-based walkthroughs and tests
    - `download_and_replay_notebooks.ipynb` – Demonstrates downloading, filtering, and replaying a few example notebooks step by step (cell by cell)


## Getting Started


## Notes

- To extract a sufficient number of **well-structured and replayable notebooks**, the pipeline may need to crawl **hundreds of GitHub repositories** per operator.
- This implementation is currently set to process **only 20 repositories per operator** as a lightweight simulation.
- In real-world use, you should:
  - Increase the number of pages crawled in `github_crawler.py`
  - Broaden search queries (e.g., include operator combinations or fallback terms)
  - Tolerate failed extractions due to broken notebooks, missing data files, or complex operator chains
- You can adjust the crawling and filtering logic in `github_data_collection.py` to scale up and tune output quality.
