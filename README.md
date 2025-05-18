# auto-suggest-offline
Offline component of the Auto-Suggest system for learning pandas operations from real-world Jupyter notebooks.

## Contents

- `data/` – Supporting data and examples
  - `examples` - Contains all dummy Jupyter notebooks and corresponding datasets used in the tutorial notebooks
  - `extracted_data.zip` - Includes 100 replayed samples per operator (groupby, melt, pivot, merge) in the format expected by the system:
    - `data.csv` (or `left.csv` / `right.csv` for merge) - Input tables for the operator
    - `param.json` - Parameters used in the operator call
      
    **Note**: Due to GitHub file size limits, this dataset is not included in the repository.
    If you are interested in accessing it, feel free to contact me and I’ll share it via Google Drive.

  - `test_data` - Contains unseen `.csv` tables used to verify whether a trained single-operator model (`groupby`, `melt`, `pivot`, `merge`) can correctly predict the appropriate operator parameters on new, unobserved inputs.

- `src/` – Core source code
  - `data/` – Scripts for notebook collection, processing, and replay
    - `sample_loader.py` - Loads individual or batch operator samples from extracted directories (CSV and param.json files)
    - `github_crawler.py` – Crawls GitHub for notebooks using pandas operators
    - `process_random_repos.py` – Filters notebooks and resolves local/remote datasets
    - `prepare_replay_notebooks.py` – Creates replay folders (one notebook + dataset(s))
    - `replay_notebooks.py` – Executes notebooks and extracts operator metadata
    - `github_data_collection.py` – Runs the full **data manipulation pipeline**: crawl → filter → prepare → replay
    - `list_archive_contents.py` – Inspects .tgz archives from the Auto-Suggest dataset without extracting them and reports file counts
    - `extract_archives.py` – Extracts a small, unique subset of high-quality operator samples (e.g., 30–100) from the full archive

  - `features/` – Feature extraction process per operator
    - `join_features.py` – Extracts join key compatibility features (e.g., distinct value ratio, leftness, value overlap, etc)
    - `groupby_features.py` – Extracts features for identifying groupby candidates (e.g., distinct value counts, column data type, etc)
    - `pivot_features.py` - Builds affinity matrices between candidate pivot columns
    - `unpivot_features.py` - Measures compatibility for unpivot group detection 
   
  - `models/` – Model logic and operator-specific algorithms
    - `join_col_model.py` - ML-based join key predictor
    - `join_type_model.py` - ML-based Join type classifier (inner, left, outer)
    - `groupby_model.py` - ML-based Groupby classifier (dimension/measures columns)
    - `pivot_model.py` - Pivot column grouping using AMPT (graph-based partitioning optimization)
    - `unpivot_model.py` - Unpivot column prediction using CMUT (graph-based partitioning optimization)
   
  - `baselines/` – Heuristic baseline implementations for each operator
    - `join_baselines.py` - Heuristic methods like ML-FK, PowerPivot, Multi, Holistic, Max-Overlap snd Vendors
    - `groupby_baselines.py` - Heuristic methods like SQL history, Coarse-grained types, Fine-grained types and Min-cardinality
    - `pivot_baselines.py` - Heuristic methods like Affinity, Type-rules, Min-emptiness and Balanced-split
    - `unpivot_baselines.py` - Heuristics like Pattern, Column-name, Data-type and Contiguous-type similarities
   
  - `utils/` – Shared utilities
     - `evaluation.py` - Several evaluation utilities for join and group predictions 
     - `join_recommendation_pipeline.py` - Pipeline to combine join column and join type predictions
     - `model_utils` - Shared helper functions for saving and loading models

  - `tutorials/` – Notebook-based walkthroughs and tests
    - `download_and_replay_notebooks.ipynb` – Demonstrates downloading, filtering, and replaying a few example notebooks step by step (cell by cell)
    - `toy_recommendation_workflow_simulations.ipynb` – Simulates the full Auto-Suggest recommendation pipeline using toy data. Covers feature extraction, single-operator prediction, RNN/n-gram modeling, and final MLP-based next-operator prediction.

- `models/` – Trained ML-based models (100 samples per operator)
  - `join_col_model.pkl` – Join column prediction model
  - `join_type_model.pkl` – Join type classification model
  - `groupby_model.pkl` – Groupby column prediction model

- `workflows/` – Execution flow guides for each operator. It contains operator-specific .txt guides outlining the key steps and function calls for training, evaluation, and recommendation workflows 
   across join, groupby, pivot, and unpivot modules.

## Getting Started


## Notes

- To extract a sufficient number of **well-structured and replayable notebooks**, the pipeline may need to crawl **hundreds of GitHub repositories** per operator.
- This implementation is currently set to process **only 20 repositories per operator** as a lightweight simulation.
- In real-world use, you should:
  - Increase the number of pages crawled in `github_crawler.py`
  - Broaden search queries (e.g., include operator combinations or fallback terms)
  - Tolerate failed extractions due to broken notebooks, missing data files, or complex operator chains
- You can adjust the crawling and filtering logic in `github_data_collection.py` to scale up and tune output quality.
