# auto-suggest-offline
Offline component of the Auto-Suggest system for learning pandas operations from real-world Jupyter notebooks.

## Contents

- `data/` – Supporting data and examples
  - `examples/` - Contains all dummy Jupyter notebooks and corresponding datasets used in the tutorial notebooks
  - `extracted_data.zip` - Includes 100 replayed samples per operator (groupby, melt, pivot, merge) in the format expected by the system:
    - `data.csv` (or `left.csv` / `right.csv` for merge) - Input tables for the operator
    - `param.json` - Parameters used in the operator call
  - `generated_data/` - Contains combined data, operator sequences, and related statistics for training and evaluating next operator prediction models
  - `test_data/` - Contains unseen .csv tables and pre-split train/validation/test sets for join, groupby, and the final combined operator model
 
  **Note**: Due to GitHub file size limits, this dataset is not included in the repository. If you are interested in accessing it, feel free to contact me and I’ll 
  share it via Google Drive. We have only uploaded the examples and test ```.csv``` files.

- `src/` – Core source code
  - `data/` – Scripts for notebook collection, processing, and replay
    - `sample_loader.py` - Loads individual or batch operator samples from extracted directories (CSV and param.json files)
    - `github_crawler.py` – Crawls GitHub for notebooks using pandas operators
    - `process_random_repos.py` – Filters notebooks and resolves local/remote datasets
    - `prepare_replay_notebooks.py` – Creates replay folders (one notebook + dataset(s))
    - `replay_notebooks.py` – Executes notebooks and extracts operator metadata
    - `github_data_collection.py` – Runs the full **data manipulation pipeline**: crawl → filter → prepare → replay
    - `generate_op_sequences.py` - Generates synthetic sequences of operator calls for training/evaluating N-gram and RNN models (next-operator prediction components)
    - `generate_combined_data` - Builds feature vectors for the final MLP model by combining RNN predictions with operator-specific scores (groupby, pivot, unpivot, join)
    - `list_archive_contents.py` – Inspects .tgz archives from the Auto-Suggest dataset without extracting them and reports file counts
    - `extract_archives.py` – Extracts a small, unique subset of high-quality operator samples (e.g., 30–100) from the full archive

  - `models/` – Model logic and operator-specific algorithms
    - `join_col_model.py` - ML-based join key predictor
    - `join_type_model.py` - ML-based Join type classifier (inner, left, outer)
    - `groupby_model.py` - ML-based Groupby classifier (dimension/measures columns)
    - `pivot_model.py` - Pivot column grouping using AMPT (graph-based partitioning optimization)
    - `unpivot_model.py` - Unpivot column prediction using CMUT (graph-based partitioning optimization)
    - `ngram_rnn_models.py` - Defines and runs (train, evaluate, predict) N-gram and RNN next-operator prediction models
   
  - `baselines/` – Heuristic baseline implementations for each operator
    - `join_baselines.py` - Heuristic methods like ML-FK, PowerPivot, Multi, Holistic, Max-Overlap snd Vendors
    - `groupby_baselines.py` - Heuristic methods like SQL history, Coarse-grained types, Fine-grained types and Min-cardinality
    - `pivot_baselines.py` - Heuristic methods like Affinity, Type-rules, Min-emptiness and Balanced-split
    - `unpivot_baselines.py` - Heuristics like Pattern, Column-name, Data-type and Contiguous-type similarities
   
  - `utils/` – Shared utilities
     - `model_utils` - Shared helper utilities for model persistence, evaluation, and prediction visualization

  - `tutorials/` – Notebook-based walkthroughs and tests
    - `download_and_replay_notebooks.ipynb` – Demonstrates downloading, filtering, and replaying a few example notebooks step by step (cell by cell)
    - `toy_recommendation_workflow_simulations.ipynb` – Simulates the full Auto-Suggest recommendation pipeline using toy data. Covers feature extraction, single-operator prediction, RNN/n-gram modeling, and final MLP-based next-operator prediction

- `models/` – Contains the trained ML-based models for join, groupby, and next-operator prediction, including both traditional and deep learning approaches. This also includes sequence models (RNN and N-gram) for next-operator prediction and affinity regression models for pivot/unpivot tasks.

- `results/` - Training configurations, evaluation outputs and visualizations
   - `metrics/` – Training configurations, evaluation results and comparison with heuristic methods CSVs
   - `figures/` - Visualizations from training process for ML-based predictors
   - `logs/` - Terminal output logs (.txt files) for training, evaluation and recommendation for both recommendation tasks 

- `docs` - Reference materials
   - `Auto-Suggest_SIGMOD2020.pdf` - Original research paper describing the Auto-Suggest system (SIGMOD 2020)
   - `auto_suggest_soft_implementation.pdf` - A minimal implementation and simulation of the Auto-Suggest system on a small dataset, providing validation and insights on the core ideas


## Getting Started


## Notes

- To extract a sufficient number of **well-structured and replayable notebooks**, the pipeline may need to crawl **hundreds of GitHub repositories** per operator.
- This implementation is currently set to process **only 20 repositories per operator** as a lightweight simulation
- In real-world use, you should:
  - Increase the number of pages crawled in `github_crawler.py`
  - Broaden search queries (e.g., include operator combinations or fallback terms)
  - Tolerate failed extractions due to broken notebooks, missing data files, or complex operator chains
- You can adjust the crawling and filtering logic in `github_data_collection.py` to scale up and tune output quality
