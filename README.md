# Auto-Suggest-Offline
Offline component of the Auto-Suggest system for learning pandas operations from real-world Jupyter notebooks, and recommending top choices of how to perform a data preparation operation and what operation to apply next.

## Quickstart

Clone the repository and run everything with:

```python -m src.auto_suggest```

## Architecture Overview

The offline component has two main phases:

  - **Data Collection** – Crawls GitHub notebooks, downloads and replays them to log pandas operator usage  
  - **Models Training & Evaluation** – Trains and evaluates both ML-based and optimization-based models to recommend how to perform each operator (e.g., Join, GroupBy, Pivot, Unpivot). Additionally, trains and evaluates the final model to predict the next likely operator in a sequence.

The pipeline combines these phases into a fully reproducible workflow for learning and predicting pandas operations.

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

- `docs` - Reference materials
   - `Auto-Suggest_SIGMOD2020.pdf` - Original research paper describing the Auto-Suggest system (SIGMOD 2020)
   - `auto_suggest_soft_implementation.pdf` - A report of this soft implementation of Auto-Suggest offline system on a small dataset, providing validation and insights on the core ideas
   - `auto_pipeline_intro` - This short presentation introduces how Auto-Suggest fits into larger pipeline automation systems like Auto-Pipeline (VLDB 2021), and outlines future directions

- `models/` – Contains the trained ML-based models for join, groupby, and next-operator prediction, including both traditional and deep learning approaches. This also includes sequence models (RNN and N-gram) for next-operator prediction and affinity regression models for pivot/unpivot tasks.

- `results/` - Training configurations, evaluation outputs and visualizations
   - `metrics/` – Training configurations, evaluation results and comparison with heuristic methods CSVs
   - `figures/` - Visualizations from training process for ML-based predictors
   - `logs/` - Terminal output logs (.txt files) for training, evaluation and recommendation for both recommendation tasks 

- `src/` – Core source code
  - `data/` – Scripts for notebook collection, processing, and replay
    - `sample_loader.py` - Loads individual or batch operator samples from extracted directories (CSV and param.json files)
    - `github_crawler.py` – Crawls GitHub for notebooks using pandas operators
    - `process_random_repos.py` – Filters notebooks, resolves local/remote datasets, and prepares isolated replay folders (one notebook + dataset(s))
    - `replay_notebooks.py` – Executes notebooks and extracts operator metadata
    - `list_archive_contents.py` – Inspects .tgz archives from the Auto-Suggest dataset without extracting them and reports file counts
    - `extract_archives.py` – Extracts a small, unique subset of high-quality operator samples (e.g., 30–100) from the full archive
    - `generate_data.py` – Generates synthetic operator sequences, prepares datasets for N-gram/RNN/MLP models, and computes statistics

  - `models/` – Model logic and operator-specific algorithms
    - `join_col_model.py` - ML-based join key predictor
    - `join_type_model.py` - ML-based Join type classifier (inner, left, outer)
    - `groupby_model.py` - ML-based Groupby classifier (dimension/measures columns)
    - `pivot_model.py` - Pivot column grouping using AMPT (graph-based partitioning optimization)
    - `unpivot_model.py` - Unpivot column prediction using CMUT (graph-based partitioning optimization)
    - `ngram_rnn_models.py` - Defines and runs (train, evaluate, predict) N-gram and RNN next-operator prediction models
    - `next_operation_predictor.py` - Builds, trains, and evaluates the final MLP model for next-operator prediction using RNN-based operator sequence modeling and single operator-specific prediction scores.
   
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

  - `main.py` - Main driver script for the first recommendation task implementation (single-operator prediction). Provides functions for training, evaluating, and predicting each operator, along with command-line support and overall workflow orchestration.
  - `auto_suggest.py` - Runs everything


## Getting Started

Once the environment is set up, this section provides a complete walkthrough of all `.py` files in the project.

First, **crawl, download, process, and replay notebooks** to extract pandas operator usage. This example uses a **small number of samples per operator**:

```
python -m src.data.github_crawler
python -m src.data.process_random_repos
python -m src.data.replay_notebooks
```

This is an example pipeline for a small set of notebooks. To scale up to more samples, you will need more compute resources and may need to adapt the code to better handle complex or messy notebooks to ensure you get well-structured and replayable data.

To apply Auto-Suggest using some pre-processed data provided by the authors, you can list and extract sample data:

```
# See and list some samples
python src/data/list_archive_contents

# Extract a small, high-quality subset for testing
python src/data/extract_archives_small
```

For the first recommendation task (how to perform a pandas operation), you can run all cases at once:

```python -m src.main --operator all --mode all ```

Or run each operator individually:

```
# Train join models
python -m src.main --operator join --mode train

# Evaluate join models
python -m src.main --operator join --mode eval

# Predict on new unseen data
python -m src.main --operator join --mode predict --left_file data/test_data/join_customers.csv --right_file data/test_data/join_orders.csv

# Train groupby model
python -m src.main --operator groupby --mode train

# Evaluate groupby model
python -m src.main --operator groupby --mode eval

# Predict on new unseen data
python -m src.main --operator groupby --mode predict --input_file data/test_data/groupby_sales_data.csv

# Run/Evaluate pivot model
python -m src.main --operator pivot --mode train
python -m src.main --operator pivot --mode eval

# Predict on new unseen data
python -m src.main --operator pivot --mode predict --input_file data/test_data/pivot_financial_data.csv

# Or specify a different aggregation function (default is mean)
python -m src.main --operator pivot --mode predict --input_file data/test_data/pivot_financial_data.csv --aggfunc sum

# Run/Evaluate unpivot model
python -m src.main --operator unpivot --mode train
python -m src.main --operator unpivot --mode eval

# Predict on new unseen data
python -m src.main --operator unpivot --mode predict --input_file data/test_data/unpivot_product_sales.csv
python -m src.main --operator unpivot --mode predict --input_file data/test_data/unpivot_regional_metrics.csv
```

For the second recommendation task (predicting the next likely operator), there is a need to generate artificial operator sequences, as such data is not provided by authors, and the combined feature data for the final MLP model:

```
# Generate sequence data for sequence models as well as combined data for the final MLP model
python -m src.data.generate_data
```

To train and evaluate sequence models (N-gram and RNN) on these synthetic sequences:

```
# Train/Evaluate/Predict N-gram or RNN models all at once
python -m src.models.ngram_rnn_models --model ngram --mode all
python -m src.models.ngram_rnn_models --model rnn --mode all

# Or train/evaluate/predict them separately
python -m src.models.ngram_rnn_models --model ngram train
python -m src.models.ngram_rnn_models --model ngram eval
python -m src.models.ngram_rnn_models --model ngram predict

python -m src.models.ngram_rnn_models --model rnn train
python -m src.models.ngram_rnn_models --model rnn eval
python -m src.models.ngram_rnn_models --model rnn predict

```

Finally, to train, evaluate, and predict with the final Auto-Suggest model, which combines the RNN sequence context and single-operator predictions in a final MLP layer:

```
# Train/Evaluate/Predict next operator all at once
python -m src.models.next_operation_predictor --mode all

# Or run each phase separately
python -m src.models.next_operation_predictor --mode train
python -m src.models.next_operation_predictor --mode eval
python -m src.models.next_operation_predictor --mode predict --input_file data/test_data/unpivot_product_sales.csv --history "dropna, merge, pivot"

```

## Notes

- To extract a sufficient number of **well-structured and replayable notebooks**, the pipeline may need to crawl **hundreds of GitHub repositories** per operator.
- This implementation is currently set to process only a **few notebooks per operator** as a lightweight simulation.
- In real-world use, you should:
  - Increase the number of pages crawled in `github_crawler.py`
  - Broaden search queries (e.g., include operator combinations or fallback terms)
  - Tolerate failed extractions due to broken notebooks, missing data files, or complex operator chains
- **Important:** This is a minimal simulation for a small number of datasets. It does not guarantee success with large-scale or unstructured repositories.
- To fully understand the process of **crawling, downloading, replaying notebooks**, as well as **training and evaluating all models**, I recommend exploring the tutorials provided in the `tutorials/` folder:
  - `download_and_replay_notebooks.ipynb` – Demonstrates downloading, filtering, and replaying a few example notebooks step by step (cell by cell)
  - `toy_recommendation_workflow_simulations.ipynb` – Simulates the full Auto-Suggest recommendation pipeline using toy data. Covers feature extraction, single-operator prediction, RNN/n-gram modeling, and final MLP-based next-operator prediction
- The original paper is in `docs/`, as well as a detailed report regarding our implementation: what we did differently, what we followed, and further insights.

## Contributing

Contributions and feedback are welcome! Feel free to open issues or pull requests.
