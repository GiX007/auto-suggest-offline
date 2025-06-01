#
# auto_suggest.py
#
# Runs everything.
#

import subprocess

def run_cmd(command):
    print(f"\nRunning: {' '.join(command)}\n")
    subprocess.run(command, check=True)

if __name__ == "__main__":
    # Step 1: Crawl GitHub for notebooks (per operator)
    run_cmd(["python", "-m", "src.data.github_crawler"])

    # Step 2: Filter notebooks and resolve datasets
    run_cmd(["python", "-m", "src.data.process_random_repos"])

    # Step 3: Replay notebooks and extract operators metadata
    run_cmd(["python", "-m", "src.data.replay_notebooks"])

    # Step 4: Examine archive files (https://github.com/congy/AutoSuggest) without extracting them
    run_cmd(["python", "src/data/list_archive_contents.py"])

    # Step 5: Extract a small dataset for testing models
    run_cmd(["python", "src/data/extract_archives_small.py"])

    # Step 6: Train/Eval/Predict All Operator Models at Once
    run_cmd(["python", "-m", "src.main", "--operator", "all", "--mode", "all"])

    # Step 7: Generate synthetic data for MLP model
    run_cmd(["python", "-m", "src.data.generate_data"])

    # Step 8: Train/Eval/Predict N-gram and RNN sequence models
    run_cmd(["python", "-m", "src.models.ngram_rnn_models", "--model", "ngram", "--mode", "all"])
    run_cmd(["python", "-m", "src.models.ngram_rnn_models", "--model", "rnn", "--mode", "all"])

    # Step 9: Train/Eval/Predict Next Operator Prediction model
    run_cmd(["python", "-m", "src.models.next_operation_predictor", "--mode", "all"])

    print("\nAll steps of Auto-Suggest have completed successfully.\n")
