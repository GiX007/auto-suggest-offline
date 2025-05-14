# src/data/extract_archives.py
#
# Based on the Auto-Suggest paper, the expected sample counts are:
# - merge (join): 11,200 samples (paper) vs 24,575 samples (actual archive)
# - groupby: 8,900 samples (paper) vs 12,939 samples (actual archive)
# - pivot: 7,700 samples (paper) vs 35,974 samples (actual archive)
# - melt (unpivot): 2,900 samples (paper) vs 4,942 samples (actual archive)
#
"""
This script:
1. Extracts a small number of unique samples for each operator (merge, groupby, pivot, melt)
2. Filters out samples with tables that have fewer than 5 rows
3. Ensures tables are unique by checking their content fingerprints
4. Preserves original notebook names for extracted folders
5. Organizes outputs with consistent file naming (left.csv/right.csv for merge, data.csv for others)
"""

import os
import tarfile
import random
import json
import pandas as pd
import hashlib
import time

# Define paths
ARCHIVE_DIR = r"C:\Users\giorg\Auto_Suggest\data\archives"  # Directory with .tgz files
OUTPUT_DIR = r"C:\Users\giorg\Auto_Suggest\data\extracted"  # Output directory

# Operator archive filenames
OPERATORS = {
    "merge": "merge.tgz",
    "groupby": "groupby.tgz",
    "pivot": "pivot.tgz",
    "melt": "melt.tgz"  # Unpivot is called melt in Pandas
}

# Number of samples to extract for each operator
NUM_SAMPLES = 100
# Minimum rows required in tables
MIN_ROWS = 5


def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"Created directory: {directory}")


def get_table_hash(dataframe):
    """
    Generate a hash for a dataframe to identify duplicate tables.
    We use a sample of the data to create a fingerprint of the table.
    """
    if dataframe is None or len(dataframe) < 1:
        return "empty"

    # Convert a sample of the dataframe to string and hash it
    sample_str = str(dataframe.head(3).values) + str(dataframe.shape)
    return hashlib.md5(sample_str.encode()).hexdigest()


def extract_samples():
    """
    Extract unique samples from archives for each operator.
    Ensures samples are unique and tables have minimum rows.
    """
    # Create output directory
    create_directory(OUTPUT_DIR)

    # Process each operator
    for operator, archive_name in OPERATORS.items():
        print(f"\nExtracting notebook samples for {operator} operator...")

        # Path to the archive
        archive_path = os.path.join(ARCHIVE_DIR, archive_name)

        # Check if archive exists
        if not os.path.exists(archive_path):
            print(f"Archive not found: {archive_path}")
            continue

        # Create operator directory
        operator_dir = os.path.join(OUTPUT_DIR, operator)
        create_directory(operator_dir)

        # Keep track of unique tables using hashes
        unique_hashes = set()
        # List of valid sample directories
        valid_samples = []

        # Open the archive
        with tarfile.open(archive_path, "r:gz") as tar:
            # Get all members
            members = tar.getmembers()

            # Count different file types (for consistent counting with list_archive_contents__.py)
            data_csv_count = sum(1 for m in members if m.name.endswith('data.csv'))
            param_json_count = sum(1 for m in members if m.name.endswith('param.json'))
            left_csv_count = sum(1 for m in members if m.name.endswith('left.csv'))
            right_csv_count = sum(1 for m in members if m.name.endswith('right.csv'))

            # Report numbers using the same method as list_archive_contents__.py
            if operator == "merge":
                # Estimate based on left/right CSV files count
                if left_csv_count > 0 and right_csv_count > 0:
                    estimated_samples = min(left_csv_count, right_csv_count)
                    print(f"Found {estimated_samples} potential notebook samples")
            else:
                # For other operators, use data.csv count
                print(f"Found {data_csv_count} potential notebook samples")

            # Process quietly without verbose logging
            all_dirs = [m.name for m in members if m.isdir()]
            sample_dirs = []
            for member in members:
                if member.isdir():
                    parts = member.name.split('/')
                    if len(parts) >= 1:
                        sample_dirs.append(member.name)

            sample_dirs = sorted(list(set(sample_dirs)))

            # Find sample directories that contain CSV files
            valid_directories = []
            for path in sample_dirs:
                if operator == "merge":
                    # For merge, look for left.csv and right.csv
                    left_files = [m for m in members
                                  if m.name.startswith(path + '/') and
                                  m.name.endswith('left.csv') and
                                  not m.isdir()]
                    right_files = [m for m in members
                                   if m.name.startswith(path + '/') and
                                   m.name.endswith('right.csv') and
                                   not m.isdir()]

                    if left_files and right_files:
                        csv_files = left_files + right_files
                        valid_directories.append((path, csv_files))
                else:
                    # For other operators, look for data.csv
                    csv_files = [m for m in members
                                 if m.name.startswith(path + '/') and
                                 m.name.endswith('.csv') and
                                 not m.isdir()]
                    if csv_files:
                        valid_directories.append((path, csv_files))

            # Shuffle to get a random selection
            random.shuffle(valid_directories)

            # Extract up to NUM_SAMPLES valid samples
            for sample_dir, csv_files in valid_directories:
                # Skip if we already have enough samples
                if len(valid_samples) >= NUM_SAMPLES:
                    break

                # Check if tables are big enough and unique
                is_unique = True
                valid_csv_files = []

                for csv_file in csv_files:
                    # Extract and read the CSV
                    f = tar.extractfile(csv_file)
                    if f is None:
                        continue

                    # Load into pandas
                    try:
                        df = pd.read_csv(f, low_memory=False)

                        # Skip if table is too small
                        if len(df) < MIN_ROWS:
                            is_unique = False
                            break

                        # Generate hash and check for uniqueness
                        table_hash = get_table_hash(df)
                        if table_hash in unique_hashes:
                            is_unique = False
                            break

                        unique_hashes.add(table_hash)
                        valid_csv_files.append(csv_file)
                    except Exception as e:
                        is_unique = False
                        break

                # If sample is valid, add to our list
                if is_unique and valid_csv_files:
                    valid_samples.append((sample_dir, valid_csv_files))

            # Extract the valid samples
            #print(f"{len(valid_samples)} samples extracted")

            for idx, (sample_dir, csv_files) in enumerate(valid_samples, 1):
                # Use the original notebook name for the folder instead of sample_1, sample_2, etc.
                # Extract just the last part of the path to use as folder name
                original_folder_name = os.path.basename(sample_dir)
                sample_output_dir = os.path.join(operator_dir, original_folder_name)
                create_directory(sample_output_dir)

                # Find param.json file if any
                param_files = [m for m in members
                               if m.name.startswith(sample_dir + '/') and
                               m.name.endswith('param.json') and
                               not m.isdir()]

                # Extract CSV files and param file
                files_to_extract = csv_files + param_files

                for file in files_to_extract:
                    # Extract the file
                    f = tar.extractfile(file)
                    if f is None:
                        continue

                    # Get just the filename without path
                    file_basename = os.path.basename(file.name)
                    output_path = os.path.join(sample_output_dir, file_basename)

                    # Save the file
                    with open(output_path, 'wb') as out_file:
                        out_file.write(f.read())

                # For merge operator, rename to left.csv and right.csv if needed
                if operator == "merge":
                    csv_files = [f for f in os.listdir(sample_output_dir) if f.endswith('.csv')]
                    if len(csv_files) >= 2:
                        # Sort by file size (smaller file is typically left table)
                        csv_files.sort(key=lambda f: os.path.getsize(os.path.join(sample_output_dir, f)))

                        # Rename if not already named correctly
                        if "left.csv" not in csv_files and "right.csv" not in csv_files:
                            os.rename(
                                os.path.join(sample_output_dir, csv_files[0]),
                                os.path.join(sample_output_dir, "left.csv")
                            )
                            os.rename(
                                os.path.join(sample_output_dir, csv_files[1]),
                                os.path.join(sample_output_dir, "right.csv")
                            )

                # For other operators, rename main CSV to data.csv if needed
                elif operator in ["groupby", "pivot", "melt"]:
                    csv_files = [f for f in os.listdir(sample_output_dir) if f.endswith('.csv')]
                    if csv_files and "data.csv" not in csv_files:
                        os.rename(
                            os.path.join(sample_output_dir, csv_files[0]),
                            os.path.join(sample_output_dir, "data.csv")
                        )

            print(f"Extracted {len(valid_samples)} unique {operator} samples")


if __name__ == "__main__":
    start_time = time.time()
    print("Starting extraction of unique samples...")
    extract_samples()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format time in a human-readable way
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} seconds"
    elif elapsed_time < 3600:
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes} minutes and {seconds} seconds"
    else:
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours} hours, {minutes} minutes and {seconds} seconds"

    print("\nExtraction complete!")
    print(f"Total extraction time: {time_str}")