# list_archive_contents.py
#
# This script examines the contents of .tgz archive files without extracting them.
# It shows:
# 1. The size of each archive file
# 2. One sample notebook path to understand the structure
# 3. How many data.csv/left.csv/right.csv and param.json files exist in the archive
#

import os
import tarfile
import time

def list_archive_contents(archive_path):
    """List key contents of a tar archive without showing all members"""
    print(f"\nExamining archive: {os.path.basename(archive_path)}")
    try:
        with tarfile.open(archive_path, 'r:gz') as tar:
            # Get all members
            members = tar.getmembers()

            # Count different file types
            data_csv_count = sum(1 for m in members if m.name.endswith('data.csv'))
            param_json_count = sum(1 for m in members if m.name.endswith('param.json'))
            left_csv_count = sum(1 for m in members if m.name.endswith('left.csv'))
            right_csv_count = sum(1 for m in members if m.name.endswith('right.csv'))

            # Find a sample entry for display
            sample_entry = None

            # For merge.tgz (look for left.csv/right.csv)
            if "merge" in os.path.basename(archive_path).lower():
                merge_samples = {}
                for member in members:
                    if member.name.endswith(('left.csv', 'right.csv', 'param.json')):
                        notebook_path = os.path.dirname(member.name)
                        if notebook_path not in merge_samples:
                            merge_samples[notebook_path] = []
                        merge_samples[notebook_path].append(os.path.basename(member.name))

                # Find a complete merge notebook
                complete_samples = [path for path, files in merge_samples.items()
                                    if
                                    len(files) >= 3 and 'left.csv' in files and 'right.csv' in files and 'param.json' in files]

                if complete_samples:
                    sample_path = complete_samples[0]
                    print("\nSample notebook (merge format):")
                    print(f"  {sample_path}")
                    print(f"  {sample_path}/param.json")
                    print(f"  {sample_path}/left.csv")
                    print(f"  {sample_path}/right.csv")

            # For other operations (look for data.csv)
            else:
                sample_notebooks = {}
                for member in members:
                    if member.name.endswith(('data.csv', 'param.json')):
                        notebook_path = os.path.dirname(member.name)
                        if notebook_path not in sample_notebooks:
                            sample_notebooks[notebook_path] = []
                        sample_notebooks[notebook_path].append(os.path.basename(member.name))

                # Find a complete notebook (with both data.csv and param.json)
                complete_notebooks = [path for path, files in sample_notebooks.items()
                                      if len(files) >= 2 and 'data.csv' in files and 'param.json' in files]

                # Print one sample notebook
                if complete_notebooks:
                    sample_path = complete_notebooks[0]
                    print("\nSample notebook:")
                    print(f"  {sample_path}")
                    print(f"  {sample_path}/param.json")
                    print(f"  {sample_path}/data.csv")

            # Print file counts
            print(f"\nFound {data_csv_count} data.csv files")
            print(f"Found {param_json_count} param.json files")

            # For merge operation, also show left/right counts
            if left_csv_count > 0 or right_csv_count > 0:
                print(f"Found {left_csv_count} left.csv files")
                print(f"Found {right_csv_count} right.csv files")

                # Estimate number of complete samples
                if left_csv_count > 0 and right_csv_count > 0:
                    estimated_samples = min(left_csv_count, right_csv_count)
                    print(f"Estimated number of merge samples: {estimated_samples}")
            # For other operators, estimate based on param.json and data.csv counts
            else:
                estimated_samples = min(data_csv_count, param_json_count)
                operator_name = os.path.basename(archive_path).replace('.tgz', '')
                print(f"Estimated number of {operator_name} samples: {estimated_samples}")

    except Exception as e:
        print(f"Error examining archive: {e}")

# Directory containing the archives
archive_dir = r"C:\Users\giorg\Auto_Suggest\data\archives"

# Find all .tgz files
tgz_files = [os.path.join(archive_dir, f) for f in os.listdir(archive_dir) if f.endswith('.tgz')]

start_time = time.time()
print("Starting archive analysis...")

for archive_path in tgz_files:
    # Get file size in MB
    size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"\n{'=' * 50}")
    print(f"Archive: {os.path.basename(archive_path)}")
    print(f"Size: {size_mb:.2f} MB")

    list_archive_contents(archive_path)

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

print(f"\nTotal analysis time: {time_str}")