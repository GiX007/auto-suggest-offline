"""
replay_notebooks.py

This script replays each notebook from:
  data/prepared_replay_notebooks/<operator>/<repo>__<notebook>/

It detects pandas operations of interest:
  groupby, merge, melt, pivot

For each operation, it saves the following to:
  data/extracted_github_data/<operator>/<notebook>__cellX/
    - param.json       → extracted input parameters
    - op_seq.json      → upstream operations
    - data.csv         → input dataframe for the operator

The maximum execution time per cell is 5 minutes.
Each replayed notebook is isolated and produces one output folder per operator call.

"""

import os
import json
import hashlib
import nbformat
import pandas as pd
import ast
import contextlib
import re
import importlib
import subprocess
import sys
import threading
import warnings
import logging
from graphviz import Digraph

# === PLOT OUTPUT SUPPRESSION (Matplotlib & Plotly) ===

# Suppress matplotlib plots entirely (e.g., plt.show())
import matplotlib
matplotlib.use("Agg")  # suppress all popups

import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None  # disable popup windows
os.environ["PLOTLY_RENDERER"] = "none"


# Suppress Plotly from opening browser windows
import plotly.io as pio
pio.renderers.default = 'svg'  # render in background

# Suppress tqdm progress bars from notebooks being replayed
import builtins
builtins.tqdm = lambda iterable=None, *a, **k: iterable if iterable is not None else iter([])


# === PATH CONFIGURATION ===
PREPARED_DIR = r"C:\Users\giorg\Auto_Suggest\data\prepared_replay_notebooks"
OUTPUT_DIR = r"C:\Users\giorg\Auto_Suggest\data\extracted_github_data"

track_ops = ['dropna', 'fillna', 'concat', 'merge', 'melt', 'pivot', 'pivot_table', 'groupby']
main_ops = ['groupby', 'merge', 'melt', 'pivot', 'pivot_table']


# === UTILITY FUNCTIONS ===

class ExecutionTimeout(Exception):
    """
    Custom exception raised when a code cell exceeds the maximum execution time.
    Used to abort notebook replay when a long-running cell hangs or loops indefinitely.
    """
    pass

def run_with_timeout(code, env, timeout=300):   # 300 seconds for 5 minutes max execution time per cell
    """
    Executes a code snippet in a separate thread and aborts if it exceeds the timeout.

    Args:
      code (str): Python code to execute
      env (dict): Execution environment (globals dictionary)
      timeout (int): Maximum time (in seconds) allowed for execution

    Raises:
      ExecutionTimeout: If the execution exceeds the allowed timeout
      Exception: Any exception raised within the executed code
    """
    def target():
        try:
            exec(code, env)
        except Exception as e:
            env['_error'] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise ExecutionTimeout
    if '_error' in env:
        raise env['_error']


@contextlib.contextmanager
def suppress_output():
    """
    Temporarily suppresses standard output and error streams. Useful for hiding noisy outputs (e.g., from exec, warnings, or imports) during notebook replay or dynamic code execution.
    Usage: with suppress_output():
                exec(code)
    """

    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def try_import_or_install(module_name):
    """
    Tries to import a module: if not found, attempts to install it via pip.
    Returns True if the module is importable after install, else False.
    """
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        print(f"\nMissing package '{module_name}', attempting to install...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", module_name], check=True)
            importlib.invalidate_caches()
            importlib.import_module(module_name)
            print(f"Successfully installed and imported '{module_name}'")
            return True
        except Exception as e:
            print(f"Failed to install '{module_name}': {e}")
            return False

def capture_dataframes(env):
    """
    Scans the given execution environment for pandas DataFrame objects.

    For each DataFrame found:
      - Computes a hash of its content using md5
      - Stores a copy of the DataFrame along with its hash

    Returns:
       dict[str, (pd.DataFrame, str)]: A mapping from variable name to a tuple of (copied DataFrame, hash string)

    Useful for:
      - Tracking DataFrame versions after each cell execution
      - Building a dataflow graph based on content-level identity
    """
    df_map = {}
    for name, obj in env.items():
        if isinstance(obj, pd.DataFrame):
            h = hashlib.md5(pd.util.hash_pandas_object(obj, index=True).values).hexdigest()
            df_map[name] = (obj.copy(), h)
    return df_map


def trace_full_upstream(df_hash, graph, visited=None):
    """
    Recursively traces the full upstream sequence of operations that led to a given DataFrame in the dataflow graph.

    Args:
      df_hash (str): Hash of the target DataFrame node to trace from.
      graph (dict): A dictionary representing the dataflow graph. Each node maps to a list of (parent_hash, operation) pairs.
      visited (set, optional): Tracks already visited nodes to avoid cycles.

    Returns: list[str]: A deduplicated list of operations (in order) that produced the target DataFrame.

    Useful for:
      - Reconstructing the preprocessing pipeline leading to a given result
      - Saving `op_seq.json` files as done in Auto-Suggest
    """
    if visited is None:
        visited = set()
    if df_hash in visited or df_hash not in graph:
        return []
    visited.add(df_hash)
    ops = []
    for parent_hash, op in graph[df_hash]['parents']:
        ops += trace_full_upstream(parent_hash, graph, visited)
        ops.append(op)
    return list(dict.fromkeys(ops))


def extract_groupby_params(line):
    """
    Extracts key parameters from a line of code that performs a pandas groupby operation.

    Specifically:
        - Parses the `by` argument passed to `groupby(...)`
        - Attempts to identify the value column being accessed, either via:
          - bracket notation (e.g., df.groupby(...)[value])
          - chained method access (e.g., df.groupby(...).value)

    Args: line (str): A line of code containing a groupby operation.

    Returns: dict: { 'by': List of column(s) used to group,
                     'value': List containing the accessed value column, if found }

    Robust to:
      - Single or list-style groupby arguments
      - Partial errors in eval()
      - Common aggregation methods like mean(), sum(), etc.

    Example Output: {'by': ['Sex'], 'value': ['Survived']}
    """
    by_cols, value_cols = [], []
    match = re.search(r'groupby\((.*?)\)', line)
    if match:
        groupby_arg = match.group(1).strip()
        try:
            by_cols = eval(groupby_arg) if groupby_arg.startswith('[') else [groupby_arg]
        except:
            by_cols = [groupby_arg.strip("'").strip('"')]
    match_val = re.search(r'\)\s*\[\s*[\'"](.+?)[\'"]\s*\]', line)
    if match_val:
        value_cols = [match_val.group(1)]
    else:
        match_dot = re.search(r'\)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if match_dot:
            candidate = match_dot.group(1)
            if candidate.lower() not in ['mean', 'sum', 'count', 'max', 'min', 'median', 'std', 'var']:
                value_cols = [candidate]
    return {'by': by_cols, 'value': value_cols}


def extract_melt_params(line):
    """
    Extracts the `id_vars` and `value_vars` arguments from a line of code containing a call to `pd.melt(...)`.

    Specifically:
      - Parses assignments like: id_vars=["col1", "col2"], value_vars="col3"
      - Supports both list and string formats
      - Handles typical usage patterns from pandas melt calls

    Args: line (str): A line of Python code that calls pd.melt() or DataFrame.melt()

    Returns: dict: { 'id_vars': List of identifier variables (columns to keep fixed),
                     'value_vars': List of columns to unpivot }

    Robust to:
      - Single or multiple column assignments
      - eval() fallback if input isn't valid Python literal

    Example Output: {'id_vars': ['country', 'year'], 'value_vars': ['pop', 'gdp']}
    """
    id_vars, value_vars = [], []
    id_match = re.search(r'id_vars\s*=\s*(\[.*?\]|\".*?\"|\'.*?\')', line)
    val_match = re.search(r'value_vars\s*=\s*(\[.*?\]|\".*?\"|\'.*?\')', line)
    try:
        if id_match:
            id_vars = eval(id_match.group(1))
            if isinstance(id_vars, str): id_vars = [id_vars]
        if val_match:
            value_vars = eval(val_match.group(1))
            if isinstance(value_vars, str): value_vars = [value_vars]
    except:
        pass
    return {'id_vars': id_vars, 'value_vars': value_vars}


def extract_merge_params(line):
    """
    Extracts join key parameters from a line of code that performs a pandas merge operation.

    Supports both:
      - Symmetric merges using `on=...`
      - Asymmetric merges using `left_on=...` and `right_on=...`
      - Single-column and multi-column keys

    Args: line (str): A line of code containing a pandas merge or join operation.

    Returns: dict: { 'left_on': str or list of str — column(s) used as left join key,,
                     'right_on': str or list of str — column(s) used as right join key }

    Behavior:
      - If `on=` is present and `left_on`/`right_on` are not, it is used for both sides
      - Uses eval() to parse lists (e.g., on=["id", "date"])

    Example Outputs:
        {'left_on': 'user_id', 'right_on': 'uid'}
        {'left_on': ['id', 'timestamp'], 'right_on': ['uid', 'ts']}

    Limitations:
      - Assumes keys are expressed directly in the line as literals (e.g., on="id" or on=["id", "date"])
      - Does not resolve variables or function calls used in join keys (e.g., on=join_keys or on=get_key_columns(df))
    """
    left_on, right_on, on = None, None, None
    on_match = re.search(r'on\s*=\s*[\'"](\w+)[\'"]', line)
    left_match = re.search(r'left_on\s*=\s*[\'"](\w+)[\'"]', line)
    right_match = re.search(r'right_on\s*=\s*[\'"](\w+)[\'"]', line)
    if on_match:
        on = on_match.group(1)
    if left_match:
        left_on = left_match.group(1)
    if right_match:
        right_on = right_match.group(1)
    if on and not (left_on or right_on):
        left_on = right_on = on
    return {'left_on': left_on, 'right_on': right_on}


def extract_pivot_params(line):
    """
    Extracts key arguments from a line of code that performs a pandas pivot or pivot_table operation.

    Specifically:
      - Parses the `index=`, `columns=`, and `values=` parameters
      - Supports both single-column (string) and multi-column (list) formats

    Args: line (str): A line of Python code containing a call to pivot or pivot_table.

    Returns: dict: { 'index': List of column(s) used as row index,
                     'columns': List of column(s) to spread across columns,
                     'values': List of column(s) to populate the values }

    Robust to:
      - Variants like: index='col', index=['col1', 'col2']
      - Syntax inconsistencies (basic eval fallback)

    Example Output: {'index': ['date'], 'columns': ['region'], 'values': ['sales']}
    """
    index, columns, values = [], [], []
    idx_match = re.search(r'index\s*=\s*(\[.*?\]|\".*?\"|\'.*?\')', line)
    col_match = re.search(r'columns\s*=\s*(\[.*?\]|\".*?\"|\'.*?\')', line)
    val_match = re.search(r'values\s*=\s*(\[.*?\]|\".*?\"|\'.*?\')', line)
    try:
        if idx_match:
            index = eval(idx_match.group(1))
            if isinstance(index, str): index = [index]
        if col_match:
            columns = eval(col_match.group(1))
            if isinstance(columns, str): columns = [columns]
        if val_match:
            values = eval(val_match.group(1))
            if isinstance(values, str): values = [values]
    except:
        pass
    return {'index': index, 'columns': columns, 'values': values}


# === MAIN REPLAY LOOP ===

# Suppress common warnings and notebook-generated logs
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

# Try to override tqdm from libraries (e.g. pandas, transformers, etc.)
try:
    import tqdm
    tqdm.tqdm = lambda *a, **k: a[0] if a else iter([])
except ImportError:
    pass


for operator in os.listdir(PREPARED_DIR):

    print(f"\nReplaying notebooks for {operator} operator...\n")
    operator_path = os.path.join(PREPARED_DIR, operator)
    if not os.path.isdir(operator_path):
        continue

    for folder in os.listdir(operator_path):
        nb_dir = os.path.join(operator_path, folder)
        if not os.path.isdir(nb_dir):
            continue

        nb_file = next((f for f in os.listdir(nb_dir) if f.endswith(".ipynb")), None)
        if not nb_file:
            continue

        nb_path = os.path.join(nb_dir, nb_file)

        # print(f"\nProcessing: {nb_path}")

        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
                skip_notebook = False
        except Exception as e:
            # print(f"Failed to open {nb_file}: {e}")
            continue

        execution_env = {}
        synthetic_groupby_idx = 0
        dataflow_graph = {}
        var_hash_map = {}
        output_map = {}

        for idx, cell in enumerate(nb.cells):
            if skip_notebook:
                break

            if cell.cell_type != 'code':
                continue

            code = cell.source
            cell.outputs = []  # Clear all outputs

            # Skip notebook if it uses non-importable modules like src.print or utils.*
            if "src." in code or "utils." in code:
                print(f"Skipping notebook {nb_file}: Contains non-importable module like src.* or utils.*")
                skip_notebook = True
                break

            try:
                ast_tree = ast.parse(code)
            except:
                continue

            for node in ast.walk(ast_tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node

            try:
                try:
                    run_with_timeout(code, execution_env, timeout=300)
                except ExecutionTimeout:
                    print(f"\nSkipping notebook {nb_file}: Cell {idx} exceeded 5 min timeout.")
                    skip_notebook = True
                    break
            except ImportError as e:
                missing_pkg = str(e).split("'")[1]
                if try_import_or_install(missing_pkg):
                    try:
                        with suppress_output():
                            exec(code, execution_env)
                    except Exception as retry_e:
                        # print(f"Error after retrying cell {idx}: {retry_e}")
                        break
                else:
                    # print(f"Could not import {missing_pkg}, skipping cell.")
                    break
            except Exception as e:
                # print(f"Error executing cell {idx}: {e}")
                break

            df_state = capture_dataframes(execution_env)
            for name, (df_obj, h) in df_state.items():
                var_hash_map[name] = h

            for node in ast.walk(ast_tree):
                if hasattr(node, 'targets') and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    val = execution_env.get(var_name)
                    if isinstance(val, pd.DataFrame):
                        h = hashlib.md5(pd.util.hash_pandas_object(val, index=True).values).hexdigest()
                        var_hash_map[var_name] = h
                        df_state[var_name] = (val, h)

            df_state_all = {
                name: (execution_env[name], var_hash_map[name])
                for name in var_hash_map
                if name in execution_env and isinstance(execution_env[name], pd.DataFrame)
            }


            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Call):
                    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                        node.func.id if isinstance(node.func, ast.Name) else None)
                    if func_name not in track_ops:
                        continue

                    output_name = None
                    if hasattr(node.parent, 'targets') and isinstance(node.parent.targets[0], ast.Name):
                        output_name = node.parent.targets[0].id

                    output_hash = None
                    if output_name in df_state_all:
                        output_hash = df_state_all[output_name][1]
                        output_map[output_hash] = output_name

                    base_vars = set()
                    for arg in node.args:
                        if isinstance(arg, ast.Name) and arg.id in var_hash_map:
                            base_vars.add(arg.id)
                    if hasattr(node.func, 'value'):
                        base = node.func.value
                        if isinstance(base, ast.Name):
                            base_vars.add(base.id)
                        elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                            base_vars.add(base.value.id)

                    if not output_hash and func_name == 'groupby':
                        synthetic_groupby_idx += 1
                        output_hash = f"__groupby_result_{nb_file}_cell{idx}_{synthetic_groupby_idx}__"
                        output_map[output_hash] = f"groupby_result_{synthetic_groupby_idx}"

                    target_hash = output_hash or f"__implicit_{idx}_{func_name}__"
                    if target_hash not in dataflow_graph:
                        dataflow_graph[target_hash] = {'parents': []}
                    for base_var in base_vars:
                        parent_hash = var_hash_map.get(base_var)
                        if parent_hash:
                            dataflow_graph[target_hash]['parents'].append((parent_hash, func_name))

            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Call):
                    if hasattr(node, 'parent') and isinstance(node.parent, ast.Call):
                        continue

                    func_name = node.func.attr if isinstance(node.func, ast.Attribute) else None
                    if func_name not in main_ops:
                        continue

                    input_name = None
                    if func_name in ['melt', 'merge']:
                        for arg in node.args:
                            if isinstance(arg, ast.Name) and arg.id in df_state_all:
                                input_name = arg.id
                                break
                    else:
                        if isinstance(node.func.value, ast.Name):
                            input_name = node.func.value.id
                        elif isinstance(node.func.value, ast.Attribute):
                            if isinstance(node.func.value.value, ast.Name):
                                input_name = node.func.value.value.id

                    if not input_name or input_name not in df_state_all:
                        continue

                    input_df, input_hash = df_state_all[input_name]
                    op_sequence = trace_full_upstream(input_hash, dataflow_graph)

                    if func_name == 'groupby':
                        params = extract_groupby_params(code)
                    elif func_name == 'melt':
                        params = extract_melt_params(code)
                    elif func_name == 'merge':
                        params = extract_merge_params(code)
                    elif func_name in ['pivot', 'pivot_table']:
                        params = extract_pivot_params(code)
                        func_name = 'pivot'

                    if skip_notebook:
                        continue  # skip saving results and DAGs

                    save_dir = os.path.join(OUTPUT_DIR, func_name, f"{folder}_cell{idx}")
                    os.makedirs(save_dir, exist_ok=True)

                    with open(os.path.join(save_dir, 'param.json'), 'w') as f:
                        json.dump(params, f, indent=4)
                    with open(os.path.join(save_dir, 'op_seq.json'), 'w') as f:
                        json.dump(op_sequence, f, indent=4)

                    input_df.reset_index(drop=True).to_csv(os.path.join(save_dir, 'data.csv'), index=False)

                    # print(f"{func_name} → Saved to: {save_dir}")


                    # === Render DAG (save only, no display) ===
                    op_dot = Digraph(comment=f"{func_name.capitalize()} DAG Cell {idx}", format='png')
                    visited_nodes = set()


                    def add_node_recursive(h):
                        if h in visited_nodes or h not in dataflow_graph:
                            return
                        visited_nodes.add(h)
                        op_dot.node(h[:6], output_map.get(h, h[:6]))
                        for parent_h, op in dataflow_graph[h]['parents']:
                            op_dot.node(parent_h[:6], output_map.get(parent_h, parent_h[:6]))
                            op_dot.edge(parent_h[:6], h[:6], label=op)
                            add_node_recursive(parent_h)



                    dag_root = output_hash if 'output_hash' in locals() and output_hash else None
                    if not dag_root:
                        for df_name, (_, h) in df_state_all.items():
                            dag_root = h
                            break

                    if dag_root:
                        add_node_recursive(dag_root)
                        dag_dir = os.path.join(save_dir, "DAGs")
                        os.makedirs(dag_dir, exist_ok=True)
                        dag_filename = f"{nb_file.replace('.ipynb', '')}_cell{idx}_{func_name}"
                        graph_path = os.path.join(dag_dir, dag_filename)
                        op_dot.render(graph_path, cleanup=True)
                        # print(f"📈 DAG saved as: {graph_path}.png")

# Summary of Number of extracted folders per operator
print("\nSummary of Extracted Data Files per Operator:")
for operator in ["groupby", "melt", "merge", "pivot"]:
    try:
        op_path = os.path.join(OUTPUT_DIR, operator)
        if not os.path.isdir(op_path):
            print(f"{operator}: 0")
            continue
        count = len([
            name for name in os.listdir(op_path)
            if os.path.isdir(os.path.join(op_path, name))
        ])
        print(f"{operator}: {count}")
    except Exception as e:
        print(f"{operator}: error reading folder ({e})")

# Clean exit: mark all non-main threads as daemon and exit
for t in threading.enumerate():
    if t is not threading.main_thread():
        t.daemon = True

print("\nReplay completed successfully.")
sys.exit(0)