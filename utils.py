#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

import math
import pathlib as pl
from collections import Counter, defaultdict
from warnings import deprecated

import beartype.typing as ty
import numpy as np
import pandas as pd
import torch
from beartype import beartype as type_checked

PathLike = str | pl.Path

def _detect_headers(
  filepath: PathLike, 
  separator: str = '\t', 
  has_header: bool | None = None
) -> bool:
  if has_header is None:
    # Conservative guesses data has a header only if all values 
    # in the first row:
    # - Are alphabetic strings (isalpha()), or
    # - Contain an underscore (like col_name)
    # Use with caution as this is a heuristic.
    try:
      preview = pd.read_csv(
        filepath, sep=separator, nrows=2, header=None, dtype=str)
      
      if preview.shape[0] < 2:
        return False

      return all(x.isalpha() or "_" in x for x in preview.iloc[0])  # minimal clue
    except Exception:
      return False
  return has_header

@type_checked
def fast_tab_data_read(
  filepath: PathLike,
  chunksize: int = None,
  factor: int = 4,
  dtype: ty.Any = None,
  separator: str = ',',
  has_header: bool = None
) -> pd.DataFrame:
  import psutil as ps
  if isinstance(filepath, str):
    filepath = pl.Path(filepath)
  
  # Checks whether the first row look like a list of 
  # field names rather than real data.
  has_header = _detect_headers(filepath, separator, has_header)
  header_param = 0 if has_header else None
  
  # (Reused from SIGNAL)
  # IMPORTANT: in chunk reading be careful as pandas might infer a columns dtype 
  # as different for diff chunk. As such specifying a dtype while
  # reading by giving params to read_csv maybe better. Label encoding
  # will fail if half the rows for same column is int and rest are str.
  if chunksize is None:
    try:
      # estimate the available memory for the dataframe chunks
      chunksize = (
        ps.virtual_memory().available // (pd.read_csv(
          str(resolved_file_path), 
          sep=separator, nrows=1).memory_usage(deep=True).sum() * factor))
    except Exception:
      chunksize = 1000
  df = pd.DataFrame()
  for chunk in pd.read_csv(
    str(filepath), 
    sep=separator, 
    chunksize=chunksize, 
    dtype=dtype, 
    header=header_param,
  ):
    if df.empty:
      df = chunk
    else:
      df = pd.concat([df, chunk], ignore_index=True)
  
  if not has_header:
    df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
  
  return df

@type_checked
def find_groups_in_data(
  csv_file: pl.Path
) -> list[list[dict]]:
  if not csv_file.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_file}")
  
  df = fast_tab_data_read(
    filepath=csv_file,
    separator=',',
    has_header=True,
    # Note: this wont work if has_header=False,
    # so make sure the CSV has a header row 
    # and you set has_header=True.
    dtype={
      "trial": str,
      "model": str,
      "task": str,
      "quality": float,
      "accuracy": float,
      "strategy": str,
    }
  )
  
  # Group by trial and strategy to create batches
  batches = []
  
  # Create a dictionary to group rows by (trial, strategy) combination
  batch_groups = {}
  
  for _, row in df.iterrows():
    # Create a key from trial and strategy
    batch_key = (row["trial"], row["strategy"])
    
    # Initialize the batch group if it doesn't exist
    if batch_key not in batch_groups:
      batch_groups[batch_key] = []
    
    # Convert the pandas Series to a dictionary and add to the batch group
    row_dict = row.to_dict()
    batch_groups[batch_key].append(row_dict)
  
  # Convert the grouped data into the final list of lists format
  for batch_key, batch_rows in batch_groups.items():
    batches.append(batch_rows)
  
  return batches

@type_checked
def group_by_strategy(
  csv_file: pl.Path
) -> dict[str, dict[str, list[dict]]]:
  """
  Group trials by strategy and then by trial number within each strategy.
  
  Returns:
    Dictionary structure: {strategy: {trial_num: [rows]}}
    Example: {"RANDOM": {"0": [row1, row2], "1": [row3, row4]}}
  """
  if not csv_file.exists():
    raise FileNotFoundError(f"CSV file not found: {csv_file}")
  
  all_groups: list[list[dict]] = find_groups_in_data(csv_file)
  
  # Use defaultdict with lambda to create nested defaultdict structure
  strategy_groups = defaultdict(lambda: defaultdict(list))

  for batch in all_groups:
    for row in batch:
      # Get trial and extract the trial number
      trial = row["trial"]
      
      # Handle different trial formats by getting the last part after split
      # Works for "Trial_519 0", "Trial 1 0", etc.
      trial_parts = trial.split()
      if len(trial_parts) < 2:
        # Skip rows where trial format is unexpected
        continue
        
      trial_num = trial_parts[-1]  # Get the last part as the number
      
      # Group by strategy first, then by trial number
      strategy_groups[row["strategy"]][trial_num].append(row)

  # Convert defaultdict to regular dict for cleaner output
  return {
    strategy: dict(trial_nums) 
    for strategy, trial_nums in strategy_groups.items()
  }

@type_checked
def trial_groups_by_strategy(
  strategy: str,
  grouped_data: dict[str, dict[str, list[dict]]]
) -> list[list[dict]]:
  strategy = strategy or 'LOCAL'
  data = grouped_data[strategy]
  trial_grouped = {}
  for _, trials in data.items():
    for trial in trials:
      trial_key = trial['trial']
      if trial_key not in trial_grouped:
        trial_grouped[trial_key] = []
      trial_grouped[trial_key].append(trial)
  return list(trial_grouped.values())

@type_checked
def find_top_k(
  filepath: pl.Path, 
  Ks: ty.Optional[ty.Iterable[int]] = None
) -> list[set[str]]:
  """
  Given a file with tabular data, return the top k most effective models 
  by 'accuracy'. Uses fast_tab_data_read to read the file and returns 
  a set of model names.
  """
  if Ks is None:
    Ks = (3, 5, 10)
  df = fast_tab_data_read(filepath)
  
  # Check for required columns
  required_columns = {'model', 'accuracy'}
  if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(
      f"Required columns {missing} not found in file. "
      f"Available columns: {list(df.columns)}")
  
  # Sort by effectiveness (highest first) and keep only the best instance of each model
  # This ensures that for duplicate model names, we keep the one with higher effectiveness
  df_sorted = df.sort_values('accuracy', ascending=False)
  df_deduplicated = df_sorted.drop_duplicates(subset=['model'], keep='first')
  unique_sorted_models = df_deduplicated['model']
  
  results: list[set[str]] = []
  for k in Ks:
    if k <= 0:
      results.append(set())
    elif k >= len(unique_sorted_models):
      # If k is larger than available unique models, return all
      results.append(set(unique_sorted_models))
    else:
      top_models = unique_sorted_models.head(k)
      results.append(set(top_models))
  
  return results

@type_checked
def find_top_k_rows(
  filepath: pl.Path, 
  Ks: list[int] = [3, 5, 10]
) -> list[list[dict]]:
  """
  Given a file with tabular data, return the top k most effective models 
  by 'accuracy'. Uses fast_tab_data_read to read the file and returns 
  the whole record (row) as a dict for each model.
  """
  df = fast_tab_data_read(filepath)
  
  # Check for required columns
  required_columns = {'model', 'accuracy'}
  if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(
      f"Required columns {missing} not found in file. "
      f"Available columns: {list(df.columns)}"
    )
  
  # Sort by effectiveness (highest first), keep best instance per model
  df_sorted = df.sort_values('accuracy', ascending=False)
  df_deduplicated = df_sorted.drop_duplicates(subset=['model'], keep='first')
  
  results: list[list[dict]] = []
  for k in Ks:
    if k <= 0:
      results.append([])
    elif k >= len(df_deduplicated):
      top_k_rows = df_deduplicated.to_dict(orient='records')
      results.append(top_k_rows)
    else:
      top_k_rows = df_deduplicated.head(k).to_dict(orient='records')
      results.append(top_k_rows)

  return results

@type_checked
def tokenize(text: str) -> list[str]:
  return text.lower().split()

@type_checked
def compute_sparse_tfidf_matrix(
  tasks: list[str]
) -> tuple[torch.Tensor, list[str]]:
  """
  Creates a Normalized Sparse TF-IDF Tensor (N, Vocab)
  """
  tokenized_docs = [tokenize(t) for t in tasks]

  # 1. Build Vocab & Sparse Indices (Python side is faster for string hashing)
  vocab = {}
  indices = [] # [[row, col], ...]
  values = []  # [count, ...]

  for doc_idx, doc in enumerate(tokenized_docs):
    counts = Counter(doc)
    for token, count in counts.items():
      if token not in vocab:
        vocab[token] = len(vocab)

      indices.append([doc_idx, vocab[token]])
      values.append(count)

  if not indices:
    return torch.empty(0), []

  # 2. Create Base Sparse Tensor (TF)
  # Shape: (Num_Docs, Vocab_Size)
  i = torch.LongTensor(indices).t()  # Size (2, NNZ)
  v = torch.FloatTensor(values)      # Size (NNZ)

  num_docs = len(tasks)
  vocab_size = len(vocab)

  # 3. Compute IDF (Vectorized)
  # Since we used Counter above, 'i[1]' contains the column (word) index 
  # exactly once for every document that contains that word.
  # Therefore, bincount on column indices gives us Document Frequency (DF).
  df = torch.bincount(i[1], minlength=vocab_size).float()

  # IDF = log(N / (1 + DF))
  idf = torch.log(num_docs / (1 + df))

  # 4. Apply IDF to TF values
  # We multiply the TF 'values' by the IDF corresponding to their column index
  tf_idf_values = v * idf[i[1]]

  # 5. L2 Normalization (Vectorized on Sparse Data)
  # Calculate sum of squares per row (document)
  # scatter_add adds values into the specific indices of the target
  row_sq_sums = torch.zeros(num_docs)
  row_sq_sums.scatter_add_(0, i[0], tf_idf_values ** 2)
  row_norms = torch.sqrt(row_sq_sums)

  # Handle zero division safely
  row_norms[row_norms == 0] = 1.0

  # Normalize the values
  # Divide each value by the norm of its specific row
  norm_values = tf_idf_values / row_norms[i[0]]

  # 6. Construct Final Sparse Tensor
  # We use coalesce() to ensure indices are sorted and optimized for math
  sparse_tensor = torch.sparse_coo_tensor(i, norm_values, (num_docs, vocab_size))
  return sparse_tensor.coalesce(), tasks

@type_checked
def cluster_tasks_sparse(
  tasks: list[str], 
  threshold: float = 0.85
) -> dict[str, str]:
  if not tasks:
    return {}

  # 1. Get the master sparse matrix (N, V)
  # This uses minimal memory even for huge vocabularies
  matrix, _ = compute_sparse_tfidf_matrix(tasks)

  task_to_prototype = {}

  # We keep prototypes as a list of indices to avoid complex sparse slicing
  prototype_indices = []

  # 2. Greedy Clustering Loop
  # Optimization: We cannot easily do Sparse @ Sparse in a loop efficiently.
  # However, Sparse_Matrix @ Dense_Vector is extremely fast.
  # We densify ONLY the single current vector being compared (size = V),
  # while keeping the Prototypes Matrix sparse.

  for i in range(len(tasks)):
    matched = False

    if prototype_indices:
      # A. Create a sparse sub-matrix of just the prototypes
      # index_select works on sparse tensors in newer PyTorch,
      # but masking is safer for "drop-in" compatibility across versions.
      # Ideally, we'd use slicing (CSR), but sticking to COO logic:
      proto_matrix = torch.index_select(matrix, 0, torch.tensor(prototype_indices))

      # B. Get current vector (Dense)
      # Densifying one vector (Size V) is cheap (KB/MBs), unlike densifying the whole matrix.
      current_vec_dense = matrix[i].to_dense().unsqueeze(1)  # Shape (V, 1)

      # C. Compute Cosine Similarity
      # (Num_Protos, V) sparse @ (V, 1) dense -> (Num_Protos, 1) dense
      sim_scores = torch.sparse.mm(proto_matrix, current_vec_dense).flatten()

      # D. Check threshold
      best_val, best_idx = torch.max(sim_scores, dim=0)

      if best_val.item() >= threshold:
        matched = True
        proto_real_idx = prototype_indices[best_idx]
        task_to_prototype[tasks[i]] = tasks[proto_real_idx]

    if not matched:
      prototype_indices.append(i)
      task_to_prototype[tasks[i]] = tasks[i]

  return task_to_prototype

# @type_checked
@deprecated("cluster_tasks() was replaced by cluster_tasks_sparse().")
def compute_idf(docs: list[list[str]]) -> dict[str, float]:
  df = defaultdict(int)
  total_docs = len(docs)
  for doc in docs:
    unique_tokens = set(doc)
    for token in unique_tokens:
      df[token] += 1
  return {token: math.log(total_docs / (1 + df[token])) for token in df}

# @type_checked
@deprecated("cluster_tasks() was replaced by cluster_tasks_sparse().")
def compute_tf_idf(
  doc_tokens: list[str], 
  idf: dict[str, float], 
  vocab: dict[str, int]
) -> np.ndarray:
  vec = np.zeros(len(vocab))
  tf = Counter(doc_tokens)
  for token, count in tf.items():
    if token in vocab:
      vec[vocab[token]] = count * idf.get(token, 0.0)
  return vec

# @type_checked
@deprecated("cluster_tasks() was replaced by cluster_tasks_sparse().")
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
  denom = np.linalg.norm(a) * np.linalg.norm(b)
  return float(np.dot(a, b) / denom) if denom != 0 else 0.0

# @type_checked
@deprecated("Use cluster_tasks_sparse() instead. This will be removed soon.")
def cluster_tasks(tasks: list[str], threshold: float = 0.85) -> dict[str, str]:
  tokenized = [tokenize(t) for t in tasks]
  idf = compute_idf(tokenized)

  # Build vocabulary
  vocab = {token: i for i, token in enumerate(sorted(idf))}
  vectors = [compute_tf_idf(toks, idf, vocab) for toks in tokenized]

  task_to_prototype = {}
  prototypes = []

  for task, vec in zip(tasks, vectors):
    matched = False
    for proto_task, proto_vec in prototypes:
      if cosine_similarity(vec, proto_vec) >= threshold:
        task_to_prototype[task] = proto_task
        matched = True
        break
    if not matched:
      prototypes.append((task, vec))
      task_to_prototype[task] = task

  return task_to_prototype

if __name__ == "__main__":
  find_groups_in_data(pl.Path("runs.csv"))
