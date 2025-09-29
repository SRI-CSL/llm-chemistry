#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

import pathlib as pl
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import beartype.typing as ty
from beartype import beartype as type_checked

from utils import cluster_tasks, fast_tab_data_read

SetLike = set[str] | frozenset[str]
PathLike = str | pl.Path

# No need for recomputing chemistry for 
# models that we have previously established 
# strong chemistry.
SKIP_CHEMISTRY_STRATEGY_ROWS = True

# Controls what is considered what is in 
# a "used" models set in MIG nodes.
THRESHOLD_FOR_USAGE = 0.5

# Task 1
TRIAL_DATA_FILE = "runs.csv"

@type_checked
def set_trial_data_param(file: PathLike) -> None:
  """Set the trial data file path (filename only)."""
  global TRIAL_DATA_FILE
  TRIAL_DATA_FILE = pl.Path(file).name

@type_checked
def set_usage_threshold_params(usage_threshold: float) -> None:
  """Set the usage threshold."""
  global THRESHOLD_FOR_USAGE
  if usage_threshold == THRESHOLD_FOR_USAGE:
    return
  if not (0.0 <= usage_threshold <= 1.0):
    raise ValueError("Usage threshold must be in [0.0, 1.0].")
  THRESHOLD_FOR_USAGE = usage_threshold

@dataclass(frozen=True)
class Model:
  name: str
  # quality of answer Model produced
  q_i: float # quality in [0.0, 10.0]
  # performance score (e.g., accuracy, effectiveness?)
  # of Model to produce answer
  a_i: float # Compound accuracy value in [0.0, 1.0] (See paper)
  trial: ty.Optional[str] = None

@type_checked
def cost_q(X: set[Model]) -> float:
  # Sort by quality (highest first)
  X_sorted = sorted(X, key=lambda x: x.q_i, reverse=True)

  cost = 0.0
  for i, model in enumerate(X_sorted, start=1):
    q_norm = model.q_i / 10.0  # normalize to [0, 1]
    a = model.a_i
    penalty = (1 - q_norm) * (1 - a)
    weight = 1 / i  # w_i = 1 / rank (basic weight computation)
    cost += weight * penalty

  return cost

class MIG:
  """Model Interaction Graph (MIG) implementation.
  This representation is inspired by the Index Benefit Graph (IBG)
  and is used to track the interactions between models.
  IBGs are a representation and technique introduced in:
  M. R. Frank, E. Omiecinski, and S. B. Navathe. Adaptive
  and automated index selection in RDBMS. In EDBT, pages
  277-292, 1992.
  """

  @type_checked
  def __init__(self) -> None:
    """Initialize an empty MIG."""
    self.nodes: dict[frozenset[str], tuple[frozenset[str], float]] = {}

  @type_checked
  def add_node(
    self,
    index_set: SetLike,
    used_indices: SetLike,
    cost: float
  ) -> None:
    """Add a node to the MIG."""
    key = frozenset(index_set)
    self.nodes[key] = (frozenset(used_indices), cost)

  @type_checked
  def get_used_indices(
    self,
    index_set: SetLike
  ) -> frozenset[str] | None:
    """Get used indices for the given index set."""
    node = self.nodes.get(frozenset(index_set))
    return node[0] if node else None

  @type_checked
  def get_cost(
    self,
    index_set: ty.AbstractSet[str]
  ) -> float | None:
    """Get cost for the given index set."""
    node = self.nodes.get(frozenset(index_set))
    return node[1] if node else None

  @type_checked
  def find_covering_node(
    self,
    index_set: SetLike
  ) -> frozenset[str] | None:
    """
    Find a node Y such that used(Y) ⊆ X ⊆ Y,
    where X is the provided index_set.
    """
    x_set = frozenset(index_set)
    for node_set, (used_indices, _) in self.nodes.items():
      if used_indices <= x_set <= node_set:
        return node_set
    return None

  @type_checked
  def print_mig(self) -> None:
    if not self.nodes:
      print("Empty MIG")
      return
    
    # Find the root note (should be the largest set)
    max_size = max(len(node_set) for node_set in self.nodes.keys())
    print("MIG Contents:")
    print("-" * 60)
    
    sorted_nodes = sorted(
      self.nodes.items(), 
      key=lambda x: len(x[0]), 
      reverse=True
    )
    
    for node_set, (used_indices, cost) in sorted_nodes:
      models = sorted(node_set)
      used = sorted(used_indices)
      # Mark the root
      is_root = len(node_set) == max_size
      root_marker = " [ROOT]" if is_root else ""
      print(f"Available: {models}{root_marker}")
      print(f"Used:      {used}")
      print(f"Cost:      {cost:.2f}")
      print()

@type_checked
def build_mig(
  q: str, 
  S: set[str], 
  # This function assumes we have indexed all execution
  # trials (dataframe?) and given Q (the task or prompt or request),
  # and all the LLMs supported by our experiment, we can fetch the LLMs 
  # that participated in the trial along with their associated cost.
  find_models_for_q: ty.Callable[[str, set[str]], tuple[set[str], float]]
) -> MIG:
  @type_checked
  def enqueue_models(
    current_set: set[str],
    chosen_models: set[str], 
    processed: set[frozenset[str]], 
    queue: list[set[str]]
  ) -> list[set[str]]:
    target_set = (
      chosen_models 
      if chosen_models
      # When no models chosen (i.e., not chosen_models is True), 
      # still explore smaller sets to reach all subsets (set target_set to current_set).
      # This ensures ChemE has all the MIG nodes it needs. 
      else current_set
    )
    for model in target_set:
      new_set = current_set - {model}
      if frozenset(new_set) not in processed:
        queue.append(new_set)
    return queue
  
  mig = MIG()
  queue = [S]
  processed = set()
  while queue:
    current_set = queue.pop(0)
    current_set_frozen = frozenset(current_set)
    if current_set_frozen in processed:
      continue
    processed.add(current_set_frozen)
    chosen_models, cost = find_models_for_q(q, current_set)
    mig.add_node(current_set, chosen_models, cost)

    queue = enqueue_models(
      current_set, chosen_models, processed, queue)
  
  return mig

@type_checked
def ChemE(
  S: set[str],
  mig: MIG, # MIG for q
  # For sake of completeness, we allow tau to be set to 0.0.
  # This means all chemistry interactions are considered, 
  # including those with no interaction.
  tau: float = 0.0
) -> dict[tuple[str, str], float]:
  """ChemE algorithm solves LLMCP (See paper).
  Subset iteration is still expensive but with reduced
  cost per iteration due to memoization. In other words,
  memoization ensures each covering node and cost retrieval 
  is performed only once per distinct subset.
  """
  # Pre-cache covering nodes and costs for subsets
  @lru_cache(maxsize=None)
  def get_node_and_cost(
    subset: frozenset[str]
  ) -> tuple[frozenset[str], float] | None:
    node = mig.find_covering_node(set(subset))
    if node is None:
      return None
    cost = mig.get_cost(node)
    if cost is None or cost == 0.:
      return None
    return node, cost
  
  chem_table = {} # This is updated based on >= \tau
  
  S_list = list(S)
  n = len(S_list)
  
  # Incremental subset generation (still exponential, but memoized)
  for subset_size in range(n + 1):
    for X in combinations(S_list, subset_size):
      X_set = frozenset(X)
      node_cost_Y = get_node_and_cost(X_set)
      if node_cost_Y is None:
        continue
      _, cost_Y = node_cost_Y
      not_in_X = S - X_set
      for a, b in combinations(not_in_X, 2):
        Xa, Xb, Xab = (
          X_set | frozenset({a}),
          X_set | frozenset({b}),
          X_set | frozenset({a, b})
        )
        
        node_cost_Ya = get_node_and_cost(Xa)
        node_cost_Yb = get_node_and_cost(Xb)
        node_cost_Yab = get_node_and_cost(Xab)
        
        if not node_cost_Ya or not node_cost_Yb or not node_cost_Yab:
          continue
        
        _, cost_Ya = node_cost_Ya
        _, cost_Yb = node_cost_Yb
        _, cost_Yab = node_cost_Yab

        # Interaction effect relative to individual costs
        interaction = abs(cost_Yab - cost_Ya - cost_Yb + cost_Y)
        d = interaction / cost_Yab
        if d > 1.0:
          d = min(d, 1.0)
        
        if d >= tau:
          # update both (a,b) and (b,a) to handle pair symmetry explicitly
          chem_table[(a, b)] = max(chem_table.get((a, b), 0.0), d)
          chem_table[(b, a)] = max(chem_table.get((b, a), 0.0), d)
  
  return chem_table

@type_checked
def load_trial_data(
  csv_file: pl.Path
) -> tuple[dict[str, str], dict[str, dict[str, Model]]]:
  """
  Load trial data from a CSV file and organize it by clustered tasks.

  The input CSV is expected to contain one row per (trial, model, task) combination,
  with associated performance and metadata fields. Because tasks (values of task column) 
  may be semantically similar but not textually identical, we cluster them using semantic 
  similarity and select a representative task (prototype) per cluster.
  
  Returns a 2-tuple:
    - index: A dictionary mapping each original task string to its prototype task.
        This allows querying with any known task string to locate its cluster.
    - trial_data: A nested dictionary mapping:
        mappings: prototype task -> model name -> Model instance  
  
  'csv_file' is expected to contain the following columns:
    - trial (str): Trial name or identifier.
    - model (str): Name of the model (e.g., "a", "b", etc.).
    - task (str): Task text (prompt or request); may have near-duplicates.
    - latency (float): Latency in seconds for generating the result.
    - temperature (float): LLM sampling temperature.
    - id (str): Unique identifier for the output/answer.
    - result (str): Text result or answer generated by the model.
    - quality (float): Quality score of the answer.
    - generation_accuracy (float): Generation accuracy score of the model's output.
    - variance (float): Variance score (confidence dispersion).
    - review_accuracy (float): Review accuracy computed by MVLE.
    - accuracy (float): Accuracy score, combined generation and review accuracy (75%/25%, 
      if ground truth available; otherwise 25%/75%).
    - elapsed (str): Human-readable elapsed time for generation.
    - created (str): Timestamp of result creation in human-readable form.
  
  A minor optimization step: we are implementing a token-based clustering function, 
  which clusters tasks based on syntactic similarity using embeddings. Each cluster 
  is identified by a prototype task, and all results are grouped under that prototype. 
  This allows downstream queries to generalize across near-duplicate prompts reliably.
  """
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
  
  if SKIP_CHEMISTRY_STRATEGY_ROWS:
    df = df[df["strategy"] != "CHEMISTRY"]

  tasks = df["task"].drop_duplicates().tolist()
  index = cluster_tasks(tasks)

  trial_data: dict[str, dict[str, Model]] = defaultdict(dict)
  for row in df.itertuples(index=False):
    orig_task = row.task
    proto_task = index[orig_task]
    model_name = row.model

    model = Model(
      model_name,
      q_i=row.quality,
      a_i=max(0.1, row.accuracy),
      trial=row.trial
    )

    # There are multiple models of same kind (provider); 
    # pick model instance with max accuracy a_i
    trial_data[proto_task][model_name] = max(
      trial_data[proto_task].get(model_name, model), 
      model, key=lambda m: m.a_i)

  return index, dict(trial_data)

@lru_cache(maxsize=1)
def lazy_load_trial_data(data_file: ty.Optional[str] = None) -> tuple[dict[str, str], dict[str, dict[str, Model]]]:
  # Note: this path could change depending on where this module is placed.
  trial_data_file = TRIAL_DATA_FILE if data_file is None else data_file
  trial_data_file = pl.Path.cwd() / trial_data_file
  if not trial_data_file.exists():
    raise FileNotFoundError(f"Trial data file not found: {trial_data_file}")
  return load_trial_data(trial_data_file)

@type_checked
def get_trial_data(data_file: ty.Optional[str] = None) -> dict[str, dict[str, Model]]:
  _, trial_data = lazy_load_trial_data(data_file=data_file)
  return trial_data


@type_checked
def find_models_for_q(
  q: str,
  candidate_models: set[str]
) -> tuple[set[str], float]:
  # Fetch models for this query
  trial_data = get_trial_data()
  if not trial_data:
    raise ValueError(
      "Trial data is empty. "
      "Ensure the CSV file is loaded correctly."
    )

  if q not in trial_data:
    raise ValueError(f"Unknown P: {q}")

  available_models = trial_data[q]
  
  # Filter to valid candidates with data
  chosen_models = {
    available_models[m] 
    for m in candidate_models 
    if m in available_models
  }

  # Heuristic: for each node in MIG, focus on models with a_i > THRESHOLD_FOR_USAGE.
  # These models are likely to be the most effective in that group.
  used_models = {m for m in chosen_models if m.a_i > THRESHOLD_FOR_USAGE}

  # Compute cost from used_models
  cost = cost_q(used_models) if used_models else 1.0  # fallback cost if none used

  return {m.name for m in used_models}, cost

@type_checked
def load_past_runs(q: str) -> set[frozenset[str]]:
  # Returns a set of sets of models that participated in the 
  # past runs (trials) for the given Q.
  # Assumes Models are first grouped by task (See TRIAL_DATA),
  # then models of a task can be grouped by trial. The latter
  # represents the subset of LLMs that were used at that trial
  # to answer the query/task Q.
  index, trial_data = lazy_load_trial_data()
  proto = index.get(q)
  if proto is None:
    raise ValueError("Query/task: {p} not found in index.")
  models = trial_data.get(proto)
  if models is None:
    raise ValueError("No models found for query/task: {p}.")
  # Grouped models by trial
  trials: defaultdict[str, set[str]] = defaultdict(set)
  for model in models.values():
    if model.trial:
      trials[model.trial].add(model.name)
  return {frozenset(m) for m in trials.values()}

@type_checked
def generate_S() -> set[str]:
  # Get of all model names from trial data.
  try:
    _, trial_data = lazy_load_trial_data()
    # it is the names of the models that matter
    return {
      model.name 
      for models in trial_data.values() 
      for model in models.values()}
  except FileNotFoundError:
    print("Trial data file not found. Using example data.")
    return set()
  except Exception as e:
    print(f"Error loading trial data: {e}")
    return set()

@type_checked
def generate_neighbors(
  current: set[str], 
  universe: set[str]
) -> set[frozenset[str]]:
  neighbors = set()
  # Remove one model from current
  for m in current:
    neighbors.add(frozenset(current - {m}))
  # Add one model from universe not in current
  for m in universe - current:
    neighbors.add(frozenset(current | {m}))
  # Swap one model in and one out
  for m_to_remove in current:
    for m_to_add in universe - current:
      swapped = frozenset(current - {m_to_remove} | {m_to_add})
      if swapped != current:
        neighbors.add(swapped)
  return neighbors


@type_checked
def fast_max_inter_chem(
  S: set[str], 
  chem_table: dict[tuple[str, str], float]
) -> float:
  # It ranks models by degree (total chemistry), split them greedily, and 
  # sums their their cross-partition chemistry. Specifically, it approximates the 
  # theoretical 'maximal inter-group chemistry' by greedily splitting highly connected 
  # elements into two balanced subsets and summing their cross-partition chemistry.
  S = list(S)
  degrees = {
    s: sum(
      chem_table.get((s, t), 0.0) or chem_table.get((t, s), 0.0)
      for t in S if t != s
    ) for s in S
  }
  sorted_S = sorted(S, key=lambda x: -degrees[x])
  A = set(sorted_S[::2])
  B = set(sorted_S[1::2])
  
  return sum(
    chem_table.get((a, b), 0.0) or chem_table.get((b, a), 0.0)
    for a in A
    for b in B
  )

@type_checked
def loss(
  X: set[str], 
  S: set[str], 
  chem_table: dict[tuple[str, str], float], 
  max_total_chem: float, 
  max_inter_chem: float, 
  alpha: float,
  beta: float = 0.5
) -> float:
  # Compute intra-chemistry (within X)
  intra_chem = sum(
    chem_table[(a, b)] for a, b in combinations(sorted(X), 2)
    if (a, b) in chem_table
  )
  loss_intra = max_total_chem - intra_chem
  
  # Compute inter-chemistry (between X and S - X)
  S_minus_X = S - X
  actual_inter_chem = sum(
    chem_table.get((a, b), 0.0) or chem_table.get((b, a), 0.0)
    for a in X
    for b in S_minus_X
  )
  loss_inter = max_inter_chem - actual_inter_chem
  
  # Feasibility check
  if loss_intra < 0 or loss_inter < 0:
    return float('inf')
  
  # Calculate final loss with subset size penalty (beta)
  # Higher beta → favors smaller subsets.
  size_penalty = beta * len(X)

  return alpha * loss_inter + (1 - alpha) * loss_intra + size_penalty

# ----------------------
@type_checked
def recommend_randomized(
  S: set[str], 
  chem_table: dict[tuple[str, str], float],
  X_q: set[frozenset[str]],
  alpha: float = 0.5,
  rand_cnt: int = 10,  # Number of randomized iterations
  min_size: int = 2
) -> set[str] | None:
  """
  Randomized recommendation inspired by the choosePartition algorithm.
  Performs multiple randomized iterations to find the best chemistry-based solution.
  """
  if len(chem_table) == 0:
    return None

  max_total_chem = float(sum(
    chem_table.get((a, b), 0.0) or chem_table.get((b, a), 0.0)
    for a, b in combinations(sorted(S), 2)
  ))

  # max_inter_chem = max_total_chem
  max_inter_chem = fast_max_inter_chem(S, chem_table)
  
  # print(f"DEBUG: max_total_chem = {max_total_chem}")
  # print(f"DEBUG: max_inter_chem = {max_inter_chem}")
  # print(f"DEBUG: Number of configurations to evaluate: {len(X_q)}")

  best_solution = None
  best_loss = float('inf')

  # Try baseline solutions from X_q
  for X_i in X_q:
    if len(X_i) >= min_size:
      current_sol = set(X_i)
      current_loss = loss(current_sol, S, chem_table, max_total_chem, max_inter_chem, alpha)
      if current_loss < best_loss:
        best_solution = current_sol
        best_loss = current_loss

  # Randomized iterations
  for _ in range(rand_cnt):
    # Stage 1: Start with high-chemistry pairs
    candidate = _build_chemistry_based_solution(S, chem_table, min_size)

    if candidate and len(candidate) >= min_size:
      candidate_loss = loss(candidate, S, chem_table, max_total_chem, max_inter_chem, alpha)
      if candidate_loss < best_loss:
        best_solution = candidate
        best_loss = candidate_loss

  return best_solution

def _build_chemistry_based_solution(
  S: set[str], 
  chem_table: dict[tuple[str, str], float],
  min_size: int
) -> set[str] | None:
  """
  Build a solution by iteratively adding models with high chemistry interactions.
  """
  # Get all pairs with their chemistry weights
  pairs_with_weights = []
  for a, b in combinations(S, 2):
    weight = chem_table.get((a, b), 0.0) or chem_table.get((b, a), 0.0)
    if weight > 0:
      pairs_with_weights.append(((a, b), weight))
  
  if not pairs_with_weights:
    return None
  
  # Stage 1: Select initial pair weighted by chemistry strength
  weights = [w for _, w in pairs_with_weights]
  selected_pair = random.choices(pairs_with_weights, weights=weights, k=1)[0][0]
  solution = set(selected_pair)
  
  # Stage 2: Iteratively add models that have good chemistry with existing solution
  remaining = S - solution
  
  while remaining and len(solution) < len(S):
    # Calculate weights for adding each remaining model
    add_weights = []
    candidates = []
    
    for candidate in remaining:
      # Calculate average chemistry with existing solution members
      total_chemistry = sum(
        chem_table.get((candidate, existing), 0.0) or chem_table.get((existing, candidate), 0.0)
        for existing in solution
      )
      
      if total_chemistry > 0:
        # Weight inversely proportional to solution size (favor smaller sets)
        weight = total_chemistry / (len(solution) + 1)
        add_weights.append(weight)
        candidates.append(candidate)
    
    if not candidates:
      break
      
    # Randomly select next model to add (weighted by chemistry)
    if random.random() < 0.3:  # 30% chance to stop growing (favor smaller sets)
      break
      
    selected = random.choices(candidates, weights=add_weights, k=1)[0]
    solution.add(selected)
    remaining.remove(selected)
  
  return solution if len(solution) >= min_size else None

def _has_meaningful_chemistry(
  subset: frozenset[str], 
  chem_table: dict[tuple[str, str], float]
) -> bool:
  """Check if a subset has meaningful chemistry interactions."""
  models = list(subset)
  for i in range(len(models)):
    for j in range(i + 1, len(models)):
      a, b = models[i], models[j]
      if chem_table.get((a, b), 0.0) > 0 or chem_table.get((b, a), 0.0) > 0:
        return True
  return False
# ----------------------

@type_checked
def recommend(
  S: set[str], 
  chem_table: dict[tuple[str, str], float],
  X_q: set[frozenset[str]], # past LLM configurations for p
  alpha: float = 0.5,
  iters: int = 10,
  min_size: int = 2,
  verbose: bool = False,
  # explicit policies and classification thresholds for 
  # fallback behavior (to handle homogenous performance cases)
  model_scores: ty.Optional[dict[str, tuple[float, float]]] = None,
  fallback_policy_good: str = "single",  # "single" or "all"
  fallback_policy_bad: str = "none",     # "none" or "single"
  hi_a: float = 0.9, hi_q: float = 9.0,  # all-good if >= these
  lo_a: float = 0.5, lo_q: float = 5.0   # all-bad  if <= these
) -> set[str] | None:
  
  def _U_from_Xq() -> set[str]:
    return set().union(*X_q) if X_q else set()
  
  def _best_single_from(U: set[str]) -> set[str] | None:
    if not U:
      return None
    if not model_scores:
      return {sorted(U)[0]}  # deterministic fallback if no scores
    best = max(
      U,
      key=lambda n: (
        model_scores.get(n, (0.0, 0.0))[0],
        model_scores.get(n, (0.0, 0.0))[1],
        -len(n)
      )
    )
    return {best}
  
  def _classify(U: set[str]) -> tuple[bool, bool]:
    # returns (all_good, all_bad)
    if not model_scores or not U:
      return (False, False)
    a_vals = [model_scores.get(n, (0.0, 0.0))[0] for n in U]
    q_vals = [model_scores.get(n, (0.0, 0.0))[1] for n in U]
    all_good = (min(a_vals) >= hi_a) and (min(q_vals) >= hi_q)
    all_bad  = (max(a_vals) <= lo_a) and (max(q_vals) <= lo_q)
    return (all_good, all_bad)
  
  if len(chem_table) == 0:
    U = _U_from_Xq()
    all_good, all_bad = _classify(U)
    if all_bad:
      # Enforce "no meaningful improvement"
      if fallback_policy_bad == "none":
        return None
      # or choose least-bad singleton (best among bad)
      return _best_single_from(U)
    # return None
    if all_good:
      # Enforce your preferred "all-good" policy
      if fallback_policy_good == "all":
        return U if U else S
      return _best_single_from(U)
    
    # Mixed but chemistry still empty → pick single best from X_q
    return _best_single_from(U)
  
  max_total_chem = float(sum(
    chem_table.get((a, b), 0.0) or chem_table.get((b, a), 0.0)
    for a, b in combinations(sorted(S), 2)
  ))
  
  max_inter_chem = fast_max_inter_chem(S, chem_table)
  
  if verbose:
    print(f"DEBUG: max_total_chem = {max_total_chem}")
    print(f"DEBUG: max_inter_chem = {max_inter_chem}")
    print(f"DEBUG: Number of configurations to evaluate: {len(X_q)}")
  
  # If you remove randomized recommend, remove this X_q filtering thingy
  X_q_filtered = {
    x for x in X_q
    if len(x) >= min_size and _has_meaningful_chemistry(x, chem_table)
  }
  
  if not X_q_filtered:
    print("No valid configurations found in X_q.")
    return None
  
  X_q = X_q_filtered

  
  best_sol = None
  min_loss = float("inf")
  
  for X_i in X_q:
    current_sol = set(X_i)
    current_loss = loss(
      current_sol, S, chem_table, 
      max_total_chem, max_inter_chem, alpha)
    
    if current_loss < min_loss:
      best_sol = current_sol
      min_loss = current_loss
    
    for _ in range(iters):
      best_neighbor = None
      min_neighbor_loss = float("inf")
      for neighbor in generate_neighbors(current_sol, S):
        neighbor = set(neighbor)
        # Ensures hill climbing only considers valid neighbors.
        if len(neighbor) < min_size:
          continue
        neighbor_loss = loss(
          neighbor, S, chem_table, 
          max_total_chem, max_inter_chem, alpha)
        if neighbor_loss < min_neighbor_loss:
          best_neighbor = neighbor
          min_neighbor_loss = neighbor_loss
      if min_neighbor_loss >= current_loss:
        break
      current_sol = best_neighbor
      current_loss = min_neighbor_loss
      if current_loss < min_loss:
        best_sol = current_sol
        min_loss = current_loss
  
  return set(best_sol) if best_sol else None

if __name__ == "__main__":
  pass
