#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

# This script will produce chemistry maps 
# for model ensembles.
# Notes to the reader:
# Why are we calling these Chemistry Maps?
# Why not just Complementarity Maps? or 'Marginal' Complementarity Maps?
# Let me elaborate on the naming.. 
# LLM Chemistry is a pairwise quantity: chem(a, b),
# which is based on marginal contribution of 'b' when added 
# to a coalition with 'a', normalized by cost. Recall 
# cost is built directly on accuracy and quality benefits.
# (It's fundamentally about interaction between models.)
# ...
# Now, keep this explanation about Chemistry in your mind. 
# Good? let's now bring our complementarity index (Hypervolume + Rao).
# CI is a group-level metric: measures ensemble complementarity as a
# whole. It uses accuracy/quality to compute coverage (hypervolume) 
# and spread (Rao). CI is not a definition of LLM Chemistry per se, 
# but a proxy measure of how complementary the ensemble of LLMs is.
# Are you with me? Let's move on. 
# So what about the maps? The map we are generating for each ensemble 
# E, it actually visualizes: 
# \Delta \text{CI}(x) = \text{CI}(E \cup \{x\}) - \text{CI}(E)
# I.e., the marginal gain in complementarity if you add a candidate
# model x to ensemble E. So strictly speaking, they are 
# complementarity maps (marginal CI maps).
# So why are we calling them Chemistry Maps?
# First off, please understand that our theorem and corollary, 
# in the paper, tie chemistry to heterogeneity:
# Homogeneity \Delta = 0 ⇒ chem = 0
# Heterogeneity \Delta > 0 ⇒ chem > 0
# In that sense, these maps are a visual instantiation of the 
# theorem/corollary:
# Dark = homogeneity (chemistry vanishes).
# Bright = heterogeneity (potential for chemistry).
# So even though the formula behind the map is CI-based, the
# interpretation is "where chemistry could arise."
# I.e., these maps are chemistry potential maps. So, 
# Formally: they are complementarity maps, but interpretively: they
# are chemistry maps, because they show when and where chemistry 
# (as per your theorem/corollary) becomes nonzero.

# %%
# Experiment configuration constants
THRESHOLD_FOR_SUCCESS = 0.5
N_CONFIGURATIONS = 10
TRIAL_DATA_FILE = "runs2.csv"
VERBOSE_MODE = False
NO_RAND_SAMPLING = True
TASK1 = "classify a short statement into a category of fakeness"
TASK2 = "fix buggy program"
TASK3 = "generate a concise, clinically formal summary"
TASKS = [TASK1, TASK2, TASK3]

# %%
from chemistry import set_trial_data_param

# HACK: handles one task at a time
if TRIAL_DATA_FILE == "runs.csv":
  TASKS = TASKS[:1]
elif TRIAL_DATA_FILE == "runs2.csv":
  TASKS = TASKS[1:2]
elif TRIAL_DATA_FILE == "runs3.csv":
  TASKS = TASKS[2:3]

print(TASKS)

set_trial_data_param(TRIAL_DATA_FILE)

# %%
import pathlib as pl
from functools import lru_cache

# %%
# Global dependencies
import beartype.typing as ty
from beartype import beartype as type_checked

# %%
from utils import group_by_strategy, trial_groups_by_strategy


# Utility functions for loading trial data
@lru_cache(maxsize=1)
def lazy_load_data() -> dict[str, dict[str, list[dict]]]:
  trial_data_file = pl.Path(TRIAL_DATA_FILE)
  if not trial_data_file.exists():
    raise FileNotFoundError(f"Trial data file not found: {trial_data_file}")
  return group_by_strategy(trial_data_file)

def remote_data() -> list[list[dict]]:
  data = lazy_load_data()
  return trial_groups_by_strategy("REMOTE", data)

def random_data() -> list[list[dict]]:
  data = lazy_load_data()
  return trial_groups_by_strategy("RANDOM", data)

def local_data() -> list[list[dict]]:
  data = lazy_load_data()
  return trial_groups_by_strategy("LOCAL", data)

def performance_data() -> list[list[dict]]:
  data = lazy_load_data()
  return trial_groups_by_strategy("PERFORMANCE", data)

def chemistry_data() -> list[list[dict]]:
  data = lazy_load_data()
  return trial_groups_by_strategy("CHEMISTRY", data)

# %%
#  Utility functions for organizing ensemble data into groups
@type_checked
def generate_configs_by_group(
  data: list[list[dict]], 
  group_id: int,
  num_configs: int = N_CONFIGURATIONS,
  no_rand_sampling: bool = NO_RAND_SAMPLING
) -> list[list[dict]]:
  """Generate configurations for a specific group"""
  if group_id == 1:  # Target N=3
    group_data = [item for item in data if 1 < len(item) <= 3]
  elif group_id == 2:  # Target N=5  
    group_data = [item for item in data if 3 < len(item) <= 5]
  elif group_id == 3:  # Target N=10
    group_data = [item for item in data if len(item) > 5]
    for item in group_data:
      if len(item) > 10:
        print(item)
    stats = [len(item) for item in group_data]
    print(f"Group 3 stats: min={min(stats)}, max={max(stats)}, mean={np.mean(stats):.2f}, median={np.median(stats)}")
  else:
    raise ValueError(f"Invalid group_id: {group_id}. Must be 1, 2, or 3")

  if no_rand_sampling:
    return group_data
  return random.sample(group_data, min(num_configs, len(group_data)))

# %%
from chemistry import Model


# Build ensembles for each task
@type_checked
def get_models_in_config(task: str, config: list[dict]) -> list[Model]:
  """Get models for a specific task, maintaining deterministic order"""
  models = []
  seen = set()

  for row in config:
    if row['task'] == task:
      model = Model(
        name=row['model'], 
        q_i=row['quality'], 
        a_i=row['effectiveness'], 
        trial=row['trial'])

      model_id = (model.name, model.q_i, model.a_i, model.trial)
      if model_id not in seen:
        models.append(model)
        seen.add(model_id)

  return models

# %%
# Representative ensemble selection

@type_checked
def select_representatives(
  items: ty.List[ty.Dict], 
  k_weak: float = 0.1, 
  k_mid: float = 0.5, 
  k_strong: float = 0.9
) -> ty.Tuple[ty.Optional[ty.Dict], ty.Optional[ty.Dict], ty.Optional[ty.Dict]]:
  if not items:
    return None, None, None
  cis = np.array([x['CI'] for x in items], dtype=float)
  q_weak, q_mid, q_strong = np.quantile(cis, [k_weak, k_mid, k_strong])
  def closest_to(q):
    return min(items, key=lambda x: abs(x['CI'] - q))
  return closest_to(q_weak), closest_to(q_mid), closest_to(q_strong)

# %%
import os

import numpy as np

# Utility function for summarizing ensemble results, avoiding overfitting,
# and ensuring diversity

@type_checked
def summarize_for_tables(ensembles: list[dict], ci_threshold: float = 0.5):
  kept = [e for e in ensembles if e['CI'] > ci_threshold]
  # ... compute your table rows from `kept`
  return kept  # or whatever you need

@type_checked
def task_wide_ref_bounds(
  all_point_sets: ty.List[np.ndarray]
) -> ty.Tuple[ty.Tuple[float, float], ty.Tuple[float, float]]:
  """
  all_point_sets: iterable of np.ndarray (M_i, 2) for a given task
  Returns REF, BOUNDS with small margins so nothing sits exactly on a bound.
  This is really important to avoid overfitting.
  """
  all_pts = np.vstack([P for P in all_point_sets if len(P) > 0])
  amin, qmin = all_pts.min(axis=0)
  amax, qmax = all_pts.max(axis=0)

  # 1% margins (or use percentiles e.g. p1/p99 if you prefer)
  eps_a = 0.01 * max(1e-6, amax - amin)
  eps_q = 0.01 * max(1e-6, qmax - qmin)

  REF    = (max(0.0, amin - eps_a), max(0.0, qmin - eps_q))
  BOUNDS = (min(1.0, amax + eps_a), min(1.0, qmax + eps_q))
  return REF, BOUNDS

@type_checked
def slug(s: str) -> str:
  return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")


# %%
from collections import defaultdict

import numpy as np

from metrics import complementarity_index


@type_checked
def collect_ensembles_for_task(
  current_task: str, 
  groups: list[tuple[int, int]] = [(1,3), (2,5), (3,10)]
):
  """Return per-group/per-strategy items with Complementary Index (CI) & points, 
  and all point sets."""
  per_group_strategy = defaultdict(lambda: defaultdict(list))
  all_point_sets = []
  
  for group_id, target_n in groups:
    print(f"\nEvaluating Group {group_id} (Target N={target_n})...")
    strategies = {
      'RANDOM':       generate_configs_by_group(random_data(),      group_id=group_id),
      'PERFORMANCE':  generate_configs_by_group(performance_data(), group_id=group_id),
      'CHEMISTRY':    generate_configs_by_group(chemistry_data(),   group_id=group_id),
      'REMOTE':       generate_configs_by_group(remote_data(),      group_id=group_id),
      'LOCAL':        generate_configs_by_group(local_data(),       group_id=group_id),
    }

    for strat_name, configs in strategies.items():
      for config in configs:
        models = get_models_in_config(current_task, config)
        if not models: continue
        pts = np.array([[m.a_i, m.q_i] for m in models], float)
        if pts.size == 0: continue
        ci  = complementarity_index(pts)  # assumes metrics use fixed REF/BOUNDS later
        per_group_strategy[group_id][strat_name].append(
          {"CI": ci, "points": pts, "task": current_task,
           "group_id": group_id, "strategy": strat_name}
        )
        all_point_sets.append(pts)

  return per_group_strategy, all_point_sets

# %%
import os

import maps as maps_mod
from metrics import (
  complementarity_index, 
  hypervolume_2d_normalized,
  rao_quadratic_entropy_normalized
)


@type_checked
def render_maps_for_reps(
  current_task: str, 
  groups: list[tuple[int, int]] = [(1,3), (2,5), (3,10)],
  filter_ensembles: bool = False, 
  allow_overwrite: bool = False,
  output_root: str = "out"
):
  # Recall that groups are defined as (group_id, target_n):
  # Group 1: Target N=3, actual ≤3
  # Group 2: Target N=5, actual 3<x≤5
  # Group 3: Target N=10, actual >5
  
  # (1) collect items (no filter)
  per_group_strategy, all_point_sets = collect_ensembles_for_task(
    current_task, groups=groups)

  # (2) set fixed REF/BOUNDS for the whole task
  REF, BOUNDS = task_wide_ref_bounds(all_point_sets)
  maps_mod.REF = REF
  maps_mod.BOUNDS = BOUNDS

  os.makedirs(output_root, exist_ok=True)

  # (3) loop strategies, pick reps, make maps
  for group_id, strat_dict in per_group_strategy.items():
    for strat_name, items in strat_dict.items():
      if not items: continue

      # Tables (filtered) – optional
      if filter_ensembles:
        items = summarize_for_tables(items, ci_threshold=0.5)

      # Representatives (unfiltered)
      weak, mid, strong = select_representatives(items)
      reps = [("weak", weak), ("mid", mid), ("strong", strong)]
      reps = [(name, rep) for name, rep in reps if rep is not None]

      if not reps: continue

      for rank_name, rep in reps:
        out_dir = os.path.join(
          output_root,
          f"task_{slug(current_task)}",
          f"group{group_id}_{strat_name.lower()}",
          rank_name,
        )

        # if outdir/ exists, we can use it
        if os.path.exists(out_dir) and not allow_overwrite:
          print(f"Maps already exists in: {out_dir}")
          continue
        
        os.makedirs(out_dir, exist_ok=True)

        # One ensemble at a time
        maps_mod.build_chemistry_maps(
          ensembles=[rep["points"]],
          hypervolume_2d_normalized=hypervolume_2d_normalized,
          rao_quadratic_entropy_normalized=rao_quadratic_entropy_normalized,
          composite_index=complementarity_index,
          output_dir=out_dir,
        )

# %%
import random

random.seed(42); 
np.random.seed(42)

# Build maps for all tasks (and All groups by default in collect_ensembles_for_task)
for task in TASKS:
  print(f"\n=== Task: {task} ===")
  render_maps_for_reps(task)

# %%
print("Done.")

# %%
