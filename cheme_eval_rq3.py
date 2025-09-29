#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

# Evaluates LLM Chemistry Estimation framework 
# for multi-LLM Recommendation: RQ3.

# %%
import pathlib as pl
import random
from functools import lru_cache

import beartype.typing as ty
import numpy as np
from beartype import beartype as type_checked

from chemistry import (ChemE, Model, build_mig, cost_q, generate_S,
                       get_trial_data, set_trial_data_param)
from utils import group_by_strategy, trial_groups_by_strategy

# %%
THRESHOLD_FOR_SUCCESS = 0.5
N_CONFIGURATIONS = 10 # 10 config per #LLMs in setup: 3, 5, 10
TRIAL_DATA_FILE = "runs3.csv"
VERBOSE_MODE = False

TASK1 = "classify a short statement into a category of fakeness"
TASK2 = "generate a concise, clinically formal summary"
TASK3 = "fix buggy program"
TASKS = [TASK1, TASK2, TASK3]

# %%
def vprint(*args, **kwargs) -> None:
  if VERBOSE_MODE:
    print(*args, **kwargs)

# %%
def task_from_file(
  task_file: str,
  all_tasks: list[str] = TASKS,
) -> list[str]:
  tasks = []
  if task_file == "runs.csv":
    tasks = all_tasks[:1]
  elif task_file == "runs3.csv":
    tasks = all_tasks[1:2]
  elif task_file == "runs2.csv":
    tasks = all_tasks[2:3]
  return tasks

# %%
@type_checked
def safe_pearsonr(
  x: ty.Iterable[float], 
  y: ty.Iterable[float]
) -> tuple[float, float]:
  from scipy.stats import pearsonr
  x = np.asarray(list(x), dtype=float)
  y = np.asarray(list(y), dtype=float)

  # Remove pairs with NaN/inf
  mask = np.isfinite(x) & np.isfinite(y)
  x, y = x[mask], y[mask]

  # Need at least 3 points for a meaningful Pearson r in your code's logic
  if x.size < 3 or y.size < 3:
    return 0.0, 1.0

  # Avoid ConstantInputWarning and NaNs by short-circuiting when variance is zero
  if np.all(x == x[0]) or np.all(y == y[0]):
    return 0.0, 1.0

  r, p = pearsonr(x, y)
  # Guard against any residual NaNs
  if not np.isfinite(r) or not np.isfinite(p):
    return 0.0, 1.0
  return float(r), float(p)

# %%
@lru_cache(maxsize=1)
def lazy_load_data() -> dict[str, dict[str, list[dict]]]:
  # Note: this path could change depending on where this module is placed.
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
@type_checked
def generate_configs_by_group(
  data: list[list[dict]], 
  group_id: int,  # 1, 2, or 3 corresponding to target N=3, 5, 10
  num_configs: int = N_CONFIGURATIONS,
  no_rand_sampling: bool = False
) -> list[list[dict]]:
  """Generate configurations for a specific group (accounts for LLM failures via using ranges)"""
  # Use your original robust grouping logic that handles LLM failures
  if group_id == 1:  # Target N=3
    group_data = [item for item in data if len(item) <= 3]
  elif group_id == 2:  # Target N=5  
    group_data = [item for item in data if 3 < len(item) <= 5]
  elif group_id == 3:  # Target N=10
    group_data = [item for item in data if len(item) > 5]
  else:
    raise ValueError(f"Invalid group_id: {group_id}. Must be 1, 2, or 3")
  
  if no_rand_sampling:
    return group_data
    
  # Sample up to num_configs from this specific group
  return random.sample(group_data, min(num_configs, len(group_data)))

@type_checked
def get_models_in_config(task: str, config: list[dict]) -> set[Model]:
  models = set()
  for row in config:
    if row['task'] == task:
      model = Model(
        name=row['model'], 
        q_i=row['quality'], 
        a_i=row['effectiveness'], 
        trial=row['trial'])
      models.add(model)
  return models

# %%
@type_checked
def x_find_models_for_q(
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

  # Heuristic: for each node in MIG, focus on models with a_i > 0.5.
  # These models are likely to be the most effective in that group.
  used_models = {m for m in chosen_models if m.a_i > 0.5}

  # Compute cost from used_models
  cost = cost_q(used_models) if used_models else 1.0  # fallback cost if none used

  return {m.name for m in used_models}, cost

# %%
@type_checked
def x_load_past_runs_by_group(q: str, group_id: int, no_rand_sampling: bool = False) -> set[frozenset[Model]]:
  """Load past runs for a specific group"""
  all_configs = []
  
  # Get configs for this specific group only
  remote_configs = generate_configs_by_group(remote_data(), group_id, no_rand_sampling=no_rand_sampling)
  local_configs = generate_configs_by_group(local_data(), group_id, no_rand_sampling=no_rand_sampling)
  random_configs = generate_configs_by_group(random_data(), group_id, no_rand_sampling=no_rand_sampling)
  perf_configs = generate_configs_by_group(performance_data(), group_id, no_rand_sampling=no_rand_sampling)
  
  # Union of all configurations for this group
  all_configs.extend(remote_configs + local_configs + random_configs + perf_configs)
  
  X_q = set()
  for config in all_configs:
    models = get_models_in_config(q, config)
    X_q.add(frozenset(models))
  
  return X_q

# %%
@type_checked
def calculate_configuration_chemistry(
  config: set[Model], 
  chem_table: dict[tuple[str, str], float]
) -> float:
  """Calculate overall chemistry score for a configuration"""
  if len(config) < 2:
    vprint("Configuration must have at least 2 models for chemistry calculation.")
    return 0.0

  chemistry_scores = []
  config_list = list(config)

  # Get chemistry scores for all pairs in configuration
  for i in range(len(config_list)):
    for j in range(i + 1, len(config_list)):
      model_a, model_b = config_list[i], config_list[j]

      # Check both directions in chemistry table
      pair_key1 = (model_a.name, model_b.name)
      pair_key2 = (model_b.name, model_a.name)

      chemistry_score = chem_table.get(pair_key1, chem_table.get(pair_key2, 0.0))
      chemistry_scores.append(chemistry_score)

  if not chemistry_scores:
    vprint("No chemistry scores found for configuration:", config)
    return 0.0
  
  # Return average chemistry across all pairs
  return sum(chemistry_scores) / len(chemistry_scores)

# %%
@type_checked
def calculate_coordination_metrics(models: set[Model]) -> dict[str, float]:
  """Calculate coordination-focused metrics"""
  if len(models) < 2:
    return {
      'ensemble_complementarity': 1.0,
      'ensemble_effectiveness': 1.0
    }
  
  from metrics import complementarity_index
  points = [[m.a_i, m.q_i] for m in models]
  points = np.array(points)
  ensemble_complementarity_2 = complementarity_index(points)
  if not np.isfinite(ensemble_complementarity_2):
    ensemble_complementarity_2 = 0.0 # neutral fallback
  
  votes = [m.a_i for m in models]
  N = len(votes)
  # number of votes in votes above 0.7
  M = sum(1 for v in votes if v > 0.7)
  effectiveness_score = M / N if N > 0 else 0.0

  return {
    'ensemble_complementarity': ensemble_complementarity_2,
    'ensemble_effectiveness': effectiveness_score
  }

# %%
def get_benchmark_complexity(task: str) -> str:
  """Get benchmark-level complexity classification"""
  complexity_mapping = {
    'classify a short statement into a category of fakeness': 'low',
    'fix buggy program': 'high',  # Update with exact task name
    'generate a concise, clinically formal summary': 'medium'  # Update with exact task name
  }
  return complexity_mapping.get(task, 'unknown')

# %%
# TODO TO BE DELETED
@type_checked
def calculate_chemistry_importance(
  chemistry_scores: list[float], 
  ensemble_complementarity_scores: list[float]
) -> float:
  """Calculate how important chemistry is for ensemble complementarity at this complexity/group level"""
  from scipy.stats import pearsonr

  if len(chemistry_scores) < 3:
    return 0.0

  try:
    # Note: This can fail if inputs are constant or too few points; => nan values
    # correlation, p_value = pearsonr(chemistry_scores, ensemble_complementarity_scores)
    correlation, p_value = safe_pearsonr(chemistry_scores, ensemble_complementarity_scores)
    
    # Importance = correlation strength weighted by significance
    if p_value < 0.05:
      importance = abs(correlation)
    else:
      importance = abs(correlation) * 0.5  # Reduce importance if not significant
      
    return importance
  except:
    return 0.0

# %%
@type_checked
def analyze_task_complexity_effects_by_group(
  task: str, 
  group_id: int, 
  target_n: int
) -> dict[str, ty.Any]:
  """Analyze complexity effects for a specific task and group"""
  
  complexity_level = get_benchmark_complexity(task)
  
  # Build chemistry analysis for this task
  S = generate_S()
  mig = build_mig(task, S, x_find_models_for_q)
  chem_table = ChemE(S, mig)
  
  # Get configurations for this specific group
  X_q = x_load_past_runs_by_group(task, group_id, no_rand_sampling=True)
  
  chemistry_scores = []
  ensemble_complementarity_scores = []
  # performance_consistency_scores = []
  ensemble_effectiveness_scores = []

  for config in X_q:
    config_set = set(config)
    
    if len(config_set) < 2:
      continue
      
    # Calculate chemistry score
    chemistry_score = calculate_configuration_chemistry(config_set, chem_table)
    
    # Calculate coordination metrics
    coord_metrics = calculate_coordination_metrics(config_set)
    
    chemistry_scores.append(chemistry_score)
    ensemble_complementarity_scores.append(coord_metrics['ensemble_complementarity'])
    ensemble_effectiveness_scores.append(coord_metrics['ensemble_effectiveness'])
    # performance_consistency_scores.append(coord_metrics['performance_consistency'])
  
  if not chemistry_scores:
    return {
      'complexity_level': complexity_level,
      'group_id': group_id,
      'target_n': target_n,
      'sample_size': 0
    }
  
  # Calculate chemistry importance for ensemble complementarity
  complementarity_importance = calculate_chemistry_importance(
    chemistry_scores, ensemble_complementarity_scores)
  # consistency_importance = calculate_chemistry_importance(
  #   chemistry_scores, performance_consistency_scores)
  effectiveness_importance = calculate_chemistry_importance(
    chemistry_scores, ensemble_effectiveness_scores)
  
  # Overall chemistry importance (average of significant effects)
  overall_importance = (
    complementarity_importance 
    # + consistency_importance 
    + effectiveness_importance) / 2 # 3
  
  def corr_calc(cheme_scores, other_scores):
    from scipy.stats import pearsonr
    try:
      # Note: This can fail if inputs are constant or too few points; => nan values
      # corr, p_val = pearsonr(cheme_scores, other_scores)
      # FIXED
      corr, p_val = safe_pearsonr(cheme_scores, other_scores)
    except:
      corr, p_val = 0.0, 1.0
    return corr, p_val

  # Calculate correlation for reporting
  complementarity_corr, complementarity_p = corr_calc(
    chemistry_scores, ensemble_complementarity_scores)
  effective_corr, effective_p = corr_calc(
    chemistry_scores, ensemble_effectiveness_scores)
  
  return {
    'complexity_level': complexity_level,
    'group_id': group_id,
    'target_n': target_n,
    'sample_size': len(chemistry_scores),
    'chemistry_importance': overall_importance,
    'ensemble_complementarity_correlation': complementarity_corr,
    'ensemble_complementarity_p_value': complementarity_p,
    'ensemble_effective_correlation': effective_corr,
    'ensemble_effective_p_value': effective_p,
    'avg_chemistry_score': np.mean(chemistry_scores),
    'avg_ensemble_complementarity': np.mean(ensemble_complementarity_scores),
    'avg_ensemble_effectiveness': np.mean(ensemble_effectiveness_scores),
    'chemistry_score_std': np.std(chemistry_scores)
  }

# %%
@type_checked
def evaluate_rq3_by_group(tasks: list[str], group_id: int, target_n: int) -> dict[str, ty.Any]:
  """Evaluate RQ3 for a specific group"""
  results = {
    'group_id': group_id,
    'target_n': target_n,
    'task_complexity_effects': {}
  }
  
  for task in tasks:
    vprint(f"Analyzing complexity effects for task: {task}, Group {group_id}")
    
    task_effects = analyze_task_complexity_effects_by_group(task, group_id, target_n)
    results['task_complexity_effects'][task] = task_effects
  
  return results

# %%
@type_checked
def analyze_complexity_group_interactions(group_results: dict) -> dict[str, ty.Any]:
  """Analyze how complexity and group size interact to affect chemistry importance"""
  
  # Organize data by complexity and group
  complexity_group_matrix = {}
  
  for group_name, group_data in group_results.items():
    group_id = group_data['group_id']
    target_n = group_data['target_n']
    
    for task, task_effects in group_data['task_complexity_effects'].items():
      complexity = task_effects['complexity_level']
      chemistry_importance = task_effects['chemistry_importance']
      sample_size = task_effects['sample_size']
      
      if complexity not in complexity_group_matrix:
        complexity_group_matrix[complexity] = {}
      
      complexity_group_matrix[complexity][f'N{target_n}'] = {
        'importance': chemistry_importance,
        'sample_size': sample_size,
        'task': task
      }
  
  # Identify patterns
  patterns = {
    'complexity_trends': {},
    'group_size_trends': {},
    'interaction_effects': []
  }
  
  # Analyze trends by complexity level
  for complexity, group_data in complexity_group_matrix.items():
    importance_scores = [data['importance'] for data in group_data.values() if data['sample_size'] > 5]
    
    if importance_scores:
      patterns['complexity_trends'][complexity] = {
        'avg_importance': np.mean(importance_scores),
        'max_importance': np.max(importance_scores),
        'min_importance': np.min(importance_scores),
        'importance_range': np.max(importance_scores) - np.min(importance_scores),
        'groups_analyzed': len(importance_scores)
      }
  
  # Identify strong interaction effects
  for complexity, group_data in complexity_group_matrix.items():
    for group_size, data in group_data.items():
      if data['importance'] > 0.3 and data['sample_size'] > 10:  # Threshold for meaningful effect
        patterns['interaction_effects'].append({
          'complexity': complexity,
          'group_size': group_size,
          'importance': data['importance'],
          'task': data['task'],
          'sample_size': data['sample_size']
        })
  
  # Sort interaction effects by importance
  patterns['interaction_effects'].sort(key=lambda x: x['importance'], reverse=True)
  
  return patterns

# %%
@type_checked
def evaluate_rq3(tasks: list[str]) -> dict[str, ty.Any]:
  """Evaluate RQ3 across all groups and tasks as required by protocol"""
  # Group definitions that handle LLM failures
  groups = [
    (1, 3),   # Group 1: Target N=3, actual ≤3
    (2, 5),   # Group 2: Target N=5, actual 3<x≤5  
    (3, 10)   # Group 3: Target N=10, actual >5
  ]
  
  all_results = {
    'by_group': {},
    'summary': {},
    'interaction_analysis': {}
  }
  
  # Evaluate each group separately
  for group_id, target_n in groups:
    group_results = evaluate_rq3_by_group(tasks, group_id, target_n)
    all_results['by_group'][f'Group_{group_id}_N{target_n}'] = group_results
  
  # Cross-task and cross-group interaction analysis
  all_results['interaction_analysis'] = analyze_complexity_group_interactions(
    all_results['by_group'])
  
  # Summary information
  all_results['summary'] = {
    'groups_evaluated': [(group_id, target_n) for group_id, target_n in groups],
    'tasks_evaluated': tasks,
    'complexity_levels': [get_benchmark_complexity(task) for task in tasks],
    'grouping_logic': {
      'Group_1_N3': 'len(models) ≤ 3',
      'Group_2_N5': '3 < len(models) ≤ 5', 
      'Group_3_N10': 'len(models) > 5'
    }
  }
  
  return all_results

# %%
def clear_data_file_cache():
  lazy_load_data.cache_clear()


def summarize_interaction_effects(rq3_results: dict):
  print("=== RQ3: Task Complexity x Chemistry Interaction Effects ===")
  print()
  
  print(f"{'Complexity':<12} {'Group':<8} {'Chemistry Importance':<20} {'Ensemble Complementarity (a)':<18} {'Ensemble Effectiveness (b)':<18} {'Sample Size':<12} {'(a) Significant?':<12} {'(b) Significant?':<12}")
  print("-" * 95)
  
  def significance_indicator(p_value: float) -> str:
    if p_value < 0.001:
      return "Yes***"
    elif p_value < 0.01:
      return "Yes**"
    elif p_value < 0.05:
      return "Yes*"
    else:
      return "No"

  # Display results in matrix format
  for group_name, group_data in rq3_results['by_group'].items():
    # Clean group name
    if 'Group_1_N3' in group_name:
      group_display = "N≤3"
    elif 'Group_2_N5' in group_name:
      group_display = "3<N≤5"
    elif 'Group_3_N10' in group_name:
      group_display = "N>5"
    else:
      group_display = group_name
    
    for task, task_effects in group_data['task_complexity_effects'].items():
      complexity = task_effects['complexity_level']
      importance = task_effects['chemistry_importance']
      complementarity_corr = task_effects['ensemble_complementarity_correlation']
      complementarity_p = task_effects['ensemble_complementarity_p_value']
      effective_corr = task_effects['ensemble_effective_correlation']
      effective_p = task_effects['ensemble_effective_p_value']

      # Significance indicator
      sample_size = task_effects['sample_size']
      sig_indicator = significance_indicator(complementarity_p)
      sig_indicator2 = significance_indicator(effective_p)
      
      print(f"{complexity.capitalize():<12} {group_display:<8} {importance:<20.3f} {complementarity_corr:<18.3f} {effective_corr:<18.3f} {sample_size:<12} {sig_indicator:<12} {sig_indicator2:<12}")
  
  print("-" * 95)
  print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
  
  # Summary analysis
  print()
  print("=== Key Findings ===")
  
  interaction_analysis = rq3_results['interaction_analysis']
  
  # Complexity trends
  if 'complexity_trends' in interaction_analysis:
    print("Chemistry Importance by Complexity Level:")
    for complexity, trends in interaction_analysis['complexity_trends'].items():
      avg_importance = trends['avg_importance']
      importance_range = trends['importance_range']
      print(f"  • {complexity.capitalize()}: Average importance = {avg_importance:.3f} (range = {importance_range:.3f})")
  
  # Strong interaction effects
  if 'interaction_effects' in interaction_analysis and interaction_analysis['interaction_effects']:
    print()
    print("Strongest Interaction Effects:")
    for effect in interaction_analysis['interaction_effects'][:3]:  # Top 3
      print(f"  • {effect['complexity'].capitalize()} complexity, {effect['group_size']}: "
            f"Chemistry importance = {effect['importance']:.3f} (n={effect['sample_size']})")
  
  print()

# %%
def perform_rq3_analysis(data_file: str):
  global TRIAL_DATA_FILE, TASKS
  TRIAL_DATA_FILE = data_file
  TASKS_p = task_from_file(TRIAL_DATA_FILE)
  TASKS = TASKS_p
  set_trial_data_param(data_file)
  rq3_results = evaluate_rq3(TASKS)
  summarize_interaction_effects(rq3_results)

# %%
def main() -> None:
  for data_file in ["runs.csv", "runs2.csv", "runs3.csv"]:
    perform_rq3_analysis(data_file)
    clear_data_file_cache()

# %%
# main()
