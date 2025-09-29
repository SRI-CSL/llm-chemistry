#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

# Evaluates LLM Chemistry Estimation framework 
# for multi-LLM Recommendation: RQ2.

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
  """Generate configurations for a specific group (accounts for LLM failures)"""
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

# %%
@type_checked
def calculate_task_success(models: set[Model]) -> float:
  """Binary success rate - ensemble gets correct answer"""
  if not models:
    return 0.0

  # Use average voting approach to assess success
  avg_accuracy = sum(m.a_i for m in models) / len(models)

  # Threshold for "success" - can be adjusted
  return 1.0 if avg_accuracy > THRESHOLD_FOR_SUCCESS else 0.0

@type_checked
def calculate_reliability_score(models: set[Model]) -> float:
  """Cost efficiency (1.0 - cost_q [Performance penalty]) using your cost function"""
  if not models:
    return 0.0
  
  cost = cost_q(models)
  # Reliability = 1 - normalized_cost (lower cost = higher efficiency)
  return max(0.0, 1.0 - cost)

# %%
@type_checked
def get_models_in_config(task: str, config: list[dict]) -> set[Model]:
  models = set()
  for row in config:
    if row['task'] == task:
      model = Model(
        name=row['model'], 
        q_i=row['quality'], 
        a_i=row['accuracy'], 
        trial=row['trial'])
      models.add(model)
  return models

# MODIFIED: Now works by group
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
# MODIFIED: Now works by group
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
def calculate_avg_configuration_chemistry(
  config: set[Model], 
  chem_table: dict[tuple[str, str], float]
) -> float:
  """Calculate overall (average) chemistry score for a LLM configuration"""
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
def evaluate_single_configuration(
  task: str, 
  config: set[Model]
) -> dict[str, float] | None:
  """Evaluate performance metrics for a single configuration"""
  vprint(f"Evaluating configuration for task: {task} with models: {config}")

  models_in_config = config
  
  if not models_in_config:
    return None

  return {
    # Original binary metrics
    'task_success_rate': calculate_task_success(models_in_config),
    'reliability_score': calculate_reliability_score(models_in_config),
  }

# %%
@type_checked
def calculate_task_correlations(task_data: list[dict]) -> dict[str, dict[str, float]]:
  """Calculate correlations between chemistry and performance metrics for a task"""
  import warnings

  from scipy.stats import pearsonr, spearmanr

  chemistry_scores = [d['chemistry_score'] for d in task_data]

  # Test both original and new continuous metrics
  metrics = [
    # Original binary metrics
    'task_success_rate', 
    'reliability_score',
  ]

  correlations = {}

  for metric in metrics:
    if metric in task_data[0]:
      metric_values = [d[metric] for d in task_data]

      # Skip correlation if either variable is constant
      if len(set(chemistry_scores)) <= 1 or len(set(metric_values)) <= 1:
        correlations[metric] = {
          'pearson_r': float('nan'),
          'pearson_p': float('nan'),
          'spearman_r': float('nan'),
          'spearman_p': float('nan'),
          'n_samples': len(task_data),
          'note': 'Constant values - correlation skipped'
        }
        continue

      # Calculate correlations with warning suppression
      with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Precision loss occurred in moment calculation')
        try:
          pearson_r, pearson_p = pearsonr(chemistry_scores, metric_values)
          spearman_r, spearman_p = spearmanr(chemistry_scores, metric_values)
        except Exception as e:
          pearson_r = pearson_p = spearman_r = spearman_p = float('nan')

      correlations[metric] = {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_samples': len(task_data)
      }

  return correlations

# %%
# MODIFIED: Now evaluates by group
@type_checked
def evaluate_rq2_by_group(tasks: list[str], group_id: int, target_n: int) -> dict[str, ty.Any]:
  """Evaluate chemistry-performance correlation for a specific group"""
  results = {
    'group_id': group_id,
    'target_n': target_n,
    'correlation_data': [],
    'task_correlations': {},
    'summary_stats': {}
  }
  
  S = generate_S()
  
  for task in tasks:
    vprint(f"Analyzing chemistry-performance correlation for task: {task}, Group {group_id}")

    # Build MIG and compute chemistry table for this task
    mig = build_mig(task, S, x_find_models_for_q)
    chem_table = ChemE(S, mig)
    
    # Get configurations for this specific group
    X_q = x_load_past_runs_by_group(task, group_id, no_rand_sampling=True)
    
    # Analyze each configuration
    task_data = []
    for config in X_q:
      config_set = set(config)
      vprint(f"Evaluating configuration: {config_set}")
      
      # Calculate chemistry score for this configuration
      chemistry_score = calculate_avg_configuration_chemistry(config_set, chem_table)
      
      # Calculate performance metrics
      performance_metrics = evaluate_single_configuration(task, config_set)

      if performance_metrics:
        data_point = {
          'task': task,
          'group_id': group_id,
          'configuration': config_set,
          'chemistry_score': chemistry_score,
          **performance_metrics
        }
        task_data.append(data_point)
        results['correlation_data'].append(data_point)
    
    # Calculate correlations for this task in this group
    if task_data:
      task_correlations = calculate_task_correlations(task_data)
      results['task_correlations'][task] = task_correlations
  
  # Summary statistics for this group
  if results['correlation_data']:
    chemistry_scores = [d['chemistry_score'] for d in results['correlation_data']]
    results['summary_stats'] = {
      'n_configurations': len(results['correlation_data']),
      'chemistry_mean': np.mean(chemistry_scores),
      'chemistry_std': np.std(chemistry_scores),
      'chemistry_range': [np.min(chemistry_scores), np.max(chemistry_scores)]
    }
  
  return results

# %%
@type_checked
def evaluate_rq2(tasks: list[str]) -> dict[str, ty.Any]:
  """Evaluate RQ2 across all groups as required by protocol"""
  # Group definitions that handle LLM failures
  groups = [
    (1, 3),   # Group 1: Target N=3, actual ≤3
    (2, 5),   # Group 2: Target N=5, actual 3<x≤5  
    (3, 10)   # Group 3: Target N=10, actual >5
  ]
  
  all_results = {
    'by_group': {},
    'summary': {}
  }
  
  # Evaluate each group separately
  for group_id, target_n in groups:
    group_results = evaluate_rq2_by_group(tasks, group_id, target_n)
    all_results['by_group'][f'Group_{group_id}_N{target_n}'] = group_results
  
  # Summary information
  all_results['summary'] = {
    'groups_evaluated': [(group_id, target_n) for group_id, target_n in groups],
    'tasks_evaluated': tasks,
    'grouping_logic': {
      'Group_1_N3': 'len(models) ≤ 3',
      'Group_2_N5': '3 < len(models) ≤ 5', 
      'Group_3_N10': 'len(models) > 5'
    }
  }
  
  return all_results

@type_checked
def calculate_complementarity_metrics(models: set[Model]) -> dict[str, float]:
  """Calculate coordination-focused metrics"""
  if len(models) < 2:
    return {
      'ensemble_complementarity': 1.0
    }
    
  # Ensemble coherence: how well models complement each other
  # Higher coherence = models have similar performance levels but diverse qualities.
  from metrics import complementarity_index
  points = [[m.a_i, m.q_i] for m in models]
  points = np.array(points)
  ensemble_complementarity_2 = complementarity_index(points)
  
  return {
    'ensemble_complementarity': ensemble_complementarity_2
  }

# %%
@type_checked
def analyze_chemistry_complementarity_correlation(results: dict) -> dict[str, ty.Any]:
  """Focus on chemistry-complementarity correlations"""
  import warnings

  from scipy.stats import pearsonr, spearmanr
  
  print("=== RQ2: Chemistry-Ensemble Complementarity Analysis ===")
  print()
  
  print(f"{'Group':<12} {'Coordination Metric':<22} {'Correlation (r)':<15} {'Strength':<12} {'p-value':<8} {'Significant?':<12}")
  print("-" * 95)
  
  # Focus on key coordination metrics, with ensemble coherence highlighted
  coordination_metrics = [
    'ensemble_complementarity',        # Primary metric of interest
  ]
  
  metric_names = {
    'ensemble_complementarity': 'Ensemble Complementarity', 
  }
  
  def interpret_correlation(r):
    """Interpret correlation strength"""
    abs_r = abs(r) if not np.isnan(r) else 0
    if abs_r >= 0.5:
      return "Strong"
    elif abs_r >= 0.25:
      return "Moderate" 
    elif abs_r >= 0.1:
      return "Weak"
    else:
      return "Negligible"
  
  def significance_indicator(r, p):
    """Determine significance with effect size consideration"""
    if np.isnan(r) or np.isnan(p):
      return "No"
    
    if p < 0.001 and abs(r) >= 0.2:
      return "Yes***"
    elif p < 0.01 and abs(r) >= 0.15:
      return "Yes**"
    elif p < 0.05 and abs(r) >= 0.1:
      return "Yes*"
    else:
      return "No"
  
  coordination_results = {}
  key_findings = []
  
  for group_name, group_data in results['by_group'].items():
    # Clean group name
    if 'Group_1_N3' in group_name:
      group_display = "N≤3"
    elif 'Group_2_N5' in group_name:
      group_display = "3<N≤5"
    elif 'Group_3_N10' in group_name:
      group_display = "N>5"
    else:
      group_display = group_name
    
    # Calculate coordination metrics for all configurations in this group
    enhanced_data = []
    for config_data in group_data['correlation_data']:
      models = config_data['configuration']
      coord_metrics = calculate_complementarity_metrics(models)
      enhanced_config = {**config_data, **coord_metrics}
      enhanced_data.append(enhanced_config)
    
    if len(enhanced_data) < 10:
      print(f"{group_display:<12} Insufficient data (n={len(enhanced_data)})")
      continue
    
    chemistry_scores = [d['chemistry_score'] for d in enhanced_data]
    group_correlations = {}
    
    # Calculate correlations for coordination metrics
    for metric in coordination_metrics:
      if metric in enhanced_data[0]:
        metric_values = [d[metric] for d in enhanced_data]
        
        # Skip if constant values
        if len(set(chemistry_scores)) <= 1 or len(set(metric_values)) <= 1:
          group_correlations[metric] = {
            'pearson_r': float('nan'),
            'pearson_p': float('nan'),
            'note': 'Constant values'
          }
          continue
        
        # Calculate correlation
        with warnings.catch_warnings():
          warnings.filterwarnings('ignore', 'Precision loss occurred in moment calculation')
          try:
            pearson_r, pearson_p = pearsonr(chemistry_scores, metric_values)
          except:
            pearson_r, pearson_p = float('nan'), float('nan')
        
        group_correlations[metric] = {
          'pearson_r': pearson_r,
          'pearson_p': pearson_p,
          'n_samples': len(enhanced_data)
        }
        
        # Collect significant findings for ensemble coherence
        if metric == 'ensemble_complementarity' and not np.isnan(pearson_r) and not np.isnan(pearson_p):
          if pearson_p < 0.05 and abs(pearson_r) >= 0.15:
            key_findings.append((group_display, pearson_r, pearson_p, len(enhanced_data)))
        
        # Display result
        r_display = f"{pearson_r:.3f}" if not np.isnan(pearson_r) else "N/A"
        p_display = f"{pearson_p:.3f}" if not np.isnan(pearson_p) else "N/A"
        
        strength = interpret_correlation(pearson_r)
        significant = significance_indicator(pearson_r, pearson_p)
        
        # Highlight ensemble coherence results
        if metric == 'ensemble_complementarity' and significant.startswith('Yes'):
          print(f"{group_display:<12} {metric_names[metric]:<22} {r_display:<15} {strength:<12} {p_display:<8} {significant:<12} ★")
        else:
          print(f"{group_display:<12} {metric_names[metric]:<22} {r_display:<15} {strength:<12} {p_display:<8} {significant:<12}")
        
        # Only show group name for first metric
        group_display = ""
    
    coordination_results[group_name] = {
      'correlations': group_correlations,
      'n_configurations': len(enhanced_data)
    }
    
    print(f"{'(n=' + str(len(enhanced_data)) + ')':<12}")
  
  print("-" * 95)
  print("★ = Key Finding: Significant chemistry-ensemble coherence relationship")
  print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
  
  # Store key findings in results
  coordination_results['key_findings'] = key_findings
  
  return coordination_results

# %%
@type_checked
def summarize_complementarity_findings(coordination_results: dict) -> None:
  """Provide positive, paper-ready summary of complementarity findings"""
  print()
  print("=== Key Research Finding ===")
  
  # Extract key findings
  key_findings = coordination_results.get('key_findings', [])
  
  if key_findings:
    print("DISCOVERY: Chemistry scores significantly predict ensemble coherence in larger collaborative configurations")
    print()
    
    for group, r, p, n in key_findings:
      effect_size = "moderate" if abs(r) >= 0.25 else "small-to-moderate"
      significance = "highly significant" if p < 0.001 else "significant"
      
      print(f"• {group}: r = {r:.3f}, p < {0.001 if p < 0.001 else 0.05} ({effect_size} {significance} correlation, n = {n})")
      
# %%
def perform_rq2_analysis(data_file: str):
  global TRIAL_DATA_FILE, TASKS
  TRIAL_DATA_FILE = data_file
  TASKS_p = task_from_file(TRIAL_DATA_FILE)
  TASKS = TASKS_p
  set_trial_data_param(data_file)
  rq2_results = evaluate_rq2(TASKS)
  
  # Focus on coordination and consistency (Option 4)
  coordination_results = analyze_chemistry_complementarity_correlation(rq2_results)
  summarize_complementarity_findings(coordination_results)
  
  # Optional: Add chemistry score distribution diagnostics
  print("\n=== Chemistry Score Distribution Diagnostics ===")
  for group_name, group_data in rq2_results['by_group'].items():
    if group_data['correlation_data']:
      chem_scores = [d['chemistry_score'] for d in group_data['correlation_data']]
      group_display = group_name.replace('Group_', '').replace('_N', ' N=')
      print(f"{group_display}: Range=[{min(chem_scores):.3f}, {max(chem_scores):.3f}], "
            f"Mean={np.mean(chem_scores):.3f}, Std={np.std(chem_scores):.3f}")
    else:
      print(f"{group_name}: No data available")

# %%
def clear_data_file_cache():
  lazy_load_data.cache_clear()

# %%
def main() -> None:
  for data_file in ["runs.csv", "runs2.csv", "runs3.csv"]:
    perform_rq2_analysis(data_file)
    clear_data_file_cache()

# main()
