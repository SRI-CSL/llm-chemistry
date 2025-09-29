#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

# Evaluates LLM Chemistry Estimation framework 
# for multi-LLM Recommendation: RQ1.

# %%
# Experiment configuration constants
THRESHOLD_FOR_SUCCESS = 0.5
N_CONFIGURATIONS = 10
VERBOSE_MODE = False
NO_RAND_SAMPLING = True

BASELINES = ['RANDOM', 'PERFORMANCE', 'REMOTE', 'LOCAL']
CHEMISTRY = ['CHEMISTRY']

# Queries set Qs
TASK1 = "classify a short statement into a category of fakeness"
TASK2 = "generate a concise, clinically formal summary"
TASK3 = "fix buggy program"
TASKS = [TASK1, TASK2, TASK3]

# %%
def task_from_file(
  task_file: str,
  all_tasks: list[str],
) -> list[str]:
  tasks = []
  if task_file == "runs.csv":
    tasks = all_tasks[:1]
  elif task_file == "runs3.csv":
    tasks = all_tasks[1:2]
  elif task_file == "runs2.csv":
    tasks = all_tasks[2:3]
  return tasks

def task_display_name(data_file: str) -> str:
  if data_file == "runs.csv":
    return "(Task 1)"
  elif data_file == "runs2.csv":
    return "(Task 3)" # In paper is Task 3 
  elif data_file == "runs3.csv":
    return "(Task 2)"
  return "(Unknown Task)"

# %%
def abbreviate(word: str, length: int = 4) -> str:
  word = word.capitalize()
  if len(word) > length:
    return word[:length] + "."
  return word

# %%
import pathlib as pl
import random
from collections import defaultdict
from functools import lru_cache

# %%
# Global dependencies
import beartype.typing as ty
import numpy as np
import pandas as pd
from beartype import beartype as type_checked
from scipy import stats

# %%
from utils import group_by_strategy, trial_groups_by_strategy


# Utility functions for loading trial data
@lru_cache(maxsize=1)
def lazy_load_data(data_file: str) -> dict[str, dict[str, list[dict]]]:
  trial_data_file = pl.Path(data_file)
  if not trial_data_file.exists():
    raise FileNotFoundError(f"Trial data file not found: {trial_data_file}")
  return group_by_strategy(trial_data_file)

def remote_data(data_file: str) -> list[list[dict]]:
  data = lazy_load_data(data_file)
  return trial_groups_by_strategy("REMOTE", data)

def random_data(data_file: str) -> list[list[dict]]:
  data = lazy_load_data(data_file)
  return trial_groups_by_strategy("RANDOM", data)

def local_data(data_file: str) -> list[list[dict]]:
  data = lazy_load_data(data_file)
  return trial_groups_by_strategy("LOCAL", data)

def performance_data(data_file: str) -> list[list[dict]]:
  data = lazy_load_data(data_file)
  return trial_groups_by_strategy("PERFORMANCE", data)

def chemistry_data(data_file: str) -> list[list[dict]]:
  data = lazy_load_data(data_file)
  return trial_groups_by_strategy("CHEMISTRY", data)

# %%
def vprint(*args, **kwargs) -> None:
  if VERBOSE_MODE:
    print(*args, **kwargs)

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
        vprint(item)
    stats = [len(item) for item in group_data]
    vprint(f"Group 3 stats: min={min(stats)}, max={max(stats)}, mean={np.mean(stats):.2f}, median={np.median(stats)}")
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
@type_checked
def load_actual_performance_data(
  data_file: str
) -> tuple[dict[str, dict[str, float]], list[str]]:
  """
  Load actual model performance data from runs.csv
  Returns: (model_performance_dict, list_of_items)
  """
  df = pd.read_csv(data_file)

  # Get unique items/test cases
  items = sorted(df['id'].unique().tolist())

  # For each model-item pair, get the effectiveness/accuracy
  model_performance = defaultdict(dict)

  for _, row in df.iterrows():
    model_name = row['model']
    item_id = row['id']

    # Use effectiveness as the "correctness" probability for this item
    # In real evaluation, you'd have actual predictions vs ground truth
    effectiveness = row['effectiveness']

    model_performance[model_name][item_id] = effectiveness

  return dict(model_performance), items

# %%
@type_checked
def evaluate_ensemble_effectiveness(
  selected_models: list[str], 
  model_performance: dict[str, dict[str, float]], 
  items: list[str]
) -> float:
  """
  Correctly evaluate ensemble effectiveness:
  For each item, simulate a majority vote (Soft voting) 
  where each model contributes its effectiveness score
  as a probability of voting correctly. The ensemble is 
  correct if the average vote > 0.5.
  """
  if not selected_models or not items:
    return 0.0

  total_correct = 0.0
  valid_items = 0

  for item in items:
    votes = []
    for model in selected_models:
      if model in model_performance and item in model_performance[model]:
        # Use effectiveness as probability of voting correctly
        votes.append(model_performance[model][item])

    if not votes:
      continue

    # Soft voting (averaging model correctness probabilities with a 0.5 threshold
    ensemble_vote = np.mean(votes)
    total_correct += 1 if ensemble_vote > THRESHOLD_FOR_SUCCESS else 0
    valid_items += 1

  return total_correct / valid_items if valid_items > 0 else 0.0

# %%
@type_checked
def evaluate_improvement_over_best(
  selected_models: list[str],
  model_performance: dict[str, dict[str, float]],
  items: list[str]
) -> float:
  """
  FIXED: Use consistent effectiveness values for improvement calculation
  """
  if not selected_models or not items:
    return 0.0

  # Get effectiveness values from the same source as ensemble calculation
  all_configs_data = []
  data = lazy_load_data()
  
  # Combine all strategies
  strategies = BASELINES + CHEMISTRY

  for strategy in strategies:
    strategy_data = trial_groups_by_strategy(strategy, data)
    for group in strategy_data:
      all_configs_data.extend(group)

  model_effectiveness_map = {}
  for row in all_configs_data:
    model_name = row['model']
    effectiveness = row['effectiveness']
    model_effectiveness_map[model_name] = effectiveness

  # Calculate individual effectiveness
  individual_effectiveness = []
  for model in selected_models:
    if model in model_effectiveness_map:
      individual_effectiveness.append(model_effectiveness_map[model])

  if not individual_effectiveness:
    return 0.0

  best_individual = max(individual_effectiveness)
  ensemble_effectiveness = evaluate_ensemble_effectiveness(
    selected_models, model_performance, items)

  return ensemble_effectiveness - best_individual

# %%
@type_checked
def evaluate_strategy_configurations(
  task: str, 
  configs: list[list[dict]],
  model_performance: dict[str, dict[str, float]],
  items: list[str]
) -> list[dict[str, float]]:
  """Evaluate collaborative performance metrics for each LLM configuration"""
  if not configs:
    return []

  results = []

  for config in configs:
    models_in_config = get_models_in_config(task, config)

    if not models_in_config:
      continue

    model_names = [m.name for m in models_in_config]

    vprint(f'  Models in Ensemble:')
    seen_before = {}
    unique_models = []
    for name in model_names:
      if name not in seen_before:
        unique_models.append(name)
        seen_before[name] = 1

    for name in unique_models:
      vprint(f'    - {name}')
    ensemble_effectiveness = evaluate_ensemble_effectiveness(model_names, model_performance, items)

    vprint(f'  Ensemble Effectiveness: {ensemble_effectiveness:.3f}')
    vprint('')

    # Collaborative performance metrics using actual data
    metrics = {
      'ensemble_effectiveness': ensemble_effectiveness,
      'num_models': len(model_names),
      # Individual metrics for comparison
      'avg_individual_effectiveness': np.mean([m.a_i for m in models_in_config]),
      'best_individual_effectiveness': max([m.a_i for m in models_in_config])
    }

    results.append(metrics)

  return results

# %%
@type_checked
def evaluate_rq1(tasks: list[str], data_file: str) -> dict[str, ty.Any]:
  """RQ1: Evaluates whether Chemistry-based selection improves 
  Ensemble Effectiveness compared to baseline strategies."""

  # Set seed for reproducibility
  random.seed(42)
  np.random.seed(42)

  groups = [
    (1, 3),   # Group 1: Target N=3, (actual ≤3 to account for failures)
    (2, 5),   # Group 2: Target N=5, (actual 3<x≤5 to account for failures)
    (3, 10)   # Group 3: Target N=10, (actual 5<x≤10 to account for failures)
  ]

  all_results = {
    'by_group': {},
    'summary': {}
  }

  # Load actual performance data
  vprint("Loading actual performance data from runs.csv...")
  model_performance, items = load_actual_performance_data(data_file=data_file)
  vprint(f"Loaded performance data for {len(model_performance)} models on {len(items)} items")

  for group_id, target_n in groups:
    vprint(f"\nEvaluating Group {group_id} (Target N={target_n})...")

    group_results = {
      'group_id': group_id,
      'target_n': target_n,
      # ensemble effectiveness results per strategy
      'strategy_performance': defaultdict(list)
    }

    for task in tasks:
      vprint(f"  Task: {task}")

      # Get configurations for each strategy
      strategies = {
        'RANDOM': generate_configs_by_group(random_data(data_file), group_id),
        'PERFORMANCE': generate_configs_by_group(performance_data(data_file), group_id),
        'CHEMISTRY': generate_configs_by_group(chemistry_data(data_file), group_id),
        'REMOTE': generate_configs_by_group(remote_data(data_file), group_id),
        'LOCAL': generate_configs_by_group(local_data(data_file), group_id)
      }

      for strategy, configs in strategies.items():
        vprint(f" {strategy}: {len(configs)} configurations")

        strategy_metrics = evaluate_strategy_configurations(
          task, configs, model_performance, items
        )
        
        vprint(type(strategy_metrics))
        group_results['strategy_performance'][strategy].extend(strategy_metrics)

    all_results['by_group'][f'Group_{group_id}_N{target_n}'] = group_results

  return all_results

# %%
@type_checked
def statistical_analysis_collaborative(
  results: dict[str, ty.Any], 
  *,
  task_display: str
) -> None:
  """Statistical analysis for collaborative performance"""
  
  print("\n" + "="*85)
  print(f"RQ1: ENSEMBLE EFFECTIVENESS IMPROVEMENT ANALYSIS {task_display}")
  print("="*85)

  # Key collaborative metrics
  key_metrics = ['ensemble_effectiveness']
  metric_names = {
    'ensemble_effectiveness': 'Ensemble Effectiveness',
  }

  print(f"\n{'Group Size (N)':<12} {'Metric':<25} {'Chemistry':<10} {'Best Baseline':<12} {'Improvement (Δ%)':<8}")
  print("-" * 95)

  significant_improvements = 0
  total_comparisons = 0

  group_best_map = {}
  for group_name, group_data in results['by_group'].items():
    # Extract group size 3, 5, and 10 from text like Group_1_N3, Group_2_N5, Group_3_N10
    # so instead N=3, N=5, N=10, we just print 3, 5, 10
    group_display = group_name.replace('Group_', 'Group ').replace('_N', ' N ')
    # if N = 3, 5 this works fine, but if N = 10, we want to just show 10
    group_display = group_display.split()  # Split into parts
    group_display = group_display[-1]

    for metric in key_metrics:
      # Get data for all strategies
      strategy_data = {}
      for strategy, metrics_list in group_data['strategy_performance'].items():
        values = [m[metric] for m in metrics_list if metric in m]
        if values:
          strategy_data[strategy] = values

      if 'CHEMISTRY' not in strategy_data:
        continue

      chemistry_values = strategy_data['CHEMISTRY']
      chemistry_mean = np.mean(chemistry_values)

      # DEBUG: Print what strategies we're comparing
      # uppercase first letter in group_display
      group_display = group_display.upper()
      vprint(f"DEBUG - N = {group_display} {metric_names[metric]}:")
      for strat, vals in strategy_data.items():
        vprint(f"  {strat}: mean={np.mean(vals):.3f}, n={len(vals)}")

      # Find best baseline
      baseline_means = {}
      for strategy, values in strategy_data.items():
        if strategy != 'CHEMISTRY':
          baseline_means[strategy] = np.mean(values)

      if not baseline_means:
        continue

      best_baseline_strategy = max(baseline_means, key=baseline_means.get)
      best_baseline_values = strategy_data[best_baseline_strategy]
      best_baseline_mean = np.mean(best_baseline_values)
      
      abbreviated_strategy_display = abbreviate(best_baseline_strategy)
      group_best_map[group_display] = abbreviated_strategy_display

      # Improvement of mean Chemistry over mean Best baseline
      improvement = chemistry_mean - best_baseline_mean

      # Statistical test
      if len(chemistry_values) > 1 and len(best_baseline_values) > 1:
        try:
          t_stat, p_value = stats.ttest_ind(chemistry_values, best_baseline_values)
        except:
          p_value = 1.0
      else:
        p_value = 1.0

      # Significance
      if p_value < 0.001:
        sig = "***"
      elif p_value < 0.01:
        sig = "**"
      elif p_value < 0.05:
        sig = "*"
      else:
        sig = "ns"

      # Count significant improvements
      total_comparisons += 1
      if p_value < 0.05 and improvement > 0:
        significant_improvements += 1

      # Display
      # Improvement (Δ%)
      improvement_delta_percent = (improvement / best_baseline_mean * 100) if best_baseline_mean != 0 else float('inf')
      print(
        f"{group_display:<12} {metric_names[metric]:<25} {chemistry_mean:<10.3f} "
        f"{best_baseline_mean:<12.3f} {improvement_delta_percent:<8.3f}"
      )

      group_display = ""  # Only show group name once
  print("-" * 95)
  print("Where:")
  # Print group_best_map
  for group, strategy in group_best_map.items():
    print(f"- Best Baseline for Group {group}: {strategy}")
  print("-" * 95)

# %%
def perform_rq1_analysis(
  tasks: list[str], 
  task_display: str = "Task 1",
  data_file: str = "runs.csv"
) -> None:
  print(f"RQ1 Evaluation Starting for {task_display}...")
  print("="*60)

  try:
    # Run collaborative evaluation
    results = evaluate_rq1(tasks, data_file=data_file)
    if not results or 'by_group' not in results:
      print("No results to analyze.")
      return

    # Statistical analysis
    statistical_analysis_collaborative(results, task_display=task_display)

  except Exception as e:
    print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()

# %%
from chemistry import set_trial_data_param

def main():
  # Research Question 1 (RQ1)

  # 'Performance histories' files
  # Task 1 (Classification) = "runs.csv"
  # Task 2 (Summarization) = "runs3.csv"
  # Task 3 (Program Repair) = "runs2.csv"

  for data_file in ["runs.csv", "runs3.csv", "runs2.csv"]:
    set_trial_data_param(data_file)
    # Filter tasks based on selected trial data file
    tasks = task_from_file(data_file, TASKS)
    perform_rq1_analysis(
      tasks, 
      task_display=task_display_name(data_file), 
      data_file=data_file
    )

# %%
# main()

# %%
print("Done.")

# %%
pass