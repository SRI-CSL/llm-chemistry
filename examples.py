#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

import argparse

from beartype import beartype as type_checked

from chemistry import ChemE, Model, build_mig, cost_q, recommend


def execute(trial_data: dict, verbose: bool) -> None:
  @type_checked
  def example_find_models_for_q(
    q: str,
    candidate_models: set[str]
  ) -> tuple[set[str], float]:

    # Fetch models for this query
    if q not in trial_data:
      raise ValueError(f"Unknown P: {q}")

    available_models = trial_data[q]

    # Filter to valid candidates with data
    chosen_models = {
      available_models[m] 
      for m in candidate_models 
      if m in available_models
    }

    # Heuristic: use only models with a_i > 0.7
    used_models = {m for m in chosen_models if m.a_i > 0.7}

    # Compute cost from used_models
    cost = cost_q(used_models) if used_models else 1.0  # fallback cost if none used

    return {m.name for m in used_models}, cost

  @type_checked
  def example_load_past_runs(
    q: str
  ) -> set[frozenset[str]]:
    if q not in trial_data:
      return set()
    
    # Simulated past runs for the example
    return {
      frozenset({'a', 'b'}),
      frozenset({'a', 'c'}),
      frozenset({'b', 'c', 'd'}),
      frozenset({'a', 'd'})
    }
  
  S = {'a', 'b', 'c', 'd'}
  q = "example-request"
  mig = build_mig(q, S, example_find_models_for_q)
  if verbose:
    print("Printing MIG:")
    mig.print_mig()
  
  # Chemistry table
  chem_table = ChemE(S, mig)
  
  if verbose:
    print("ChemE results:")
    for (a, b), chem in sorted(chem_table.items()):
      print(f"chem_Q({a}, {b}) = {chem}")
    
    print("\n" + "-" * 40 + "\n")
  
  # Load historical subsets (mocked)
  X_q = example_load_past_runs(q)
  model_scores = { name: (m.a_i, m.q_i) for name, m in trial_data[q].items() }
  # Run recommendation
  best_subset = recommend(S, chem_table, X_q, alpha=0.5, model_scores=model_scores)
  sorted_subset = sorted(best_subset) if best_subset else 'None'
  print(f"Recommended subset for '{q}': {sorted_subset}")

def worst_scenario(verbose: bool) -> None:
  # Simulated CoLLM trial data for testing
  TRIAL_DATA: dict[str, dict[str, Model]] = {
    "example-request": {
      'a': Model(name='a', q_i=3.866269393389262, a_i=0.483156169435019, trial=None), 
      'd': Model(name='d', q_i=4.23545831646782, a_i=0.4169803693485891, trial=None), 
      'b': Model(name='b', q_i=7.0754943315754275, a_i=0.8701096088884712, trial=None), 
      'c': Model(name='c', q_i=5.498006799042209, a_i=0.5805627039013722, trial=None)}
  }
  
  execute(TRIAL_DATA, verbose)

def best_scenario(verbose: bool) -> None:
  # Simulated CoLLM trial data for testing
  TRIAL_DATA: dict[str, dict[str, Model]] = {
    "example-request": {
      'a': Model(name='a', q_i=10.0, a_i=1.0, trial=None), 
      'd': Model(name='d', q_i=10.0, a_i=1.0, trial=None), 
      'b': Model(name='b', q_i=10.0, a_i=1.0, trial=None), 
      'c': Model(name='c', q_i=10.0, a_i=1.0, trial=None)}
  }
  
  execute(TRIAL_DATA, verbose)


def random_scenario(verbose: bool) -> None:
  from random import uniform

  # Simulated CoLLM trial data for testing
  TRIAL_DATA: dict[str, dict[str, Model]] = {
    "example-request": {
      name: Model(name, q_i=uniform(3.0, 10.0), a_i=uniform(0.4, 1.0))
      for name in {'a', 'b', 'c', 'd'}
    }
  }
  
  execute(TRIAL_DATA, verbose)

@type_checked
def main() -> None:
  parser = argparse.ArgumentParser(
    description="Run ChemE-based Multi-LLM Recommendation."
  )
  parser.add_argument(
    "--verbose", "-v", action="store_true",
    help="Print detailed ChemE results and intermediate steps."
  )
  CHOICES = ["random", "worst", "best"]
  parser.add_argument(
    "--choice", "-c", choices=CHOICES, default="random",
    help="Choose the recommendation strategy."
  )
  
  args = parser.parse_args()
  
  if args.choice == "random":
    print("Running random scenario...")
    random_scenario(args.verbose)
  elif args.choice == "worst":
    print("Running worst scenario...")
    worst_scenario(args.verbose)
  elif args.choice == "best":
    print("Running best scenario...")
    best_scenario(args.verbose)
  else:
    print(f"Scenario '{args.choice}' is not implemented yet.")

if __name__ == "__main__":
  main()