#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

import argparse
import pathlib as pl

from beartype import beartype as type_checked

from chemistry import (
  ChemE, 
  build_mig, 
  find_models_for_q, 
  generate_S,
  load_past_runs, 
  recommend, 
  set_trial_data_param,
  set_usage_threshold_params,
  get_trial_data,
)
from utils import find_top_k


TRIAL_TASKS: dict[str, list[str]] = {
  "runs.csv": ["classify a short statement into a category of fakeness"],
  "runs2.csv": ["fix buggy program"],
  "runs3.csv": ["generate a concise, clinically formal summary"]
}

def vprint(args: argparse.Namespace, msg: str) -> None:
  if getattr(args, "verbose", False):
    print(msg)

def cwd() -> pl.Path:
  return pl.Path.cwd()

@type_checked
def cmd_recommend(args: argparse.Namespace) -> None:
  tasks = TRIAL_TASKS.get(args.data)
  if tasks is None:
    print(f"Data for {args.data} not found.")
    return
  if not tasks:
    print(f"No tasks found for {args.data}.")
    return

  tasks = tasks if args.task is None else [args.task]

  data_file = cwd() / args.data
  if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")

  print(f"Setting: {args.data} as trial data file.")
  set_trial_data_param(args.data)
  
  trial_data = get_trial_data()

  print(f"Setting: {args.usage_threshold} as usage threshold.")
  set_usage_threshold_params(args.usage_threshold)

  S = generate_S()
  for q in tasks:
    mig = build_mig(q, S, find_models_for_q)
    chem_table = ChemE(S, mig)
    if args.verbose:
      print("-" * 40)
      print(f"ChemE results for '{q}':")
      for (a, b), chem in sorted(chem_table.items()):
        print(f"  chem({a}, {b}) = {chem:.4f}")
      print("-" * 40)
    X_q = load_past_runs(q)
    
    # Build per-model scores for this task q
    models_for_q = trial_data.get(q, {})
    model_scores = { name: (m.a_i, m.q_i) for name, m in models_for_q.items() }
    
    best_subset = recommend(
      S, chem_table, X_q, verbose=args.verbose, model_scores=model_scores)
    sorted_subset = sorted(best_subset) if best_subset else ['None']
    print("-" * 40)
    print(f"Recommended subset for '{q}':")
    for i, elem in enumerate(sorted_subset, 1):
      print(f"  {i}. {elem}")

@type_checked
def cmd_rank(args: argparse.Namespace) -> None:
  data_file = cwd() / args.data
  if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")

  print(f"Finding best performing models from: {args.data}")
  topk_results: list[set[str]] = find_top_k(data_file, Ks=args.ks)
  for k, models in zip(args.ks, topk_results):
    sorted_models = sorted(models) if models else ['None']
    print(f"Top-{k} models:")
    for i, model in enumerate(sorted_models, 1):
      print(f"  {i}. {model}")
    print("-" * 40)

@type_checked
def cmd_eval(args: argparse.Namespace) -> None:
  if args.question == 'rq1':
    from cheme_eval_rq1 import main
    main()
  elif args.question == 'rq2':
    from cheme_eval_rq2 import main
    main()
  elif args.question == 'rq3':
    from cheme_eval_rq3 import main
    main()

def build_eval_subparser(subparsers: argparse._SubParsersAction) -> None:
  p_eval = subparsers.add_parser(
    "eval",
    help="evaluates LLM Chemistry-based recommendation.",
  )
  p_eval.add_argument(
    "--question",
    "-q",
    type=str,
    choices=['rq1', 'rq2', 'rq3'],
    default="rq1",
    help="select a research question to eval.",
  )
  p_eval.set_defaults(func=cmd_eval)

def build_recommend_subparser(subparsers: argparse._SubParsersAction) -> None:
  p_rec = subparsers.add_parser(
    "recommend",
    help="recommend optimal model subsets for a given task.",
  )
  p_rec.add_argument(
    "--data",
    "-d",
    type=str,
    default="runs.csv",
    help="path to the trial data CSV file.",
  )
  p_rec.add_argument(
    "--usage-threshold",
    "-u",
    type=float,
    default=0.5,
    help="threshold for usage (0.0 to 1.0).",
  )
  p_rec.add_argument(
    "--task",
    "-t",
    type=str,
    default=None,
    help="run recommendation for a specific task (prototype).",
  )
  p_rec.set_defaults(func=cmd_recommend)

def build_rank_subparser(subparsers: argparse._SubParsersAction) -> None:
  p_rank = subparsers.add_parser(
    "rank",
    help="list top-K models by performance from the CSV file.",
  )
  p_rank.add_argument(
    "--data",
    "-d",
    type=str,
    default="runs.csv",
    help="path to the trial data CSV file.",
  )
  p_rank.add_argument(
    "--ks",
    "-k",
    type=int,
    nargs="+",
    default=[3, 5, 10],
    help="list of top-K values to retrieve. E.g., -k 3 5 10",
  )
  p_rank.set_defaults(func=cmd_rank)

def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run LLM Chemistry Estimation for Multi-LLM Recommendation."
  )
  parser.add_argument(
    "--verbose",
    "-v",
    action="store_true",
    help="print detailed results and intermediate steps.",
  )

  subparsers = parser.add_subparsers(
    dest="command", required=True)

  # recommend
  build_recommend_subparser(subparsers)
  # rank
  build_rank_subparser(subparsers)
  # eval
  build_eval_subparser(subparsers)

  return parser
  
def main() -> None:
  parser = build_parser()
  args = parser.parse_args()
  if hasattr(args, "func"):
    args.func(args)
  else:
    parser.print_help()

if __name__ == "__main__":
  main()
