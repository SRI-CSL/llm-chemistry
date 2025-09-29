#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

import beartype.typing as ty
import numpy as np
from beartype import beartype as type_checked

@type_checked
def hypervolume_2d(
  points: np.ndarray,
  ref: ty.Tuple[float, float] = (0.0, 0.0),
  bounds: ty.Tuple[float, float] = (1.0, 1.0),
) -> float:
  """
  Hypervolume in 2D (maximize both axes):
  
  Naujoks, B., Beume, N., & Emmerich, M. (2005, September). 
  Multi-objective optimisation using S-metric selection: Application 
  to three-dimensional solution spaces. In 2005 IEEE Congress on 
  Evolutionary Computation (Vol. 2, pp. 1282-1289). IEEE.
  
  Compute the dominated hypervolume (area) of 2D points for a max-max problem,
  relative to a reference point `ref`, clipped to an upper bound `bounds`.

  Interpretation: area of union of rectangles from `ref` to each point.

  Requirements:
    - Larger is better on both axes (accuracy, quality).
    - ref <= points <= bounds coordinate-wise (we clip if needed).
    - If your scores aren't in [0,1], set `ref=(min_a,min_q)`, `bounds=(max_a,max_q)`.

  Returns:
    Hypervolume (area) in the same units as the rectangle [ref, bounds].
  """
  P = np.asarray(points, dtype=float)
  if P.size == 0:
    return 0.0

  # Clip to [ref, bounds]
  lo = np.array(ref, dtype=float)
  hi = np.array(bounds, dtype=float)
  P = np.minimum(np.maximum(P, lo), hi)

  # Remove points that are weakly dominated by others (optional but faster/cleaner)
  # A point x is dominated if exists y with y>=x on both coords and y>x on at least one.
  # Simple O(M^2) filter (fine for modest M).
  keep = np.ones(len(P), dtype=bool)
  for i in range(len(P)):
    if not keep[i]:
      continue
    # Creates a boolean mask dom that identifies which rows 
    # in the 2D array P "dominate" the row P[i] according 
    # to Pareto dominance.
    dom = (P >= P[i]).all(axis=1) & (P > P[i]).any(axis=1)
    if dom.any():
      keep[i] = False
  P = P[keep]
  if len(P) == 0:
    return 0.0

  # Sort by x (accuracy) descending; build the increasing skyline in y (quality)
  P = P[np.argsort(-P[:, 0], kind="mergesort")]  # stable
  hv = 0.0
  x_prev = hi[0]
  y_best = lo[1]

  # Sweep from high x down to ref.x, accumulating disjoint vertical strips
  for x, y in P:
    if y > y_best:
      # Add strip area between current x and previous x, at height y_best
      hv += (x_prev - x) * (y_best - lo[1])
      # Update frontier
      y_best = y
      x_prev = x

  # Final strip from last x to ref.x at height y_best
  hv += (x_prev - lo[0]) * (y_best - lo[1])

  # Clip to rectangle area (numeric safety)
  rect_area = max(0.0, (hi[0] - lo[0]) * (hi[1] - lo[1]))
  return float(min(max(hv, 0.0), rect_area))

@type_checked
def hypervolume_2d_normalized(
  points: np.ndarray,
  ref: ty.Tuple[float, float] = (0.0, 0.0),
  bounds: ty.Tuple[float, float] = (1.0, 1.0),
) -> float:
  """HV normalized to [0,1] by dividing by the rectangle area [ref,bounds]."""
  area = hypervolume_2d(points, ref=ref, bounds=bounds)
  rect_area = max(0.0, (bounds[0] - ref[0]) * (bounds[1] - ref[1]))
  return 0.0 if rect_area == 0 else float(area / rect_area)

@type_checked
def rao_quadratic_entropy(
  points: np.ndarray,
  metric: ty.Literal["euclidean", "mahalanobis"] = "euclidean",
  cov: np.ndarray | None = None,
) -> float:
  """
  Rao's Quadratic Entropy or RaoQ (aka, average pairwise distance):
  
  Botta-Dukát, Z. (2005). Rao's quadratic entropy as a measure of 
  functional diversity based on multiple traits. Journal of 
  vegetation science, 16(5), 533-540.
  
  RaoQ (with uniform weights) = average pairwise distance among the points.
  points: (M, 2) array of (accuracy, quality).
  """
  X = np.asarray(points, dtype=float)
  M = X.shape[0]
  if M < 2:
    return 0.0

  if metric == "euclidean":
    diffs = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diffs * diffs, axis=2))
  elif metric == "mahalanobis":
    if cov is None:
      cov = np.cov(X.T, bias=True)
    # Compute the (Moore-Penrose) pseudo-inverse of a matrix.
    VI = np.linalg.pinv(cov)
    diffs = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.einsum("...i,ij,...j->...", diffs, VI, diffs))
  else:
    raise ValueError("metric must be 'euclidean' or 'mahalanobis'")

  # Average with uniform weights w_i = 1/M:
  # RaoQ = sum_{i,j} w_i w_j d(x_i,x_j) = (1/M^2) * sum_{i,j} d_ij
  return float(D.sum() / (M * M))

@type_checked
def rao_quadratic_entropy_normalized(points, ref, bounds):
  X = np.asarray(points, float)
  if X.shape[0] < 2: return 0.0
  # average pairwise distance (include i=j zeros; harmless):
  diffs = X[:,None,:] - X[None,:,:]
  D = np.sqrt(np.sum(diffs*diffs, axis=2))
  rao = D.sum() / (X.shape[0] * X.shape[0])

  # FIXED normalization by rectangle diagonal (must not depend on points!)
  dx = max(0.0, bounds[0] - ref[0])
  dy = max(0.0, bounds[1] - ref[1])
  dmax = (dx**2 + dy**2) ** 0.5
  return 0.0 if dmax == 0.0 else float(np.clip(rao / dmax, 0.0, 1.0))

@type_checked
def complementarity_index(
  points: np.ndarray,
  ref: ty.Tuple[float, float] = (0.0, 0.0),
  bounds: ty.Tuple[float, float] = (1.0, 1.0),
  alpha: float = 0.5,
) -> float:
  """
  A single-number complementarity score in [0,1]:
    CI = alpha * HV_norm + (1 - alpha) * RaoQ_norm

  - HV_norm captures ensemble-level coverage of the accuracy–quality space.
  - RaoQ_norm captures member-level non-redundancy (spread).
  - alpha ∈ [0,1] balances coverage vs non-redundancy (default 0.5).
  """
  hv_n = hypervolume_2d_normalized(points, ref=ref, bounds=bounds)
  rao_n = rao_quadratic_entropy_normalized(points, ref=ref, bounds=bounds)
  alpha = float(np.clip(alpha, 0.0, 1.0))
  return float(alpha * hv_n + (1.0 - alpha) * rao_n)

if __name__ == "__main__":
  # Example usage
  # Points are (accuracy, quality) for each ensemble member
  ensemble_weak = np.array([
    [0.90, 0.78],
    [0.92, 0.76],
    [0.88, 0.80],
    [0.91, 0.77],
    [0.89, 0.79],
  ], dtype=float)
  ensemble_moderate = np.array([
    [0.92, 0.55],
    [0.88, 0.70],
    [0.81, 0.85],
    [0.95, 0.40],
    [0.86, 0.78],
  ], dtype=float)
  ensemble_strong = np.array([
    [0.98, 0.30],  # accuracy specialist
    [0.30, 0.98],  # quality specialist
    [0.91, 0.90],  # strong generalist
    [0.85, 0.55],
    [0.70, 0.80],
  ], dtype=float)

  # What we are after is a metric for Ensemble complementarity. 
  # Ensemble complementarity is the extent to which ensemble members 
  # bring different and useful strengths so that, taken together, 
  # the ensemble provides better coverage of the performance 
  # space (accuracy–quality trade-offs in your case) than any 
  # single member could alone.
  # Intuition:
  # - If members of an ensemble are too similar, they mostly reinforce each 
  # other's strengths and weaknesses → little added value.
  # - If members are different in useful ways, one member's weakness 
  # is offset by another's strength.
  # - Thus, complementarity is about:
  #  1. Coverage: members occupy different regions of the performance trade-off space.
  #  2. Non-redundancy: members are not clustered but spread in a way 
  # that their contributions add new value.
  # In other words, complementarity is the property that ensemble members 
  # compensate for each other's weaknesses and extend the ensemble's overall 
  # capabilities by being both diverse and useful.

  inputs = [ensemble_weak, ensemble_moderate, ensemble_strong]
  for points in inputs:
    hv_norm = hypervolume_2d_normalized(points)
    print(f"Normalized Hypervolume: {hv_norm}")
    rao = rao_quadratic_entropy_normalized(points)  # in [0,1]
    print(f"Normalized Rao's Quadratic Entropy: {rao}")
    ci  = complementarity_index(points, alpha=0.5)
    print(f"Complementarity Index: {ci}")

  from maps import build_chemistry_maps
  ensembles = [ensemble_weak, ensemble_moderate, ensemble_strong]
  files = build_chemistry_maps(ensembles, hypervolume_2d_normalized, rao_quadratic_entropy_normalized, complementarity_index)
  for i, f in enumerate(files, start=1):
    print(f"Map for Ensemble {i}: {f}")