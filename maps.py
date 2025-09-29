#!/usr/bin/env python
# Copyright Huascar Sanchez, 2024-.

import os

import beartype.typing as ty
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as type_checked

# Parameters for visualizing the accuracy-quality trade-offs
REF: ty.Tuple[float, float] = (0.0, 0.0) # Reference point
BOUNDS: ty.Tuple[float, float] = (1.0, 1.0) # Upper bounds
ALPHA: float = 0.5  # weight for HV in composite CI
GRID_RES: int = 150 # grid resolution for maps (higher = smoother, slower)

Hypervolume2dNorm = ty.Callable[
  [np.ndarray, ty.Tuple[float, float], ty.Tuple[float, float]],
  float
]
RaoNorm = ty.Callable[
  [np.ndarray, ty.Tuple[float, float], ty.Tuple[float, float]],
  float
]
CompositeIndex = ty.Callable[
  [np.ndarray, ty.Tuple[float, float], ty.Tuple[float, float], float],
  float
]

@type_checked
def build_maps(
  E: np.ndarray, 
  hypervolume_2d_normalized: Hypervolume2dNorm, 
  rao_quadratic_entropy_normalized: RaoNorm, 
  composite_index: CompositeIndex
) -> tuple:
  # Baselines
  HV0 = hypervolume_2d_normalized(E, REF, BOUNDS)
  Rao0 = rao_quadratic_entropy_normalized(E, REF, BOUNDS)
  CI0 = composite_index(E, REF, BOUNDS, ALPHA)

  xs = np.linspace(REF[0], BOUNDS[0], GRID_RES)
  ys = np.linspace(REF[1], BOUNDS[1], GRID_RES)
  XX, YY = np.meshgrid(xs, ys)
  grid = np.stack([XX.ravel(), YY.ravel()], axis=1)

  # Marginal gains at candidate point x
  def d_rao(x):
    return rao_quadratic_entropy_normalized(np.vstack([E, x]), REF, BOUNDS) - Rao0
  def d_hv(x):
    return hypervolume_2d_normalized(np.vstack([E, x]), REF, BOUNDS) - HV0
  def d_ci(x):
    return composite_index(np.vstack([E, x]), REF, BOUNDS, ALPHA) - CI0

  dR = np.array([d_rao(x) for x in grid]).reshape(XX.shape)
  dH = np.array([d_hv(x) for x in grid]).reshape(XX.shape)
  dC = np.array([d_ci(x) for x in grid]).reshape(XX.shape)

  centroid = E.mean(axis=0)
  return (XX, YY, dR, dH, dC, centroid, Rao0, HV0, CI0)

def classify_field(Z, tiny=1e-12, low_contrast=1e-8):
  """
  Classify a ΔCI field for diagnostics without changing values,
  where CI is the complementarity index.

  Params:
  -----------------
  tiny: values below this are effectively zero everywhere
  low_contrast:range below this means "almost constant"
  """
  import numpy as np
  Z = np.asarray(Z, float)
  zmin = np.nanmin(Z); zmax = np.nanmax(Z)
  if not np.isfinite(zmin) or not np.isfinite(zmax):
    return "invalid"
  rng = zmax - zmin
  if zmax <= tiny:
    return "saturated_all_dark"   # ΔCI ≈ 0 everywhere
  if rng <= low_contrast:
    return "uniform_bright_like"  # nearly constant > 0 (suspicious)
  return "normal"


@type_checked
def save_map(XX, YY, Z, E, centroid, title, cbar_label, outfile):
  import matplotlib.pyplot as plt
  import numpy as np
  from matplotlib.ticker import ScalarFormatter

  # Fixed: DO NOT modify Z numerically (no clipping)
  Z = np.asarray(Z, float)
  zmin = float(np.nanmin(Z))
  zmax = float(np.nanmax(Z))
  if not np.isfinite(zmin): zmin = 0.0
  if not np.isfinite(zmax): zmax = 0.0
  if zmax <= zmin:               # purely constant field (e.g., all zeros)
    zmax = zmin + 1e-12          # plotting-only epsilon

  status = classify_field(Z)
  if status == "saturated_all_dark":
    subtitle = " (saturated: ΔCI≈0 everywhere)"
  elif status == "uniform_bright_like":
    subtitle = " (uniform field: nearly constant)"
  else:
    subtitle = ""

  fig, ax = plt.subplots(figsize=(4.2, 3.6))

  # Colormap: ensure near-zero shows as darkest color, not white background
  cmap = plt.get_cmap("viridis").copy()
  cmap.set_under(cmap(0.0))          # color for values < first level

  # Always include the true data min/max; use 'extend=min' to paint under-range
  cf = ax.contourf(
    XX, YY, Z,
    levels=20,
    cmap=cmap,
    vmin=zmin, vmax=zmax,
    extend='min',                  # show darkest for near-zero ΔCI
    antialiased=True, zorder=0
  )
  ax.set_facecolor(cmap(0.0))        # background = darkest (no white gaps)

  # Axes & title
  ax.set_xlim(REF[0], BOUNDS[0])
  ax.set_ylim(REF[1], BOUNDS[1])
  ax.margins(0)
  ax.set_xlabel('Accuracy')
  ax.set_ylabel('Quality')
  
  if subtitle == "":
    formatted_title = title + "\n"
  else:
    formatted_title = title + subtitle + "\n"
  ax.set_title(formatted_title, fontsize=10)

  # Points (clipped to bounds, visible on any background)
  if E is not None and len(E) > 0:
    E = np.asarray(E, float)
    E = E[np.isfinite(E).all(axis=1)]
    lo, hi = np.array(REF, float), np.array(BOUNDS, float)
    E = np.minimum(np.maximum(E, lo), hi)
    ax.scatter(
      E[:, 0], E[:, 1], s=50, facecolors="white",
      edgecolors="black", linewidths=0.8, zorder=3, clip_on=True
    )

  # Centroid
  if centroid is not None and np.isfinite(centroid).all():
    cx = min(max(float(centroid[0]), REF[0]), BOUNDS[0])
    cy = min(max(float(centroid[1]), REF[1]), BOUNDS[1])
    ax.scatter(
      [cx], [cy], s=80, marker="X", c="#e53935",
      edgecolors="white", linewidths=0.9, zorder=4, clip_on=True
    )

  # Colorbar with non-intrusive exponent
  cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
  cb.set_label(cbar_label, fontsize=9)
  fmt = ScalarFormatter(useMathText=True)
  fmt.set_powerlimits((-2, 2))
  cb.formatter = fmt
  cb.update_normal(cf)

  fig.tight_layout()
  fig.savefig(outfile, dpi=300, bbox_inches="tight")
  plt.close(fig)

@type_checked
def validate_complementarity_index(
  points, 
  composite_index: CompositeIndex,
  ref=(0.0, 0.0), 
  bounds=(1.0, 1.0), 
  alpha=0.5,
  tol=1e-9
):
  """
  Compute the complementarity index (CI) and check that it lies within [0, 1].
  If not, raise a ValueError with details.

  Args:
    points : np.ndarray
      Ensemble as an (n,2) array of (accuracy, quality) pairs.
    ref : tuple
      Reference point for normalization.
    bounds : tuple
      Upper bounds for normalization.
    alpha : float
      Weight between HV and Rao components.
    tol : float
      Tolerance for floating point drift.

  Returns:
    float : CI score (validated).
  """
  ci = composite_index(points, ref=ref, bounds=bounds, alpha=alpha)

  if not (0.0 - tol <= ci <= 1.0 + tol):
    raise ValueError(
      f"Complementarity index out of bounds: {ci:.6f} "
      f"(expected within [0,1], ref={ref}, bounds={bounds}, alpha={alpha})"
    )
  # Clip tiny numerical drift (e.g., -1e-12 or 1.0000000001)
  return float(np.clip(ci, 0.0, 1.0))

@type_checked
def build_chemistry_maps(
  ensembles: ty.List[np.ndarray],
  hypervolume_2d_normalized: Hypervolume2dNorm, 
  rao_quadratic_entropy_normalized: RaoNorm, 
  composite_index: CompositeIndex,
  output_dir: str = "output", 
  trial: str = "trial", 
  debug: bool = False
) -> ty.List[str]:
  output_paths = []
  for idx, E in enumerate(ensembles, start=1):
    validate_complementarity_index(E, composite_index, REF, BOUNDS, ALPHA)
    if debug:
      print("DEBUG: |E|=", len(E))
      print(
        "DEBUG: HV0, Rao0, CI0 =",
        hypervolume_2d_normalized(E, REF, BOUNDS),
        rao_quadratic_entropy_normalized(E, REF, BOUNDS),
        composite_index(E, REF, BOUNDS, ALPHA)
      )

    # Pick a candidate x far away (e.g., [0.05, 0.95])
    x = np.array([0.05, 0.95])
    if debug:
      print(
        "DEBUG: HV(E∪{x}) vs HV(E) =",
        hypervolume_2d_normalized(np.vstack([E, x]), REF, BOUNDS),
        hypervolume_2d_normalized(E, REF, BOUNDS))
      print(
        "DEBUG: Rao(E∪{x}) vs Rao(E) =",
        rao_quadratic_entropy_normalized(np.vstack([E, x]), REF, BOUNDS),
        rao_quadratic_entropy_normalized(E, REF, BOUNDS))
    XX, YY, dR, dH, dC, centroid, Rao0, HV0, CI0 = build_maps(
      E, 
      hypervolume_2d_normalized, 
      rao_quadratic_entropy_normalized, 
      composite_index
    )
    # make directory exist Ok
    os.makedirs(output_dir, exist_ok=True)
    base = f"{output_dir}/{trial}_ensemble_{idx:02d}"
    p1 = f"{base}_rao_delta_map.png"
    p2 = f"{base}_hv_delta_map.png"
    p3 = f"{base}_composite_delta_map.png"
    save_map(
      XX, YY, dR, E, centroid,
      title=f"Rao-only Marginal Map (ΔRaoQ_norm)\nBaseline RaoQ_norm={Rao0:.3f}",cbar_label="ΔRaoQ_norm when adding a model at (x,y)",
      outfile=p1
    )
    save_map(
      XX, YY, dH, E, centroid,
      title=f"Hypervolume-only Marginal Map (ΔHV_norm)\nBaseline HV_norm={HV0:.3f}",
      cbar_label="ΔHV_norm when adding a model at (x,y)",
      outfile=p2)
    save_map(
      XX, YY, dC, E, centroid,
      title=f"Complementarity Marginal Map (ΔCI, λ={ALPHA:.2f})\nBaseline CI={CI0:.3f}",
      cbar_label="ΔCI when adding a model at (x,y)",
      outfile=p3
    )
    output_paths.extend([p1, p2, p3])
  return output_paths

if __name__ == "__main__":
  pass
