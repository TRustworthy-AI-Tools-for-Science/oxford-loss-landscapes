# VR-PCA Research Utilities

This document describes the benchmark and analysis scripts that accompany the
VR-PCA implementation in `oxford_loss_landscapes`. They generate synthetic
regression problems, run both solvers with matched HVP budgets, and report
runtime, accuracy, and scaling trends.

> **Status**: VR-PCA is fully integrated into the main package. These scripts
> are research and validation tools, not part of the published API.

## Scripts

### `examples/vrpca_benchmark.py`

CLI entry point that constructs synthetic MLP models, runs `eigsh` and/or VR-PCA,
and prints eigenvalue estimates with timing.

```bash
python examples/vrpca_benchmark.py --solver both --width 64 --samples 512
```

Options: `--width`, `--input-dim`, `--samples`, `--epochs`, `--batch-size`,
`--solver {eigsh,vrpca,both}`, `--seed`.

### `scripts/scaling_analysis.py`

Post-processing tool that fits power-law scaling curves and visualises the
runtime vs accuracy trade-off using a CSV emitted by `vrpca_benchmark.py`.

```bash
python scripts/scaling_analysis.py vrpca_results.csv --plot-type all \
    --scaling-plot scaling.png \
    --error-plot runtime_vs_error.png \
    --combined-plot combined.png
```

### `scripts/combined_scaling_analysis.py`

Variant of the scaling script focused on producing a single publication-style
plot plus textual interpretation of the fitted exponents.

```bash
python scripts/combined_scaling_analysis.py vrpca_results.csv \
    --output vrpca_scaling.png --verbose
```

## Background: VR-PCA vs eigsh

`oxford_loss_landscapes.hessian` wraps SciPy's `eigsh`, which applies a Lanczos
iteration requiring repeated full-batch HVPs. The VR-PCA algorithm reduces cost by:

1. Periodic **snapshot** computations of the full HVP at the current iterate.
2. **Mini-batch corrections** approximating the HVP on small random data subsets.
3. A **variance-reduced update** adding the mini-batch difference to the snapshot,
   yielding an unbiased HVP estimate with lower noise.
4. **Normalization** after every step to keep the iterate on the unit sphere.

When the leading Hessian eigenvalue has a moderate gap (λ) and the dataset is
large, VR-PCA converges in `O(n + 1/λ²) log(1/ε)` HVPs — favourable compared to
`O(n · log(1/ε) / λ)` for Lanczos.

## Installation

```bash
python -m pip install -e .[dev]
```

Runtime dependencies: `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`,
`scikit-learn`.

## Example workflow

1. Install the project in editable mode with `[dev]` extras.
2. Run a single benchmark comparison with `vrpca_benchmark.py`.
3. For scaling sweeps, adapt `vrpca_benchmark.py` to loop over model widths and
   write rows to a CSV, then use `scaling_analysis.py` for plots.
4. Use `combined_scaling_analysis.py` for a quick read-out of fitted exponents.

## Troubleshooting

- **SciPy or scikit-learn import errors**: run `pip install -e .[dev]` in a clean
  virtual environment.
- **Plots not showing**: the scripts always save figures to disk; open the PNGs
  manually or add `plt.show()` in a notebook context.
