# jax-galsim-benchmark

A quick repository to test jax-galsim's numerical fidelity and time performance
against galsim, using a simulation pipeline modeled on
www.github.com/s-Sayan/ShearNet's dataset generation
(`shearnet/core/dataset.py`, as used by `shear_bias/m/main.py`).

Unlike ShearNet, this repo does **not** perform any shape measurement
(ngmix, metacalibration, ShearNet model evaluation). It only generates a
galaxy image dataset -- once with `galsim`, once with `jax_galsim`, using
identical random draws for both -- and then compares the two:

- **Pixel-level agreement**: max/mean absolute pixel difference and flux
  agreement between the two renderings of each object.
- **Performance**: wall time per object for each backend. Before the timed
  jax-galsim run, a separate, untimed warmup pass renders
  `comparison.jax_warmup_galaxies` (default 50) throwaway galaxies through
  the exact same code path to trigger JIT compilation, so the timed
  comparison itself is apples-to-apples (galsim needs no such warmup). The
  default `simulation.n_obs` is 10,000, large enough to amortize any
  remaining per-call overhead and saturate a GPU-backed jaxlib.

## Real data (COSMOS catalog + empirical PSFs)

Like ShearNet, this pipeline draws galaxy ellipticities from a real COSMOS
catalog and PSFs from real SuperBIT PSFEx models when they're available,
falling back to synthetic ellipticities / an analytic Gaussian PSF (with a
warning) otherwise -- the same fallback ShearNet itself uses. To use the
real data, place these at the repo root (siblings of `src/`):

- `cosmos_catalog_eval.fits` -- a FITS table with `G1`/`G2`/`HLR`/`FLUX`
  columns.
- `psf_data/emp_psfs_best/psfex-output/` -- a directory of `.psf` PSFEx
  model files (e.g. copied from ShearNet's `psf_data/`).

jax-galsim has no PSFEx/WCS support, so the empirical PSF for each object is
always evaluated once with galsim and rendered to a pixel stamp; both
backends then represent that stamp as an `InterpolatedImage`, keeping the
same physical PSF input for both while still exercising each backend's own
`Convolve`/`shear`/`drawImage` code path. Paths are configured under `paths:`
in `src/config.yaml`.

## Usage

```bash
conda env create -f environment.yml
conda activate jax-galsim-benchmark
cd src
./run.sh                       # uses config.yaml as-is (10,000 objects)
./run.sh --n-obs 500            # override the number of objects
./run.sh --jax-warmup 100       # override the untimed JIT-warmup galaxy count
./run.sh --save-datasets        # also dump rendered images to results/*.npz
```

Simulation parameters (galaxy HLR/flux, PSF FWHM, pixel scale, stamp size,
noise level, number of objects, applied shear, catalog/PSF paths) live in
`src/config.yaml`.

Output (a JSON report plus comparison plots) is written to `src/results/` by
default.

## Layout

- `src/dataset.py` -- shared image-simulation code, parameterized by backend
  module (`galsim` or `jax_galsim`).
- `src/compare.py` -- pixel-diff and timing comparison between two rendered
  datasets.
- `src/plotting.py` -- diagnostic plots (example image triplet, pixel-diff
  histogram, timing).
- `src/main.py` -- CLI entry point tying the above together.
