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
- **Performance**: wall time per object for each backend.

### jax-galsim rendering strategy

jax-galsim can be rendered two ways, selected by `comparison.jax_mode`:

- **`batched`** (default) -- `jax.jit(jax.vmap(...))` over batches of
  `jax_batch_size` objects with a pinned FFT size. This is the **only** way to
  actually use a GPU: it compiles once and renders a whole batch in parallel.
- **`eager`** -- the naive one-object-at-a-time drop-in. Correct, and a
  faithful mirror of the galsim loop, but it does *not* saturate a GPU:
  each object dispatches ~1,500 serial kernel launches. An untimed warmup pass
  (`comparison.jax_warmup_galaxies`, default 50) precedes the timed eager run.
- **`both`** -- run eager and batched head-to-head.

If you ran the benchmark and jax-galsim took *minutes* for 10,000 objects, you
were on the eager path. See  `src/diagnose_jax.py` to reproduce the diagnosis on
your hardware:

```bash
cd src
python diagnose_jax.py            # op counts, compile counts, eager vs batched
python main.py --jax-mode both    # head-to-head in the real benchmark
```

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
./run.sh                       # uses config.yaml as-is (10,000 objects, batched jax)
./run.sh --n-obs 500            # override the number of objects
./run.sh --jax-mode both        # eager vs batched jax-galsim, head-to-head
./run.sh --batch-size 1024      # objects per jit(vmap) call (GPU occupancy knob)
./run.sh --save-datasets        # also dump rendered images to results/*.npz
```

Simulation parameters (galaxy HLR/flux, PSF FWHM, pixel scale, stamp size,
noise level, number of objects, applied shear, catalog/PSF paths) live in
`src/config.yaml`.

Output (a JSON report plus comparison plots) is written to `src/results/` by
default.

## Layout

- `src/dataset.py` -- shared image-simulation code, parameterized by backend
  module (`galsim` or `jax_galsim`); includes both the eager one-at-a-time
  renderer and the batched `jit(vmap)` renderer.
- `src/compare.py` -- pixel-diff and timing comparison between two rendered
  datasets.
- `src/plotting.py` -- diagnostic plots (example image triplet, pixel-diff
  histogram, timing).
- `src/main.py` -- CLI entry point tying the above together.
- `src/diagnose_jax.py` -- standalone script that measures op counts, compile
  counts, and eager-vs-batched timing to explain the performance gap.
