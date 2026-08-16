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

**Both backends are fully independent.** Each reads the PSFEx model with its
own reader (`galsim.des.DES_PSFEx` / `jax_galsim.des.DES_PSFEx`), builds its
own `TanWCS`, evaluates the PSF at the chosen focal-plane position, and runs
its own `Convolve`/`shear`/`drawImage`. Only the *choices* are shared — which
catalog row, which PSFEx file, which position, the sub-pixel offset and the
noise realizations — all drawn once with numpy so the two pipelines can be
compared pixel-by-pixel on identical inputs. Nothing is rendered by galsim on
behalf of jax-galsim. Paths are configured under `paths:` in `src/config.yaml`.

> This requires `jax_galsim.des`, added in
> [JAX-GalSim PR #261](https://github.com/GalSim-developers/JAX-GalSim/pull/261).
> Until that merges, install jax-galsim from that branch.

Because the PSFEx model is fit to observed (already-pixelized) stars, it
already includes one convolution by the pixel response. Everything on the
`superbit` PSF path is therefore drawn with `drawImage(method='no_pixel')`,
so GalSim/JAX-GalSim does not convolve by the pixel a second time. The
analytic Gaussian (`ideal`) PSF does not include a pixel and keeps the
default `method='auto'`.

## Fidelity: how closely do the two agree?

`src/analyze_agreement.py` renders noiseless stamps with both backends and
scores them against **JAX-GalSim's own image-comparison cutoffs** (from
`tests/jax/test_spergel_comp_galsim.py`): `rtol=0` with `atol=1e-9` when the
two renderings may differ in FFT grid, and `atol=1e-16` for analytic profiles
on an identical grid. Those are defined on a *unit-flux* image, so the script
compares `max|Δ|/flux`.

```bash
cd src
python analyze_agreement.py -c config.yaml --n-obs 50
```

Over 40 objects spanning 40 distinct PSFEx models:

| quantity | result | cutoff |
|---|---|---|
| PSF stamp, `max\|Δ\|/flux` | 2.5e-15 | 1e-9 ✅ |
| galaxy stamp, `max\|Δ\|/flux` | 1.4e-11 | 1e-9 ✅ |
| relative flux error | 3.2e-09 | — |
| HSM `\|Δe1\|`, `\|Δe2\|` | **0 exactly** | — |
| HSM `\|Δsigma\|` | 2.4e-07 pix (max) | — |

Both backends pick identical `stepk`/`maxk`, so they use the same FFT grid.
The galaxy residual sits above the 1e-16 analytic ideal because this pipeline
interpolates an empirically-sampled PSF (Lanczos) and applies a WCS transform
— extra float operations XLA may reorder relative to C++ — but it is ~70x
inside the 1e-9 tolerance. The headline is the last two rows: adaptive-moment
ellipticities come out **bit-identical**, so shear estimates are unaffected.

### Caveat on the batched path

The pixel differences printed by `main.py --jax-mode batched` are *not* a
backend comparison: `vmap` needs fixed-shape inputs, so the batched path
resamples the PSF onto a stamp and wraps it in an `InterpolatedImage`, while
the galsim reference convolves the native PSFEx profile. Doing both *within
galsim* attributes the gap to the stamp round-trip (~5e-1) rather than the
pinned FFT size (~2e-5). Use `analyze_agreement.py` for fidelity, and the
batched mode for throughput.

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
- `src/analyze_agreement.py` -- stamp-level fidelity analysis scored against
  JAX-GalSim's own accuracy cutoffs, including HSM shape agreement.
