"""Dataset generation shared by both the galsim and jax-galsim backends.

This is a stripped-down port of s-Sayan/ShearNet's `shearnet/core/dataset.py`
(the reusable simulation module the `shear_bias/m` bias test is itself built
on): a single Exponential galaxy with an intrinsic ellipticity drawn from a
COSMOS catalog, sheared by +/-shear_true, convolved with either an empirical
SuperBIT PSFEx PSF or an analytic Gaussian PSF, drawn onto a postage stamp
and given Gaussian pixel noise. The ngmix Observation packaging, PSF-leakage
perturbation images, SNR/clean-image extras and the metacalibration
bootstrapper have all been removed -- this only produces the images, since
dataset generation (not shape measurement) is what's being benchmarked.

Real inputs (matching ShearNet's layout, both expected at the repo root,
i.e. a sibling of `src/`):
  - `cosmos_catalog_eval.fits`: a FITS table with G1/G2/HLR/FLUX columns.
  - `psf_data/emp_psfs_best/psfex-output/`: a directory of `.psf` PSFEx
    model files, one of which is selected at random per object (matching
    ShearNet's `import_psf`), together with a random position in a
    SuperBIT-sized focal plane.

Both backends are fully independent: each reads the PSFEx model with its own
reader (``galsim.des.DES_PSFEx`` / ``jax_galsim.des.DES_PSFEx``), builds its
own TanWCS, evaluates the PSF at the chosen focal-plane position and runs its
own Convolve/shear/drawImage. Only the *choices* (which catalog row, which
PSFEx file, which position, the sub-pixel offset and the noise realizations)
are drawn once with numpy and shared, so the two pipelines can be compared
pixel-by-pixel on identical inputs. Nothing is rendered by galsim on behalf
of jax-galsim.

(This requires ``jax_galsim.des``, added in JAX-GalSim PR #261. If either the
catalog or the PSF directory is missing, generation falls back to synthetic
ellipticities / an analytic Gaussian PSF -- the same fallback ShearNet itself
uses when the real data files aren't available.)

All randomness (catalog index, intrinsic ellipticity fallback, PSF
file/position, sub-pixel offset, noise realizations) is drawn once with
numpy and then reused for both backends, so that the two backends can be
compared pixel-by-pixel on identical inputs.
"""
import glob
import os
import time
from types import SimpleNamespace

import numpy as np

# SuperBIT focal-plane geometry used by ShearNet to place empirical PSFEx
# evaluations (shearnet/core/dataset.py: WCS_PARAMS / MARGIN).
WCS_PARAMS = {
    "image_xsize": 9600,
    "image_ysize": 6422,
    "pixel_scale": 0.1408,
    "center_ra": 13.3,
    "center_dec": 33.1,
    "theta": 0.0,
}
MARGIN = 200


def _create_wcs_from_params(mod, params):
    """Port of shearnet.utils.simutils.create_wcs_from_params.

    Parameterized by backend module, so each backend builds its own WCS out of
    its own classes. jax-galsim implements the whole TanWCS/AffineTransform/
    CelestialCoord chain, so no galsim objects leak into the jax-galsim path.
    """
    xsize = params["image_xsize"]
    ysize = params["image_ysize"]
    pixel_scale = params["pixel_scale"]
    center_ra = params["center_ra"] * mod.hours
    center_dec = params["center_dec"] * mod.degrees
    theta = params.get("theta", 0.0) * mod.degrees

    fiducial_full_image = mod.ImageF(xsize, ysize)

    dudx = np.cos(theta) * pixel_scale
    dudy = -np.sin(theta) * pixel_scale
    dvdx = np.sin(theta) * pixel_scale
    dvdy = np.cos(theta) * pixel_scale

    affine = mod.AffineTransform(
        dudx, dudy, dvdx, dvdy, origin=fiducial_full_image.true_center
    )
    sky_center = mod.CelestialCoord(ra=center_ra, dec=center_dec)
    return mod.TanWCS(affine, sky_center, units=mod.arcsec)


# DES_PSFEx instances are cached per (backend, file): reading a PSFEx model is
# host-side FITS I/O that would otherwise be repeated for every object.
_PSFEX_CACHE = {}


def get_psfex(mod, psf_file):
    """Return a backend-native DES_PSFEx for ``psf_file``, cached.

    Both galsim and jax-galsim (with the ``jax_galsim.des`` module) can read
    PSFEx files directly, so neither backend needs help from the other.
    """
    key = (mod.__name__, psf_file)
    if key not in _PSFEX_CACHE:
        if mod.__name__ == "galsim":
            import galsim.des  # noqa: F401  (populates galsim.des)
        wcs = _create_wcs_from_params(mod, WCS_PARAMS)
        _PSFEX_CACHE[key] = mod.des.DES_PSFEx(psf_file, wcs=wcs)
    return _PSFEX_CACHE[key]


def find_psf_files(psf_data_dir):
    """Locate PSFEx `.psf` files, mirroring shearnet.core.dataset.search_psf_files."""
    if not psf_data_dir:
        return []
    if os.path.isfile(psf_data_dir):
        return [psf_data_dir]
    if os.path.isdir(psf_data_dir):
        return sorted(glob.glob(os.path.join(psf_data_dir, "*.psf")))
    return []


def draw_psf_choice(rng, psf_files):
    """Pick a PSFEx model file and a focal-plane position (mirrors ShearNet's import_psf).

    Only the *choice* is made here, with numpy, so both backends evaluate the
    same model at the same position from their own PSFEx reader.
    """
    x = MARGIN + (WCS_PARAMS["image_xsize"] - 2 * MARGIN) * rng.uniform()
    y = MARGIN + (WCS_PARAMS["image_ysize"] - 2 * MARGIN) * rng.uniform()
    psf_file = psf_files[rng.randint(len(psf_files))]
    return psf_file, x, y


def load_cosmos_catalog(cat_path, seed, ellipticity_sigma, hlr, flux, n_synthetic=5000, quiet=False):
    """Load G1/G2/HLR/FLUX from a COSMOS catalog FITS file, or synthesize one.

    Mirrors the fallback in shearnet.core.dataset._load_cosmos_cat: if the
    real catalog isn't available, draw G1/G2 ~ Normal(0, ellipticity_sigma)
    and use constant HLR/FLUX, so the pipeline still runs end-to-end.
    """
    if cat_path and os.path.exists(cat_path):
        from astropy.io import fits

        with fits.open(cat_path) as hdul:
            data = hdul[1].data
        return dict(
            G1=np.asarray(data["G1"], dtype=np.float64),
            G2=np.asarray(data["G2"], dtype=np.float64),
            HLR=np.asarray(data["HLR"], dtype=np.float64),
            FLUX=np.asarray(data["FLUX"], dtype=np.float64),
        ), True

    if not quiet:
        print(
            f"WARNING: cosmos catalog not found at {cat_path!r}; using a synthetic "
            f"G1/G2 catalog (same fallback shearnet/core/dataset.py uses)."
        )
    rng = np.random.RandomState(seed)
    return dict(
        G1=rng.normal(0.0, ellipticity_sigma, n_synthetic),
        G2=rng.normal(0.0, ellipticity_sigma, n_synthetic),
        HLR=np.full(n_synthetic, hlr),
        FLUX=np.full(n_synthetic, flux),
    ), False


def pregenerate_truth(cfg, paths, quiet=False):
    """Draw all per-object randomness up front so both backends see the same inputs."""
    rng = np.random.RandomState(cfg["seed"])
    n = cfg["n_obs"]
    scale = cfg["scale"]
    npix = cfg["npix"]
    npix_psf = cfg["psf_npix"]
    noise_sd = cfg["noise_sd"]
    psf_noise = cfg["psf_noise"]

    catalog, used_real_catalog = load_cosmos_catalog(
        paths.get("cosmos_cat_fname"),
        seed=cfg["seed"],
        ellipticity_sigma=cfg["ellipticity_sigma"],
        hlr=cfg["hlr"],
        flux=cfg["flux"],
        quiet=quiet,
    )
    n_cat = len(catalog["G1"])

    psf_mode = cfg["psf_mode"]
    psf_files = find_psf_files(paths.get("psf_data_dir")) if psf_mode == "superbit" else []
    if psf_mode == "superbit" and not psf_files:
        if not quiet:
            print(
                f"WARNING: no PSFEx files found under {paths.get('psf_data_dir')!r}; "
                f"falling back to an analytic Gaussian PSF (psf.fwhm)."
            )
        psf_mode = "ideal"

    truth = []
    for _ in range(n):
        idx = rng.randint(n_cat)
        g1 = float(catalog["G1"][idx])
        g2 = float(catalog["G2"][idx])
        hlr = float(catalog["HLR"][idx]) if cfg["hlr_type"] == "catalog" else cfg["hlr"]
        flux = float(catalog["FLUX"][idx]) if cfg["flux_type"] == "catalog" else cfg["flux"]

        dx, dy = rng.uniform(low=-scale / 2, high=scale / 2, size=2)
        noise_p = rng.normal(scale=noise_sd, size=(npix, npix))
        noise_m = rng.normal(scale=noise_sd, size=(npix, npix))
        noise_psf = rng.normal(scale=psf_noise, size=(npix_psf, npix_psf))

        psf_file, psf_x, psf_y = (None, 0.0, 0.0)
        if psf_mode == "superbit":
            psf_file, psf_x, psf_y = draw_psf_choice(rng, psf_files)

        truth.append(
            dict(
                g1=g1,
                g2=g2,
                dx=dx,
                dy=dy,
                hlr=hlr,
                flux=flux,
                noise_p=noise_p,
                noise_m=noise_m,
                noise_psf=noise_psf,
                psf_file=psf_file,
                psf_x=psf_x,
                psf_y=psf_y,
            )
        )

    return truth, psf_mode, used_real_catalog


def make_one(mod, t, cfg, psf_mode, dtype=np.float64):
    """Render one galaxy (sheared +/- shear_true) and its PSF with the given backend module."""
    scale = cfg["scale"]
    npix = cfg["npix"]
    npix_psf = cfg["psf_npix"]
    shear_true = cfg["shear_true"]

    gsp = mod.GSParams(maximum_fft_size=32768)

    if psf_mode == "superbit":
        # Each backend reads the PSFEx model itself and evaluates it at the
        # chosen focal-plane position, so nothing is shared between them but
        # the choice of file and position.
        psfex = get_psfex(mod, t["psf_file"])
        psf = psfex.getPSF(mod.PositionD(t["psf_x"], t["psf_y"]))
    else:
        psf = mod.Gaussian(fwhm=cfg["psf_fwhm"])

    obj0 = mod.Exponential(half_light_radius=t["hlr"], flux=t["flux"]).shear(g1=t["g1"], g2=t["g2"])
    objp = obj0.shear(g1=shear_true, g2=0.0).shift(dx=t["dx"], dy=t["dy"])
    objm = obj0.shear(g1=-shear_true, g2=0.0).shift(dx=t["dx"], dy=t["dy"])

    conv_p = mod.Convolve(psf, objp, gsparams=gsp)
    conv_m = mod.Convolve(psf, objm, gsparams=gsp)

    # The empirical PSFEx model is fit to observed (already-pixelized) stars, so
    # the profile returned by getPSF already includes one convolution by the
    # pixel response. Drawing with the default 'auto' (=fft)
    # would convolve by the pixel *again*, over-smoothing the result. drawImage's
    # 'no_pixel' method samples the profile without adding that extra pixel; it is
    # exactly the case the docstring calls out ("a PSF that already includes a
    # convolution by the pixel response ... e.g. a PSF from an observed image of a
    # star"). The analytic Gaussian ('ideal') PSF does NOT include a pixel, so it
    # keeps 'auto' and lets GalSim add the pixel integration.
    draw_method = "no_pixel" if psf_mode == "superbit" else "auto"

    psf_im = np.array(
        psf.drawImage(nx=npix_psf, ny=npix_psf, scale=scale, dtype=dtype, method=draw_method).array
    ) + t["noise_psf"]
    im_p = np.array(
        conv_p.drawImage(nx=npix, ny=npix, scale=scale, dtype=dtype, method=draw_method).array
    ) + t["noise_p"]
    im_m = np.array(
        conv_m.drawImage(nx=npix, ny=npix, scale=scale, dtype=dtype, method=draw_method).array
    ) + t["noise_m"]

    return psf_im, im_p, im_m


def warmup_jit(mod, cfg, paths, psf_mode, n_warmup, dtype=np.float64, seed_offset=999983):
    """Render n_warmup throwaway objects to trigger JIT compilation, untimed images discarded.

    Uses a distinct seed so the warmup objects are disjoint from the ones
    used in the timed comparison, but the same psf_mode/config so it
    exercises the exact code path (shapes, PSF representation) that the
    timed run will hit -- this is what makes the *timed* run apples-to-apples
    between galsim (which needs no warmup) and jax-galsim (which does).
    Returns the wall-clock time spent warming up.
    """
    warmup_cfg = dict(cfg)
    warmup_cfg["n_obs"] = n_warmup
    warmup_cfg["seed"] = cfg["seed"] + seed_offset
    warmup_truth, _, _ = pregenerate_truth(warmup_cfg, paths, quiet=True)

    t_start = time.perf_counter()
    for t in warmup_truth:
        make_one(mod, t, warmup_cfg, psf_mode, dtype=dtype)
    return time.perf_counter() - t_start


def generate_dataset(mod, truth_list, cfg, psf_mode, dtype=np.float64):
    """Render the full dataset with one backend, timing each object individually.

    Callers that need jax-galsim's JIT warmed up first should call
    `warmup_jit` before this, so every object timed here is post-warmup.
    """
    n = len(truth_list)
    psf_ims = np.empty((n, cfg["psf_npix"], cfg["psf_npix"]))
    im_ps = np.empty((n, cfg["npix"], cfg["npix"]))
    im_ms = np.empty((n, cfg["npix"], cfg["npix"]))
    per_object_time = np.empty(n)

    t_start = time.perf_counter()
    for i, t in enumerate(truth_list):
        t0 = time.perf_counter()
        psf_im, im_p, im_m = make_one(mod, t, cfg, psf_mode, dtype=dtype)
        per_object_time[i] = time.perf_counter() - t0
        psf_ims[i] = psf_im
        im_ps[i] = im_p
        im_ms[i] = im_m
    total_time = time.perf_counter() - t_start

    return SimpleNamespace(
        psf_images=psf_ims,
        images_p=im_ps,
        images_m=im_ms,
        per_object_time=per_object_time,
        total_time=total_time,
        mean_ms=float(per_object_time.mean() * 1e3),
        std_ms=float(per_object_time.std() * 1e3),
    )


# ---------------------------------------------------------------------------
# Batched jax-galsim rendering (the GPU-saturating path)
#
# The one-at-a-time `make_one` loop above is a faithful drop-in port of the
# galsim code, but it is pathologically slow for jax-galsim on a GPU: each
# drawImage expands to ~580 primitive XLA ops, and rendering objects one at a
# time in a Python loop dispatches those ~1,500 kernels (3 draws/object)
# serially per object, gated by kernel-launch + host<->device-copy latency,
# with a single 53x53 stamp using a rounding-error fraction of the device.
#
# To actually use the GPU we follow the pattern from JAX-GalSim's "sharp bits"
# docs: pin the FFT size (minimum_fft_size == maximum_fft_size) so every
# object has identical array shapes, then `jax.vmap` the render over a batch
# and `jax.jit` it so the whole batch compiles ONCE and runs as fused kernels
# over a batch axis wide enough to fill the SMs. See ANALYSIS.md.
# ---------------------------------------------------------------------------

def _psf_stamps(mod, truth_list, cfg, fft_size, chunk=256):
    """Render each object's PSF stamp with the backend's own PSFEx reader.

    The batched renderer needs fixed-shape PSF inputs, so the empirical PSF is
    sampled onto a stamp here. For jax-galsim this is done with jit(vmap(...))
    over a stacked batch of DES_PSFEx models -- possible because the PSFEx data
    is traced (JAX-GalSim PR #261), so models from *different* files share a
    tree structure. Doing it object-by-object in eager JAX instead costs ~460
    ms/object (~75 minutes for 10,000) versus ~2 ms/object batched.
    """
    npix_psf, scale = cfg["psf_npix"], cfg["scale"]

    def eager_stamp(t):
        return np.asarray(
            get_psfex(mod, t["psf_file"])
            .getPSF(mod.PositionD(t["psf_x"], t["psf_y"]))
            .drawImage(nx=npix_psf, ny=npix_psf, scale=scale, method="no_pixel")
            .array
        )

    if mod.__name__ != "jax_galsim":
        return np.stack([eager_stamp(t) for t in truth_list])

    import jax
    import jax.numpy as jnp

    gsp = mod.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)

    def one(model, x, y):
        return model.getPSF(mod.PositionD(x, y), gsparams=gsp).drawImage(
            nx=npix_psf, ny=npix_psf, scale=scale, method="no_pixel"
        ).array

    render = jax.jit(jax.vmap(one))
    out = np.empty((len(truth_list), npix_psf, npix_psf))
    for start in range(0, len(truth_list), chunk):
        sub = truth_list[start : start + chunk]
        models = [get_psfex(mod, t["psf_file"]) for t in sub]
        batch = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *models)
        xs = jnp.array([t["psf_x"] for t in sub])
        ys = jnp.array([t["psf_y"] for t in sub])
        out[start : start + len(sub)] = np.asarray(render(batch, xs, ys))
    return out


def _stack_truth(mod, truth_list, cfg, psf_mode, fft_size):
    """Stack the per-object truth dicts into contiguous arrays for vmap."""
    n = len(truth_list)
    hlr = np.array([t["hlr"] for t in truth_list])
    flux = np.array([t["flux"] for t in truth_list])
    g1 = np.array([t["g1"] for t in truth_list])
    g2 = np.array([t["g2"] for t in truth_list])
    dx = np.array([t["dx"] for t in truth_list])
    dy = np.array([t["dy"] for t in truth_list])
    if psf_mode == "superbit":
        psf_stamps = _psf_stamps(mod, truth_list, cfg, fft_size)
    else:
        psf_stamps = np.zeros((n, cfg["psf_npix"], cfg["psf_npix"]))
    return dict(hlr=hlr, flux=flux, g1=g1, g2=g2, dx=dx, dy=dy, psf_stamps=psf_stamps)


def _make_batched_render_fn(mod, cfg, psf_mode, fft_size):
    """Build a jit(vmap(render_one)) callable with a pinned, static FFT size."""
    scale = cfg["scale"]
    npix = cfg["npix"]
    npix_psf = cfg["psf_npix"]
    shear_true = cfg["shear_true"]
    psf_fwhm = cfg["psf_fwhm"]
    # Pinning min == max makes the k-space FFT grid a compile-time constant, so
    # the batch compiles once instead of retracing on every object's size.
    gsp = mod.GSParams(minimum_fft_size=fft_size, maximum_fft_size=fft_size)
    # See make_one: the empirical PSF already includes the pixel response, so
    # draw with 'no_pixel'; the analytic Gaussian keeps the default pixel-adding
    # 'auto' method.
    draw_method = "no_pixel" if psf_mode == "superbit" else "auto"

    def render_one(psf_stamp, hlr, flux, g1, g2, dx, dy):
        if psf_mode == "superbit":
            psf = mod.InterpolatedImage(mod.Image(psf_stamp, scale=scale), gsparams=gsp)
        else:
            psf = mod.Gaussian(fwhm=psf_fwhm, gsparams=gsp)
        obj0 = mod.Exponential(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2)
        objp = obj0.shear(g1=shear_true, g2=0.0).shift(dx=dx, dy=dy)
        objm = obj0.shear(g1=-shear_true, g2=0.0).shift(dx=dx, dy=dy)
        psf_im = psf.drawImage(nx=npix_psf, ny=npix_psf, scale=scale, method=draw_method).array
        im_p = mod.Convolve(psf, objp, gsparams=gsp).drawImage(nx=npix, ny=npix, scale=scale, method=draw_method).array
        im_m = mod.Convolve(psf, objm, gsparams=gsp).drawImage(nx=npix, ny=npix, scale=scale, method=draw_method).array
        return psf_im, im_p, im_m

    import jax
    return jax.jit(jax.vmap(render_one))


def generate_dataset_batched(mod, truth_list, cfg, psf_mode, batch_size, fft_size, dtype=np.float64):
    """Render the dataset with jax-galsim using jit+vmap over batches.

    Splits the objects into chunks of `batch_size` (to bound the FFT
    intermediate memory: each chunk holds ~batch_size * fft_size^2 complex
    values on the device), rendering each chunk in one fused, pre-compiled
    kernel. Timing is reported per chunk; the first chunk's time includes the
    one-time XLA compilation and is tracked separately.
    """
    import jax.numpy as jnp

    n = len(truth_list)
    stacked = _stack_truth(mod, truth_list, cfg, psf_mode, fft_size)
    render = _make_batched_render_fn(mod, cfg, psf_mode, fft_size)

    psf_ims = np.empty((n, cfg["psf_npix"], cfg["psf_npix"]))
    im_ps = np.empty((n, cfg["npix"], cfg["npix"]))
    im_ms = np.empty((n, cfg["npix"], cfg["npix"]))
    per_chunk_time = []

    t_start = time.perf_counter()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        args = [jnp.asarray(stacked[k][start:stop]) for k in ("psf_stamps", "hlr", "flux", "g1", "g2", "dx", "dy")]
        t0 = time.perf_counter()
        psf_b, p_b, m_b = render(*args)
        # block so the timer captures actual device execution, not just dispatch
        p_b.block_until_ready()
        per_chunk_time.append(time.perf_counter() - t0)
        psf_ims[start:stop] = np.asarray(psf_b)
        im_ps[start:stop] = np.asarray(p_b)
        im_ms[start:stop] = np.asarray(m_b)
    total_time = time.perf_counter() - t_start

    # add the same precomputed noise the eager path uses
    psf_ims += np.stack([t["noise_psf"] for t in truth_list])
    im_ps += np.stack([t["noise_p"] for t in truth_list])
    im_ms += np.stack([t["noise_m"] for t in truth_list])

    per_chunk_time = np.array(per_chunk_time)
    compile_time = float(per_chunk_time[0]) if len(per_chunk_time) else 0.0
    steady = per_chunk_time[1:] if len(per_chunk_time) > 1 else per_chunk_time
    steady_objs = max(n - batch_size, batch_size)

    return SimpleNamespace(
        psf_images=psf_ims,
        images_p=im_ps,
        images_m=im_ms,
        per_object_time=np.repeat(per_chunk_time / batch_size, 1),  # per-chunk, coarse
        total_time=total_time,
        compile_time=compile_time,
        mean_ms=float(steady.sum() / steady_objs * 1e3),
        std_ms=float((steady / batch_size).std() * 1e3) if len(steady) else 0.0,
    )
