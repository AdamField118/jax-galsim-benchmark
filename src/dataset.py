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

jax-galsim has no PSFEx/WCS support, so the empirical PSF is always
evaluated once with galsim and rendered to a pixel stamp; both backends
then represent that stamp as an InterpolatedImage, which keeps the *same*
physical PSF input to both while still exercising each backend's own
Convolve/shear/drawImage code path. If either the catalog or the PSF
directory is missing, this falls back to synthetic ellipticities /
an analytic Gaussian PSF -- the same fallback ShearNet itself uses when
the real data files aren't available (e.g. in CI).

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


def _create_wcs_from_params(params):
    """Port of shearnet.utils.simutils.create_wcs_from_params (galsim-only)."""
    import galsim

    xsize = params["image_xsize"]
    ysize = params["image_ysize"]
    pixel_scale = params["pixel_scale"]
    center_ra = params["center_ra"] * galsim.hours
    center_dec = params["center_dec"] * galsim.degrees
    theta = params.get("theta", 0.0) * galsim.degrees

    fiducial_full_image = galsim.ImageF(xsize, ysize)

    dudx = np.cos(theta) * pixel_scale
    dudy = -np.sin(theta) * pixel_scale
    dvdx = np.sin(theta) * pixel_scale
    dvdy = np.cos(theta) * pixel_scale

    affine = galsim.AffineTransform(
        dudx, dudy, dvdx, dvdy, origin=fiducial_full_image.true_center
    )
    sky_center = galsim.CelestialCoord(ra=center_ra, dec=center_dec)
    return galsim.TanWCS(affine, sky_center, units=galsim.arcsec)


def find_psf_files(psf_data_dir):
    """Locate PSFEx `.psf` files, mirroring shearnet.core.dataset.search_psf_files."""
    if not psf_data_dir:
        return []
    if os.path.isfile(psf_data_dir):
        return [psf_data_dir]
    if os.path.isdir(psf_data_dir):
        return sorted(glob.glob(os.path.join(psf_data_dir, "*.psf")))
    return []


def render_empirical_psf_stamp(rng, psf_files, wcs, npix_psf, scale):
    """Evaluate a random empirical PSFEx model at a random focal-plane position.

    Only galsim can read PSFEx files (jax-galsim has no `des` module), so
    this always uses galsim and hands back a plain pixel array that either
    backend can wrap in an InterpolatedImage.
    """
    import galsim
    import galsim.des

    x = MARGIN + (WCS_PARAMS["image_xsize"] - 2 * MARGIN) * rng.uniform()
    y = MARGIN + (WCS_PARAMS["image_ysize"] - 2 * MARGIN) * rng.uniform()
    image_pos = galsim.PositionD(x=x, y=y)

    psf_file = psf_files[rng.randint(len(psf_files))]
    psfex = galsim.des.DES_PSFEx(psf_file, wcs=wcs)
    psf_obj = psfex.getPSF(image_pos)

    stamp = psf_obj.drawImage(nx=npix_psf, ny=npix_psf, scale=scale, dtype=np.float64).array
    return np.ascontiguousarray(stamp)


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
    wcs = _create_wcs_from_params(WCS_PARAMS) if psf_mode == "superbit" else None

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

        psf_stamp = None
        if psf_mode == "superbit":
            psf_stamp = render_empirical_psf_stamp(rng, psf_files, wcs, npix_psf, scale)

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
                psf_stamp=psf_stamp,
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
        psf = mod.InterpolatedImage(mod.Image(t["psf_stamp"], scale=scale))
    else:
        psf = mod.Gaussian(fwhm=cfg["psf_fwhm"])

    obj0 = mod.Exponential(half_light_radius=t["hlr"], flux=t["flux"]).shear(g1=t["g1"], g2=t["g2"])
    objp = obj0.shear(g1=shear_true, g2=0.0).shift(dx=t["dx"], dy=t["dy"])
    objm = obj0.shear(g1=-shear_true, g2=0.0).shift(dx=t["dx"], dy=t["dy"])

    conv_p = mod.Convolve(psf, objp, gsparams=gsp)
    conv_m = mod.Convolve(psf, objm, gsparams=gsp)

    psf_im = np.array(
        psf.drawImage(nx=npix_psf, ny=npix_psf, scale=scale, dtype=dtype).array
    ) + t["noise_psf"]
    im_p = np.array(
        conv_p.drawImage(nx=npix, ny=npix, scale=scale, dtype=dtype).array
    ) + t["noise_p"]
    im_m = np.array(
        conv_m.drawImage(nx=npix, ny=npix, scale=scale, dtype=dtype).array
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
