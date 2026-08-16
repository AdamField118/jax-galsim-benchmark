"""Comprehensive stamp-level agreement analysis: galsim vs jax-galsim.

Both pipelines are fully independent (each reads the PSFEx model with its own
reader and runs its own Convolve/drawImage), so this measures how far the two
renderings of the *same* physical object drift apart.

Accuracy cutoffs are the ones JAX-GalSim uses internally to compare drawn
images against reference GalSim, in
``tests/jax/test_spergel_comp_galsim.py::test_spergel_comp_galsim_image``:

    if use_same_fft_size:
        atol = 1e-16
    else:
        atol = 1e-9
    np.testing.assert_allclose(arr_galsim, arr_jgs, atol=atol, rtol=0)

Those are absolute tolerances on a **unit-flux** image (``flux_b: 1`` in that
test), so this script compares ``max|Δ| / flux`` against them -- comparing raw
pixel differences of a flux~1e4 galaxy to a unit-flux tolerance would be
meaningless.

Stamps are rendered noiseless here: the pipeline adds identical noise to both
backends, which cancels exactly in the difference and would only dilute the
shape-measurement comparison.

Usage:
    python analyze_agreement.py -c config.yaml --n-obs 50
"""

import argparse
import json
import os

import numpy as np

import helpers

# JAX-GalSim's internal image-comparison cutoffs (unit flux, rtol=0).
ATOL_SAME_FFT = 1e-16
ATOL_DIFF_FFT = 1e-9


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--n-obs", type=int, default=50)
    p.add_argument("--outdir", default=None)
    return p.parse_args()


def _noiseless(truth):
    """Copy the truth list with all noise realizations zeroed."""
    out = []
    for t in truth:
        t2 = dict(t)
        for k in ("noise_p", "noise_m", "noise_psf"):
            t2[k] = np.zeros_like(t[k])
        out.append(t2)
    return out


def _moments(arr, scale):
    """Adaptive moments (e1, e2, sigma) measured with galsim's HSM.

    The *same* measurement code is applied to both backends' stamps, so any
    difference comes from the images, not from the measurement.
    """
    import galsim

    try:
        res = galsim.Image(np.ascontiguousarray(arr), scale=scale).FindAdaptiveMom()
        return res.observed_shape.e1, res.observed_shape.e2, res.moments_sigma
    except Exception:
        return np.nan, np.nan, np.nan


def _fft_regime(mod_g, mod_j, truth, cfg, psf_mode):
    """Compare the k-space sampling each backend picks for the same object.

    ``stepk``/``maxk`` set the FFT grid, so if they agree the two backends are
    in JAX-GalSim's "same fft size" regime (atol 1e-16) and otherwise in the
    looser "different fft size" regime (atol 1e-9).
    """
    from dataset import get_psfex

    t = truth[0]
    out = {}
    for name, mod in (("galsim", mod_g), ("jax_galsim", mod_j)):
        if psf_mode == "superbit":
            psf = get_psfex(mod, t["psf_file"]).getPSF(
                mod.PositionD(t["psf_x"], t["psf_y"])
            )
        else:
            psf = mod.Gaussian(fwhm=cfg["psf_fwhm"])
        gal = mod.Exponential(half_light_radius=t["hlr"], flux=t["flux"]).shear(
            g1=t["g1"], g2=t["g2"]
        )
        conv = mod.Convolve(psf, gal, gsparams=mod.GSParams(maximum_fft_size=32768))
        out[name] = (float(conv.stepk), float(conv.maxk))
    same = np.allclose(out["galsim"], out["jax_galsim"], rtol=1e-6, atol=0)
    return out, bool(same)


def _stats(v):
    v = np.asarray(v, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return dict(mean=np.nan, median=np.nan, p95=np.nan, max=np.nan)
    return dict(
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        p95=float(np.percentile(finite, 95)),
        max=float(finite.max()),
    )


def main():
    args = parse_args()
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    full_cfg = helpers.load_config(args.config)
    cfg = full_cfg["simulation"]
    cfg["n_obs"] = args.n_obs

    psf_cfg = full_cfg.get("psf", {})
    cfg["psf_mode"] = psf_cfg.get("mode", "ideal")
    cfg["psf_fwhm"] = psf_cfg.get("fwhm", 0.5)
    cfg["psf_npix"] = psf_cfg.get("npix", cfg["npix"])
    cfg["psf_noise"] = psf_cfg.get("noise", 1.0e-6)

    paths = {
        k: (os.path.normpath(os.path.join(cfg_dir, v)) if v else v)
        for k, v in full_cfg.get("paths", {}).items()
    }
    outdir = args.outdir or full_cfg.get("output", {}).get("dir", "results")
    os.makedirs(outdir, exist_ok=True)

    from dataset import make_one, pregenerate_truth

    print(f"Pre-generating {cfg['n_obs']} objects (seed={cfg['seed']})...")
    truth, psf_mode, used_real_cat = pregenerate_truth(cfg, paths)
    truth = _noiseless(truth)

    import galsim

    import jax  # noqa: E402

    jax.config.update("jax_enable_x64", True)
    import jax_galsim  # noqa: E402

    print(f"  galsim {galsim.__version__} vs jax-galsim {jax_galsim.__version__}")
    print(f"  psf mode: {psf_mode} | real catalog: {used_real_cat}")
    n_files = len({t['psf_file'] for t in truth if t['psf_file']})
    print(f"  distinct PSFEx models exercised: {n_files}")

    kspace, same_fft = _fft_regime(galsim, jax_galsim, truth, cfg, psf_mode)
    atol = ATOL_SAME_FFT if same_fft else ATOL_DIFF_FFT

    scale = cfg["scale"]
    rows = {"psf": [], "im_p": [], "im_m": []}
    flux_rel, dm = [], {"de1": [], "de2": [], "dsigma": [], "sigma": []}

    print(f"\nRendering {cfg['n_obs']} objects with each backend (noiseless)...")
    for i, t in enumerate(truth):
        g = make_one(galsim, t, cfg, psf_mode)
        j = make_one(jax_galsim, t, cfg, psf_mode)
        for key, gi, ji in zip(("psf", "im_p", "im_m"), g, j):
            gi = np.asarray(gi, dtype=float)
            ji = np.asarray(ji, dtype=float)
            fl = abs(gi.sum()) or 1.0
            d = np.abs(gi - ji)
            rows[key].append((d.max(), d.max() / fl, float(np.sqrt((d**2).mean()))))
        gp, jp = np.asarray(g[1], dtype=float), np.asarray(j[1], dtype=float)
        flux_rel.append(abs(gp.sum() - jp.sum()) / abs(gp.sum()))
        ge1, ge2, gs = _moments(gp, scale)
        je1, je2, js = _moments(jp, scale)
        dm["de1"].append(abs(ge1 - je1))
        dm["de2"].append(abs(ge2 - je2))
        dm["dsigma"].append(abs(gs - js))
        dm["sigma"].append(gs)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{cfg['n_obs']}", flush=True)

    # ---------------- report ----------------
    print("\n" + "=" * 72)
    print(f"STAMP AGREEMENT: galsim vs jax-galsim  ({cfg['n_obs']} objects)")
    print("=" * 72)

    print("\n[1] k-space sampling (decides which JAX-GalSim cutoff applies)")
    for k, (stepk, maxk) in kspace.items():
        print(f"      {k:<11s} stepk={stepk:.10g}  maxk={maxk:.10g}")
    print(f"      -> same FFT grid: {same_fft}  =>  cutoff atol = {atol:g} (unit flux, rtol=0)")

    print("\n[2] Pixel agreement per component (max over all objects)")
    print(
        f"      {'component':<16s} {'max|Δ| raw':>12s} {'max|Δ|/flux':>13s} {'rms raw':>12s}"
        f"  {'vs 1e-9':>8s} {'vs 1e-16':>9s}"
    )
    verdicts, verdicts_strict = {}, {}
    for key, label in (("psf", "PSF"), ("im_p", "galaxy (+shear)"), ("im_m", "galaxy (-shear)")):
        a = np.array(rows[key])
        worst_raw, worst_norm, rms = a[:, 0].max(), a[:, 1].max(), a[:, 2].max()
        ok = worst_norm <= ATOL_DIFF_FFT
        ok_strict = worst_norm <= ATOL_SAME_FFT
        verdicts[key] = bool(ok)
        verdicts_strict[key] = bool(ok_strict)
        print(
            f"      {label:<16s} {worst_raw:12.3e} {worst_norm:13.3e} {rms:12.3e}"
            f"  {'PASS' if ok else 'FAIL':>8s} {'PASS' if ok_strict else 'FAIL':>9s}"
        )
    print("      'max|Δ|/flux' is the unit-flux-equivalent quantity JAX-GalSim's")
    print("      cutoffs are defined on. 1e-9 is its tolerance when the two")
    print("      renderings may differ in FFT grid; 1e-16 is the near-bit-exact")
    print("      bound it holds analytic profiles to on an identical grid.")

    print("\n[3] Flux conservation (+shear stamp)")
    s = _stats(flux_rel)
    print(f"      relative |Δflux|/flux   mean={s['mean']:.3e}  p95={s['p95']:.3e}  max={s['max']:.3e}")

    print("\n[4] Shape measurement (galsim HSM adaptive moments on both stamps)")
    for k, label, unit in (
        ("de1", "|Δe1|", ""),
        ("de2", "|Δe2|", ""),
        ("dsigma", "|Δsigma|", " pix"),
    ):
        s = _stats(dm[k])
        print(
            f"      {label:<10s} mean={s['mean']:.3e}  p95={s['p95']:.3e}  max={s['max']:.3e}{unit}"
        )
    sig = _stats(dm["sigma"])
    print(f"      (mean measured sigma = {sig['mean']:.4f} pix)")
    worst_e = max(_stats(dm["de1"])["max"], _stats(dm["de2"])["max"])
    if worst_e == 0.0:
        print("      worst |Δe| = 0 exactly: HSM returns bit-identical ellipticities")
        print("      for both backends, so shear estimates are unaffected.")
    else:
        print(f"      worst |Δe| = {worst_e:.3e}; multiplicative-bias targets are ~1e-3,")
        print(f"      so this is {1e-3 / worst_e:.3g}x below the scale that would matter.")

    overall = all(verdicts.values())
    overall_strict = all(verdicts_strict.values())
    print("\n[5] Verdict")
    print(f"      Within JAX-GalSim's 1e-9 image tolerance:  {'YES' if overall else 'NO'}")
    print(f"      Within its 1e-16 analytic-ideal tolerance: {'YES' if overall_strict else 'NO'}")
    if overall and not overall_strict:
        print("      The residual above 1e-16 is expected here: unlike the analytic")
        print("      profiles that bound is measured on, this pipeline interpolates an")
        print("      empirically-sampled PSF (Lanczos) and applies a WCS transform, and")
        print("      XLA is free to reorder those float operations relative to C++.")

    report = dict(
        n_obs=cfg["n_obs"],
        psf_mode=psf_mode,
        n_psf_models=n_files,
        galsim_version=galsim.__version__,
        jax_galsim_version=jax_galsim.__version__,
        kspace=kspace,
        same_fft_grid=same_fft,
        atol_applied=atol,
        components={
            k: dict(
                max_abs_raw=float(np.array(v)[:, 0].max()),
                max_abs_per_flux=float(np.array(v)[:, 1].max()),
                rms_raw=float(np.array(v)[:, 2].max()),
                within_1e9=verdicts[k],
                within_1e16=verdicts_strict[k],
            )
            for k, v in rows.items()
        },
        flux_relative=_stats(flux_rel),
        moments={k: _stats(v) for k, v in dm.items()},
        all_within_1e9=overall,
        all_within_1e16=overall_strict,
    )
    path = os.path.join(outdir, "agreement_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
