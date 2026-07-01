"""Pixel-level and performance comparison between two rendered datasets."""
import numpy as np


def _diff_stats(a, b):
    diff = a - b
    return dict(
        max_abs=float(np.max(np.abs(diff))),
        mean_abs=float(np.mean(np.abs(diff))),
        rms=float(np.sqrt(np.mean(diff**2))),
    )


def _timing_stats(ds):
    return dict(
        total_s=ds.total_time,
        warmup_s=ds.warmup_time,
        steady_state_mean_ms=ds.steady_state_mean * 1e3,
        steady_state_std_ms=ds.steady_state_std * 1e3,
    )


def compare_datasets(ds_galsim, ds_jax, label_a="galsim", label_b="jax_galsim"):
    flux_p_a = ds_galsim.images_p.sum(axis=(1, 2))
    flux_p_b = ds_jax.images_p.sum(axis=(1, 2))
    flux_rel_diff = np.abs(flux_p_a - flux_p_b) / np.abs(flux_p_a)

    return dict(
        n_obs=len(ds_galsim.images_p),
        image_p_diff=_diff_stats(ds_galsim.images_p, ds_jax.images_p),
        image_m_diff=_diff_stats(ds_galsim.images_m, ds_jax.images_m),
        psf_diff=_diff_stats(ds_galsim.psf_images, ds_jax.psf_images),
        flux_relative_diff_mean=float(flux_rel_diff.mean()),
        flux_relative_diff_max=float(flux_rel_diff.max()),
        timing={label_a: _timing_stats(ds_galsim), label_b: _timing_stats(ds_jax)},
        speedup_galsim_over_jax=ds_jax.steady_state_mean / ds_galsim.steady_state_mean,
    )


def print_report(report, label_a="galsim", label_b="jax_galsim"):
    n = report["n_obs"]
    print(f"\n===== Comparison over {n} objects =====")

    print("\nPixel-level agreement (sheared image, backend A - backend B):")
    for name, key in (("galaxy (+shear)", "image_p_diff"), ("galaxy (-shear)", "image_m_diff"), ("PSF", "psf_diff")):
        s = report[key]
        print(f"  {name:<16s} max|diff|={s['max_abs']:.3e}  mean|diff|={s['mean_abs']:.3e}  rms={s['rms']:.3e}")

    print("\nFlux agreement (relative difference, +shear image):")
    print(f"  mean={report['flux_relative_diff_mean']:.3e}  max={report['flux_relative_diff_max']:.3e}")

    print("\nPerformance:")
    for label in (label_a, label_b):
        t = report["timing"][label]
        print(
            f"  {label:<12s} total={t['total_s']:.3f}s  "
            f"steady-state={t['steady_state_mean_ms']:.3f}"
            f"+/-{t['steady_state_std_ms']:.3f} ms/obj  "
            f"warmup={t['warmup_s']:.3f}s"
        )
    print(f"\n  {label_b} is {report['speedup_galsim_over_jax']:.1f}x slower than {label_a} per object (steady-state)")
