"""Generate a weak-lensing-style galaxy dataset with both galsim and
jax-galsim, using identical random draws for each object, then compare the
rendered images and the per-object render time.

This is a stripped-down version of s-Sayan/ShearNet's dataset generation
(shearnet/core/dataset.py, as used by shear_bias/m/main.py): the ngmix
metacalibration shape measurement and the ShearNet model evaluation have
been removed, but the real COSMOS catalog and SuperBIT PSFEx model are
still used when available (see dataset.py for the fallback behavior when
they aren't). Only image simulation remains, since that's what's being
benchmarked here.
"""
import argparse
import json
import os

import helpers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare galsim vs jax-galsim dataset generation"
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--n-obs", type=int, default=None, help="Override simulation.n_obs from the config"
    )
    parser.add_argument(
        "--outdir", default=None, help="Override output.dir from the config"
    )
    parser.add_argument(
        "--save-datasets", action="store_true",
        help="Also save the rendered images to <outdir>/dataset_<backend>.npz",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip generating comparison plots"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = os.path.dirname(os.path.abspath(args.config))
    full_cfg = helpers.load_config(args.config)
    cfg = full_cfg["simulation"]
    if args.n_obs is not None:
        cfg["n_obs"] = args.n_obs

    psf_cfg = full_cfg.get("psf", {})
    cfg["psf_mode"] = psf_cfg.get("mode", "ideal")
    cfg["psf_fwhm"] = psf_cfg.get("fwhm", 0.5)
    cfg["psf_npix"] = psf_cfg.get("npix", cfg["npix"])
    cfg["psf_noise"] = psf_cfg.get("noise", 1.0e-6)

    # Resolve paths (cosmos catalog, PSF data dir) relative to the config file's
    # directory, so the defaults in config.yaml correctly point at the repo root.
    paths = {}
    for key, value in full_cfg.get("paths", {}).items():
        paths[key] = os.path.normpath(os.path.join(config_dir, value)) if value else value

    outdir = args.outdir or full_cfg.get("output", {}).get("dir", "results")
    os.makedirs(outdir, exist_ok=True)

    warmup = full_cfg.get("comparison", {}).get("jax_warmup", 3)

    from dataset import pregenerate_truth, generate_dataset

    print(f"Pre-generating truth values for {cfg['n_obs']} objects (seed={cfg['seed']})...")
    truth, psf_mode, used_real_catalog = pregenerate_truth(cfg, paths)
    print(f"  cosmos catalog: {'real (' + paths.get('cosmos_cat_fname', '') + ')' if used_real_catalog else 'synthetic fallback'}")
    print(f"  psf: {psf_mode}" + ("" if psf_mode == cfg["psf_mode"] else f" (requested {cfg['psf_mode']})"))

    import galsim

    print(f"\nRendering {cfg['n_obs']} objects with galsim {galsim.__version__}...")
    ds_galsim = generate_dataset(galsim, truth, cfg, psf_mode, warmup=0)
    print(
        f"  total={ds_galsim.total_time:.3f}s  "
        f"steady-state={ds_galsim.steady_state_mean * 1e3:.3f}"
        f"+/-{ds_galsim.steady_state_std * 1e3:.3f} ms/obj"
    )

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax_galsim

    print(f"\nRendering {cfg['n_obs']} objects with jax-galsim {jax_galsim.__version__}...")
    ds_jax = generate_dataset(jax_galsim, truth, cfg, psf_mode, warmup=warmup)
    print(
        f"  total={ds_jax.total_time:.3f}s  "
        f"steady-state={ds_jax.steady_state_mean * 1e3:.3f}"
        f"+/-{ds_jax.steady_state_std * 1e3:.3f} ms/obj "
        f"(excludes first {min(warmup, cfg['n_obs'])} call(s), JIT compilation)"
    )

    if args.save_datasets:
        helpers.save_dataset_npz(os.path.join(outdir, "dataset_galsim.npz"), ds_galsim)
        helpers.save_dataset_npz(os.path.join(outdir, "dataset_jax_galsim.npz"), ds_jax)
        print(f"\nSaved rendered datasets to {outdir}/dataset_*.npz")

    from compare import compare_datasets, print_report

    report = compare_datasets(ds_galsim, ds_jax)
    report_path = os.path.join(outdir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"\nSaved comparison report to {report_path}")

    if not args.no_plots:
        import plotting

        plotting.make_plots(ds_galsim, ds_jax, report, outdir)
        print(f"Saved comparison plots to {outdir}/")


if __name__ == "__main__":
    main()
