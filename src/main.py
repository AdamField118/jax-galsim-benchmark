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
        "--jax-warmup", type=int, default=None,
        help="Override comparison.jax_warmup_galaxies from the config "
        "(number of untimed galaxies rendered with jax-galsim to trigger JIT compilation "
        "before the timed comparison)",
    )
    parser.add_argument(
        "--jax-mode", choices=["eager", "batched", "both"], default=None,
        help="Override comparison.jax_mode: 'eager' (one object at a time, no GPU "
        "saturation), 'batched' (jit+vmap over batches, saturates a GPU), or 'both'",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override comparison.jax_batch_size"
    )
    parser.add_argument(
        "--fft-size", type=int, default=None, help="Override comparison.jax_fft_size"
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

    comp_cfg = full_cfg.get("comparison", {})
    n_warmup = args.jax_warmup if args.jax_warmup is not None else comp_cfg.get("jax_warmup_galaxies", 50)
    jax_mode = args.jax_mode or comp_cfg.get("jax_mode", "batched")
    batch_size = args.batch_size or comp_cfg.get("jax_batch_size", 512)
    fft_size = args.fft_size or comp_cfg.get("jax_fft_size", 256)

    from dataset import (
        pregenerate_truth, generate_dataset, generate_dataset_batched, warmup_jit,
    )

    print(f"Pre-generating truth values for {cfg['n_obs']} objects (seed={cfg['seed']})...")
    truth, psf_mode, used_real_catalog = pregenerate_truth(cfg, paths)
    print(f"  cosmos catalog: {'real (' + paths.get('cosmos_cat_fname', '') + ')' if used_real_catalog else 'synthetic fallback'}")
    print(f"  psf: {psf_mode}" + ("" if psf_mode == cfg["psf_mode"] else f" (requested {cfg['psf_mode']})"))

    import galsim

    print(f"\nRendering {cfg['n_obs']} objects with galsim {galsim.__version__}...")
    ds_galsim = generate_dataset(galsim, truth, cfg, psf_mode)
    print(f"  total={ds_galsim.total_time:.3f}s  {ds_galsim.mean_ms:.3f}+/-{ds_galsim.std_ms:.3f} ms/obj")

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax_galsim

    print(f"\njax-galsim {jax_galsim.__version__} on devices: {jax.devices()}")

    jax_jit_warmup_s = None
    ds_jax = None

    if jax_mode in ("batched", "both"):
        print(
            f"\n[batched] Rendering {cfg['n_obs']} objects with jit(vmap(...)), "
            f"batch_size={batch_size}, fft_size={fft_size} (GPU-saturating path)..."
        )
        ds_jax_batched = generate_dataset_batched(
            jax_galsim, truth, cfg, psf_mode, batch_size=batch_size, fft_size=fft_size
        )
        print(
            f"  total={ds_jax_batched.total_time:.3f}s  "
            f"{ds_jax_batched.mean_ms:.4f} ms/obj (steady-state)  "
            f"first-chunk compile+run={ds_jax_batched.compile_time:.3f}s"
        )
        ds_jax = ds_jax_batched

    if jax_mode in ("eager", "both"):
        if n_warmup > 0:
            print(
                f"\n[eager] Warming up jax-galsim's JIT with {n_warmup} throwaway "
                f"galaxies (untimed)..."
            )
            jax_jit_warmup_s = warmup_jit(jax_galsim, cfg, paths, psf_mode, n_warmup)
            print(f"  warmup wall time: {jax_jit_warmup_s:.3f}s")

        print(f"\n[eager] Rendering {cfg['n_obs']} objects one-at-a-time (post-warmup)...")
        ds_jax_eager = generate_dataset(jax_galsim, truth, cfg, psf_mode)
        print(f"  total={ds_jax_eager.total_time:.3f}s  {ds_jax_eager.mean_ms:.3f}+/-{ds_jax_eager.std_ms:.3f} ms/obj")
        if jax_mode == "both":
            print(
                f"\n  batched is {ds_jax_eager.mean_ms / ds_jax_batched.mean_ms:.1f}x "
                f"faster per object than eager"
            )
        else:
            ds_jax = ds_jax_eager

    if args.save_datasets:
        helpers.save_dataset_npz(os.path.join(outdir, "dataset_galsim.npz"), ds_galsim)
        helpers.save_dataset_npz(os.path.join(outdir, "dataset_jax_galsim.npz"), ds_jax)
        print(f"\nSaved rendered datasets to {outdir}/dataset_*.npz")

    from compare import compare_datasets, print_report

    report = compare_datasets(ds_galsim, ds_jax, jax_jit_warmup_s=jax_jit_warmup_s)
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
