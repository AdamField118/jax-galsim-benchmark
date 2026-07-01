"""Diagnostic plots for the galsim vs jax-galsim dataset comparison."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_plots(ds_galsim, ds_jax, report, outdir, label_a="galsim", label_b="jax_galsim"):
    os.makedirs(outdir, exist_ok=True)

    _plot_example_triplet(ds_galsim, ds_jax, outdir, label_a, label_b)
    _plot_diff_histogram(ds_galsim, ds_jax, outdir, label_a, label_b)
    _plot_timing(ds_galsim, ds_jax, outdir, label_a, label_b)


def _plot_example_triplet(ds_galsim, ds_jax, outdir, label_a, label_b, index=0):
    a = ds_galsim.images_p[index]
    b = ds_jax.images_p[index]
    diff = a - b

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(a, origin="lower")
    axes[0].set_title(label_a)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(b, origin="lower")
    axes[1].set_title(label_b)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, origin="lower", cmap="RdBu_r")
    axes[2].set_title(f"{label_a} - {label_b}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Example object #{index} (+shear image)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "example_triplet.png"), dpi=150)
    plt.close(fig)


def _plot_diff_histogram(ds_galsim, ds_jax, outdir, label_a, label_b):
    diff = (ds_galsim.images_p - ds_jax.images_p).ravel()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(diff, bins=100)
    ax.set_xlabel(f"pixel value: {label_a} - {label_b}")
    ax.set_ylabel("count")
    ax.set_title("Per-pixel difference across all objects (+shear image)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pixel_diff_histogram.png"), dpi=150)
    plt.close(fig)


def _plot_timing(ds_galsim, ds_jax, outdir, label_a, label_b):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(
        [label_a, label_b],
        [ds_galsim.steady_state_mean * 1e3, ds_jax.steady_state_mean * 1e3],
        yerr=[ds_galsim.steady_state_std * 1e3, ds_jax.steady_state_std * 1e3],
    )
    axes[0].set_ylabel("ms / object (steady-state)")
    axes[0].set_title("Per-object render time")

    n = len(ds_galsim.per_object_time)
    axes[1].plot(np.arange(n), ds_galsim.per_object_time * 1e3, label=label_a)
    axes[1].plot(np.arange(n), ds_jax.per_object_time * 1e3, label=label_b)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("object index")
    axes[1].set_ylabel("ms")
    axes[1].set_title("Per-object time (log scale, shows JIT warmup)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "timing.png"), dpi=150)
    plt.close(fig)
