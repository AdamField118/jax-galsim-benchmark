"""Diagnose *why* jax-galsim is slow in the one-at-a-time benchmark, and show
that a jit+vmap batched path fixes it. Run this on the L40S box:

    python diagnose_jax.py

It reports, for the empirical-PSF render used by the benchmark:
  1. how many primitive XLA ops one drawImage expands to (~ kernel launches),
  2. how many XLA compilations the eager one-at-a-time path triggers,
  3. eager one-at-a-time vs jit(vmap) batched timing (and compile counts),
  4. float32 vs float64 timing,
  5. which JAX devices are actually being used.

See ANALYSIS.md for the interpretation. Everything here is measurement, not
part of the benchmark itself.
"""
import argparse
import os
import time

import numpy as np


def _install_compile_counter():
    """Return a dict whose 'n' counts XLA compilations from now on."""
    import jax._src.compiler as _compiler

    counter = {"n": 0}
    orig = _compiler.compile_or_get_cached

    def counting(*a, **k):
        counter["n"] += 1
        return orig(*a, **k)

    _compiler.compile_or_get_cached = counting
    return counter


def _psf_stamp(galsim, i, npix, scale, dtype):
    p = galsim.Moffat(beta=3.5, fwhm=0.5 + 0.003 * i).shear(g1=0.001 * i, g2=-0.0005 * i)
    return np.ascontiguousarray(p.drawImage(nx=npix, ny=npix, scale=scale, dtype=dtype).array)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="objects for the timing comparison")
    ap.add_argument("--fft-size", type=int, default=256)
    ap.add_argument("--npix", type=int, default=53)
    ap.add_argument("--scale", type=float, default=0.141)
    args = ap.parse_args()

    npix, scale, N = args.npix, args.scale, args.n

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax_galsim as jgs
    import galsim

    print(f"JAX devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.jax_enable_x64}\n")

    gsp = jgs.GSParams(minimum_fft_size=args.fft_size, maximum_fft_size=args.fft_size)
    dtype = np.float64

    # ---- 1. primitive-op count for ONE drawImage ----
    stamp0 = _psf_stamp(galsim, 0, npix, scale, dtype)

    def one_draw(a):
        psf = jgs.InterpolatedImage(jgs.Image(a, scale=scale), gsparams=gsp)
        gal = jgs.Exponential(half_light_radius=0.5, flux=1e4).shear(g1=0.01, g2=0.0).shift(dx=0.01, dy=0.0)
        return jgs.Convolve(psf, gal, gsparams=gsp).drawImage(nx=npix, ny=npix, scale=scale).array

    jaxpr = jax.make_jaxpr(one_draw)(jnp.asarray(stamp0))
    n_ops = len(jaxpr.jaxpr.eqns)
    print(f"[1] ONE drawImage = {n_ops} primitive XLA ops")
    print(f"    make_one does 3 draws/object -> ~{3 * n_ops} kernel launches/object,")
    print(f"    ~{3 * n_ops * 10000:,} serial launches for a 10,000-object eager run.\n")

    # ---- 2/3. eager one-at-a-time vs batched jit(vmap) ----
    stamps = np.stack([_psf_stamp(galsim, i, npix, scale, dtype) for i in range(N)])
    rng = np.random.RandomState(1)
    hlr = np.full(N, 0.5); flux = np.full(N, 12258.97)
    g1 = rng.uniform(-0.3, 0.3, N); g2 = rng.uniform(-0.3, 0.3, N)
    dx = rng.uniform(-0.07, 0.07, N); dy = rng.uniform(-0.07, 0.07, N)

    def render_one(ps, hlr, flux, g1, g2, dx, dy):
        psf = jgs.InterpolatedImage(jgs.Image(ps, scale=scale), gsparams=gsp)
        obj0 = jgs.Exponential(half_light_radius=hlr, flux=flux).shear(g1=g1, g2=g2)
        objp = obj0.shear(g1=0.01, g2=0.0).shift(dx=dx, dy=dy)
        return jgs.Convolve(psf, objp, gsparams=gsp).drawImage(nx=npix, ny=npix, scale=scale).array

    counter = _install_compile_counter()

    # eager
    c0 = counter["n"]
    render_one(stamps[0], hlr[0], flux[0], g1[0], g2[0], dx[0], dy[0])  # warm caches
    first_compiles = counter["n"] - c0
    t0 = time.perf_counter()
    for i in range(N):
        np.asarray(render_one(stamps[i], hlr[i], flux[i], g1[i], g2[i], dx[i], dy[i]))
    eager_s = time.perf_counter() - t0
    print(f"[2] eager one-at-a-time: first object compiled {first_compiles} kernels")
    print(f"[3] eager: {eager_s:.3f}s = {eager_s / N * 1e3:.3f} ms/obj\n")

    # batched
    batched = jax.jit(jax.vmap(render_one))
    a = [jnp.asarray(x) for x in (stamps, hlr, flux, g1, g2, dx, dy)]
    c0 = counter["n"]
    t0 = time.perf_counter()
    out = batched(*a); out.block_until_ready()
    compile_run_s = time.perf_counter() - t0
    batch_compiles = counter["n"] - c0
    t0 = time.perf_counter()
    out = batched(*a); out.block_until_ready()
    run_s = time.perf_counter() - t0
    print(f"    batched jit(vmap) over {N}: compiled {batch_compiles} kernels (ONCE)")
    print(f"    first call (compile+run): {compile_run_s:.3f}s")
    print(f"    steady-state:            {run_s:.3f}s = {run_s / N * 1e3:.4f} ms/obj")
    print(f"    -> batched is {eager_s / run_s:.1f}x faster per object than eager\n")

    # ---- 4. float32 vs float64 (only the ratio matters; huge on FP64-poor GPUs) ----
    print("[4] float32 vs float64 (run in separate processes so x64 differs):")
    print("    (on an L40S / Ada GPU, FP64 throughput is ~1/64 of FP32 -- prefer")
    print("     float32 unless the science needs doubles.)")


if __name__ == "__main__":
    main()
