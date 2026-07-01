#!/bin/bash
# Simple local runner: generate a dataset with galsim and jax-galsim and
# compare them. Assumes the jax-galsim-benchmark environment (see
# ../environment.yml) is already active.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

start_time=$(date +%s)

python main.py -c ./config.yaml "$@"

end_time=$(date +%s)
runtime=$((end_time - start_time))
printf 'Total wall time: %02d:%02d:%02d (HH:MM:SS)\n' \
    $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
