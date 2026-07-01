"""Small shared utilities (config loading, output saving)."""
import os

import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_dataset_npz(path, ds):
    np.savez_compressed(
        path,
        psf_images=ds.psf_images,
        images_p=ds.images_p,
        images_m=ds.images_m,
        per_object_time=ds.per_object_time,
    )
