"""Process-level environment setup.

This module must be imported before TensorFlow/Keras/NumPy-heavy imports.
Do not import TensorFlow, Keras, NumPy, pandas, sklearn, or screendl modules here.
"""

from __future__ import annotations

import os
import sys


def assert_tensorflow_not_imported() -> None:
    """Raise if TensorFlow/Keras has already been imported."""
    bad = [name for name in ("tensorflow", "keras", "tf_keras") if name in sys.modules]
    if bad:
        raise RuntimeError(
            "TensorFlow/Keras was imported before configure_process_env(): "
            + ", ".join(bad)
        )


def configure_process_env(
    *,
    default_threads: str = "4",
    inter_op_threads: str = "1",
    malloc_arena_max: str = "2",
    force: bool = False,
    strict_import_order: bool = True,
) -> None:
    """Configure env vars that must be set before TensorFlow/BLAS imports.

    Parameters
    ----------
    default_threads
        Fallback thread count when SLURM_CPUS_PER_TASK is unset.
    inter_op_threads
        TensorFlow inter-op parallelism. Usually keep low for sweeps.
    malloc_arena_max
        Limit glibc allocator arenas on Linux. Helps reduce retained RSS.
    force
        If True, overwrite existing env vars. Default preserves cluster/user env.
    strict_import_order
        If True, fail if TensorFlow/Keras was already imported.
    """
    if strict_import_order:
        assert_tensorflow_not_imported()

    n_threads = os.environ.get("SLURM_CPUS_PER_TASK", default_threads)

    values = {
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "TF_USE_LEGACY_KERAS": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "HYDRA_FULL_ERROR": "1",
        "MALLOC_ARENA_MAX": malloc_arena_max,
        "OMP_NUM_THREADS": n_threads,
        "TF_NUM_INTRAOP_THREADS": n_threads,
        "TF_NUM_INTEROP_THREADS": inter_op_threads,
        "MKL_NUM_THREADS": n_threads,
        "OPENBLAS_NUM_THREADS": n_threads,
        "VECLIB_MAXIMUM_THREADS": n_threads,
        "NUMEXPR_NUM_THREADS": n_threads,
    }

    for key, value in values.items():
        if force:
            os.environ[key] = str(value)
        else:
            os.environ.setdefault(key, str(value))
