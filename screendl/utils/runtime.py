"""TensorFlow runtime utilities."""

from __future__ import annotations

import gc
import logging
import os
import typing as t

import tensorflow as tf

import tensorflow.keras.backend as K  # type: ignore

from .logging import configure_tqdm_logging

log = logging.getLogger(__name__)


def configure_tensorflow_runtime() -> None:
    """Apply conservative TensorFlow CPU runtime settings.

    Environment variables controlling BLAS/OpenMP threading must be set before
    importing TensorFlow. This function only applies TensorFlow runtime settings.
    """
    # try:
    #     tf.config.threading.set_intra_op_parallelism_threads(
    #         int(os.environ["TF_NUM_INTRAOP_THREADS"])
    #     )
    #     tf.config.threading.set_inter_op_parallelism_threads(
    #         int(os.environ["TF_NUM_INTEROP_THREADS"])
    #     )
    # except RuntimeError:
    #     # Runtime was already initialized. Env vars may still help BLAS/OpenMP.
    #     log.warning(
    #         "TensorFlow runtime already initialized; thread settings unchanged."
    #     )

    tf.config.optimizer.set_jit(False)


def cleanup_objects(*objects: t.Any) -> None:
    """Run garbage collection after caller-side references are no longer needed.

    Note that this does not delete variables in the caller's scope. Callers should
    avoid using the objects after passing them here, or explicitly set them to None.
    """
    del objects
    gc.collect()


def reset_keras_runtime() -> None:
    """Best-effort cleanup of global Keras state."""
    # K.clear_session(free_memory=True)
    K.clear_session()
    gc.collect()


def get_current_rss_gb() -> float:
    """Return current resident memory in GB on Linux."""
    status_path = "/proc/self/status"

    try:
        with open(status_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    # Format: VmRSS:   123456 kB
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return float("nan")

    return float("nan")


def log_memory(label: str) -> None:
    """Log current resident memory."""
    log.info("%s | current RSS: %.2f GB", label, get_current_rss_gb())


def configure_runtime() -> None:
    """Configure runtime behavior shared across runner entrypoints."""
    configure_tqdm_logging()
    reset_keras_runtime()
    configure_tensorflow_runtime()
