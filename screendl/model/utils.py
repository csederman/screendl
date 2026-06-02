"""Keras model utilities."""

from __future__ import annotations

import gc
import typing as t
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from cdrpy.mapper import BatchedResponseGenerator

if t.TYPE_CHECKING:
    from cdrpy.datasets import Dataset


def _iter_child_layers(layer: keras.layers.Layer) -> t.Iterable[keras.layers.Layer]:
    """Yield direct child layers from either `.layers` or `.sub_layers`."""
    seen: set[int] = set()

    for attr in ("layers", "sub_layers"):
        children = getattr(layer, attr, None)
        if not children:
            continue

        for child in children:
            if isinstance(child, keras.layers.Layer) and id(child) not in seen:
                seen.add(id(child))
                yield child


TrainableState = list[tuple[str, bool]]


def iter_layers_recursive(
    model: keras.Model | keras.layers.Layer,
) -> t.Iterable[keras.layers.Layer]:
    """Yield a model/layer and all nested child layers exactly once."""
    visited: set[int] = set()

    def _walk(layer: keras.layers.Layer) -> t.Iterator[keras.layers.Layer]:
        if id(layer) in visited:
            return

        visited.add(id(layer))
        yield layer

        for child in _iter_child_layers(layer):
            yield from _walk(child)

    yield from _walk(model)


def get_trainable_state(
    model: keras.Model | keras.layers.Layer,
) -> TrainableState:
    """Capture recursive trainability flags in traversal order."""
    return [(layer.name, layer.trainable) for layer in iter_layers_recursive(model)]


def set_trainable_state(
    model: keras.Model | keras.layers.Layer,
    trainable_state: TrainableState,
    *,
    strict: bool = True,
) -> None:
    """Restore recursive trainability flags captured by `get_trainable_state`."""
    layers = list(iter_layers_recursive(model))

    if strict and len(layers) != len(trainable_state):
        raise ValueError(
            f"Layer count changed: model has {len(layers)} layers, "
            f"state has {len(trainable_state)} layers."
        )

    for layer, (expected_name, trainable) in zip(layers, trainable_state):
        if strict and layer.name != expected_name:
            raise ValueError(
                f"Layer order/name changed: expected {expected_name}, got {layer.name}."
            )
        layer.trainable = trainable


def freeze_layers(
    model: keras.Model | keras.layers.Layer,
    names: str | tuple[str, ...] | None = None,
    prefixes: str | tuple[str, ...] | None = None,
) -> keras.Model:
    """Recursively freeze layers matching either `names` or `prefixes`."""
    if isinstance(names, str):
        names = (names,)
    if isinstance(prefixes, str):
        prefixes = (prefixes,)

    visited: set[int] = set()

    def _freeze(layer: keras.layers.Layer) -> None:
        if id(layer) in visited:
            return
        visited.add(id(layer))

        if names is not None and layer.name in names:
            layer.trainable = False
        elif prefixes is not None and layer.name.startswith(prefixes):
            layer.trainable = False

        for child in _iter_child_layers(layer):
            _freeze(child)

    _freeze(model)
    return model


def clone_model_from_weights(
    model: keras.Model,
    weights: t.Any | None = None,
    *,
    trainable_state: TrainableState | None = None,
    strict_trainable_state: bool = True,
) -> keras.Model:
    """Clone a model and optionally restore weights/trainability.

    The returned model does not share layer objects with the input model.
    """
    cloned = keras.models.clone_model(model)

    if weights is None:
        weights = model.get_weights()

    cloned.set_weights(weights)

    if trainable_state is not None:
        set_trainable_state(
            cloned,
            trainable_state,
            strict=strict_trainable_state,
        )

    return cloned


def configure_transfer_model(
    base_model: keras.Model,
    initial_weights: t.Any | None = None,
    frozen_layer_names: str | tuple[str, ...] | None = None,
    frozen_layer_prefixes: str | tuple[str, ...] | None = None,
    trainable_state: TrainableState | None = None,
) -> keras.Model:
    """Configure an isolated transfer model."""
    model = clone_model_from_weights(base_model, weights=initial_weights)

    model.trainable = True

    if trainable_state is not None:
        set_trainable_state(model, trainable_state, strict=True)

    freeze_layers(model, frozen_layer_names, frozen_layer_prefixes)
    return model


def _as_callback_list(callbacks: t.Any) -> list[t.Any]:
    """Normalize callbacks to a mutable list."""
    if callbacks is None:
        return []
    if isinstance(callbacks, list):
        return list(callbacks)
    return [callbacks]


def fit_transfer_model(
    model: keras.Model,
    dataset: Dataset,
    batch_size: int = 256,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float | None = None,
    fit_kwargs: dict[t.Any, t.Any] | None = None,
    loss: t.Any = "mean_squared_error",
    **kwargs: t.Any,
) -> keras.Model:
    """Train a transfer model.

    The model is assumed to already be initialized with the desired starting
    weights before this function is called.
    """
    del kwargs

    if fit_kwargs is None:
        fit_kwargs = {}

    batch_gen = None
    batch_seq = None

    try:
        batch_gen = BatchedResponseGenerator(dataset, batch_size)
        batch_seq = batch_gen.flow_from_dataset(dataset, shuffle=True, seed=1441)

        def scheduler(epoch: int, lr: float) -> float:
            if epoch <= 2:
                return lr
            return float(lr * tf.math.exp(-0.1))

        callbacks = _as_callback_list(fit_kwargs.get("callbacks", []))
        callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))

        fit_kwargs = dict(fit_kwargs)
        fit_kwargs["callbacks"] = callbacks

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
            loss=loss,
            jit_compile=False,
        )

        model.fit(batch_seq, epochs=epochs, verbose=0, **fit_kwargs)
        return model

    finally:
        del batch_seq
        del batch_gen
        gc.collect()


def configure_screenahead_model(
    base_model: keras.Model,
    initial_weights: t.Any | None = None,
    frozen_layer_names: str | tuple[str, ...] | None = None,
    frozen_layer_prefixes: str | tuple[str, ...] | None = None,
    training: bool = False,
    trainable_state: TrainableState | None = None,
) -> keras.Model:
    """Configure an isolated ScreenAhead model.

    The base model is cloned first, then wrapped with `training=...` so dropout
    or noise layers can be disabled during ScreenAhead adaptation.
    """
    model = clone_model_from_weights(
        base_model,
        weights=initial_weights,
    )

    model.trainable = True

    if trainable_state is not None:
        set_trainable_state(model, trainable_state, strict=True)

    inputs = model.inputs
    outputs = model(inputs, training=training)
    wrapped = keras.Model(inputs, outputs, name=f"{model.name}_screenahead")

    freeze_layers(wrapped, frozen_layer_names, frozen_layer_prefixes)
    return wrapped


def fit_screenahead_model(
    model: keras.Model,
    dataset: Dataset,
    batch_size: int | None = None,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float | None = None,
    callbacks: list[t.Any] | None = None,
    loss: t.Any = "mean_squared_error",
    **kwargs: t.Any,
) -> keras.Model:
    """Train a ScreenAhead model.

    The model is assumed to already be initialized with the desired starting
    weights before this function is called.
    """
    del kwargs

    if batch_size is None:
        batch_size = dataset.n_drugs

    batch_gen = None
    batch_seq = None

    try:
        batch_gen = BatchedResponseGenerator(dataset, batch_size)
        batch_seq = batch_gen.flow_from_dataset(dataset, shuffle=True, seed=1441)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
            loss=loss,
            jit_compile=False,
        )

        model.fit(
            batch_seq,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
        )
        return model

    finally:
        del batch_seq
        del batch_gen
        gc.collect()


def clear_compiled_model(model: keras.Model | None) -> None:
    """Best-effort cleanup of compiled Keras state."""
    if model is None:
        return

    for attr in (
        "train_function",
        "test_function",
        "predict_function",
    ):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:
                pass

    try:
        model.stop_training = True
    except Exception:
        pass

    try:
        model.optimizer = None  # type: ignore[misc]
    except Exception:
        pass


def trim_malloc() -> None:
    """Return freed heap pages to the OS on Linux/glibc when possible."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def safe_path_token(value: t.Any) -> str:
    """Make a filesystem-safe token from a cell/model identifier."""
    text = str(value)
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in text)


def save_keras_model(model: keras.Model, path: str | Path) -> None:
    """Save a Keras model, creating parent directories first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
