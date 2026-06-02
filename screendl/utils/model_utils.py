"""Keras model utilities."""

from __future__ import annotations

import tensorflow as tf
import typing as t

from tensorflow import keras

from cdrpy.mapper import BatchedResponseGenerator

if t.TYPE_CHECKING:
    from cdrpy.data import Dataset


def _iter_child_layers(layer: keras.layers.Layer) -> t.Iterable[keras.layers.Layer]:
    """Yields direct child layers from either `.layers` or `.sub_layers`."""
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
    names: str | t.Tuple[str] | None = None,
    prefixes: str | t.Tuple[str] | None = None,
) -> keras.Model:
    """Recursively freeze layers matching either `names` or `prefixes`.

    Parameters
    ----------
    model : keras.Model | keras.layers.Layer
        The model object.
    names : str | t.Tuple[str] | None, optional
        The layer name(s) to freeze, by default None
    prefixes : str | t.Tuple[str] | None, optional
        Layer name prefixes to freeze, by default None

    Returns
    -------
    keras.Model
        The model with frozen layers.
    """
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


def configure_transfer_model(
    base_model: keras.Model,
    frozen_layer_names: str | tuple[str, ...] | None = None,
    frozen_layer_prefixes: str | tuple[str, ...] | None = None,
) -> keras.Model:
    """Configure an isolated transfer model."""
    model = keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    model.trainable = True
    model = freeze_layers(model, frozen_layer_names, frozen_layer_prefixes)
    return model


def fit_transfer_model(
    ft_model: keras.Model,
    initial_weights: t.Any,
    dataset: Dataset,
    batch_size: int = 256,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float | None = None,
    fit_kwargs: t.Dict[t.Any, t.Any] | None = None,
    loss: t.Any = "mean_squared_error",
    **kwargs,
) -> keras.Model:
    """Trains the transfer model.

    Parameters
    ----------
    base_model : keras.Model
        The pretrained model.
    dataset : Dataset
        The training dataset.
    batch_size : int, optional
        The batch size to use, by default 256
    epochs : int, optional
        Number of training epochs, by default 10
    learning_rate : float, optional
        Optimizer learning rate, by default 1e-4
    weight_decay : float | None, optional
        Weight decay rate for adam optimizer, by default None

    Returns
    -------
    keras.Model
        The resulting model.
    """
    if fit_kwargs is None:
        fit_kwargs = dict()

    batch_gen = BatchedResponseGenerator(dataset, batch_size)
    batch_seq = batch_gen.flow_from_dataset(dataset, shuffle=True, seed=1441)

    # model = configure_transfer_model(
    #     base_model,
    #     keras.optimizers.Adam(learning_rate, weight_decay=weight_decay),
    #     **kwargs,
    # )

    def scheduler(epoch: int, lr: float) -> float:
        if epoch <= 2:
            return lr
        return float(lr * tf.math.exp(-0.1))

    callbacks = fit_kwargs.get("callbacks", [])
    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, list):
        callbacks = [callbacks]
    else:
        callbacks = list(callbacks)

    callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))
    fit_kwargs = dict(fit_kwargs)
    fit_kwargs["callbacks"] = callbacks

    ft_model.set_weights(initial_weights)
    ft_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate, weight_decay=weight_decay),
        loss=loss,
        jit_compile=False,
    )

    _ = ft_model.fit(batch_seq, epochs=epochs, verbose=0, **fit_kwargs)

    del batch_seq
    del batch_gen

    return ft_model


def configure_screenahead_model(
    base_model: keras.Model,
    frozen_layer_names: str | tuple[str, ...] | None = None,
    frozen_layer_prefixes: str | tuple[str, ...] | None = None,
    training: bool = False,
) -> keras.Model:
    """Configure an isolated ScreenAhead model."""
    model = keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    model.trainable = True

    # turn noise layer of during screenahead
    inputs = model.inputs
    outputs = model(inputs, training=training)
    wrapped = keras.Model(inputs, outputs, name=model.name)

    wrapped = freeze_layers(wrapped, frozen_layer_names, frozen_layer_prefixes)
    return wrapped


def fit_screenahead_model(
    sa_model: keras.Model,
    initial_weights: t.Any,
    dataset: Dataset,
    batch_size: int | None = None,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float | None = None,
    callbacks: t.List[t.Any] | None = None,
    loss: t.Any = "mean_squared_error",
    **kwargs,
) -> keras.Model:
    """Trains the model using ScreenAhead.

    Parameters
    ----------
    base_model : keras.Model
        The pretrained model
    dataset : Dataset
        The ScreenAhead dataset
    batch_size : int | None, optional
        The batch size to use, by default None
    epochs : int, optional
        Number of training epochs, by default 20
    learning_rate : float, optional
        Learning rate for Adam optimizer, by default 1e-4
    weight_decay : float | None, optional
        Weight decay rate for Adam optimizer, by default None

    Returns
    -------
    keras.Model
        The resulting model.
    """
    if batch_size is None:
        batch_size = dataset.n_drugs

    batch_gen = BatchedResponseGenerator(dataset, batch_size)
    batch_seq = batch_gen.flow_from_dataset(dataset, shuffle=True, seed=1441)

    sa_model.set_weights(initial_weights)
    sa_model.compile(
        keras.optimizers.Adam(learning_rate, weight_decay=weight_decay),
        loss=loss,
        jit_compile=False,
    )

    _ = sa_model.fit(batch_seq, epochs=epochs, verbose=0, callbacks=callbacks)

    del batch_seq
    del batch_gen

    return sa_model
