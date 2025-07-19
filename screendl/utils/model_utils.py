"""Keras model utilities."""

from __future__ import annotations

import tensorflow as tf
import typing as t

from tensorflow import keras
from cdrpy.mapper import BatchedResponseGenerator

if t.TYPE_CHECKING:
    from cdrpy.data import Dataset


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

    for layer in model.layers:
        if names is not None and layer.name in names:
            layer.trainable = False
        elif prefixes is not None and layer.name.startswith(prefixes):
            layer.trainable = False
        elif hasattr(layer, "layers"):
            freeze_layers(layer, names, prefixes)

    return model


def configure_transfer_model(
    base_model: keras.Model,
    optim: t.Any,
    loss: t.Any = "mean_squared_error",
    frozen_layer_names: str | t.Tuple[str] | None = None,
    frozen_layer_prefixes: str | t.Tuple[str] | None = None,
    training: bool = False,
) -> keras.Model:
    """Configures the model for transfer learning.

    Parameters
    ----------
    model : keras.Model
        The pretrained model
    optim : t.Any
        A keras optimizer
    loss : t.Any
        The loss function to use, by default "mean_squared_error"
    frozen_layer_names : str | t.Tuple[str] | None, optional
        An iterable of layer names to be frozen, by default None
    frozen_layer_prefixes : str | t.Tuple[str] | None, optional
        A tuple of prefixes to freeze layers, by default None
    training : bool, optional
        Privileged training arg passed to the new model, by default False

    Returns
    -------
    keras.Model
        The configured model.
    """

    base_model.trainable = True

    # inputs = base_model.inputs
    # output = base_model(inputs, training=training)
    # model = keras.Model(inputs, output, name=base_model.name)

    model = freeze_layers(base_model, frozen_layer_names, frozen_layer_prefixes)
    model.compile(optim, loss)

    return model


def fit_transfer_model(
    base_model: keras.Model,
    dataset: Dataset,
    batch_size: int = 256,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float | None = None,
    fit_kwargs: t.Dict[t.Any, t.Any] | None = None,
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

    model = configure_transfer_model(
        base_model,
        keras.optimizers.Adam(learning_rate, weight_decay=weight_decay),
        **kwargs,
    )

    def scheduler(epoch: int, lr: float) -> float:
        if epoch <= 2:
            return lr
        return lr * tf.math.exp(-0.1)

    callbacks = fit_kwargs.get("callbacks", [])
    if not isinstance(callbacks, list):
        callbacks = [callbacks]
    callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))
    fit_kwargs["callbacks"] = callbacks

    _ = model.fit(batch_seq, epochs=epochs, verbose=0, **fit_kwargs)

    return model


def configure_screenahead_model(
    base_model: keras.Model,
    optim: t.Any,
    loss: t.Any = "mean_squared_error",
    frozen_layer_names: str | t.Tuple[str] | None = None,
    frozen_layer_prefixes: str | t.Tuple[str] | None = None,
    training: bool = False,
) -> keras.Model:
    """Configures a model for ScreenAhead.

    Parameters
    ----------
    base_model : keras.Model
        The pretrained model
    optim : t.Any
        A keras optimizer
    loss : t.Any
        The loss function to use, by default "mean_squared_error"
    frozen_layer_names : str | t.Tuple[str] | None, optional
        An iterable of layer names to be frozen, by default None
    frozen_layer_prefixes : str | t.Tuple[str] | None, optional
        A tuple of prefixes to freeze layers, by default None
    training : bool, optional
        Privileged training arg passed to the new model, by default False

    Returns
    -------
    keras.Model
        The configured model.
    """
    base_model.trainable = True

    inputs = base_model.inputs
    output = base_model(inputs, training=training)
    model = keras.Model(inputs, output, name=base_model.name)

    model = freeze_layers(model, frozen_layer_names, frozen_layer_prefixes)
    model.compile(optim, loss)

    return model


def fit_screenahead_model(
    base_model: keras.Model,
    dataset: Dataset,
    batch_size: int | None = None,
    epochs: int = 20,
    learning_rate: float = 1e-4,
    weight_decay: float | None = None,
    callbacks: t.List[t.Any] | None = None,
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

    optim = keras.optimizers.Adam(learning_rate, weight_decay=weight_decay)
    model = configure_screenahead_model(base_model, optim, **kwargs)

    _ = model.fit(batch_seq, epochs=epochs, verbose=0, callbacks=callbacks)

    return model
