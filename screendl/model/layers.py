"""Basic layer blocks."""

from __future__ import annotations

import typing as t

from tensorflow import keras


@keras.utils.register_keras_serializable()
class MLPBlock(keras.layers.Layer):
    """Configurable MLP block."""

    def __init__(
        self,
        dense: keras.layers.Dense,
        activation: keras.layers.Activation,
        dropout: keras.layers.Dropout | None = None,
        batch_norm: keras.layers.BatchNormalization | None = None,
        **kwargs,
    ) -> None:
        super(MLPBlock, self).__init__(**kwargs)
        self.dense = dense
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm

        # NOTE: this is here to enable recursive setting of certain attributes
        self.layers = list(
            filter(None, [self.dense, self.activation, self.dropout, self.batch_norm])
        )

    def call(self, inputs: t.Any, training: bool = True) -> t.Any:
        """"""
        x = self.dense(inputs)
        if training and self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if training and self.dropout is not None:
            x = self.dropout(x)
        return x

    def get_config(self) -> t.Dict[str, t.Any]:
        """Returns config for serialization."""
        base_config = super().get_config()
        config = {
            "activation": keras.utils.serialize_keras_object(self.activation),
            "dense": keras.utils.serialize_keras_object(self.dense),
            "dropout": keras.utils.serialize_keras_object(self.dropout),
            "batch_norm": keras.utils.serialize_keras_object(self.batch_norm),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config: t.Dict[str, t.Any]):
        dense_config = config.pop("dense")
        dense_layer = keras.layers.deserialize(dense_config)

        act_config = config.pop("activation")
        act_layer = keras.layers.deserialize(act_config)

        dropout_config = config.pop("dropout")
        dropout_layer = keras.layers.deserialize(dropout_config)

        batch_norm_config = config.pop("batch_norm")
        batch_norm_layer = keras.layers.deserialize(batch_norm_config)

        return cls(dense_layer, act_layer, dropout_layer, batch_norm_layer, **config)


def make_mlp_block(
    units: int,
    activation: t.Any = "relu",
    use_l2: bool = False,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    l2_factor: float = 0.01,
    **kwargs,
) -> MLPBlock:
    """Builds an MLPBlock with the specified configuration.

    Parameters
    ----------
    units : int
        The number of units for the dense layer.
    activation : t.Any, optional
        The activation function, by default "relu"
    use_batch_norm : bool, optional
        Whether or not to use batch normalization, by default False
    use_dropout : bool, optional
        Whether or not to use dropout, by default False
    dropout_rate : float, optional
        The dropout rate (ignored if use_dropout is False), by default 0.1

    Returns
    -------
    MLPBlock
        The configured MLPBlock instance.
    """
    kernel_regularizer = None if not use_l2 else keras.regularizers.L2(l2_factor)

    act_layer = keras.layers.Activation(keras.activations.get(activation))
    dense_layer = keras.layers.Dense(units, kernel_regularizer=kernel_regularizer)
    dropout_layer = keras.layers.Dropout(dropout_rate) if use_dropout else None
    batch_norm_layer = keras.layers.BatchNormalization() if use_batch_norm else None

    return MLPBlock(dense_layer, act_layer, dropout_layer, batch_norm_layer, **kwargs)
