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
        normalization: (
            keras.layers.BatchNormalization | keras.layers.LayerNormalization | None
        ) = None,
        **kwargs,
    ) -> None:
        super(MLPBlock, self).__init__(**kwargs)
        self.dense = dense
        self.activation = activation
        self.dropout = dropout
        self.normalization = normalization
        self.sub_layers = list(
            filter(
                None, [self.dense, self.activation, self.dropout, self.normalization]
            )
        )

    def call(self, inputs: t.Any, training: bool = None) -> t.Any:
        """"""
        x = self.dense(inputs)
        if self.normalization is not None:
            x = self.normalization(x, training=training)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        return x

    def get_config(self) -> t.Dict[str, t.Any]:
        """Returns config for serialization."""
        base_config = super().get_config()
        config = {
            "activation": keras.utils.serialize_keras_object(self.activation),
            "dense": keras.utils.serialize_keras_object(self.dense),
            "dropout": keras.utils.serialize_keras_object(self.dropout),
            "normalization": keras.utils.serialize_keras_object(self.normalization),
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

        normalization_config = config.pop("normalization")
        normalization_layer = keras.layers.deserialize(normalization_config)

        return cls(dense_layer, act_layer, dropout_layer, normalization_layer, **config)


def get_keras_activation(
    identifier: t.Any,
) -> keras.layers.Activation | keras.layers.PReLU:
    if identifier == "prelu":
        return keras.layers.PReLU()
    else:
        return keras.layers.Activation(keras.activations.get(identifier))


NORM_LAYERS = {
    "batch": keras.layers.BatchNormalization,
    "layer": keras.layers.LayerNormalization,
}


def make_mlp_block(
    units: int,
    activation: t.Any = "relu",
    use_l2: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    l2_factor: float = 0.01,
    norm_type: str | None = None,
    **kwargs,
) -> MLPBlock:
    """Builds an MLPBlock with the specified configuration.

    Parameters
    ----------
    units : int
        The number of units for the dense layer.
    activation : t.Any, optional
        The activation function, by default "relu"
    use_normalization : bool, optional
        Whether or not to use batch normalization, by default False
    use_dropout : bool, optional
        Whether or not to use dropout, by default False
    dropout_rate : float, optional
        The dropout rate (ignored if use_dropout is False), by default 0.1
    use_l2 : bool, optional
        Whether or not to use L2 regularization, by default False
    l2_factor : float, optional
        The L2 regularization factor (ignored if use_l2 is False), by default 0.01
    norm_type : str | None, optional
        The type of normalization layer to use ("batch" or "layer", ignored if use_normalization is False), by default None

    Returns
    -------
    MLPBlock
        The configured MLPBlock instance.
    """
    kernel_regularizer = None if not use_l2 else keras.regularizers.L2(l2_factor)

    act_layer = get_keras_activation(activation)
    # act_layer = keras.layers.Activation(keras.activations.get(activation))
    dense_layer = keras.layers.Dense(units, kernel_regularizer=kernel_regularizer)
    dropout_layer = keras.layers.Dropout(dropout_rate) if use_dropout else None
    normalization_layer = (
        NORM_LAYERS.get(norm_type)() if use_normalization and norm_type else None
    )

    return MLPBlock(
        dense_layer, act_layer, dropout_layer, normalization_layer, **kwargs
    )
