"""Ensemble model utilities."""

from __future__ import annotations

import tensorflow as tf
import typing as t

from tensorflow import keras


def trimmed_mean(x: tf.Tensor, fraction: float = 0.2, axis: int = 1):
    """Compute the trimmed mean along the specified axis."""
    x_sorted = tf.sort(x, axis=axis)

    n = tf.shape(x)[axis]
    k = tf.cast(tf.math.floor(tf.cast(n, tf.float32) * fraction), tf.int32)

    inds = tf.range(k, n - k)
    x_trimmed = tf.gather(x_sorted, inds, axis=axis)

    return tf.reduce_mean(x_trimmed, axis=axis)


_ScreenDLInput = t.Any  # dummy type - here for readability


class ScreenDLEnsembleWrapper:
    """Light ScreenDL wrapper for generating ensembled predictions."""

    def __init__(self, members: t.Iterable[keras.Model]) -> None:
        self.members = members

    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return self.call(*args, **kwargs)

    def call(
        self,
        inputs: _ScreenDLInput | t.Iterable[_ScreenDLInput],
        training: bool = False,
        map_inputs: bool = False,
        trim_frac: float = 0.2,
    ) -> tf.Tensor:
        """Predictions using trimmed mean of ensemble members."""
        if map_inputs:
            # inputs have a 1-to-1 correspondence with ensemble members:
            # this is useful when we have different preprocessing pipelines
            # since models are trained on different subsets of data
            assert len(inputs) == len(self.members)
        else:
            # single input mode - each member is applied to the same input
            inputs = [inputs for _ in range(len(self.members))]

        preds = []
        for M, input in zip(self.members, inputs):
            preds.append(M(input, training=training))

        preds = tf.concat(preds, axis=1)
        return trimmed_mean(preds, fraction=trim_frac, axis=1)

    def get_weights(self) -> t.List[t.Any]:
        return [M.get_weights() for M in self.members]

    def set_weights(self, weights: t.List[t.Any]) -> None:
        for M, W in zip(self.members, weights):
            M.set_weights(W)
