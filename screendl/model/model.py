"""

"""

from __future__ import annotations

import tensorflow as tf
import typing as t

from tensorflow import keras
from keras import Model
from keras import layers


class MLPBlock(layers.Layer):
    """
    FIXME: add get_config method for the layer for saving
    """

    def __init__(
        self,
        units: int,
        name: str,
        activation: t.Any = "relu",
        use_batch_norm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super(MLPBlock, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        if self.use_dropout:
            self.dropout = layers.Dropout(self.dropout_rate)
        else:
            self.dropout = None

        if self.use_batch_norm:
            self.batch_norm = layers.BatchNormalization()
        else:
            self.batch_norm = None

        self.dense = layers.Dense(self.units)
        self.activation = layers.Activation(keras.activations.get(activation))

    def call(self, inputs: t.Any, training: bool = True, **kwargs) -> t.Any:
        """"""
        x = inputs
        if training and self.dropout is not None:
            x = self.dropout(x)
        x = self.dense(x)
        if training and self.batch_norm is not None:
            x = self.batch_norm(x)
        return self.activation(x)

    def get_config(self) -> dict[str, t.Any]:
        """Returns config for serialization."""
        return {
            "units": self.units,
            "use_batch_norm": self.use_batch_norm,
            "use_dropout": self.use_dropout,
            "dropout_rate": self.dropout_rate,
        }


def _create_expression_subnetwork(
    dim: int,
    norm_layer: layers.Normalization | None = None,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the expression subnetwork."""
    x = input_layer = layers.Input((dim,), name="exp_input")

    if norm_layer is not None:
        if not norm_layer.is_adapted:
            # FIXME: change this to a warning since you can still adapt later
            # FIXME: this needs to be decoupled from model creation - I need to
            #   refactor the models so normalization occurs during data
            #   preprocessing.
            raise ValueError("requires adapted normalization layer...")
        x = norm_layer(x)

    if hidden_dims is None:
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    # create the first FC component
    x = MLPBlock(
        dim,
        name="exp_mlp_1",
        activation="relu",
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=initial_dropout_rate,
    )(x)

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 2):
        x = MLPBlock(
            units,
            name=f"exp_mlp_{i}",
            activation=("tanh" if i == n_hidden else "relu"),
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )(x)

    return Model(inputs=input_layer, outputs=x, name="exp_subnet")


def _create_mutation_subnetwork(
    dim: int,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the mutation subnetwork."""
    x = input_layer = layers.Input((dim,), name="mut_input")

    if hidden_dims is None:
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    x = MLPBlock(
        dim,
        name="mut_mlp_1",
        activation="relu",
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=initial_dropout_rate,
    )(x)

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 2):
        x = MLPBlock(
            units,
            name=f"mut_mlp_{i}",
            activation=("tanh" if i == n_hidden else "relu"),
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )(x)

    return Model(inputs=input_layer, outputs=x, name="mut_subnet")


def _create_ontology_subnetwork(
    dim: int,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the ontology subnetwork."""
    x = input_layer = layers.Input((dim,), name="ont_input")

    if hidden_dims is None:
        hidden_dims = [dim // 2, dim // 4]

    x = MLPBlock(
        dim,
        name="ont_mlp_1",
        activation="relu",
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=initial_dropout_rate,
    )(x)

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 2):
        x = MLPBlock(
            units,
            name=f"ont_mlp_{i}",
            activation=("tanh" if i == n_hidden else "relu"),
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )(x)

    return Model(inputs=input_layer, outputs=x, name="ont_subnet")


def _create_copy_number_subnetwork(
    dim: int,
    norm_layer: layers.Normalization | None = None,
    hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the mutation subnetwork."""
    x = input_layer = layers.Input((dim,), name="cnv_input")

    if norm_layer is not None:
        if not norm_layer.is_adapted:
            # FIXME: change this to a warning since you can still adapt later
            # FIXME: this needs to be decoupled from model creation - I need to
            #   refactor the models so normalization occurs during data
            #   preprocessing.
            raise ValueError("requires adapted normalization layer...")
        x = norm_layer(x)

    if hidden_dims is None:
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    x = MLPBlock(
        dim,
        name="cnv_mlp_1",
        activation="relu",
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=initial_dropout_rate,
    )(x)

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 2):
        x = MLPBlock(
            units,
            name=f"cnv_mlp_{i}",
            activation=("tanh" if i == n_hidden else "relu"),
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )(x)

    return Model(inputs=input_layer, outputs=x, name="cnv_subnet")


def _create_cell_subnetwork(
    exp_dim: int,
    mut_dim: int | None = None,
    cnv_dim: int | None = None,
    ont_dim: int | None = None,
    exp_norm_layer: layers.Normalization | None = None,
    cnv_norm_layer: layers.Normalization | None = None,
    exp_hidden_dims: list[int] | None = None,
    mut_hidden_dims: list[int] | None = None,
    cnv_hidden_dims: list[int] | None = None,
    ont_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Model:
    """Creates the cell subnetwork.

    Parameters
    ----------
        exp_dim: The dimension of the gene expression subnetwork.
        mut_dim: The dimension of the mutation subnetwork.
        cnv_dim: The dimension of the copy number subnetwork.
        exp_norm_layer: An optional `keras.layers.Normalization` layer.
        cnv_norm_layer: An optional `keras.layers.Normalization` layer.
        exp_hidden_dims: An optional list specifying hidden layers/units for
            the gene expression subnetwork.
        mut_hidden_dims: An optional list specifying hidden layers/units for
            the mutation subnetwork.
        cnv_hidden_dims: An optional list specifying hidden layers/units for
            the copy number subnetwork.
        use_batch_norm: Whether or not to use batch normalization.
        use_dropout: Whether or not to use dropout.
        dropout_rate: The dropout rate. Ignored if `use_dropout` is `False`.

    Returns
    -------
        The drug subnetwork `keras.Model` instance.
    """
    exp_subnet = _create_expression_subnetwork(
        exp_dim,
        exp_norm_layer,
        exp_hidden_dims,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        initial_dropout_rate=initial_dropout_rate,
    )
    subnet_inputs = [exp_subnet.input]
    subnet_output = exp_subnet.output

    if mut_dim is not None:
        mut_subnet = _create_mutation_subnetwork(
            mut_dim,
            mut_hidden_dims,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        subnet_inputs.append(mut_subnet.input)
        subnet_output = layers.Concatenate()(
            [subnet_output, mut_subnet.output]
        )

    if cnv_dim is not None:
        cnv_subnet = _create_copy_number_subnetwork(
            cnv_dim,
            cnv_norm_layer,
            cnv_hidden_dims,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        subnet_inputs.append(cnv_subnet.input)
        subnet_output = layers.Concatenate()(
            [subnet_output, cnv_subnet.output]
        )

    if ont_dim is not None:
        ont_subnet = _create_ontology_subnetwork(
            ont_dim,
            ont_hidden_dims,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        subnet_inputs.append(ont_subnet.input)
        subnet_output = layers.Concatenate()(
            [subnet_output, ont_subnet.output]
        )

    return Model(
        inputs=subnet_inputs, outputs=subnet_output, name="cell_subnet"
    )


def _create_drug_subnetwork(
    mol_dim: int,
    mol_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Sequential:
    """Creates the drug subnetwork.

    Parameters
    ----------
        mol_dim: The dimension of the drug subnetwork.
        mol_hidden_dims: Optional list specifying hidden layers/units.
        use_batch_norm: Whether or not to use batch normalization.
        use_dropout: Whether or not to use dropout.
        dropout_rate: The dropout rate. Ignored if `use_dropout` is `False`.

    Returns
    -------
        The drug subnetwork `keras.Model` instance.
    """
    x = input_layer = layers.Input((mol_dim,), name="fp_input")

    if mol_hidden_dims is None:
        mol_hidden_dims = [mol_dim // 2, mol_dim // 4, mol_dim // 8]

    x = MLPBlock(
        mol_dim,
        name="mol_mlp_1",
        activation="relu",
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=initial_dropout_rate,
    )(x)

    n_hidden = len(mol_hidden_dims)
    for i, units in enumerate(mol_hidden_dims, 2):
        x = MLPBlock(
            units,
            name=f"mol_mlp_{i}",
            activation=("tanh" if i == n_hidden else "relu"),
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )(x)

    return Model(inputs=input_layer, outputs=x, name="drug_subnet")


def create_model(
    exp_dim: int,
    mol_dim: int,
    mut_dim: int | None = None,
    cnv_dim: int | None = None,
    ont_dim: int | None = None,
    exp_norm_layer: layers.Normalization | None = None,
    cnv_norm_layer: layers.Normalization | None = None,
    exp_hidden_dims: list[int] | None = None,
    mut_hidden_dims: list[int] | None = None,
    cnv_hidden_dims: list[int] | None = None,
    ont_hidden_dims: list[int] | None = None,
    mol_hidden_dims: list[int] | None = None,
    shared_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.1,
    initial_dropout_rate: float = 0.1,
) -> keras.Model:
    """"""
    cell_subnet = _create_cell_subnetwork(
        exp_dim,
        mut_dim,
        cnv_dim,
        ont_dim,
        exp_norm_layer=exp_norm_layer,
        cnv_norm_layer=cnv_norm_layer,
        exp_hidden_dims=exp_hidden_dims,
        mut_hidden_dims=mut_hidden_dims,
        cnv_hidden_dims=cnv_hidden_dims,
        ont_hidden_dims=ont_hidden_dims,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        initial_dropout_rate=initial_dropout_rate,
    )

    drug_subnet = _create_drug_subnetwork(
        mol_dim,
        mol_hidden_dims=mol_hidden_dims,
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        initial_dropout_rate=initial_dropout_rate,
    )

    if isinstance(cell_subnet.input, list):
        # FIXME: Should probably not concatenate here and pass the outputs of
        #   the cell encoders and drug encoders separately.
        subnet_inputs = [*cell_subnet.input, drug_subnet.input]
    else:
        subnet_inputs = [cell_subnet.input, drug_subnet.input]
    subnet_outputs = [cell_subnet.output, drug_subnet.output]

    latent_dim = sum(
        (cell_subnet.output_shape[1], drug_subnet.output_shape[1])
    )

    if shared_hidden_dims is None:
        shared_hidden_dims = [
            latent_dim // 2,
            latent_dim // 4,
            latent_dim // 8,
            latent_dim // 10,
        ]

    x = layers.Concatenate(name="concat")(subnet_outputs)

    x = MLPBlock(
        latent_dim,
        name="shared_mlp_1",
        activation="relu",
        use_batch_norm=use_batch_norm,
        use_dropout=use_dropout,
        dropout_rate=initial_dropout_rate,
    )(x)

    for i, units in enumerate(shared_hidden_dims, 2):
        x = MLPBlock(
            units,
            name=f"shared_mlp_{i}",
            activation="relu",
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )(x)

    output = layers.Dense(1, "linear", name="final_act")(x)

    return Model(inputs=subnet_inputs, outputs=output, name="ScreenDL")
