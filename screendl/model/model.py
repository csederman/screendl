""" """

from __future__ import annotations

import tensorflow as tf
import typing as t

from tensorflow import keras
from tensorflow.keras import Model  # type: ignore[reportMissingImports]
from tensorflow.keras import layers  # type: ignore[reportMissingImports]

from .layers import make_mlp_block


def _create_expression_subnetwork(
    dim: int,
    norm_layer: layers.Normalization | None = None,
    hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_noise: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    dropout_rate: float = 0.1,
    noise_stddev: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
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

    if use_noise:
        x = layers.GaussianNoise(noise_stddev, name="exp_noise")(x)

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        mlp_block = make_mlp_block(
            units,
            activation=("tanh" if i == n_hidden else activation),
            # activation=activation,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            name=f"exp_mlp_{i}",
        )
        x = mlp_block(x)

    return Model(inputs=input_layer, outputs=x, name="exp_subnet")


def _create_mutation_subnetwork(
    dim: int,
    hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    dropout_rate: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
) -> keras.Model:
    """Creates the mutation subnetwork."""
    x = input_layer = layers.Input((dim,), name="mut_input")

    if hidden_dims is None:
        hidden_dims = [dim // 2, dim // 4, dim // 8]

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        mlp_block = make_mlp_block(
            units,
            activation=("tanh" if i == n_hidden else activation),
            # activation=activation,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            name=f"mut_mlp_{i}",
        )
        x = mlp_block(x)

    return Model(inputs=input_layer, outputs=x, name="mut_subnet")


def _create_ontology_subnetwork(
    dim: int,
    hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    dropout_rate: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
) -> keras.Model:
    """Creates the ontology subnetwork."""
    x = input_layer = layers.Input((dim,), name="ont_input")

    if hidden_dims is None:
        hidden_dims = [dim // 2, dim // 4]

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        mlp_block = make_mlp_block(
            units,
            activation=("tanh" if i == n_hidden else activation),
            # activation=activation,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            name=f"ont_mlp_{i}",
        )
        x = mlp_block(x)

    return Model(inputs=input_layer, outputs=x, name="ont_subnet")


def _create_copy_number_subnetwork(
    dim: int,
    norm_layer: layers.Normalization | None = None,
    hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_noise: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    dropout_rate: float = 0.1,
    noise_stddev: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
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

    if use_noise:
        x = layers.GaussianNoise(noise_stddev, name="cnv_noise")(x)

    n_hidden = len(hidden_dims)
    for i, units in enumerate(hidden_dims, 1):
        mlp_block = make_mlp_block(
            units,
            activation=("tanh" if i == n_hidden else activation),
            # activation=activation,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            name=f"cnv_mlp_{i}",
        )
        x = mlp_block(x)

    return Model(inputs=input_layer, outputs=x, name="cnv_subnet")


def _create_cell_subnetwork(
    exp_dim: int,
    mut_dim: int | None = None,
    cnv_dim: int | None = None,
    ont_dim: int | None = None,
    exp_norm_layer: layers.Normalization | None = None,
    cnv_norm_layer: layers.Normalization | None = None,
    exp_hidden_dims: t.List[int] | None = None,
    mut_hidden_dims: t.List[int] | None = None,
    cnv_hidden_dims: t.List[int] | None = None,
    ont_hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_noise: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    dropout_rate: float = 0.1,
    noise_stddev: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
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
        use_normalization: Whether or not to use batch normalization.
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
        use_l2=use_l2,
        use_noise=use_noise,
        use_normalization=use_normalization,
        use_dropout=use_dropout,
        l2_factor=l2_factor,
        dropout_rate=dropout_rate,
        noise_stddev=noise_stddev,
        activation=activation,
        norm_type=norm_type,
    )
    subnet_inputs = [exp_subnet.input]
    subnet_output = exp_subnet.output

    if mut_dim is not None:
        mut_subnet = _create_mutation_subnetwork(
            mut_dim,
            mut_hidden_dims,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            activation=activation,
            norm_type=norm_type,
        )
        subnet_inputs.append(mut_subnet.input)
        subnet_output = layers.Concatenate()([subnet_output, mut_subnet.output])

    if cnv_dim is not None:
        cnv_subnet = _create_copy_number_subnetwork(
            cnv_dim,
            cnv_norm_layer,
            cnv_hidden_dims,
            use_l2=use_l2,
            use_noise=use_noise,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            noise_stddev=noise_stddev,
            activation=activation,
            norm_type=norm_type,
        )
        subnet_inputs.append(cnv_subnet.input)
        subnet_output = layers.Concatenate()([subnet_output, cnv_subnet.output])

    if ont_dim is not None:
        ont_subnet = _create_ontology_subnetwork(
            ont_dim,
            ont_hidden_dims,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            activation=activation,
            norm_type=norm_type,
        )
        subnet_inputs.append(ont_subnet.input)
        subnet_output = layers.Concatenate()([subnet_output, ont_subnet.output])

    return Model(inputs=subnet_inputs, outputs=subnet_output, name="cell_subnet")


def _create_drug_subnetwork(
    mol_dim: int,
    mol_hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    dropout_rate: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
) -> keras.Sequential:
    """Creates the drug subnetwork.

    Parameters
    ----------
        mol_dim: The dimension of the drug subnetwork.
        mol_hidden_dims: Optional list specifying hidden layers/units.
        use_normalization: Whether or not to use batch normalization.
        use_dropout: Whether or not to use dropout.
        dropout_rate: The dropout rate. Ignored if `use_dropout` is `False`.

    Returns
    -------
        The drug subnetwork `keras.Model` instance.
    """
    x = input_layer = layers.Input((mol_dim,), name="mol_input")

    if mol_hidden_dims is None:
        mol_hidden_dims = [mol_dim // 2, mol_dim // 4, mol_dim // 8]

    n_hidden = len(mol_hidden_dims)
    for i, units in enumerate(mol_hidden_dims, 1):
        mlp_block = make_mlp_block(
            units,
            activation=("tanh" if i == n_hidden else activation),
            # activation=activation,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            l2_factor=l2_factor,
            dropout_rate=dropout_rate,
            norm_type=norm_type,
            name=f"mol_mlp_{i}",
        )
        x = mlp_block(x)

    return Model(inputs=input_layer, outputs=x, name="drug_subnet")


def create_model(
    exp_dim: int,
    mol_dim: int,
    mut_dim: int | None = None,
    cnv_dim: int | None = None,
    ont_dim: int | None = None,
    exp_norm_layer: layers.Normalization | None = None,
    cnv_norm_layer: layers.Normalization | None = None,
    exp_hidden_dims: t.List[int] | None = None,
    mut_hidden_dims: t.List[int] | None = None,
    cnv_hidden_dims: t.List[int] | None = None,
    ont_hidden_dims: t.List[int] | None = None,
    mol_hidden_dims: t.List[int] | None = None,
    shared_hidden_dims: t.List[int] | None = None,
    use_l2: bool = False,
    use_noise: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    l2_factor: float = 0.01,
    noise_stddev: float = 0.1,
    dropout_rate: float = 0.1,
    activation: t.Any = "relu",
    norm_type: str | None = None,
    interaction_mode: str = "concat",
    bilinear_dim: int = 64,
    include_bilinear_product: bool = True,
    include_bilinear_score: bool = False,
) -> keras.Model:
    """Create the ScreenDL response model.

    Parameters
    ----------
    interaction_mode
        ``"concat"`` preserves the original architecture. ``"bilinear"`` adds
        low-rank tumor/drug multiplicative interaction features before the
        shared MLP.
    """
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
        use_l2=use_l2,
        use_noise=use_noise,
        use_normalization=use_normalization,
        use_dropout=use_dropout,
        l2_factor=l2_factor,
        dropout_rate=dropout_rate,
        noise_stddev=noise_stddev,
        activation=activation,
        norm_type=norm_type,
    )

    drug_subnet = _create_drug_subnetwork(
        mol_dim,
        mol_hidden_dims=mol_hidden_dims,
        use_l2=use_l2,
        use_normalization=use_normalization,
        use_dropout=use_dropout,
        l2_factor=l2_factor,
        dropout_rate=dropout_rate,
        activation=activation,
        norm_type=norm_type,
    )

    if isinstance(cell_subnet.input, list):
        # FIXME: Should probably not concatenate here and pass the outputs of
        #   the cell encoders and drug encoders separately.
        subnet_inputs = [*cell_subnet.input, drug_subnet.input]
    else:
        subnet_inputs = [cell_subnet.input, drug_subnet.input]

    z_t = layers.Activation("linear", name="cell_embed")(cell_subnet.output)
    z_d = layers.Activation("linear", name="drug_embed")(drug_subnet.output)

    subnet_outputs = [z_t, z_d]

    latent_dim = sum((cell_subnet.output_shape[1], drug_subnet.output_shape[1]))

    if shared_hidden_dims is None:
        shared_hidden_dims = [
            latent_dim // 2,
            latent_dim // 4,
            latent_dim // 8,
            latent_dim // 10,
        ]

    if interaction_mode == "concat":
        x = layers.Concatenate(name="concat")(subnet_outputs)

    elif interaction_mode == "bilinear":
        reg = keras.regularizers.l2(l2_factor) if use_l2 else None

        p_t = layers.Dense(
            bilinear_dim,
            activation=None,
            kernel_regularizer=reg,
            name="cell_bilinear_projection",
        )(z_t)

        p_d = layers.Dense(
            bilinear_dim,
            activation=None,
            kernel_regularizer=reg,
            name="drug_bilinear_projection",
        )(z_d)

        interaction_terms = [z_t, z_d]

        if include_bilinear_product:
            interaction_terms.append(
                layers.Multiply(name="bilinear_product")([p_t, p_d])
            )

        if include_bilinear_score:
            interaction_terms.append(
                layers.Dot(axes=1, name="bilinear_score")([p_t, p_d])
            )

        x = layers.Concatenate(name="concat")(interaction_terms)

    else:
        raise ValueError(
            "interaction_mode must be one of {'concat', 'bilinear'}, "
            f"got {interaction_mode!r}."
        )

    for i, units in enumerate(shared_hidden_dims, 1):
        mlp_block = make_mlp_block(
            units,
            activation=activation,
            use_l2=use_l2,
            use_normalization=use_normalization,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            l2_factor=l2_factor,
            norm_type=norm_type,
            name=f"shared_mlp_{i}",
        )
        x = mlp_block(x)

    output = layers.Dense(1, "linear", name="final_act")(x)

    return Model(inputs=subnet_inputs, outputs=output, name="ScreenDL")


def _source_layer_name(tensor: t.Any) -> str | None:
    """Return the source Keras layer name for a symbolic tensor."""
    keras_history = getattr(tensor, "_keras_history", None)
    if keras_history is None:
        return None

    # tf.keras may expose this as tuple-like or object-like.
    if isinstance(keras_history, tuple):
        layer = keras_history[0]
    else:
        layer = getattr(keras_history, "layer", None)

    return None if layer is None else layer.name


def _ensure_named_output(
    tensor: t.Any,
    *,
    name: str,
) -> t.Any:
    """Wrap tensor in an Identity layer only if it is not already named."""
    if _source_layer_name(tensor) == name:
        return tensor

    return layers.Identity(name=name)(tensor)


def add_function_auxiliary_heads(
    response_model: keras.Model,
    *,
    drug_aux_dim: int | None = None,
    cell_aux_dim: int | None = None,
    drug_embedding_layer: str = "drug_embed",
    cell_embedding_layer: str = "cell_embed",
    drug_hidden_dims: t.List[int] | None = None,
    cell_hidden_dims: t.List[int] | None = None,
    activation: t.Any = "relu",
    use_l2: bool = False,
    l2_factor: float = 0.01,
    response_output_name: str = "response",
    drug_output_name: str = "drug_function",
    cell_output_name: str = "cell_function",
) -> keras.Model:
    """Add drug/tumor functional auxiliary heads to a response model.

    The response output is wrapped in an Identity layer named ``response``, so
    output names match ``FunctionAuxResponseSequence`` target keys.
    """
    if drug_aux_dim is None and cell_aux_dim is None:
        raise ValueError("At least one of drug_aux_dim or cell_aux_dim must be set.")

    reg = keras.regularizers.l2(l2_factor) if use_l2 else None

    response = _ensure_named_output(
        response_model.output,
        name=response_output_name,
    )
    outputs = {response_output_name: response}

    if drug_aux_dim is not None:
        if drug_hidden_dims is None:
            drug_hidden_dims = []

        z_d = response_model.get_layer(drug_embedding_layer).output
        x_d = z_d

        for i, units in enumerate(drug_hidden_dims, 1):
            x_d = make_mlp_block(
                units,
                activation=activation,
                use_l2=use_l2,
                l2_factor=l2_factor,
                name=f"{drug_output_name}_mlp_{i}",
            )(x_d)

        outputs[drug_output_name] = layers.Dense(
            drug_aux_dim,
            activation="linear",
            kernel_regularizer=reg,
            name=drug_output_name,
        )(x_d)

    if cell_aux_dim is not None:
        if cell_hidden_dims is None:
            cell_hidden_dims = []

        z_t = response_model.get_layer(cell_embedding_layer).output
        x_t = z_t

        for i, units in enumerate(cell_hidden_dims, 1):
            x_t = make_mlp_block(
                units,
                activation=activation,
                use_l2=use_l2,
                l2_factor=l2_factor,
                name=f"{cell_output_name}_mlp_{i}",
            )(x_t)

        outputs[cell_output_name] = layers.Dense(
            cell_aux_dim,
            activation="linear",
            kernel_regularizer=reg,
            name=cell_output_name,
        )(x_t)

    return Model(
        inputs=response_model.inputs,
        outputs=outputs,
        name=f"{response_model.name}_with_function_aux",
    )


def get_response_model(
    model_with_aux: keras.Model,
    *,
    response_layer_name: str = "response",
    name: str = "ScreenDL",
) -> keras.Model:
    """Extract a response-only model from an auxiliary multi-output model.

    This returns a model view sharing the trained response graph/weights.
    """
    return Model(
        inputs=model_with_aux.inputs,
        outputs=model_with_aux.get_layer(response_layer_name).output,
        name=name,
    )
