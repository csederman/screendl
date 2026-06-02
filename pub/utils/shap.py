"""SHAP analysis helpers."""

from __future__ import annotations

import shap

import pandas as pd
import numpy as np
import typing as t
import tensorflow as tf

try:
    import tensorflow.keras.ops as K  # type: ignore

    convert_to_tensor = K.convert_to_tensor
except ImportError:
    import tensorflow.keras.backend as K  # type: ignore

    convert_to_tensor = tf.convert_to_tensor

from functools import partial
from scipy import stats
from tensorflow import keras
from tqdm import tqdm

from cdrpy.datasets import Dataset

from screendl.utils.ensemble import ScreenDLEnsembleWrapper


def _primary_output(y):
    """Return the response tensor from a possibly multi-output model."""
    if isinstance(y, dict):
        if "response" in y:
            return y["response"]
        if "final_act" in y:
            return y["final_act"]
        # fallback: first dict value
        return next(iter(y.values()))

    if isinstance(y, (list, tuple)):
        if len(y) == 1:
            return y[0]
        # ScreenDL main response should be the first output if AUX was appended.
        return y[0]

    return y


def create_predict_func(
    model: keras.Model,
    t_baseline: np.ndarray,
    d_baseline: np.ndarray,
) -> keras.Model:
    """Create a prediction function for the ensemble model.

    We want to interpret the omic features that contribute to the prediction of a
    specific drug response. Concretely, for a given R = f(d,t) for drug d and tumor t,
    we decompose R using a heirarchical additive model:

    R = f(d,t)
      = f_0 + f_d(d) + f_t(t) + f_dt(d,t)
      = f(t0, d0)
        + [f(t0, d) - f(t0, d0)]
        + [f(t, d0) - f(t0, d0)]
        + I(d,t)

    where f_0 is the global bias, f_d is the drug-specific bias, f_t is the
    tumor-specific bias, and f_dt is the interaction term between drug and tumor.
    Rearranging terms gives us:

    f_dt(d,t) = f(d,t) - f_0 - f_d(d) - f_t(t)
              = f(d,t) - f(t, d0) - f(t0, d) + f(t0, d0)

    So we adapt our model to return only I(d,t) for a given drug and tumor.

    Parameters
    ----------
    model : keras.Model
        The model to use for predictions.
    t_baseline : np.ndarray
        Baseline tumor features, shape (1, n_features).
    d_baseline : np.ndarray
        Baseline drug features, shape (1, n_features).

    Returns
    --------
    """
    n_t = t_baseline.shape[1]
    n_d = d_baseline.shape[1]

    x_in = keras.Input(shape=(n_t + n_d,))

    t_slice = keras.layers.Lambda(lambda x: x[:, :n_t])(x_in)
    d_slice = keras.layers.Lambda(lambda x: x[:, n_t:])(x_in)

    t_baseline_expanded = keras.layers.Lambda(
        lambda x: K.tile(
            convert_to_tensor(t_baseline),
            [K.shape(x)[0], 1],
        )
    )(x_in)

    d_baseline_expanded = keras.layers.Lambda(
        lambda x: K.tile(
            convert_to_tensor(d_baseline),
            [K.shape(x)[0], 1],
        )
    )(x_in)

    f_t_d = _primary_output(model([t_slice, d_slice], training=False))
    f_t_db = _primary_output(model([t_slice, d_baseline_expanded], training=False))
    f_tb_d = _primary_output(model([t_baseline_expanded, d_slice], training=False))
    f_tb_db = _primary_output(
        model([t_baseline_expanded, d_baseline_expanded], training=False)
    )

    out = f_t_d - f_t_db - f_tb_d + f_tb_db
    out = keras.layers.Flatten()(out)

    return keras.Model(x_in, out)


def shap_adapter(
    M: keras.Model,
    D: Dataset,
    drug_id: str,
    tumor_id: str,
    n_bg_samples: int = 10,
    n_shap_samples: int = 1000,
    sorted_tumors: t.List[str] | None = None,
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate SHAP values for a specific tumor and drug pair.

    Parameters
    ----------
    M : keras.Model
        The model to use for predictions.
    D : Dataset
        The dataset containing the drug and tumor encoders.
    drug_id : str
        The ID of the drug to analyze.
    tumor_id : str
        The ID of the tumor to analyze.
    n_bg_samples : int, optional
        Number of background samples to use for SHAP, by default 10.
    n_shap_samples : int, optional
        Number of samples to use for SHAP, by default 1000.
    sorted_tumors : list[str] | None, optional
        Pre-sorted list of tumor IDs to use as background, by default None.

    Returns
    -------
    shap_values_t : pd.DataFrame
        SHAP values for tumor features.
    shap_values_d : pd.DataFrame
        SHAP values for drug features.
    """
    t_ids = sorted(list(set(D.cell_ids)))
    d_ids = sorted(list(D.drug_encoders["mol"].keys()))

    if sorted_tumors is None:
        sorted_tumors = [x for x in t_ids if x != tumor_id]
        np.random.shuffle(sorted_tumors)

    x_all_tumors = D.cell_encoders["exp"].data.loc[t_ids]
    x_all_drugs = D.drug_encoders["mol"].data.loc[d_ids]

    _, t_dim = x_all_tumors.shape
    _, d_dim = x_all_drugs.shape

    # baselines for model inference (to be passed to the interaction predictor)
    t_baseline = x_all_tumors.drop(tumor_id).mean().values[None, :]
    d_baseline = np.zeros((1, d_dim))  # null drug -> model will predict tumor's GDS

    # background feature distributions for SHAP
    bg_candidates = [t for t in sorted_tumors if t != tumor_id]
    t_bg = x_all_tumors.drop(tumor_id).loc[bg_candidates[:n_bg_samples]].values

    # we use the interested drug as the reference since we want to interpret why this
    # specific drug would work for this tumor relative to others
    d_bg = x_all_drugs.loc[drug_id].values[None, :]
    d_bg = np.tile(d_bg, (n_bg_samples, 1))
    x_bg = np.hstack([t_bg, d_bg])

    predict_func = create_predict_func(M, t_baseline, d_baseline)
    explainer = shap.GradientExplainer(predict_func, [x_bg])

    x_t = x_all_tumors.loc[[tumor_id]].values
    x_d = x_all_drugs.loc[[drug_id]].values
    x_pair = np.hstack([x_t, x_d])
    shap_values = explainer.shap_values(x_pair, nsamples=n_shap_samples)

    shap_values_t = pd.DataFrame(
        {"feat": x_all_tumors.columns, "value": shap_values[0].flatten()[:t_dim]}
    )
    shap_values_d = pd.DataFrame(
        {"feat": x_all_drugs.columns, "value": shap_values[0].flatten()[t_dim:]}
    )

    return shap_values_t, shap_values_d


def shap_adapter_batch(
    M: keras.Model,
    D: Dataset,
    drug_id: str,
    tumor_ids: t.List[str],
    n_bg_samples: int = 10,
    n_shap_samples: int = 1000,
    sorted_tumors: t.List[str] | None = None,
    batch_size: int = 32,
) -> t.Dict[str, t.Tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate SHAP values for multiple tumors against a single drug.

    Builds the interaction model and GradientExplainer once, then batches
    all query tumors through in chunks. Uses the global mean tumor baseline
    rather than per-tumor LOO means.

    Parameters
    ----------
    M : keras.Model
        The model to use for predictions.
    D : Dataset
        The dataset containing the drug and tumor encoders.
    drug_id : str
        The ID of the drug to analyze.
    tumor_ids : list[str]
        The IDs of the tumors to analyze.
    n_bg_samples : int, optional
        Number of background samples to use for SHAP, by default 10.
    n_shap_samples : int, optional
        Number of samples to use for SHAP, by default 1000.
    sorted_tumors : list[str] | None, optional
        Pre-sorted list of tumor IDs to use as background, by default None.
    batch_size : int, optional
        Number of tumors to process per SHAP call, by default 32.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Mapping from tumor ID to (shap_values_t, shap_values_d).
    """
    t_ids = sorted(list(set(D.cell_ids)))
    if tumor_ids is None:
        tumor_ids = t_ids
    else:
        available = set(t_ids)
        tumor_ids = [t for t in tumor_ids if t in available]

    d_ids = sorted(list(D.drug_encoders["mol"].keys()))

    x_all_tumors = D.cell_encoders["exp"].data.loc[t_ids]
    x_all_drugs = D.drug_encoders["mol"].data.loc[d_ids]
    _, t_dim = x_all_tumors.shape
    _, d_dim = x_all_drugs.shape

    # Global mean baseline (includes query tumors — negligible difference at scale)
    t_baseline = x_all_tumors.mean().values[None, :]
    d_baseline = np.zeros((1, d_dim))

    # Build interaction model and explainer once
    predict_func = create_predict_func(M, t_baseline, d_baseline)

    if sorted_tumors is None:
        sorted_tumors = list(t_ids)
        np.random.shuffle(sorted_tumors)

    bg_candidates = list(sorted_tumors)
    t_bg = x_all_tumors.loc[bg_candidates[:n_bg_samples]].values
    d_bg = np.tile(x_all_drugs.loc[drug_id].values[None, :], (n_bg_samples, 1))
    x_bg = np.hstack([t_bg, d_bg])

    explainer = shap.GradientExplainer(predict_func, [x_bg])

    # Stack all query pairs
    x_d = x_all_drugs.loc[[drug_id]].values
    x_pairs = np.vstack(
        [np.hstack([x_all_tumors.loc[[tid]].values, x_d]) for tid in tumor_ids]
    )

    # Run SHAP in chunks
    shap_chunks = []
    for i in tqdm(range(0, len(tumor_ids), batch_size), desc="SHAP batches"):
        chunk = x_pairs[i : i + batch_size]
        sv = explainer.shap_values(chunk, nsamples=n_shap_samples)

        # GradientExplainer may return list-of-arrays for outputs.
        if isinstance(sv, list):
            sv = sv[0]

        shap_chunks.append(np.asarray(sv))

    shap_matrix = np.concatenate(shap_chunks, axis=0)
    shap_matrix = np.asarray(shap_matrix)

    # Common single-output shape is (n_samples, n_features, 1).
    if shap_matrix.ndim == 3 and shap_matrix.shape[-1] == 1:
        shap_matrix = shap_matrix[..., 0]

    if shap_matrix.ndim != 2:
        raise ValueError(
            f"Expected SHAP matrix to be 2D, got shape {shap_matrix.shape}"
        )

    # Split into per-tumor DataFrames
    results = {}
    for j, tid in enumerate(tumor_ids):
        sv = np.asarray(shap_matrix[j]).reshape(-1)

        sv_t = pd.DataFrame({"feat": x_all_tumors.columns, "value": sv[:t_dim]})
        sv_d = pd.DataFrame(
            {"feat": x_all_drugs.columns, "value": sv[t_dim : t_dim + d_dim]}
        )
        results[tid] = (sv_t, sv_d)

    return results


def simple_gsea(
    gene_shap: t.Iterable[t.Tuple[str, float]],
    gene_set: t.Set[str],
    top_n: int = 100,
    by_abs: bool = False,
    reversed: bool = False,
) -> t.Tuple[float, int, int, int, int]:
    """Simple GSEA with hypergeometric test."""
    all_genes = [g for g, _ in gene_shap]
    M = len(all_genes)

    gene_set_filtered = set(all_genes) & set(gene_set)
    K_eff = len(gene_set_filtered)
    if K_eff == 0:
        # no overlap with your universe => no enrichment
        return 1.0, 0, M, K_eff, top_n

    key = (lambda x: abs(x[1])) if by_abs else (lambda x: x[1])
    sorted_genes = sorted(gene_shap, key=key, reverse=True if by_abs else reversed)
    selected = {g for g, _ in sorted_genes[:top_n]}
    overlap = len(selected & gene_set_filtered)

    pval = stats.hypergeom.sf(overlap - 1, M, K_eff, top_n)
    return pval, overlap, M, K_eff, top_n


def _get_x_gexp(D: Dataset) -> pd.DataFrame:
    """Get the gene expression data for the given dataset."""
    t_ids = sorted(list(set(D.cell_ids)))
    return D.cell_encoders["exp"].data.loc[t_ids]


def _get_z_for_dataset(
    D: Dataset,
    gs: t.Iterable[str],
    tumor_id: str,
    gene_sets: t.Dict[str, t.Iterable[str]],
) -> float:
    """"""
    X = _get_x_gexp(D).filter(items=gene_sets[gs]).mean(axis=1)
    X_t = X.loc[tumor_id]
    X_rest = X.drop(tumor_id)
    return (X_t - X_rest.mean()) / X_rest.std()


def get_z(
    gs: t.Iterable[str],
    tumor_id: str,
    datasets: t.List[Dataset],
    gene_sets: t.Dict[str, t.Iterable[str]],
) -> pd.DataFrame:
    """Get z-scores for a given gene set across all datasets."""
    partial_func = partial(
        _get_z_for_dataset, gs=gs, tumor_id=tumor_id, gene_sets=gene_sets
    )
    return np.mean(list(map(partial_func, datasets)))


def _agg_shap_values(shap_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate SHAP values across ensemble members."""
    return (
        shap_df.groupby("feat", as_index=False)["value"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "value"})
        .sort_values("value")
    )


def run_shap_gsea(
    shap_df: pd.DataFrame,
    tumor_id: str,
    gene_sets: t.Dict[str, t.Iterable[str]],
    datasets: t.List[Dataset],
    top_n: int = 100,
    by_abs: bool = False,
    reversed: bool = False,
) -> pd.DataFrame:
    """Runs GSEA on SHAP values using ORA via hypergeometric test."""
    shap_df_agg = _agg_shap_values(shap_df)
    gene_shap = shap_df_agg[["feat", "value"]].values.tolist()

    res = []
    for name, genes in tqdm(gene_sets.items()):
        p, overlap, M, K_eff, top_n = simple_gsea(
            gene_shap, set(genes), top_n=top_n, by_abs=by_abs, reversed=reversed
        )
        # calculate totals
        totals = shap_df.query("feat in @genes").groupby("idx")["value"].sum()
        res.append(
            {
                "set": name,
                "pval": p,  # pval is computed on the aggregate shap values
                "overlap": overlap,
                "M": M,
                "K_eff": K_eff,
                "top_n": top_n,
                "total_mean": totals.mean(),
                "total_std": totals.std(),
                "z": get_z(name, tumor_id, datasets, gene_sets),
            }
        )

    return pd.DataFrame(res)


def _get_embedding_model(
    model: keras.Model,
    layer_name: str = "shared_mlp_2",
) -> keras.Model:
    """Get an embedding model from a possibly wrapped ScreenDL model."""

    # If this is a ScreenAhead/wrapper model, the actual ScreenDL model is nested.
    # Use the nested model's own inputs and outputs, not outer inputs + inner outputs.
    try:
        target_layer = model.get_layer(layer_name)
        return keras.Model(model.inputs, target_layer.output)
    except ValueError:
        pass

    nested_models = [layer for layer in model.layers if isinstance(layer, keras.Model)]

    for nested in nested_models:
        try:
            target_layer = nested.get_layer(layer_name)
            return keras.Model(nested.inputs, target_layer.output)
        except ValueError:
            continue

    raise ValueError(
        f"Could not find layer {layer_name!r} in model or nested model. "
        f"Top-level layers: {[layer.name for layer in model.layers]}"
    )


def get_ensemble_embeddings_for_drug(
    model: ScreenDLEnsembleWrapper,
    datasets: t.Iterable[Dataset],
    drug_id: str,
    layer_name: str = "shared_mlp_2",
) -> pd.DataFrame:
    """Get the embeddings from the ensemble model."""
    embedding_models = [_get_embedding_model(m, layer_name) for m in model.members]
    x_tumors = [_get_x_gexp(D) for D in datasets]
    x_drug = [D.drug_encoders["mol"].data.loc[[drug_id]].values for D in datasets]

    # align tumors across datasets
    t_ids = x_tumors[0].index
    x_tumors = [x.loc[t_ids].values for x in x_tumors]

    x_embed = [
        M([x_t, np.tile(x_d, (len(x_t), 1))], training=False).numpy()
        for M, x_t, x_d in zip(embedding_models, x_tumors, x_drug)
    ]

    return pd.DataFrame(np.concatenate(x_embed, axis=1), index=t_ids)
