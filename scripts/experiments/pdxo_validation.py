#!/usr/bin/env python
"""Runs domain-specific fine-tuning on the Welm breast cancer PDXO dataset."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging

import pandas as pd
import typing as t
import tensorflow.keras.backend as K  # pyright: ignore[reportMissingImports]

from sklearn.preprocessing import StandardScaler
from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from cdrpy.datasets import Dataset
from cdrpy.feat.encoders import PandasEncoder
from cdrpy.data.preprocess import normalize_responses

from screendl.pipelines.basic.screendl import (
    data_loader,
    data_splitter,
    model_trainer as base_model_trainer,
    model_builder as base_model_builder,
)
from screendl.utils.drug_selectors import SELECTORS, DrugSelectorType
from screendl.utils import evaluation as eval_utils
from screendl.utils import model_utils
from screendl.utils import data_utils

if t.TYPE_CHECKING:
    from omegaconf import DictConfig
    from cdrpy.feat.encoders import PandasEncoder


log = logging.getLogger(__name__)


def get_id_prefix(pdmc_id: str) -> str:
    """Gets matching prefixes to avoid data leak."""
    if pdmc_id.startswith("HCI"):
        return pdmc_id[:6]
    elif pdmc_id.startswith("BCM"):
        return pdmc_id[:7]
    elif pdmc_id.startswith("TOW"):
        return pdmc_id[:5]
    else:
        return pdmc_id


def loo_split_generator(
    D: Dataset,
) -> t.Generator[t.Tuple[Dataset, Dataset], None, None]:
    """Generates leave-one-pdmc-out splits for model evaluation.

    Parameters
    ----------
    D : Dataset
        The PDMC dataset to split.

    Returns
    -------
    t.Generator[t.Tuple[Dataset, Dataset], None, None]
        Generator of (train, test) dataset tuples

    Yields
    ------
    Iterator[t.Generator[t.Tuple[Dataset, Dataset], None, None]]
        Iterator of (train, test) dataset tuples
    """
    pdmc_ids = set(D.cell_ids)
    for test_id in pdmc_ids:
        id_prefix = get_id_prefix(test_id)
        train_ids = [x for x in pdmc_ids if not str(x).startswith(id_prefix)]
        train_ds = D.select_cells(train_ids, name="train")
        test_ds = D.select_cells([test_id], name="test")
        yield train_ds, test_ds


def get_screenahead_split(
    dataset: Dataset,
    drug_selector: DrugSelectorType,
    num_drugs: int,
    exclude_drugs: t.List[str] | None = None,
) -> t.Tuple[Dataset, Dataset]:
    """"""
    drug_choices = set(dataset.drug_ids)
    if exclude_drugs is not None:
        drug_choices = set(x for x in drug_choices if x not in exclude_drugs)
    screen_drugs = drug_selector.select(num_drugs, choices=drug_choices)
    holdout_drugs = drug_choices.difference(screen_drugs)

    screen_ds = dataset.select_drugs(screen_drugs, name="this_screen")
    holdout_ds = dataset.select_drugs(holdout_drugs, name="this_holdout")

    return screen_ds, holdout_ds


def apply_pdmc_drug_filters(
    cfg: DictConfig, pdmc_ds: Dataset, cell_ds: Dataset | t.Iterable[Dataset]
) -> Dataset:
    """Filters drugs based on configured parameters."""
    if isinstance(cell_ds, Dataset):
        keep_drugs = set(cell_ds.drug_ids)
    elif isinstance(cell_ds, list):
        keep_drugs = set()
        for D in cell_ds:
            keep_drugs = keep_drugs.union(D.drug_ids)

    if cfg.experiment.keep_pdmc_only_drugs is not None:
        if cfg.experiment.keep_pdmc_only_drugs == "all":
            keep_drugs = keep_drugs.union(pdmc_ds.drug_ids)
        elif isinstance(cfg.experiment.keep_pdmc_only_drugs, ListConfig):
            keep_drugs = keep_drugs.union(cfg.experiment.keep_pdmc_only_drugs)
        else:
            raise TypeError(
                "Unsupported parameter type (experiment.keep_pdmc_only_drugs)"
            )

    if cfg.experiment.min_pdmcs_per_drug is not None:
        pdmcs_per_drug = pdmc_ds.obs["drug_id"].value_counts()
        drugs_with_min_pdmcs = pdmcs_per_drug[
            pdmcs_per_drug >= cfg.experiment.min_pdmcs_per_drug
        ]
        keep_drugs = keep_drugs.intersection(drugs_with_min_pdmcs.index)

    return pdmc_ds.select_drugs(keep_drugs, name=pdmc_ds.name)


def data_preprocessor(
    cfg: DictConfig,
    train_cell_ds: Dataset,
    val_cell_ds: Dataset,
    pdmc_ds: Dataset,
) -> t.Tuple[Dataset, Dataset, Dataset]:
    """Preprocesses and split the raw data.

    Parameters
    ----------
    cfg : DictConfig
        Hidra experiment config (ignored)
    train_cell_ds : Dataset
        The cell line (source) training dataset
    val_cell_ds : Dataset
        The cell line validation dataset (used for early stopping)
    pdmc_ds : Dataset
        The PDMC dataset

    Returns
    -------
    t.Tuple[Dataset, Dataset, Dataset]
        A tuple containing the processed dataset objects.
    """

    train_cell_ids = list(set(train_cell_ds.cell_ids))
    val_cell_ids = list(set(val_cell_ds.cell_ids))
    pdmc_ids = list(set(pdmc_ds.cell_ids))

    # 1. normalize the gene expression
    exp_enc: PandasEncoder = train_cell_ds.cell_encoders["exp"]

    x_cell_train = exp_enc.data.loc[train_cell_ids]
    x_cell_val = exp_enc.data.loc[val_cell_ids]
    x_pdmc = exp_enc.data.loc[pdmc_ids]

    ss = StandardScaler().fit(x_cell_train)
    x_cell_train[:] = ss.transform(x_cell_train)
    x_cell_val[:] = ss.transform(x_cell_val)
    x_pdmc[:] = ss.transform(x_pdmc)

    exp_enc.data = pd.concat([x_cell_train, x_cell_val, x_pdmc])

    # FIXME: this could be done external to this function
    pdmc_ds = apply_pdmc_drug_filters(cfg, pdmc_ds, [train_cell_ds, val_cell_ds])

    # normalize the drug responses
    train_cell_ds, val_cell_ds, _ = normalize_responses(
        train_cell_ds, val_cell_ds, norm_method="grouped"
    )
    pdmc_ds, *_ = normalize_responses(pdmc_ds, norm_method="grouped")

    return train_cell_ds, val_cell_ds, pdmc_ds


def safe_lconfig_as_tuple(item: t.Any) -> t.Any:
    """Converts ListConfig instances to tuples or does nothing."""
    return tuple(item) if isinstance(item, ListConfig) else item


@hydra.main(
    version_base=None,
    config_path="../../conf/runners",
    config_name="pdxo_validation",
)
def run(cfg: DictConfig) -> None:
    """Runs leave-one-out cross validation for the HCI PDMC dataset.

    Parameters
    ----------
    cfg : DictConfig
        Experiment configuration.
    """
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    log.info(f"Loading {dataset_name}...")
    dataset = data_loader(cfg)
    all_drug_ids = list(dataset.drug_encoders["mol"].keys())

    log.info(f"Splitting {dataset_name}...")
    D_t_cell, D_v_cell, D_pdxo = data_splitter(cfg, dataset)

    log.info(f"Preprocessing {dataset_name}...")
    D_t_cell, D_v_cell, D_pdxo = data_preprocessor(cfg, D_t_cell, D_v_cell, D_pdxo)

    log.info(f"Building {model_name}...")
    M_base = base_model_builder(cfg, D_t_cell)

    log.info(f"Pretraining {model_name}...")
    M_base = base_model_trainer(cfg, M_base, D_t_cell, D_v_cell)

    log.info("Running experiment...")
    hp_tune = cfg.xfer.hyper
    hp_screen = cfg.screenahead.hyper
    opt_screen = cfg.screenahead.opt

    W_base = M_base.get_weights()
    split_gen = loo_split_generator(D_pdxo)

    results = []
    for D_t_pdxo, D_e_pdxo in tqdm(split_gen, total=D_pdxo.n_cells):
        # create background dataset against which we normalize the predictions
        D_bg_pdxo_full = data_utils.expand_dataset(
            D_t_pdxo,
            cell_ids=list(set(D_t_pdxo.cell_ids)),
            drug_ids=all_drug_ids,
        )

        # create the evaluation dataset (all drugs for the tumor of interest)
        D_e_pdxo_full = data_utils.expand_dataset(
            D_e_pdxo,
            cell_ids=[D_e_pdxo.cell_ids[0]],
            drug_ids=all_drug_ids,
        )

        results.append(
            eval_utils.get_predictions_vs_background(
                M=M_base,
                D_t=D_e_pdxo_full,
                D_b=D_bg_pdxo_full,
                W_t=None,
                W_b=None,
                model="base",
                was_screened=False,
            )
        )

        # initialize the drug selector
        drug_selector = SELECTORS[opt_screen.selector](
            dataset.select_cells(D_t_pdxo.cell_ids).select_drugs(D_pdxo.drug_ids),
            na_threshold=opt_screen.na_thresh,
            seed=opt_screen.seed,
        )

        M_tune = model_utils.fit_transfer_model(
            M_base,
            D_t_pdxo,
            batch_size=hp_tune.batch_size,
            epochs=hp_tune.epochs,
            learning_rate=hp_tune.learning_rate,
            weight_decay=hp_tune.weight_decay,
            frozen_layer_prefixes=safe_lconfig_as_tuple(hp_tune.frozen_layer_prefixes),
            frozen_layer_names=safe_lconfig_as_tuple(hp_tune.frozen_layer_names),
        )

        W_tune = M_tune.get_weights()
        results.append(
            eval_utils.get_predictions_vs_background(
                M=M_tune,
                D_t=D_e_pdxo_full,
                D_b=D_bg_pdxo_full,
                W_t=None,
                W_b=None,
                model="xfer",
                was_screened=False,
            )
        )

        D_s_pdxo, _ = get_screenahead_split(
            D_e_pdxo,
            drug_selector=drug_selector,
            num_drugs=opt_screen.n_drugs,
            exclude_drugs=opt_screen.exclude_drugs,
        )

        M_screen = model_utils.fit_screenahead_model(
            M_tune,
            D_s_pdxo,
            batch_size=hp_screen.batch_size,
            epochs=hp_screen.epochs,
            learning_rate=hp_screen.learning_rate,
            frozen_layer_prefixes=safe_lconfig_as_tuple(hp_screen.frozen_layer_prefixes),
            frozen_layer_names=safe_lconfig_as_tuple(hp_screen.frozen_layer_names),
            training=False,
        )

        W_screen = M_screen.get_weights()
        R_screen = eval_utils.get_predictions_vs_background(
            M=M_screen,
            D_t=D_e_pdxo_full,
            D_b=D_bg_pdxo_full,
            W_t=W_screen if cfg.experiment.background_correction else None,
            W_b=W_tune if cfg.experiment.background_correction else None,
            model="screen",
        )
        R_screen["was_screened"] = R_screen["drug_id"].isin(D_s_pdxo.drug_ids)
        results.append(R_screen)

        M_base.set_weights(W_base)

    results_df = pd.concat(results)
    results_df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    run()
    K.clear_session()
