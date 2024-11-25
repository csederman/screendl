#!/usr/bin/env python
"""Runs transfer learning experiments on the HCI PDMC dataset."""

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


@hydra.main(
    version_base=None, config_path="../../conf/runners", config_name="pdxo_validation"
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

    log.info(f"Splitting {dataset_name}...")
    train_cell_ds, val_cell_ds, pdmc_ds = data_splitter(cfg, dataset)

    log.info(f"Preprocessing {dataset_name}...")
    train_cell_ds, val_cell_ds, pdmc_ds = data_preprocessor(
        cfg, train_cell_ds, val_cell_ds, pdmc_ds
    )

    print(pdmc_ds)

    log.info(f"Building {model_name}...")
    base_model = base_model_builder(cfg, train_cell_ds)

    log.info(f"Pretraining {model_name}...")
    base_model = base_model_trainer(cfg, base_model, train_cell_ds, val_cell_ds)
    base_result = eval_utils.get_preds_vs_background(
        base_model, pdmc_ds, pdmc_ds, model="base", was_screened=False
    )

    log.info("Running experiment...")

    hp_tune = cfg.xfer.hyper
    hp_screen = cfg.screenahead.hyper
    opt_screen = cfg.screenahead.opt

    split_gen = loo_split_generator(pdmc_ds)
    base_weights = base_model.get_weights()

    selector = SELECTORS[opt_screen.selector](
        dataset.select_cells(set(train_cell_ds.cell_ids).union(val_cell_ds.cell_ids)),
        seed=opt_screen.seed,
        na_threshold=0.8,
    )

    results = []
    for train_ds, test_ds in tqdm(split_gen, total=pdmc_ds.n_cells):
        screen_ds, _ = get_screenahead_split(
            test_ds,
            selector,
            opt_screen.n_drugs,
            exclude_drugs=opt_screen.exclude_drugs,
        )

        # 1. run screenahead without initial fine-tuning
        sa_model = model_utils.fit_screenahead_model(
            base_model,
            screen_ds,
            batch_size=hp_screen.batch_size,
            epochs=hp_screen.epochs,
            learning_rate=hp_screen.learning_rate,
            frozen_layer_prefixes=(
                tuple(hp_screen.frozen_layer_prefixes)
                if isinstance(hp_screen.frozen_layer_prefixes, ListConfig)
                else hp_screen.frozen_layer_prefixes
            ),
            frozen_layer_names=(
                tuple(hp_screen.frozen_layer_names)
                if isinstance(hp_screen.frozen_layer_names, ListConfig)
                else hp_screen.frozen_layer_names
            ),
            training=False,
        )

        sa_result = eval_utils.get_preds_vs_background(
            sa_model, test_ds, train_ds, model="screen (no-fine-tune)"
        )
        sa_result["was_screened"] = sa_result["drug_id"].isin(screen_ds.drug_ids)
        results.append(sa_result)

        # 2. run the normal fine-tune + ScreenAhead stuff
        base_model.set_weights(base_weights)

        tune_model = model_utils.fit_transfer_model(
            base_model,
            train_ds,
            batch_size=hp_tune.batch_size,
            epochs=hp_tune.epochs,
            learning_rate=hp_tune.learning_rate,
            weight_decay=hp_tune.weight_decay,
            frozen_layer_prefixes=(
                tuple(hp_tune.frozen_layer_prefixes)
                if isinstance(hp_tune.frozen_layer_prefixes, ListConfig)
                else hp_tune.frozen_layer_prefixes
            ),
            frozen_layer_names=(
                tuple(hp_tune.frozen_layer_names)
                if isinstance(hp_tune.frozen_layer_names, ListConfig)
                else hp_tune.frozen_layer_names
            ),
        )

        results.append(
            eval_utils.get_preds_vs_background(
                tune_model, test_ds, train_ds, model="xfer", was_screened=False
            )
        )

        sa_model = model_utils.fit_screenahead_model(
            tune_model,
            screen_ds,
            batch_size=hp_screen.batch_size,
            epochs=hp_screen.epochs,
            learning_rate=hp_screen.learning_rate,
            frozen_layer_prefixes=(
                tuple(hp_screen.frozen_layer_prefixes)
                if isinstance(hp_screen.frozen_layer_prefixes, ListConfig)
                else hp_screen.frozen_layer_prefixes
            ),
            frozen_layer_names=(
                tuple(hp_screen.frozen_layer_names)
                if isinstance(hp_screen.frozen_layer_names, ListConfig)
                else hp_screen.frozen_layer_names
            ),
            training=False,
        )

        sa_result = eval_utils.get_preds_vs_background(
            sa_model, test_ds, train_ds, model="screen (fine-tune)"
        )
        sa_result["was_screened"] = sa_result["drug_id"].isin(screen_ds.drug_ids)

        results.append(sa_result)

        base_model.set_weights(base_weights)

    results = pd.concat([base_result, *results])
    results.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    run()
    K.clear_session()
