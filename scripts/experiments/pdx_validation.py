#!/usr/bin/env python
"""Runs transfer learning experiments on the HCI PDMC dataset."""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"

import hydra
import logging
import itertools

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


def create_background_dataset(
    pdmc_ds: Dataset, drug_ids: t.List[str], cell_ids: t.List[str]
) -> Dataset:
    """Creates a background dataset to predict against."""
    obs = list(itertools.product(cell_ids, drug_ids))
    obs = pd.DataFrame(obs, columns=["cell_id", "drug_id"])
    obs = obs.assign(id=range(len(obs)), label=1)

    return Dataset(
        obs,
        cell_meta=pdmc_ds.cell_meta,
        drug_meta=pdmc_ds.drug_meta,
        cell_encoders=pdmc_ds.cell_encoders,
        drug_encoders=pdmc_ds.drug_encoders,
        name="background-dataset",
    )


def get_screenahead_dataset(
    dataset: Dataset,
    mode: t.Literal["screen-all", "screen-selected"],
    drug_selector: DrugSelectorType,
    num_drugs: int,
    exclude_drugs: t.List[str] | None = None,
) -> Dataset:
    """"""
    if mode == "screen-all":
        screen_ds = dataset
    elif mode == "screen-selected":
        screen_ds, _ = get_screenahead_split(
            dataset, drug_selector=drug_selector, num_drugs=num_drugs
        )
    elif mode == "screen-non-pdx":
        screen_ds, _ = get_screenahead_split(
            dataset,
            drug_selector=drug_selector,
            num_drugs=num_drugs,
            exclude_drugs=exclude_drugs,
        )
    else:
        raise ValueError(f"Invalid screenahead mode (got {mode})")

    return screen_ds


def load_pdx_data(cfg: DictConfig, pdmc_ds: Dataset) -> Dataset:
    """Loads the raw PDX screening data."""
    pdx_obs = pd.read_csv(cfg.experiment.pdx_obs_path)
    pdx_obs = pdx_obs[pdx_obs["cell_id"].isin(pdmc_ds.cell_ids)]
    pdx_obs = pdx_obs[pdx_obs["drug_id"].isin(pdmc_ds.drug_ids)]
    pdx_obs["label"] = pdx_obs["mRECIST"].isin(["CR", "PR", "SD"]).astype(int)

    pdx_dataset = Dataset(
        pdx_obs,
        cell_encoders=pdmc_ds.cell_encoders,
        drug_encoders=pdmc_ds.drug_encoders,
        name="pdx_ds",
    )

    if cfg.experiment.pdx_ids is not None:
        pdx_dataset = pdx_dataset.select_cells(cfg.experiment.pdx_ids)

    return pdx_dataset


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

    pdmc_ds = apply_pdmc_drug_filters(cfg, pdmc_ds, [train_cell_ds, val_cell_ds])

    # normalize the drug responses
    train_cell_ds, val_cell_ds, _ = normalize_responses(
        train_cell_ds, val_cell_ds, norm_method="grouped"
    )
    pdmc_ds, *_ = normalize_responses(pdmc_ds, norm_method="grouped")

    return train_cell_ds, val_cell_ds, pdmc_ds


@hydra.main(
    version_base=None, config_path="../../conf/runners", config_name="pdx_validation"
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

    pdx_ds = load_pdx_data(cfg, pdmc_ds)
    pdx_ids = sorted(list(set(pdx_ds.cell_ids)))
    pdx_drug_ids = sorted(list(set(pdx_ds.drug_ids)))

    target_drugs = (
        set(pdx_ds.drug_ids).union(train_cell_ds.drug_ids).union(val_cell_ds.drug_ids)
    )
    pdmc_ds = pdmc_ds.select_drugs(target_drugs, name="pdmc_ds")

    log.info(f"Building {model_name}...")
    base_model = base_model_builder(cfg, train_cell_ds)

    # dataset for early stopping
    train_pdmc_ids = [x for x in set(pdmc_ds.cell_ids) if x not in pdx_ids]
    train_pdmc_ds = pdmc_ds.select_cells(train_pdmc_ids, name="pdmc_train_ds")

    log.info(f"Pretraining {model_name}...")
    base_model = base_model_trainer(cfg, base_model, train_cell_ds, train_pdmc_ds)
    # base_model = base_model_trainer(cfg, base_model, train_cell_ds, val_cell_ds)

    base_pdxo_result = eval_utils.get_preds_vs_background(
        base_model, pdmc_ds, pdmc_ds, model="base", was_screened=False
    )

    # background_ds = create_background_dataset(
    #     pdmc_ds,
    #     drug_ids=sorted(list(set(pdmc_ds.drug_ids))),
    #     cell_ids=train_pdmc_ids,
    # )

    # base_pdx_result = eval_utils.get_preds_vs_background(
    #     base_model,
    #     D_target=pdx_ds,
    #     D_background=train_pdmc_ds.select_drugs(pdx_drug_ids),
    #     # D_background=background_ds,
    #     model="base",
    #     was_screened=False,
    # )

    log.info("Running experiment...")

    hp_tune = cfg.xfer.hyper
    hp_screen = cfg.screenahead.hyper
    opt_screen = cfg.screenahead.opt

    base_weights = base_model.get_weights()
    split_gen = loo_split_generator(pdmc_ds)

    selector = SELECTORS[opt_screen.selector](
        dataset.select_cells(set(train_cell_ds.cell_ids).union(val_cell_ds.cell_ids)),
        seed=opt_screen.seed,
        na_threshold=0.8,
    )

    pdx_results = []
    pdxo_results = []
    for train_pdxo_ds, test_pdxo_ds in tqdm(split_gen, total=pdmc_ds.n_cells):

        test_pdx_ds = pdx_ds.select_cells(set(test_pdxo_ds.cell_ids))

        background_pdxo_ds = train_pdxo_ds.select_drugs(set(test_pdxo_ds.drug_ids))
        background_pdx_ds = train_pdxo_ds.select_drugs(set(test_pdx_ds.drug_ids))

        if test_pdx_ds.size > 0:
            pdx_results.append(
                eval_utils.get_preds_vs_background(
                    base_model,
                    D_target=test_pdx_ds,
                    D_background=background_pdx_ds,
                    # D_background=background_ds,
                    model="base",
                    was_screened=False,
                )
            )

        tune_model = model_utils.fit_transfer_model(
            base_model,
            train_pdxo_ds,
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
            training=True,
        )

        pdxo_results.append(
            eval_utils.get_preds_vs_background(
                tune_model,
                D_target=test_pdxo_ds,
                D_background=background_pdxo_ds,
                # D_background=background_ds,
                model="xfer",
                was_screened=False,
            )
        )

        if test_pdx_ds.size > 0:
            pdx_results.append(
                eval_utils.get_preds_vs_background(
                    tune_model,
                    D_target=test_pdx_ds,
                    D_background=background_pdx_ds,
                    # D_background=background_ds,
                    model="xfer",
                    was_screened=False,
                )
            )

        screen_pdxo_ds = get_screenahead_dataset(
            test_pdxo_ds,
            opt_screen.mode,
            drug_selector=selector,
            num_drugs=opt_screen.n_drugs,
            exclude_drugs=(
                pdx_drug_ids
                if opt_screen.mode == "screen-non-pdx"
                else opt_screen.exclude_drugs
            ),
        )

        sa_model = model_utils.fit_screenahead_model(
            tune_model,
            screen_pdxo_ds,
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

        sa_pdxo_result = eval_utils.get_preds_vs_background(
            sa_model,
            D_target=test_pdxo_ds,
            D_background=background_pdxo_ds,
            # D_background=background_ds,
            model="screen",
        )
        sa_pdxo_result["was_screened"] = sa_pdxo_result["drug_id"].isin(
            screen_pdxo_ds.drug_ids
        )
        pdxo_results.append(sa_pdxo_result)

        if test_pdx_ds.size > 0:
            sa_pdx_result = eval_utils.get_preds_vs_background(
                sa_model,
                D_target=test_pdx_ds,
                D_background=background_pdx_ds,
                # D_background=background_ds,
                model="screen",
            )
            sa_pdx_result["was_screened"] = sa_pdx_result["drug_id"].isin(
                screen_pdxo_ds.drug_ids
            )
            pdx_results.append(sa_pdx_result)

        base_model.set_weights(base_weights)

    pdxo_results = pd.concat([base_pdxo_result, *pdxo_results])
    pdxo_results.to_csv("predictions_pdxo.csv", index=False)

    # pdx_results = pd.concat([base_pdx_result, *pdx_results])
    pdx_results = pd.concat(pdx_results)
    pdx_results.to_csv("predictions_pdx.csv", index=False)


if __name__ == "__main__":
    run()
    K.clear_session()
