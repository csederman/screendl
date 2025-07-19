"""ScreenDL core pipelines."""

from __future__ import annotations

import random
import pickle

import typing as t
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(1771)
random.seed(1771)
tf.random.set_seed(1771)

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from omegaconf import OmegaConf

from cdrpy.data.preprocess import GroupStandardScaler
from cdrpy.data.datasets import Dataset
from cdrpy.mapper import BatchedResponseGenerator

from screendl.model import (
    load_drug_features,
    load_cell_features,
    create_model,
    train_model,
)
from screendl.utils.evaluation import make_pred_df, get_eval_metrics, ScoreDict
from ..basic.screendl import data_splitter as default_data_splitter

if t.TYPE_CHECKING:
    from omegaconf import DictConfig
    from cdrpy.feat.encoders import PandasEncoder


def load_dataset(cfg: DictConfig) -> Dataset:
    """"""
    paths = cfg.dataset.sources

    drug_encoders = load_drug_features(paths.drug.mol)
    cell_encoders = load_cell_features(paths.cell.exp)
    cell_meta = pd.read_csv(paths.cell.meta, index_col=0)
    drug_meta = pd.read_csv(paths.drug.meta, index_col=0)

    return Dataset.from_csv(
        paths.labels,
        cell_encoders=cell_encoders,
        drug_encoders=drug_encoders,
        cell_meta=cell_meta,
        drug_meta=drug_meta,
        name=cfg.dataset.name,
    )


def split_dataset(
    cfg: DictConfig, D: Dataset
) -> t.Tuple[Dataset, t.Union[Dataset, None], t.Union[Dataset, None]]:
    """Split the dataset or do nothing."""
    if cfg.pretrain.full_dataset_mode:
        # pretrain on the full dataset in single run mode
        return D, None, None
    # slit into train/val/test sets
    return default_data_splitter(cfg, D)


def preprocess_dataset(
    cfg: DictConfig,
    Dt: Dataset,
    Dv: Dataset | None,
    De: Dataset | None,
) -> t.Tuple[Dataset, Dataset | None, Dataset | None]:
    """"""

    # normalize the gene expression
    exp_enc: PandasEncoder = Dt.cell_encoders["exp"]
    exp_scaler = StandardScaler()
    exp_enc.data[:] = exp_scaler.fit_transform(exp_enc.data.values)

    with open("exp_scaler.pkl", "wb") as fh:
        pickle.dump(exp_scaler, fh)

    # normalize the responses
    resp_scaler_t = GroupStandardScaler()
    Dt.obs["label"] = resp_scaler_t.fit_transform(
        Dt.obs[["label"]], groups=Dt.obs["drug_id"]
    )

    with open("resp_scaler.t.pkl", "wb") as fh:
        pickle.dump(resp_scaler_t, fh)

    if Dv is not None:
        Dv.obs["label"] = resp_scaler_t.transform(
            Dv.obs[["label"]], groups=Dv.obs["drug_id"]
        )

    resp_scaler_e = resp_scaler_t
    if De is not None:
        if cfg.pretrain.independent_norm:
            resp_scaler_e = GroupStandardScaler().fit(
                De.obs[["label"]], groups=De.obs["drug_id"]
            )
        De.obs["label"] = resp_scaler_e.transform(
            De.obs[["label"]], groups=De.obs["drug_id"]
        )

    with open("resp_scaler.e.pkl", "wb") as fh:
        pickle.dump(resp_scaler_e, fh)

    return Dt, Dv, De


def build_model_from_config(cfg: DictConfig, exp_dim: int, mol_dim: int) -> keras.Model:
    """"""
    hparams = cfg.model.hyper
    return create_model(
        exp_dim=exp_dim,
        mol_dim=mol_dim,
        exp_hidden_dims=hparams.hidden_dims.exp,
        mol_hidden_dims=hparams.hidden_dims.mol,
        shared_hidden_dims=hparams.hidden_dims.shared,
        use_noise=hparams.use_noise,
        noise_stddev=hparams.noise_stddev,
        activation=hparams.activation,
    )


def pretrain_model_from_config(
    cfg: DictConfig, model: keras.Model, Dt: Dataset, Dv: Dataset | None
) -> None:
    """"""
    hparams = cfg.model.hyper
    optimizer = keras.optimizers.Adam(
        hparams.learning_rate, weight_decay=hparams.weight_decay
    )
    return train_model(
        model,
        optimizer,
        train_ds=Dt,
        val_ds=Dv,
        batch_size=hparams.batch_size,
        epochs=hparams.epochs,
        save_dir=("." if cfg.model.io.save is True else None),
        early_stopping=hparams.early_stopping,
    )


def evaluate_model(
    cfg: DictConfig, model: keras.Model, datasets: t.Iterable[Dataset]
) -> t.Tuple[pd.DataFrame, t.Dict[str, ScoreDict]]:
    """"""
    meta_dict = {
        "model": cfg.model.name,
        "split_id": cfg.dataset.split.id,
        "split_type": cfg.dataset.split.name,
    }

    pred_dfs = []
    scores = {}
    for D in datasets:
        gen = BatchedResponseGenerator(D, cfg.model.hyper.batch_size)
        pred_df = make_pred_df(
            D,
            model.predict(gen.flow_from_dataset(D), verbose=0),
            split_group=D.name,
            **meta_dict,
        )
        pred_dfs.append(pred_df)
        scores[D.name] = get_eval_metrics(pred_df)

    pred_df = pd.concat(pred_dfs)

    return pred_df, scores


def apply_preprocessing_pipeline(
    root_dir: str | Path, Dt: Dataset, Dv: Dataset | None, De: Dataset | None
) -> t.Tuple[Dataset, Dataset | None, Dataset | None]:
    """"""
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)

    with open(root_dir / "exp_scaler.pkl", "rb") as fh:
        exp_scaler: StandardScaler = pickle.load(fh)

    with open(root_dir / "resp_scaler.t.pkl", "rb") as fh:
        resp_scaler_t: GroupStandardScaler = pickle.load(fh)

    with open(root_dir / "resp_scaler.e.pkl", "rb") as fh:
        resp_scaler_e: GroupStandardScaler = pickle.load(fh)

    Dt.cell_encoders["exp"].data[:] = exp_scaler.transform(
        Dt.cell_encoders["exp"].data.values
    )
    Dt.obs["label"] = resp_scaler_t.transform(Dt.obs[["label"]], groups=Dt.obs["drug_id"])

    if Dv is not None:
        Dv.obs["label"] = resp_scaler_t.transform(
            Dv.obs[["label"]], groups=Dv.obs["drug_id"]
        )

    if De is not None:
        De.obs["label"] = resp_scaler_e.transform(
            De.obs[["label"]], groups=De.obs["drug_id"]
        )

    return Dt, Dv, De


def load_pretraining_configs(root_dir: str | Path) -> t.Tuple[DictConfig, DictConfig]:
    """"""
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    pretrain_cfg = OmegaConf.load(root_dir / ".hydra/config.yaml")
    pretrain_hydra_cfg = OmegaConf.load(root_dir / ".hydra/hydra.yaml")
    return pretrain_cfg, pretrain_hydra_cfg


def load_pretrained_model(root_dir: str | Path) -> keras.Model:
    """"""
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    return keras.models.load_model(root_dir / "ScreenDL-PT.model")


def load_finetuned_model(root_dir: str | Path) -> keras.Model:
    """"""
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    return keras.models.load_model(root_dir / "ScreenDL-FT.model")
