# Deep Learning-Based Cancer Drug Response Prediction with ScreenDL

## Contents

* [Requirements](#requirements)
* [Installation](#installation)
* [Running ScreenDL](#running-screendl)
    * [Data Preparation](#data-preparation)
    * [Training and Evaluation](#training-and-evaluation)
    * [Running with ScreenAhead](#running-with-screenahead)
* [Running with IMPROVE](#running-with-improve)
* [Benchmarking Experiments](#benchmarking-experiments)
* [Citing ScreenDL](#citing-screendl)
* [References](#references)
    * [Models](#models)
    * [Datasets](#datasets)

## Requirements

ScreenDL currently supports Python 3.8 and requires the following packages:
- numpy (>= 1.21)
- pandas (>= 2.0.3)
- openpyxl (== 3.1.2)
- tensorflow
- tensorflow-probability
- scikit-learn
- omegaconf
- tqdm
- rdkit
- deepchem (== 2.7.1)
- cdrpy
- scipy
- inmoose (== 0.2.1)

Most of these packages are installed automatically, however, cdrpy requires manual installation. For cdrpy install instructions visit https://github.com/csederman/cdrpy.

## Installation

### Install from GitHub source

First, clone the ScreenDL repository:

```bash
git clone https://github.com/csederman/screendl.git
```

Then, `cd` into the screendl directory and install the library:

```bash
cd screendl
pip install .
```

ScreenDL provides a number of scripts for running various experiments. To install the additional dependencies required for running these scripts run:

```bash
pip install .[scripts]
```

## Running ScreenDL

### Data Preparation

All datasets used to train and evaluate ScreenDL are available as `tar.gz` archives in the `cdrpy-data` repo.

### Training and Evaluation

To perform basic training and evaluation of ScreenDL, run:

```{bash}
python scripts/runners/run.py
```

### Running with ScreenAhead

To train and evaluate ScreenDL with ScreenAhead tumor-specific fine-tuning, run:

```{bash}
python scripts/runners/run_sa.py
```

## Running with IMPROVE

## Benchmarking Experiments

Note that running benchmarking experiments requires the additional dependencies installed by running `pip install screendl[scripts]`.

### HiDRA

Clone the HiDRA repo and checkout the `cdrpy-benchmarking` branch.

```{bash}
git clone https://github.com/csederman/HiDRA.git && cd HiDRA
git checkout cdrpy-benchmarking
```

To train HiDRA on a single fold:

```{bash}
HIDRA_ROOT="<path to HiDRA repo>" python scripts/runnners/run.py \
    model=HiDRA-legacy \
    dataset.preprocess.norm=global
```

To run HiDRA on all training folds, use:

```{bash}
HIDRA_ROOT="<path to HiDRA repo>" python scripts/runnners/run.py -m \
    model=HiDRA-legacy \
    dataset.preprocess.norm=global
```

For more information on HiDRA, checkout the original publication: [HiDRA: Hierarchical Network for Drug Response Prediction with Attention](https://doi.org/10.1021/acs.jcim.1c00706)

### DualGCN

Clone the DualGCN repo and checkout the `cdrpy-benchmarking` branch

```{bash}
git clone https://github.com/csederman/DualGCN.git && cd DualGCN
git checkout cdrpy-benchmarking
```

To train DualGCN on a single fold:

```{bash}
DUALGCN_ROOT="/<path to DualGCN repo>/code" python scripts/runnners/run.py \
    model=DualGCN-legacy \
    dataset.preprocess.norm=global
```

To run DualGCN on all training folds, use:

```{bash}
DUALGCN_ROOT="/<path to DualGCN repo>/code" python scripts/runnners/run.py -m \
    model=DualGCN-legacy \
    dataset.preprocess.norm=global
```

For more information on DualGCN, checkout the original publication: [DualGCN: a dual graph convolutional network model to predict cancer drug response](https://doi.org/10.1186/s12859-022-04664-4)

### DeepCDR

Clone the DeepCDR repo and checkout the `cdrpy-benchmarking` branch

```{bash}
git clone https://github.com/csederman/DeepCDR.git && cd DeepCDR
git checkout cdrpy-benchmarking
```

To train DeepCDR on a single fold:

```{bash}
DEEPCDR_ROOT="/<path to DeepCDR repo>/prog python scripts/runnners/run.py \
    model=DeepCDR-legacy \
    dataset.preprocess.norm=global
```

To run DeepCDR on all training folds, use:

```{bash}
DEEPCDR_ROOT="/<path to DeepCDR repo>/prog python -m scripts/runnners/run.py \
    model=DeepCDR-legacy \
    dataset.preprocess.norm=global
```

For more information on DeepCDR, checkout the original publication: [DeepCDR: a hybrid graph convolutional network for predicting cancer drug response](https://doi.org/10.1093/bioinformatics/btaa822)

## Datasets & Inputs

TODO: make data directory in the same directory as this repo and symlink files to avoid confusing paths

1. make_dataset.sh (wrapper around make_dataset.py)
2. make_labels.sh (wrapper around make_labels.py)
3. make_folds.sh (wrapper around make_folds.py)
4. make_inputs.sh (wrapper around make_inputs.py)

## Citing ScreenDL

## References

### Models

[1]: Qiao Liu, Zhiqiang Hu, Rui Jiang, Mu Zhou, DeepCDR: a hybrid graph convolutional network for predicting cancer drug response, Bioinformatics, Volume 36, Issue Supplement_2, December 2020, Pages i911–i918, [https://doi.org/10.1093/bioinformatics/btaa822](https://doi.org/10.1093/bioinformatics/btaa822)

[2]: Ma, T., Liu, Q., Li, H. et al. DualGCN: a dual graph convolutional network model to predict cancer drug response. BMC Bioinformatics 23 (Suppl 4), 129 (2022). [https://doi.org/10.1186/s12859-022-04664-4](https://doi.org/10.1186/s12859-022-04664-4)

[3]: Jin, I. & Nam, H. HiDRA: Hierarchical Network for Drug Response Prediction with Attention. J. Chem. Inf. Model. 61, 3858–3867 (2021). [https://doi.org/10.1021/acs.jcim.1c00706](https://doi.org/10.1021/acs.jcim.1c00706)

### Datasets

