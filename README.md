# Deep Learning-Based Cancer Drug Response Prediction with ScreenDL

## Table of Contents

* [Requirements](#requirements)
* [Installation](#installation)
* [Running ScreenDL](#running-screendl)
    * [Data Preparation](#data-preparation)
    * [Training and Evaluation](#training-and-evaluation)
    * [Running with ScreenAhead](#running-with-screenahead)
    * [Hyperparameter Optimization](#hyperparameter-optimization)
* [Running with IMPROVE](#running-with-improve)
* [Benchmarking Experiments](#benchmarking-experiments)
* [Citing](#citing)
* [References](#references)

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
- deepchem
- cdrpy
- scipy

Most of these packages will be installed automatically, however, cdrpy currently needs to be installed manually. For cdrpy install instructions visit https://github.com/csederman/cdrpy.

## Installation

### Install from GitHub source

First, clone the ScreenDL repository using git:

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

### Training and Evaluation

### Running with ScreenAhead

### Hyperparameter Optimization

## Running with IMPROVE

## Benchmarking Experiments

### HiDRA

1. clone repo and checkout the branch `cdrpy-benchmarking`

### DualGCN

1. clone repo and checkout the branch `cdrpy-benchmarking`


## Datasets & Inputs

TODO: make data directory in the same directory as this repo and symlink files to avoid confusing paths

1. make_dataset.sh (wrapper around make_dataset.py)
2. make_labels.sh (wrapper around make_labels.py)
3. make_folds.sh (wrapper around make_folds.py)
4. make_inputs.sh (wrapper around make_inputs.py)

## Citing

## References

- HiDRA
- DualGCN
- DeepCDR
