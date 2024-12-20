# Deep Learning-Based Cancer Drug Response Prediction with ScreenDL

## Contents

* [Requirements](#requirements)
* [Additional Hardware & Software Requirements](#additional-hardware--software-requirements)
* [Installation](#installation)
* [Running ScreenDL](#running-screendl)
    * [Data Preparation](#data-preparation)
    * [Training and Evaluation](#training-and-evaluation)
    * [Running with ScreenAhead](#running-with-screenahead)
* [Benchmarking Experiments](#benchmarking-experiments)
* [Running with IMPROVE](#running-with-improve)
* [Citing ScreenDL](#citing-screendl)
* [References](#references)
    * [Models](#models)
    * [Datasets](#datasets)

## Requirements

ScreenDL was developed using Python 3.9.13 and requires the following packages (see Installation):

- numpy (>= 1.21)
- pandas (>= 2.0.3)
- openpyxl (== 3.1.2)
- tensorflow (== 2.11.1)
- tensorflow-probability (== 0.19.0)
- scikit-learn (== 1.3.0)
- omegaconf (>= 2.2, < 2.4)
- tqdm
- rdkit
- deepchem (== 2.7.1)
- cdrpy
- scipy (== 1.8.1)
- inmoose (== 0.2.1)

## Additional Hardware & Software Requirements

All ScreenDL models were trained on GPU nodes provided by the Utah Center for High Performance Computing equipped with either NVIDIA GTX 1080 Ti GPUs with 3584 CUDA cores and 11 GB GDDR5X memory or NVIDIA A40 GPUs with 10,752 CUDA cores and 48 GB GDDR6 memory using cuda/11.3 and cudnn/8.2.0. *We note that ScreenDL can be trained using standard CPUs and does not require GPU hardware.*

## Installation

Follow the steps below to setup a python virtual environment and install `screendl`. Please note that most of ScreenDL's dependencies are installed automatically, however, `tensorflow`, `tensorflow-probability`, and our `cdrpy` library require manual installation as outlined below (<15 minutes):

1. Setup a new conda environment:

```{bash}
conda create --name screendl-env python=3.9.13
conda activate screendl-env
```

2. Install tensorflow and tensorflow-probability:

```{bash}
pip install tensorflow==2.11.1
pip install tensorflow-probability==0.19.0
```

3. Install cdrpy:

```{bash}
git clone https://github.com/csederman/cdrpy.git && cd cdrpy
pip install --upgrade .
```

4. Install screendl:

```{bash}
cd ..
git clone https://github.com/csederman/screendl.git && cd screendl
pip install --upgrade ".[scripts]"
```

## Running ScreenDL

ScreenDL uses [Hydra](https://hydra.cc/) for config management. Before running ScreenDL, follow the instructions below to update the demo config files and prepare the input data.

### Data Preparation

All datasets used to train and evaluate ScreenDL can be found as `tar.gz` archives under the `data/datasets` directory. To prepare the data for training ScreenDL, enter the `data/datasets` directory (`cd data/datasets`) and unpack that `tar.gz` archives with the following commands:

```{bash}
cd data/datasets
mkdir <dataset name>
tar -xvzf <path to dataset archive> -C <dataset name>
```

For example, to unpack the CellModelPassports-GDSCv1v2 dataset, run:

```{bash}
cd data/datasets
mkdir CellModelPassports-GDSCv1v2
tar -xvzf CellModelPassports-GDSCv1v2.tar.gz -C CellModelPassports-GDSCv1v2
```

### Configuring ScreenDL

ScreenDL's inputs and hyperparameters are configured using [Hydra](https://hydra.cc/). Config files for all scripts can be found under the `conf` subdirectory. **ScreenDL's default configuration assumes that datasets have been extracted into the `data/datasets` directory by following the steps outlined in Data Preparation above.** If you have extracted the data according to the procedure outlined above, no additional configuration is required and you may skip this section and procede directly to Training and Evaluation.

ScreenDL can also be manually configured by updating the necessary file paths. In what follows, we outline the required manual config file updates for basic ScreenDL functionality using the CellModelPassports-GDSCv1v2 dataset as an example:

#### Update the file paths in a given script's config file

- All ScreenDL scripts are configured with a `_datastore_` field that specifies the root directory for ScreenDL's outputs (and optionally inputs).
- This `_datastore_` field must be updated to point to the appropriate directory on the user's system.
- For example, to save ScreenDL's outputs under `/<path to screendl repo>/screendl/data` the corresponding `_datastore_` field for a given config should be set to `/<path to screendl repo>/screendl/data`.
- **This step tells ScreenDL where outputs should be stored.**

#### Update the corresponding dataset config file with the appropriate file paths

- We use Hydra's nested config system to manage the configuration of multiple datasets simultaneously.
- In order for ScreenDL to read a given dataset (i.e., CellModelPassports-GDSCv1v2), the config field `dataset.dir` must point to the location of the corresponding extracted `tar.gz` archive.
- *Note that we recommend using expanded file paths when updating config files.*
- For example, for CellModelPassports-GDSCv1v2, set the `dir` field under `conf/runners/dataset/CellModelPassports-GDSCv1v2.yaml` to point to the root directory of the extracted dataset.
- For example, if you extracted the CellModelPassports-GDSCv1v2 dataset to `/<path to screendl repo>/screendl/data/datasets/CellModelPassports-GDSCv1v2` set the `dir` field in `conf/runners/dataset/CellModelPassports-GDSCv1v2.yaml` to `/<path to screendl repo>/data/datasets/CellModelPassports-GDSCv1v2`.
- **This step tells ScreenDL where input data is stored.**

### Training and Evaluation

Note that, as detailed in **Installation**, running benchmarking experiments requires the additional dependencies installed by running `pip install .[scripts]`. To perform basic training and evaluation of ScreenDL, run:

```{bash}
python scripts/runners/run.py
```

The typical CPU runtime for basic training of ScreenDL is <30min for a single train/test split.

This script will generate several output files under the directories configured above:

1. `predictions.csv`: The raw predictions generated for each tumor-drug pair. This file is used as input for subsequent analyses.
2. `scores.json`: The train/validation/test metrics for the ScreenDL model.

### Running with ScreenAhead

To train and evaluate ScreenDL with **ScreenAhead tumor-specific fine-tuning**, run:

```{bash}
python scripts/runners/run_screenahead.py
```

The typical runtime for training of ScreenDL with ScreenAhead is <30min for a single train/test split.

This script will generate several output files under the directories configured above:

1. `predictions.csv`: The raw predictions generated by ScreenDL for each tumor-drug pair. This file is used as input for subsequent analyses.
1. `predictions_sa.csv`: The raw predictions generated by ScreenDL-SA (ScreenDL with tumor-specific fine-tuning) for each tumor-drug pair. This file is used as input for subsequent analyses.
2. `scores.json`: The train/validation/test metrics for the ScreenDL model.

### Running PDX/PDXO Experiments

#### PDXO Validation

To run the full PDXO validation pipeline, run the following command:

```{bash}
python scripts/experiments/pdxo_validation.py -m
```

This command will train an ensemble of 10 ScreenDL models using different subsets of cell lines and both perform domain-specific fine-tuning and ScreenAhead tumor-specific fine-tuning under leave-one-out cross-validation. To train a single model, run:

```{bash}
python scripts/experiments/pdxo_validation.py
```

#### PDX Validation

To run the PDX validation pipeline, run the following command:

```{bash}
python scripts/experiments/pdx_validation.py -m
```

This command will train an ensemble of 10 ScreenDL models using different subsets of cell lines and perform domain-specific fine-tuning and ScreenAhead tumor-specific fine-tuning under leave-one-out cross-validation. To train a single model, run:

```{bash}
python scripts/experiments/pdx_validation.py
```

## Benchmarking Experiments

Note that running benchmarking experiments requires the additional dependencies installed by running `pip install .[scripts]`.

### HiDRA

#### Cell line benchmarking

Clone the HiDRA repo and checkout the `cdrpy-benchmarking` branch.

```{bash}
git clone https://github.com/csederman/HiDRA.git && cd HiDRA
git checkout cdrpy-benchmarking
```

To train HiDRA on a single fold:

```{bash}
HIDRA_ROOT="<path to HiDRA repo>" python scripts/runnners/run.py \
    model=HiDRA-legacy \
    dataset=CellModelPassports-GDSCv1v2 \
    dataset.preprocess.norm=global
```

To run HiDRA on all training folds, use:

```{bash}
HIDRA_ROOT="<path to HiDRA repo>" python scripts/runnners/run.py -m \
    model=HiDRA-legacy \
    dataset=CellModelPassports-GDSCv1v2 \
    dataset.preprocess.norm=global
```

#### PDXO/PDX benchmarking

```{bash}
HIDRA_ROOT="<path to HiDRA repo>" python scripts/experiments/pdx_benchmarking.py -m \
    model=HiDRA-legacy \
    dataset=CellModelPassports-GDSCv1v2-HCI \
    dataset.preprocess.norm=global
```

#### More about HiDRA

For more information on HiDRA, checkout the original publication: [HiDRA: Hierarchical Network for Drug Response Prediction with Attention](https://doi.org/10.1021/acs.jcim.1c00706)

### DualGCN

#### Cell line benchmarking

Clone the DualGCN repo and checkout the `cdrpy-benchmarking` branch

```{bash}
git clone https://github.com/csederman/DualGCN.git && cd DualGCN
git checkout cdrpy-benchmarking
```

To train DualGCN on a single fold:

```{bash}
DUALGCN_ROOT="/<path to DualGCN repo>/code" python scripts/runnners/run.py \
    model=DualGCN-legacy \
    dataset=CellModelPassports-GDSCv1v2 \
    dataset.preprocess.norm=global
```

To run DualGCN on all training folds, use:

```{bash}
DUALGCN_ROOT="/<path to DualGCN repo>/code" python scripts/runnners/run.py -m \
    model=DualGCN-legacy \
    dataset=CellModelPassports-GDSCv1v2 \
    dataset.preprocess.norm=global
```

#### More about DualGCN

For more information on DualGCN, checkout the original publication: [DualGCN: a dual graph convolutional network model to predict cancer drug response](https://doi.org/10.1186/s12859-022-04664-4)

### DeepCDR

#### Cell line benchmarking

Clone the DeepCDR repo and checkout the `cdrpy-benchmarking` branch

```{bash}
git clone https://github.com/csederman/DeepCDR.git && cd DeepCDR
git checkout cdrpy-benchmarking
```

To train DeepCDR on a single fold:

```{bash}
DEEPCDR_ROOT="/<path to DeepCDR repo>/prog" python scripts/runnners/run.py \
    model=DeepCDR-legacy \
    dataset=CellModelPassports-GDSCv1v2 \
    dataset.preprocess.norm=global
```

To run DeepCDR on all training folds, use:

```{bash}
DEEPCDR_ROOT="/<path to DeepCDR repo>/prog" python scripts/runnners/run.py -m \
    model=DeepCDR-legacy \
    dataset=CellModelPassports-GDSCv1v2 \
    dataset.preprocess.norm=global
```

#### PDXO/PDX benchmarking

```{bash}
DEEPCDR_ROOT="/<path to DeepCDR repo>/prog" python scripts/experiments/pdx_benchmarking.py -m \
    model=DeepCDR-legacy \
    dataset=CellModelPassports-GDSCv1v2-HCI-Mutations \
    dataset.preprocess.norm=global
```

For more information on DeepCDR, checkout the original publication: [DeepCDR: a hybrid graph convolutional network for predicting cancer drug response](https://doi.org/10.1093/bioinformatics/btaa822)

## Running with IMPROVE

## Citing ScreenDL

[1]: Casey Sederman, Chieh-Hsiang Yang, Emilio Cortes-Sanchez, Tony Di Sera, Xiaomeng Huang, Sandra D. Scherer, Ling Zhao, Zhengtao Chu, Eliza R. White, Aaron Atkinson, Jadon Wagstaff, Katherine E. Varley, Michael T. Lewis, Yi Qiao, Bryan E. Welm, Alana L. Welm, Gabor T. Marth, A precision oncology-focused deep learning framework for personalized selection of cancer therapy, bioRxiv 2024.12.12.628190; doi: [https://doi.org/10.1101/2024.12.12.628190](https://doi.org/10.1101/2024.12.12.628190)

## References

### Models

[1]: Qiao Liu, Zhiqiang Hu, Rui Jiang, Mu Zhou, DeepCDR: a hybrid graph convolutional network for predicting cancer drug response, Bioinformatics, Volume 36, Issue Supplement_2, December 2020, Pages i911–i918, [https://doi.org/10.1093/bioinformatics/btaa822](https://doi.org/10.1093/bioinformatics/btaa822)

[2]: Ma, T., Liu, Q., Li, H. et al. DualGCN: a dual graph convolutional network model to predict cancer drug response. BMC Bioinformatics 23 (Suppl 4), 129 (2022). [https://doi.org/10.1186/s12859-022-04664-4](https://doi.org/10.1186/s12859-022-04664-4)

[3]: Jin, I. & Nam, H. HiDRA: Hierarchical Network for Drug Response Prediction with Attention. J. Chem. Inf. Model. 61, 3858–3867 (2021). [https://doi.org/10.1021/acs.jcim.1c00706](https://doi.org/10.1021/acs.jcim.1c00706)

### Datasets

