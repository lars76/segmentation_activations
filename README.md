# Effect of the output activation function in image segmentation

This repository contains the code to reproduce the results in the paper "Effect of the output activation function on the probabilities and errors in medical image segmentation".

## Installation

The tests in the paper were run on a Google Cloud server. Training all neural networks takes about 2 weeks on a single Tesla K80.

1. (Optional) Run `./google_cloud.sh` to create the server.
2. Install `segmentation-models-pytorch==0.2.0`, `albumentations==1.0.1`, `nibabel==3.2.1`, `scikit-image==0.18.2` and `pytorch 1.9.0`.
3. Download one or more datasets (see below).
4. (Optional) Run `./train_all.sh`.

### Datasets

Each dataset folder has one or more `preprocessingX.py` scripts. The preprocessing scripts create a folder "processed_training" with subdirectories 0, 1, ..., n containing the files processed_0.npy, processed_1.npy, ..., processed_n.npy (and some debug files). It is easy to add new segmentation datasets by using the same data structure and adding similar preprocessing scripts.

#### Automated Cardiac Diagnosis Challenge (ACDC)

1. Register and download the dataset from https://acdc.creatis.insa-lyon.fr/
2. Unzip the dataset inside the folder "ACDC" (input files: ACDC/training/patientXXX/*).
3. Run `python preprocess1.py`
4. Run `python preprocess2.py`
5. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "ACDC/processed_training"`

#### ISLES 2018

1. Register and download the dataset from https://www.smir.ch/ISLES/Start2018
2. Unzip the dataset inside the folder "ISLES" (input files: ISLES/TRAINING/case_X/*).
3. Run `python preprocess1.py`
4. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "ISLES/processed_training"`

#### Kvasir-SEG

1. Download the dataset from https://datasets.simula.no/kvasir-seg/#download
2. Unzip the dataset inside the folder "Kvasir" (input files: Kvasir-SEG/images/* and Kvasir-SEG/masks/*).
3. Run `python preprocess1.py`
4. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "Kvasir/processed_training"`

#### Medical Segmentation Decathlon (Prostate)

1. Download the dataset from http://medicaldecathlon.com/ (only Task05_Prostate.tar)
2. Unzip the dataset inside the folder "MSD" (input files: Task05_Prostate/imagesTr/* and Task05_Prostate/labelsTr/*)
3. Run `python preprocess1.py`
4. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "MSD/processed_training"`

## Analyzing the results

The script `train.py` saves the log and model files in individual folders e.g. MSD-DiceLoss-arctan_activation-resnet34-Unet. Each folder contains csv files "fold-k/fold-k-log.csv" (for all folds k). The script `generate_table.py` creates from these results a LaTeX table.

## Decoder and reproducibility

The default settings in `train.py` are deterministic (if the script is run on the same machine). However, changing the decoder can lead to non-deterministic results (even if you set the random seed). The reason is that segmentation_models.pytorch uses different interpolation settings depending on the architecture. U-Net with `F.interpolate` and nearest neighbor interpolation is deterministic, but bilinear interpolation as found in FPN is not necessarily deterministic. This must therefore be taken into account when choosing a decoder and using a GPU.

## Citing

```
@misc{nieradzik2021effect,
      title={Effect of the output activation function on the probabilities and errors in medical image segmentation},
      author={Lars Nieradzik and Gerik Scheuermann and Dorothee Saur and Christina Gillmann},
      year={2021},
      eprint={2109.00903},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```