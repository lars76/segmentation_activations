# Datasets

Each dataset folder has one or more `preprocessingX.py` scripts. The preprocessing scripts create a folder "processed_training" with subdirectories 0, 1, ..., n containing the files processed_0.npy, processed_1.npy, ..., processed_n.npy (and some debug files). It is easy to add new segmentation datasets by using the same data structure and adding similar preprocessing scripts.

## Automated Cardiac Diagnosis Challenge (ACDC)

1. Register and download the dataset from https://acdc.creatis.insa-lyon.fr/
2. Unzip the dataset inside the folder "ACDC" (input files: ACDC/training/patientXXX/*).
3. Run `python preprocess1.py`
4. Run `python preprocess2.py`
5. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "ACDC/processed_training"`

## ISLES 2018

1. Register and download the dataset from https://www.smir.ch/ISLES/Start2018
2. Unzip the dataset inside the folder "ISLES" (input files: ISLES/TRAINING/case_X/*).
3. Run `python preprocess1.py`
4. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "ISLES/processed_training"`

## Kvasir-SEG

1. Download the dataset from https://datasets.simula.no/kvasir-seg/#download
2. Unzip the dataset inside the folder "Kvasir" (input files: Kvasir-SEG/images/* and Kvasir-SEG/masks/*).
3. Run `python preprocess1.py`
4. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "Kvasir/processed_training"`

## Medical Segmentation Decathlon (Prostate)

1. Download the dataset from http://medicaldecathlon.com/ (only Task05_Prostate.tar)
2. Unzip the dataset inside the folder "MSD" (input files: Task05_Prostate/imagesTr/* and Task05_Prostate/labelsTr/*)
3. Run `python preprocess1.py`
4. Run `python3 train.py "sigmoid_activation" "Unet" "BCELoss" "MSD/processed_training"`

# Decoder and reproducibility

segmentation_models.pytorch uses `F.interpolate` for upsampling the image. However, this operation can be non-deterministic (even if one sets the random seed). Each run will return different results when using e.g. the architecture FPN. U-Net has not the same issue because it applies nearest neighbor interpolation on the image. Hence, when choosing the decoder and using a GPU keep this in mind. The default settings here are deterministic (when running `train.py` on the same machine).

The bash script `google_cloud.sh` creates the Google Cloud server that was used for all tests. Expect a runtime of about 2 weeks. The bash script `train_all.sh` trains all neural networks.