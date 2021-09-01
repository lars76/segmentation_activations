#!/usr/bin/env bash

declare -a activations=("arctan_activation" "softsign_activation" "sigmoid_activation" "linear_activation" "inv_square_root_activation" "cdf_activation" "hardtanh_activation")
declare -a architectures=("Unet")
declare -a losses=("BCELoss" "MSELoss" "DiceLoss")
declare -a paths=("ACDC/processed_training" "ISLES/processed_training" "Kvasir/processed_training" "MSD/processed_training")
# 7 * 1 * 3 * 4

for activation in "${activations[@]}"
do
	for architecture in "${architectures[@]}"
	do
		for loss in "${losses[@]}"
		do
			for path in "${paths[@]}"
			do
				PYTHONHASHSEED=1 python3 train.py "$activation" "$architecture" "$loss" "$path"
			done
		done
	done
done