import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from collections import defaultdict, OrderedDict

output_table = defaultdict(list)
output_table_grouped = defaultdict(dict)
for folder in tqdm(os.listdir(".")):
	elem = folder.split("-")
	if len(elem) != 5:
		continue
	dataset, loss_function, activation_function, model_name, architecture = elem

	folds = glob.glob(os.path.join(folder, "**", "*csv"))

	epochs = []
	nll = []
	total_dice = []
	best_threshold_all = []
	best_dice_all = []
	best_fp_fn_all = []
	for fold in folds:
		df = pd.read_csv(fold)

		epochs.append(df["epoch"].max())
		nll.append(df["nll"].min())
		total_dice.append(df["avg_total_dice"].max())

		columns = df.columns.values
		classes = np.unique([k[-1:] for k in columns if "class_" in k])
		#print(columns)
		thresholds = np.unique([k[5:].replace("_class_0", "") for k in columns if "_class_0" in k and k.startswith("dice_")])
		
		for c in classes:
			best_threshold = 0
			best_dice = 0
			best_fp_fn = 1e10
			for t in thresholds:
				dice = df[f"dice_{t}_class_{c}"].max()
				if best_dice < dice:
					best_dice = dice
					best_threshold = t
				best_fp_fn = min(best_fp_fn, (df[f"fn_{t}_class_{c}"] + df[f"fp_{t}_class_{c}"]).min())
			best_threshold_all.append(float(best_threshold))
			best_dice_all.append(best_dice)
			best_fp_fn_all.append(best_fp_fn)
		
	t = {"num_epochs" : epochs, "nll": nll, "total_dice": total_dice,
	     "threshold": best_threshold_all, "best_dice": best_dice_all,
	     "best_fp_fn":best_fp_fn_all,
	     "loss_function":loss_function, "activation": activation_function}
	output_table[dataset].append(t)

	key = activation_function + "#" + loss_function
	for k, v in t.items():
		if not output_table_grouped[key].get(k):
			output_table_grouped[key][k] = []
		output_table_grouped[key][k] += v

with open("latex_table.txt", "w") as f:

	f.write("\n\\section{Individual stats}\n")

	for k, v in output_table.items():
		f.write("\\begin{table}[]\n")
		f.write("\\begin{tabular}{llllll}\n")
		f.write("\\toprule\n")
		f.write("Loss & Activation & Threshold & NLL $\\downarrow$ & Dice $\\uparrow$ & Total Dice $\\uparrow$ \\\\ \\hline\n")

		v = sorted(v, key=lambda d : d['activation'] + "_" + d["loss_function"])

		for pos, result in enumerate(v):
			#print(v)
			activation = result["activation"].replace('_activation', '')
			loss = result["loss_function"].replace("Loss", "")
			f.write(f'{loss} & {activation} & ')
			for z, q in [("threshold", 0), ("nll", 0), ("best_dice", -1), ("total_dice", -1)]:
				get_best = sorted(v, key=lambda d : np.round(np.mean(d[z]), 4))[q]
				if z != "threshold" and str(result) == str(get_best):
					f.write("$\\mathbf{" + f"{np.round(np.mean(result[z]), 4)} \\pm {np.round(np.std(result[z]), 4)}" + "}$")
				else:
					f.write(f"${np.round(np.mean(result[z]), 4)} \\pm {np.round(np.std(result[z]), 4)}$")
				if z != "total_dice":
					f.write(" & ")
			f.write("\\\\ \n")

			if (pos + 1) % 3 == 0 and pos != len(v)-1:
				f.write("\\midrule\n")

		f.write("\\bottomrule\n")
		f.write("\\end{tabular}\n")
		f.write("\\caption{" + str(k) + "}\n")
		f.write("\\end{table}\n")









	f.write("\n\\section{Grouped stats}\n")

	f.write("\\begin{table}[]\n")
	f.write("\\begin{tabular}{lllllll}\n")
	f.write("\\toprule\n")
	f.write("Loss & Activation & Threshold & NLL $\\downarrow$ & Dice $\\uparrow$ & Total Dice $\\uparrow$\\\\ \\hline\n")


	output_table_grouped = OrderedDict(sorted(output_table_grouped.items()))
	for pos, (x, v) in enumerate(output_table_grouped.items()):
		activation_function, loss_function = x.split("#")
		activation = activation_function.replace('_activation', '')
		loss = loss_function.replace("Loss", "")

		f.write(f'{loss} & {activation} & ')

		for z, q in [("threshold", 0), ("nll", 0), ("best_dice", -1), ("total_dice", -1)]:
			f.write(f"${np.round(np.mean(v[z]), 4)} \\pm {np.round(np.std(v[z]), 4)}$")
			if z != "total_dice":
				f.write(" & ")
		f.write("\\\\ \n")

		if (pos + 1) % 3 == 0 and pos != len(output_table_grouped)-1:
			f.write("\\midrule\n")

	f.write("\\bottomrule\n")
	f.write("\\end{tabular}\n")
	f.write("\\caption{All datasets}\n")
	f.write("\\end{table}\n")