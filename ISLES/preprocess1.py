import nibabel as nib
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd

OUTPUT_PATH = "."

# Stroke window (W:40, L:40) or (W: 8, L:32)
# Brain window (W:80, L:40)
# Subdural window (W:200, L:80)
# Bone window (W:2800, L:600)
def window_image(img, window_width, window_center):
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2

    return np.clip(img, lower, upper)

out = defaultdict(list)
for f in glob.glob("TRAINING/**/**/*.nii"):
    if "CT_4DPWI" in f:
        continue
    start = f.find("case_")
    patient = f[start:f.find("/", start)]
    out[patient].append(f)
out = [tuple(sorted(v)) for v in out.values()]


# compute stats of dataset
nans = 0
gt_errors = 0
stats = [[1e10, -1e10] for k in range(len(out[0]) + 3)]
image_size = [1e10, 1e10]
slices = []
for images in tqdm(out):
    # 4x CT scan
    images = [images[0], images[0], images[0]] + list(images)
    for feat_map, image in enumerate(images):
        img = nib.load(image).get_fdata()
        slices.append(img.shape[-1])
        image_size[0] = min(img.shape[0], image_size[0])
        image_size[1] = min(img.shape[1], image_size[1])
        nans += np.isnan(img).sum()
        img = np.nan_to_num(img)
        # Stroke
        if feat_map == 0:
            img = window_image(img, 40, 40)
        # Brain
        elif feat_map == 1:
            img = window_image(img, 80, 40)
        # Subdural
        elif feat_map == 2:
            img = window_image(img, 200, 80)
        # Bone
        elif feat_map == 3:
            img = window_image(img, 2800, 600)
        stats[feat_map][0] = min(img.min(), stats[feat_map][0])
        stats[feat_map][1] = max(img.max(), stats[feat_map][1])

print("Stats:", stats)
print("Nans:", nans)
print("Image size:", image_size)
print("Slices:", np.mean(slices), "with std", np.std(slices), "and min", np.min(slices), "and max", np.max(slices))

for patient, images in enumerate(tqdm(out)):
    images = [images[0], images[0], images[0]] + list(images)

    path = os.path.join(OUTPUT_PATH, "processed_training", f"{patient}")
    if not os.path.exists(path):
        os.makedirs(path)

    gt = nib.load(images[-1]).get_fdata()
    # only images where ground truth has some marked pixels + 2 empty maps
    mask = np.sum(gt, axis=(0, 1)) > 0
    flat = np.flatnonzero(mask)
    if flat.size == 0:
        continue
    mask[max(flat[0]-1, 0)] = True
    mask[min(flat[-1]+1, len(mask)-1)] = True

    for feat_map, image in enumerate(images):
        img = nib.load(image).get_fdata()
        img = np.nan_to_num(img)
        img = img[...,mask]
        if feat_map == 0:
            img = window_image(img, 40, 40)
        elif feat_map == 1:
            img = window_image(img, 80, 40)
        elif feat_map == 2:
            img = window_image(img, 200, 80)
        elif feat_map == 3:
            img = window_image(img, 2800, 600)
        img = 255 * (img - stats[feat_map][0])/(stats[feat_map][1] - stats[feat_map][0])

        for z in range(img.shape[-1]):
            train_img = Image.fromarray(img[...,z].astype(np.int8), mode="L")
            train_img = np.array(train_img.resize((image_size[0], image_size[1]), resample=Image.NEAREST))
            # ground truth, make sure there are only zeros or ones
            if feat_map == len(images)-1:
                train_img[train_img > 0.2] = 1
                train_img[train_img != 1] = 0

            file_t = os.path.join(path, f"map_{feat_map}_z_{z}.npy")
            np.save(file_t, train_img)

            q = Image.fromarray(train_img.astype(np.int8), mode="L")
            q.save(os.path.join(path, f"map_{feat_map}_z_{z}.png"))

# channels = [(channel, z-axis)]
# z-axis: z = 0 (z-1), z = 1 (z), z = 2 (z + 1)
# channels: 0 - 8
# 0. CT Stroke window (W:40, L:40) or (W: 8, L:32)
# 1. CT Brain window (W:80, L:40)
# 2. CT Subdural window (W:200, L:80)
# 3. CT Bone window (W:2800, L:600)
# 4. CBF
# 5. CBV
# 6. MTT
# 7. Tmax
# 8. ground truth
channels = [(0, 1), (1, 1), (2, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]
for patient in glob.glob(os.path.join("processed_training", "**")):
    z_axis = glob.glob(os.path.join(patient, "map_0_z_*.npy"))
    z_axis = sorted([z.replace("map_0_z_", "").replace(".npy", "")[z.rfind("/")+1:] for z in z_axis])
    for z in z_axis:
        files = [os.path.join(patient, f"map_{ch}_z_{z}.npy") for ch, _ in channels]

        processed = np.zeros((256, 256, 8), dtype=np.float32)
        for j, ch in enumerate(files):
            processed[..., j] = np.load(ch)
        np.save(os.path.join(patient, f"processed_{z}.npy"), processed)