import nibabel as nib
import glob
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

image_size = [1e10, 1e10]
stats = [[1e10, -1e10], [1e10, -1e10]]
for patient, filename in enumerate(tqdm(glob.glob("Task05_Prostate/imagesTr/*.gz"))):
    img = nib.load(filename).get_fdata()
    image_size[0] = min(image_size[0], img.shape[0])
    image_size[1] = min(image_size[1], img.shape[1])

    for z in range(img.shape[2]):
        stats[0][0] = min(img[:,:,z,0].min(), stats[0][0])
        stats[0][1] = max(img[:,:,z,0].max(), stats[0][1])

        stats[1][0] = min(img[:,:,z,1].min(), stats[1][0])
        stats[1][1] = max(img[:,:,z,1].max(), stats[1][1])

print("Min image Size", image_size)
print("Stats", stats)

for patient, filename in enumerate(tqdm(glob.glob("Task05_Prostate/imagesTr/*.gz"))):
    img = nib.load(filename).get_fdata()
    label = nib.load(filename.replace("imagesTr", "labelsTr")).get_fdata()
    print(label.shape, label.min(), label.max())

    path = os.path.join("processed_training", f"{patient}")
    if not os.path.exists(path):
        os.makedirs(path)

    for z in range(img.shape[2]):
        map1 = 255 * (img[:,:,z,0] - stats[0][0])/(stats[0][1] - stats[0][0])
        map2 = 255 * (img[:,:,z,1] - stats[1][0])/(stats[1][1] - stats[1][0])

        map1 = Image.fromarray(map1.astype(np.uint8)).resize((image_size[0], image_size[1]))
        map2 = Image.fromarray(map2.astype(np.uint8)).resize((image_size[0], image_size[1]))

        label_map = Image.fromarray(label[:,:,z].astype(np.uint8))
        # NEAREST to ensure no corruption due to interpolation
        label_map = label_map.resize((image_size[0], image_size[1]), resample=Image.NEAREST)

        map1.save(os.path.join(path, f"processed_{z}_map_0.png"))
        map2.save(os.path.join(path, f"processed_{z}_map_1.png"))

        processed_img = np.concatenate((np.array(map1)[...,np.newaxis],
                                        np.array(map2)[...,np.newaxis],
                                        np.array(label_map)[...,np.newaxis]),
                                        axis=-1)
        np.save(os.path.join(path, f"processed_{z}.npy"), processed_img)

        Image.fromarray(((np.array(label_map) == 1) * 255).astype(np.uint8)).save(os.path.join(path, f"processed_{z}_gt_1.png"))
        Image.fromarray(((np.array(label_map) == 2) * 255).astype(np.uint8)).save(os.path.join(path, f"processed_{z}_gt_2.png"))