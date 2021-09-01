import numpy as np
import glob
import os
from PIL import Image

for patient, image_name in enumerate(glob.glob("Kvasir-SEG/images/*jpg")):
    gt_name = image_name.replace("images", "masks")

    image = np.array(Image.open(image_name).resize((256, 256)))

    mask = np.sum(np.array(Image.open(gt_name), dtype=np.float32), axis=-1) > 0
    mask = mask.astype(np.uint8)
    mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))[...,np.newaxis]

    processed = np.concatenate((image, mask), axis=-1)

    new_path = os.path.join("processed_training", str(patient))
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    np.save(os.path.join(new_path, "processed_0.npy"), processed)