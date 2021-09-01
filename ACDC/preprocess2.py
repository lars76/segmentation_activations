"""
Credits: https://github.com/liut969/Automated-Cardiac-Segmentation-and-Diagnosis/blob/master/data_preprocess.py
changed __main__
"""
import nibabel as nib
import cv2
import numpy as np
import os
import re
import csv
from PIL import Image

class DataPreprocess(object):
    def __init__(self, roi_x, roi_y, result_z):
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.result_z = result_z

    def get_center_point(self, csv_path='./train_center_radii.csv'):
        result = {}
        with open(csv_path, 'r') as f:
            f_csv = csv.reader(f)
            for item in f_csv:
                if f_csv.line_num == 1:
                    continue
                key = re.findall('\d+', item[0])
                result[key[0]] = item[1] + item[2]
            f.close()
        return result


    def get_roi_image(self, from_path='../data/training', center_point_csv_path='./train_center_radii.csv'):
        cases = sorted(next(os.walk(from_path))[1])
        center_points = self.get_center_point(csv_path=center_point_csv_path)
        result = np.zeros([len(cases)*2, self.result_z, self.roi_x, self.roi_y])
        for i, case in enumerate(cases):
            current_path = os.path.join(from_path, case)
            center_str_val = center_points[re.findall('\d+', case)[0]]
            center_and_radii = re.findall('\d+', center_str_val)

            for f_name in [f for f in os.listdir(current_path) if f.startswith('patient')]:
                if '4d' not in f_name and 'gt' not in f_name:
                    num = re.findall('\d+', f_name)
                    nimg = nib.load(os.path.join(current_path, f_name))
                    img = nimg.get_fdata()
                    for z in range(img.shape[2]):

                        center_point_x, center_point_y = int(center_and_radii[0]), int(center_and_radii[1])
                        radii_x, radii_y = int(center_and_radii[2]), int(center_and_radii[3])

                        x_left = int(center_point_x-self.roi_x/2)
                        x_right = int(center_point_x+self.roi_x/2)
                        y_left = int(center_point_y-self.roi_y/2)
                        y_right = int(center_point_y+self.roi_y/2)
                        if x_left < 0: x_left = 0
                        if x_right > img.shape[0]: x_right = img.shape[0]
                        if y_left < 0: y_left = 0
                        if y_right > img.shape[1]: y_right = img.shape[1]
                        roi_image = img[x_left:x_right, y_left:y_right, z]

                        if radii_x > 40 or radii_y > 40:
                            x_left = int(center_point_x-self.roi_x)
                            x_right = int(center_point_x+self.roi_x)
                            y_left = int(center_point_y-self.roi_y)
                            y_right = int(center_point_y+self.roi_y)
                            if x_left < 0: x_left = 0
                            if x_right > img.shape[0]: x_right = img.shape[0]
                            if y_left < 0: y_left = 0
                            if y_right > img.shape[1]: y_right = img.shape[1]
                            roi_image = img[x_left:x_right, y_left:y_right, z]
                            roi_image = cv2.resize(roi_image, (self.roi_x, self.roi_y), interpolation=cv2.INTER_NEAREST)

                        norm = np.uint8(cv2.normalize(roi_image, None, 0, 255, cv2.NORM_MINMAX))
                        norm = cv2.equalizeHist(norm)
                        if norm.shape[0] < self.roi_x or norm.shape[1] < self.roi_y:
                            norm = cv2.copyMakeBorder(norm, self.roi_x-norm.shape[0], 0, self.roi_y-norm.shape[1], 0, cv2.BORDER_CONSTANT, value=0)
                        if num[1] == '01':
                            result[i*2, z, ::, ::] = norm
                        else:
                            result[i*2+1, z, ::, ::] = norm
        return result / 255

    def get_roi_label(self, from_path='../data/training', center_point_csv_path='./train_center_radii.csv'):
        cases = sorted(next(os.walk(from_path))[1])
        center_points = self.get_center_point(csv_path=center_point_csv_path)
        result = np.zeros([len(cases)*2, self.result_z, self.roi_x, self.roi_y])
        for i, case in enumerate(cases):
            current_path = os.path.join(from_path, case)
            center_str_val = center_points[re.findall('\d+', case)[0]]
            center_and_radii = re.findall('\d+', center_str_val)

            for f_name in [f for f in os.listdir(current_path) if f.startswith('patient')]:
                if 'gt' in f_name:
                    num = re.findall('\d+', f_name)
                    nimg = nib.load(os.path.join(current_path, f_name))
                    img = nimg.get_fdata()
                    for z in range(img.shape[2]):

                        center_point_x, center_point_y = int(center_and_radii[0]), int(center_and_radii[1])
                        radii_x, radii_y = int(center_and_radii[2]), int(center_and_radii[3])

                        x_left = int(center_point_x-self.roi_x/2)
                        x_right = int(center_point_x+self.roi_x/2)
                        y_left = int(center_point_y-self.roi_y/2)
                        y_right = int(center_point_y+self.roi_y/2)
                        if x_left < 0: x_left = 0
                        if x_right > img.shape[0]: x_right = img.shape[0]
                        if y_left < 0: y_left = 0
                        if y_right > img.shape[1]: y_right = img.shape[1]
                        roi_label = img[x_left:x_right, y_left:y_right, z]

                        if radii_x > 40 or radii_y > 40:
                            x_left = int(center_point_x-self.roi_x)
                            x_right = int(center_point_x+self.roi_x)
                            y_left = int(center_point_y-self.roi_y)
                            y_right = int(center_point_y+self.roi_y)
                            if x_left < 0: x_left = 0
                            if x_right > img.shape[0]: x_right = img.shape[0]
                            if y_left < 0: y_left = 0
                            if y_right > img.shape[1]: y_right = img.shape[1]
                            roi_label = img[x_left:x_right, y_left:y_right, z]
                            roi_label = cv2.resize(roi_label, (self.roi_x, self.roi_y), interpolation=cv2.INTER_NEAREST)

                        if roi_label.shape[0] < self.roi_x or roi_label.shape[1] < self.roi_y:
                            roi_label = cv2.copyMakeBorder(roi_label, self.roi_x-roi_label.shape[0], 0, self.roi_y-roi_label.shape[1], 0, cv2.BORDER_CONSTANT, value=0)
                        if num[1] == '01':
                            result[i*2, z, ::, ::] = roi_label
                        else:
                            result[i*2+1, z, ::, ::] = roi_label
        return result


if __name__ == '__main__':
    data_preprocess = DataPreprocess(roi_x=128, roi_y=128, result_z=21)
    train_image = data_preprocess.get_roi_image(from_path='training', center_point_csv_path='./train_center_radii.csv')
    train_label = data_preprocess.get_roi_label(from_path='training', center_point_csv_path='./train_center_radii.csv')
    for i in range(train_image.shape[0]):
        path = os.path.join("processed_training", f"{i}")
        if not os.path.exists(path):
            os.makedirs(path)
        for z in range(train_image.shape[1]):
            if train_image[i, z].sum() > 0:
                out = np.concatenate((train_image[i, z][...,np.newaxis], train_label[i, z][...,np.newaxis]), axis=-1)
                np.save(os.path.join(path, f"processed_{z}.npy"), out)

                q = Image.fromarray((255 * train_image[i, z]).astype(np.int8), mode="L")
                q.save(os.path.join(path, f"{z}.png"))