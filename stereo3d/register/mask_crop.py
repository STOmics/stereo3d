import os
import glog
import json

import cv2 as cv
import numpy as np
import tifffile as tif

from tqdm import tqdm


def f_cut_image_by_contours(image, padding=4000):
    if isinstance(image, str):
        image = cv.imread(image, -1)

    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if np.max(image) == 1:
        image[image > 0] = 255

    height, width = image.shape[:2]
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_large = sorted(contours, key=cv.contourArea, reverse=True)[0]
    x, y, w, h = cv.boundingRect(contour_large)
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    x = max(0, x)
    y = max(0, y)

    if y + h > height: h = height - y
    if x + w > width: w = width - x

    return image[y:y + h, x:x + w], [x, y, w, h]


def cut_mask(mask_list, output_path, padding_size = 4000):
    glog.info("Cut image and h5ad file.")
    mask_cut_info = dict()
    os.makedirs(os.path.join(output_path), exist_ok=True)
    for mask_file in tqdm(mask_list, desc="Mask crop", ncols=100):
        name = os.path.basename(mask_file)
        chip = os.path.splitext(name)[0]
        cut_image, rect = f_cut_image_by_contours(mask_file, padding_size)
        tif.imwrite(os.path.join(output_path, name), cut_image)
        mask_cut_info[chip] = rect

    with open(os.path.join(output_path, "mask_cut_info.json"), 'w') as f:
        json.dump(mask_cut_info, f, indent=2)
