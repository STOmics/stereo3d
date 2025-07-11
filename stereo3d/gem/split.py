import os
import pandas as pd
import cv2 as cv
import re

from gefpy.gef_to_gem_cy import gefToGem


def cut_coord(mask_path: str, output_path: str):
    name = re.search('\w\d{5}\w\d', os.path.basename(mask_path)).group()
    mask = cv.imread(mask_path, -1)
    mask[mask > 0] = 255

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_1, contour_2 = sorted(contours, key=cv.contourArea, reverse=True)[:2]
    x1, y1, w1, h1 = cv.boundingRect(contour_1)
    x2, y2, w2, h2 = cv.boundingRect(contour_2)

    dis_x = dis_y = center_x = center_y = 0

    if x2 > x1:
        if x2 > x1 + w1:
            dis_x = x2 - (x1 + w1)
            center_x = (x2 + (x1 + w1)) / 2
            x_cut = True
        else: x_cut = False
    elif x2 < x1:
        if x2 + w2 < x1:
            dis_x = x1 - (x2 + w2)
            center_x = (x1 + (x2 + w2)) / 2
            x_cut = True
        else: x_cut = False
    else:
        x_cut = False

    if y2 > y1:
        if y2 > y1 + h1:
            dis_y = y2 - (y1 + h1)
            center_y = (y2 + (y1 + h1)) / 2
            y_cut = True
        else: y_cut = False
    elif y2 < y1:
        if y2 + h2 < y1:
            dis_y = y1 - (y2 + h2)
            center_y = (y1 + (y2 + h2)) / 2
            y_cut = True
        else: y_cut = False
    else:
        y_cut = False

    if y_cut and x_cut:
        if dis_y > dis_x:
            cut = [0, int(center_y)]
        else: cut = [int(center_x), 0]
    elif x_cut:
        cut = [int(center_x), 0]
    elif y_cut:
        cut = [0, int(center_y)]
    else:
        cut = None

    if cut[0] != 0:
        mask_1 = mask[:, :cut[0]]
        mask_2 = mask[:, cut[0]:]
    else:
        mask_1 = mask[:cut[0], :]
        mask_2 = mask[cut[0]:, :]
    cv.imwrite(os.path.join(output_path, name + '_1.tif'), mask_1)
    cv.imwrite(os.path.join(output_path, name + '_2.tif'), mask_2)
    return cut


def geftogem(gef_path, output_path=None):
    if output_path is None:
        gem_path = gef_path.replace(".gef", ".gem")
    else:
        name = os.path.basename(gef_path)
        name = name.replace(".gef", ".gem")
        gem_path = os.path.join(output_path, name)
    obj = gefToGem(gem_path, gef_path)
    obj.bgef2gem(gef_path, 1)


def split_gem(gem_path, output_path, cut):
    if os.path.splitext(gem_path)[1] == ".gef":
        name = os.path.basename(gem_path)
        _gem_path = os.path.join(output_path, "gef2gem", name.replace(".gef", ".gem"))
        # _gem_path = gem_path.replace(".gef", ".gem")
        obj = gefToGem(_gem_path, gem_path)
        obj.bgef2gem(gem_path, 1)
        gem_path = _gem_path

    name = re.search('\w\d{5}\w\d', os.path.basename(gem_path)).group()
    ext = os.path.splitext(os.path.basename(gem_path))[1]
    if ext == '.gem':
        gem = pd.read_csv(gem_path, sep='\t', skiprows=6)
    elif ext == '.txt':
        gem = pd.read_csv(gem_path, sep='\t')

    if cut[0] != 0:
        df1 = gem[gem['x'] < cut[0]]
        df2 = gem[gem['x'] >= cut[0]]
        df2['x'] = df2['x'] - cut[0]
    else:
        df1 = gem[gem['y'] < cut[1]]
        df2 = gem[gem['y'] >= cut[1]]
        df2['y'] = df2['y'] - cut[1]

    df1.to_csv(os.path.join(output_path, f"{name}_1" + ext), sep='\t', index=False)
    df2.to_csv(os.path.join(output_path, f"{name}_2" + ext), sep='\t', index=False)


if __name__ == "__main__":
    # output_path1 = r"E:\7sm\split_gem\mask"
    # output_path2 = r"E:\7sm\split_gem\gem"
    #
    # file_gem_path = r"E:\7sm\lamprey_tissuegef"
    # file_mask_path = r"E:\7sm\lamprey_mask"
    #
    # gem_list = glob(os.path.join(file_gem_path, "*.*"))
    # mask_list = glob(os.path.join(file_mask_path, "*.tif"))
    #
    # for mask_path, gem_path in zip(mask_list, gem_list):
    #     cut = cut_coord(mask_path, output_path1)
    #     split_gem(gem_path, output_path2, cut)

    data_path = r"D:\02.data\stereo3d\zp\gef"
    output_path = r"E:\m11.5_gem"
    data_list = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    for data in data_list:
        geftogem(data, output_path)

