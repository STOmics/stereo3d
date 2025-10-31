import argparse
import copy
import datetime
import json
import os.path
import platform

import cv2 as cv
import tifffile
import numpy as np
from gefpy import cell_mask_annotation
from gefpy.bgef_writer_cy import generate_bgef


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


class MultiTissue(object):
    def __init__(self, ):
        self._mask: np.ndarray = np.array([])
        self.contours: list = []
        self.boxes: list = []
        self.chip_no: str = ''
        self.savep: str = ''
        self._geojson: str = ''

    def cutout(self, mask_path: str):
        mask = tifffile.imread(mask_path)
        self._mask = mask
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)

        h0, w0 = self._mask.shape[:2]
        dct = {
            "type": "GeometryCollection",
            "geometries": [],
            "scale": 1.0,
            "center": [w0 / 2, h0 / 2, 0.5]
        }
        # Traverse all connected regions (skip background 0)
        for idx, label in enumerate(range(1, num_labels)):
            x, y, w, h, area = stats[label]
            self.boxes.append([label, x, y, w, h])
            mask1 = np.zeros_like(self._mask)
            mask1[labels == label] = 255
            contours, _ = cv.findContours(mask1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            tag = 'L{}_x{}_y{}_w{}_h{}'.format(label, x, y, w, h)
            dct_ = {
                "type": "Polygon",
                "properties": {"uID": idx,
                               "label": tag
                               },
                'coordinates': [np.squeeze(c).tolist() for c in contours],
                # 'rect': [int(i) for i in (x, y, w, h)]
            }
            dct['geometries'].append(dct_)
            # tag = 'i{}_x{}_y{}_w{}_h{}'.format(label, x, y, w, h)
            # tifffile.imwrite(
            #     os.path.join(self.savep, 'mask', '{}_{}.tif'.format(self.chip_no, tag)), mask1, compression=True)
        self._geojson = os.path.join(self.savep, '{}_{}.lasso.geojson'.format(
            self.chip_no, datetime.datetime.now().strftime('%Y%m%d%H%M%S')))
        with open(self._geojson, 'w') as fd:
            json.dump(dct, fd, indent=2)

    def cutout_gem(self, gem_path: str):
        bin_sizes = '1'
        if gem_path.endswith('gem.gz'):
            bin1_bgef_file = os.path.join(self.savep, os.path.basename(gem_path).replace('.gem.gz', '.gef'))
            generate_bgef(gem_path, bin1_bgef_file, bin_sizes=[1])
        else:
            bin1_bgef_file = gem_path
        seg = cell_mask_annotation.MaskSegmentation(
            self.chip_no, self._geojson, bin1_bgef_file, os.path.join(self.savep, 'tmp'), bin_sizes)
        seg.run_cellMask()
        # save_exp_heat_map_by_binsize(input_file=gem_path, output_png=r'fe.png', bin_size=100)

    def canvas(self, font=cv.FONT_HERSHEY_TRIPLEX, font_siz=5, font_width=3, rect_color=150):
        tmp = copy.copy(self._mask)
        tmp[tmp > 0] = 255
        for (idx, x, y, w, h) in self.boxes:
            tag = '{}'.format(idx)
            wh, _ = cv.getTextSize(tag, font, font_siz, font_width)
            cv.rectangle(tmp, (x, y - wh[1]), (x + wh[0], y), rect_color, -1)
            cv.rectangle(tmp, (x, y), (x + w, y + h), rect_color, font_width)
            cv.putText(tmp, tag, (x, y), font, font_siz, 0, font_width, cv.LINE_AA)
        tifffile.imwrite(os.path.join(self.savep, '{}.tif'.format(self.chip_no)), tmp, compression=True)

    def to_stereo3d(self, ):
        import shutil

        tmp_dir = os.path.join(self.savep, 'tmp')
        fs = search_files(tmp_dir, exts=['.gem', '.gef', '.tif'])
        for f in fs:
            if not os.path.exists(f): continue
            file_name = os.path.basename(f)
            _ = file_name.split('.')
            if '.gef' in f:
                chn, lab, le, suf = _
                shutil.copy(f, os.path.join(self.savep, 'matrix', '{}_{}.gef'.format(chn, lab)))
            elif '.gem' in f:
                chn, la, bn, lab, suf = _
                shutil.copy(f, os.path.join(self.savep, 'matrix', '{}_{}.gem'.format(chn, lab)))
            else:
                chn, la, lab, mk, suf = _
                shutil.copy(f, os.path.join(self.savep, 'mask', '{}_{}.tif'.format(chn, lab)))
            if platform.system() == 'Linux':
                os.remove(f)
        if platform.system() == 'Linux':
            shutil.rmtree(tmp_dir)


def main(args, para):
    chip_no = os.path.basename(args.matrix_path).split('.')[0]
    os.makedirs(os.path.join(args.output_path, 'tmp'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'matrix'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'mask'), exist_ok=True)
    mt = MultiTissue()
    mt.chip_no = chip_no
    mt.savep = args.output_path

    mt.cutout(args.mask_path)
    mt.canvas()
    mt.cutout_gem(gem_path=args.matrix_path)
    mt.to_stereo3d()


usage = """ Split multi tissue in one chip """
PROG_VERSION = 'v0.0.1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-matrix", "--matrix_path", action="store", dest="matrix_path", type=str, required=True,
                        help="The path of matrix file")
    parser.add_argument("-mask", "--mask_path", action="store", dest="mask_path", type=str, required=True,
                        help="The path of tissue mask")
    parser.add_argument("-output", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="output path")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)

