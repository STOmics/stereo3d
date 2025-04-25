import argparse
import copy
import os.path
import gzip
import cv2 as cv
import tifffile
import numpy as np
import pandas as pd


class MultiTissue(object):
    def __init__(self, ):
        self._mask: np.ndarray = np.array([])
        self.contours: list = []
        self.boxes: list = []

    def cutout(self, mask: np.ndarray):
        self._mask = mask
        self.contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 只检测外部轮廓

        self.boxes = []
        for cnt in self.contours:
            # 计算外接矩形（直角矩形）
            x, y, w, h = cv.boundingRect(cnt)
            self.boxes.append((x, y, w, h))

    def cutout_gem(self, gem_path: str, output: str):
        dct = {}
        if gem_path.endswith('.gef'):
            pass
        else:
            if gem_path.endswith('.gz'):
                f = gzip.open(gem_path, 'rb')
            else:
                f = open(gem_path, 'rb')

            header = ''
            num_of_header_lines = 0
            eoh = 0
            for i, l in enumerate(f):
                l = l.decode("utf-8")  # read in as binary, decode first
                if l.startswith('#'):  # header lines always start with '#'
                    header += l
                    num_of_header_lines += 1
                    eoh = f.tell()  # get end-of-header position
                else:
                    break
            f.seek(eoh)
            df = pd.read_csv(f, sep='\t', header=0)

        for idx, (x, y, w, h) in enumerate(self.boxes):
            tag = 'i{}_x{}_y{}_w{}_h{}'.format(idx + 1, x, y, w, h)
            new = df[(df.x >= x) & (df.x < x + w) & (df.y >= y) & (df.y < y + h)]

            new.columns = df.columns
            new['x'] -= x
            new['y'] -= y
            dct[tag] = new
            p = os.path.join(output, '{}.gem.gz'.format(tag))
            new.to_csv(p, compression='gzip', header=None, index=False, sep='\t')

    def tissues(self, default: int = 255) -> dict:
        dct = {}
        for idx, cnt in enumerate(self.contours):
            x, y, w, h = self.boxes[idx]
            mask1 = np.zeros((h, w), dtype=np.uint8)
            sh = copy.copy(cnt)
            sh[:, :, 0] -= x
            sh[:, :, 1] -= y
            cv.fillPoly(mask1, [sh], color=default)
            tag = 'i{}_x{}_y{}_w{}_h{}'.format(idx + 1, x, y, w, h)
            dct[tag] = mask1

        return dct

    def canvas(self, font=cv.FONT_HERSHEY_TRIPLEX, font_siz=5, font_width=3, rect_color=150):
        tmp = copy.copy(self._mask)
        tmp[tmp > 0] = 255
        for idx, (x, y, w, h) in enumerate(self.boxes):
            tag = '{}'.format(idx + 1)
            wh, _ = cv.getTextSize(tag, font, font_siz, font_width)
            cv.rectangle(tmp, (x, y - wh[1]), (x + wh[0], y), rect_color, -1)
            cv.rectangle(tmp, (x, y), (x + w, y + h), rect_color, font_width)
            cv.putText(tmp, tag, (x, y), font, font_siz, 0, font_width, cv.LINE_AA)

        return tmp


def main(args, para):
    save_p = args.output_path
    mask = tifffile.imread(args.mask_path)
    mt = MultiTissue()
    mt.cutout(mask)
    os.makedirs(os.path.join(save_p, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(save_p, 'gem'), exist_ok=True)
    chip_no = os.path.basename(args.matrix_path)[0]

    sub_t = mt.tissues()
    for tag, s in sub_t.items():
        tifffile.imwrite(os.path.join(save_p, 'mask', '{}_{}.tif'.format(chip_no, tag)), s, compression=True)

    c = mt.canvas()
    tifffile.imwrite(os.path.join(save_p, '{}.tif'.format(chip_no)), c, compression=True)

    mt.cutout_gem(args.matrix_path, output=os.path.join(save_p, 'gem'))


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
