import argparse
import os.path
import shutil

import glog
import tqdm

from stereo3d.file.slice import SliceSequence


class Saw7(object):
    def __init__(self, result: str, chip_no: str, stain: str = 'ssDNA'):
        self._result: str = result
        self._chip_no = chip_no
        self._stain: str = stain

    @property
    def gef(self, ): return os.path.join(self._result, '02.count', '{}.raw.gef'.format(self._chip_no))

    @property
    def gem(self, ): return os.path.join(self._result, '04.tissuecut', '{}.gem.gz'.format(self._chip_no))

    @property
    def tissue_mask(self, ): return os.path.join(
        self._result, '03.register', '{}_{}_tissue_cut.tif'.format(self._stain, self._chip_no))


class Saw8(object):
    def __init__(self, result: str, chip_no: str):
        self._result: str = os.path.join(result, 'result', chip_no, 'outs')
        self._chip_no = chip_no

    @property
    def gef(self, ): return os.path.join(self._result, 'feature_expression', '{}.raw.gef'.format(self._chip_no))

    @property
    def tissue_mask(self, ): return os.path.join(self._result, 'image', 'bin1_img_tissue_cut.tif')


class Naming(object):
    def __init__(self, output: str, chip_no: str):
        self._output: str = output
        self.chip_no: str = chip_no
        for it in ('gem', 'mask'):
            os.makedirs(os.path.join(self._output, it), exist_ok=True)

    @property
    def tissut_mask(self, ): return os.path.join(self._output, 'mask', '{}.tif'.format(self.chip_no))

    @property
    def gem(self, ): return os.path.join(self._output, 'gem', '{}.tif'.format(self.chip_no))

    @property
    def gef(self, ): return os.path.join(self._output, 'gem', '{}.gef'.format(self.chip_no))


def main(args, para):
    assert args.saw_version in (7, 8)
    assert args.stain in ('ssDNA', 'HE')

    ss = SliceSequence()
    ss.from_xlsx(file_path=args.record_sheet)
    chip_seq = ss.get_chip_seq()
    miss = []
    for chip_no in tqdm.tqdm(chip_seq, colour='green', ncols=100, desc='Copy', unit='files'):
        flag = 0
        n = Naming(output=args.output_path, chip_no=chip_no)
        if args.saw_version == 7:
            s = Saw7(result=args.input_path, chip_no=chip_no, stain=args.stain)
            if os.path.exists(s.gem):
                shutil.copy(s.gem, n.gem)
            else:
                flag = 1
        elif args.saw_version == 8:
            s = Saw8(result=args.input_path, chip_no=chip_no)
            if os.path.exists(s.gef):
                shutil.copy(s.gef, n.gef)
            else:
                flag = 1

        if os.path.exists(s.tissue_mask):
            shutil.copy(s.tissue_mask, n.tissut_mask)
        else:
            flag = 1
        if flag:
            miss.append(chip_no)
    miss = list(set(miss))
    if len(miss):
        glog.warning('Miss File({}): {}'.format(len(miss), miss))


usage = """ Hub saw result to stereo3d """
PROG_VERSION = 'v0.0.1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-input", "--input_path", action="store", dest="input_path", type=str, required=True,
                        help="Input saw result path")
    parser.add_argument("-stain", "--stain", action="store", dest="stain", type=str, required=True,
                        help="The stain tech(ssDNA or HE)")
    parser.add_argument("-output", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="output path")
    parser.add_argument("-record", "--record_sheet", action="store", dest="record_sheet", type=str, required=True,
                        help="Input record sheet path. ")
    parser.add_argument("-saw_version", "--saw_version", action="store", dest="saw_version", type=int, default=7,
                        help="The version of saw (7 or 8)")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)
