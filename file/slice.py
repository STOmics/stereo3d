# -*- coding: utf-8 -*-
import glog
import pandas as pd
import os.path
import tifffile


class SSDNAInfo(object):
    def __init__(self):
        self.sn = ''
        self.chip_no = ''
        self.chip_no_long = ''
        # self.memo = ''
        # self.qc = None


class HEInfo(object):
    def __init__(self):
        self.sn = ''
        # self.memo = ''
        # self.qc = None


class BlockFaceInfo(object):
    def __init__(self):
        self.no = ''
        self.should_del = False
        # self.qc = None
        # self.memo = ''


class Slice(object):
    def __init__(self):
        self.id = 0
        self.z_index = 0
        self.is_idling = False
        self.ssdna = SSDNAInfo()
        self.he = HEInfo()
        self.blockface = BlockFaceInfo()
        self.patch_time = ''
        self.face = ''

    def from_df_row(self, r):
        r = r[1]
        self.id = r['Slice_ID']
        self.z_index = r['Z_index']
        self.is_idling = r['Idling']
        # self.patch_time = r['PatchTime']
        # self.face = r['AorB']
        self.ssdna.sn = r['SSDNA_SN']
        # self.ssdna.qc = r['SSDNA_QC']
        # self.ssdna.memo = r['SSDNA_Memo']
        self.ssdna.chip_no = str(r['SSDNA_ChipNo'])
        self.ssdna.chip_no_long = str(r['SSDNA_ChipNo_long'])
        # self.he.memo = r['HE_Memo']
        # self.he.qc = r['HE_QC']
        self.he.sn = r['HE_SN']
        # self.blockface.qc = r['BlockFace_QC']
        self.blockface.no = str(r['BlockFaceNo'])
        self.blockface.should_del = r['BF_del']
        # self.blockface.memo = r['BlockFaceMemo']

    def is_idling_slice(self, ): return self.is_idling

    def has_blockface(self, ):
        if self.blockface.no == 'nan': return False
        else: return True

    def has_stereo_chip(self, ):
        if self.ssdna.chip_no == 'nan': return False
        else: return True


class SliceSequence(object):
    def __init__(self):
        self.name = 0
        self.magnification = ''
        self.size_per_pixel = ''
        self.camera_travel_distance = ''
        self.z_interval = ''
        self.sequence = dict()

    def from_xlsx(self, file_path: str):
        # meta data
        meta = pd.read_excel(file_path, sheet_name='Meta', header=0)
        dct = meta.to_dict(orient='list')
        glog.info('Meta data in record sheet: {}'.format(file_path))
        for key, value in dct.items():
            glog.info(f'{key: <25}{value}')
        self.name = dct['SampleName'][0]
        self.magnification = dct['Magnification'][0]
        self.size_per_pixel = float(dct['SizePerPixel'][0].replace('mm', ''))
        self.camera_travel_distance = dct['CameraTravelDistance'][0]
        self.z_interval = float(dct['Z-interval'][0].replace('mm', ''))

        # per slice
        slices = pd.read_excel(file_path, sheet_name='SliceSequence', header=0)
        for s in slices.iterrows():
            slice = Slice()
            slice.from_df_row(s)
            if slice.blockface.should_del == 'Y': continue
            # if slice.is_idling == 'Y': continue
            self.sequence[slice.id] = slice

    def get_slice_ids(self, ): return self.sequence.keys()

    def get_bf_chip_seq(self, k=1, long=False):
        bf_chip_seq = dict()
        for c, v in self.sequence.items():
            if v.has_stereo_chip():
                if long:
                    bf_chip_seq[v.ssdna.chip_no_long] = int(float(self.sequence[c + k].blockface.no))
                else:
                    bf_chip_seq[v.ssdna.chip_no] = int(float(self.sequence[c + k].blockface.no))
        return bf_chip_seq

    def get_z_interval(self, index='short'):
        """
        Args:
            index: str - short | long | bf
        """
        bf_chip_seq = dict()
        for c, v in self.sequence.items():
            if index == 'bf':
                if v.has_blockface():
                    bf_chip_seq[v.blockface.no] = float(v.z_index)
            else:
                if v.has_stereo_chip():
                    if index == 'long':
                        bf_chip_seq[v.ssdna.chip_no_long] = float(v.z_index)
                    elif index == 'short':
                        bf_chip_seq[v.ssdna.chip_no] = float(v.z_index)
        return bf_chip_seq

    def get_blockface_seq(self, ):
        blockface = list()
        for s, v in self.sequence.items():
            if v.has_blockface(): blockface.append(int(float(v.blockface.no)))
        return blockface

    def get_chip_seq(self, ):
        chip_seq = list()
        for c, v in self.sequence.items():
            if v.has_stereo_chip(): 
                chip_seq.append(v.ssdna.chip_no)
        return chip_seq


def main():
    xlsx = r"E:\lizepeng\m115\E-ST20220923002_slice_records_20221110.xlsx"
    ss = SliceSequence()
    ss.from_xlsx(file_path=xlsx)
    s = ss.get_chip_seq()
    bf = ss.get_blockface_seq()
    aaa = ss.get_z_interval(index='bf')
    print(s)


if __name__ == '__main__':
    main()
