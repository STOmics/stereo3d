import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import re
import glog
from openpyxl import load_workbook
from typing import Optional
import shutil
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


### parse slice records
class SliceRecordsParser:
    def __init__(self, slice_records_file):
        self.slice_records_file = slice_records_file
        wb = load_workbook(self.slice_records_file)
        sheet = wb['SliceSequence']
        col1 = []
        for col in sheet['B']:
            col1.append(col.value)
        col2 = []
        for col in sheet['D']:
            col2.append(col.value)
        col3 = []
        for col in sheet['E']:
            col3.append(col.value)

        col11 = col1[1:]
        col22 = col2[1:]
        col33 = col3[1:]

        data = pd.DataFrame([col22, col33, col11]).T
        data.columns = ['SSDNA_SN', 'SSDNA_ChipNo', 'Z_index(mm)']
        data = data.dropna(how='any', axis=0)
        self.data = data

    def extract_gef_from_sap(self, sap_input: str,
                             path_outdir: str,
                             file_suffixs = '.tissue.gef'):
        chip_id_list = self.data.SSDNA_ChipNo
        files = os.listdir(sap_input)  ##Normally，‘/data/input’
        absolute_path_list = open(os.path.join(path_outdir, 'tissuegef_path_list.txt'), 'w')
        for chip_id in chip_id_list:
            for root, dirs, files in os.walk(sap_input):
                # files = [f for f in files if f.startswith('ssDNA_'+chip_id)]
                files = [f for f in files if f.startswith(chip_id)]
                result = [f for f in files if f.endswith(file_suffixs)]
                for f in result:
                    absolute_path_list.write(os.path.join(root, f) + '\n')
        absolute_path_list.close()
        list_path = os.path.join(path_outdir, 'tissuegef_path_list.txt')
        glog.info(f'Save path list to: {list_path}')

    def extract_mask_from_sap(self, sap_input: str,
                              path_outdir: str,
                              file_suffixs = '_tissue_cut.tif'):
        chip_id_list = self.data.SSDNA_ChipNo
        files = os.listdir(sap_input)  ##Normally，‘/data/input’
        absolute_path_list = open(os.path.join(path_outdir, 'tissuemask_path_list.txt'), 'w')
        for chip_id in chip_id_list:
            for root, dirs, files in os.walk(sap_input):
                # files = [f for f in files if f.startswith('ssDNA_'+chip_id)]
                files = [f for f in files if f.startswith(chip_id)]
                result = [f for f in files if f.endswith(file_suffixs)]
                for f in result:
                    absolute_path_list.write(os.path.join(root, f) + '\n')
        absolute_path_list.close()
        list_path = os.path.join(path_outdir, 'tissuemask_path_list.txt')
        glog.info(f'Save path list to: {list_path}')


if __name__ == "__main__":
    cutter = SliceRecordsParser(
        slice_records_file='/data/work/E95_bin50/01/E-ST20220923002_slice_records_20230508.xlsx')
    cutter.extract_gef_from_sap(sap_input='/data/input/', path_outdir='/data/work/notebooktest_1007/txt/')
    cutter.extract_mask_from_sap(sap_input='/data/input/', path_outdir='/data/work/notebooktest_1007/txt/')