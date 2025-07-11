import warnings
warnings.filterwarnings('ignore')
import anndata as ad
import anndata
import pandas as pd
import numpy as np
import os
import glog
from openpyxl import load_workbook
from typing import Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import re
import gc


def adata_list_input(
    h5ad_path: str,
    slice_records_file: str,
    out_path: str,
    spatial_mm_key: str = 'spatial_mm',
    spatial_key: str = 'spatial',
    bin_type: Literal["square_bin", "cell_bin"] = "square_bin",
    binsize: Optional[int] = None,
    x_y_unit: str = '500nm',
    z_data: bool = False,
    z_unit: str = '1mm',
    z_thickness: str = '10um',
):
    """
    Transform h5ad file into input format required by 3D analysis pipeline.

    Args:   
        h5ad_path: path of input of .h5ad files.
        slice_records_file:
        spatial_mm_key:
        bin_type:
        binsize:
        x_y_unit:
        z_data:
        z_unit:
        z_thickness:
        out_path:  path of output of update .h5ad files.
        spatial_key: The column key in .obsm, default to be 'spatial'.

    Returns:
        a list of adata. and update adata to outpath which format that meets the requirements of 3D flow analysis.
        
    """

    # 01.sort h5ad file names
    A = SliceRecordsParser(slice_records_file)
    # h5ad File Renaming
    A.rename_h5adfile(h5ad_path)
    sorted_file_names = sort_file_names(h5ad_path, suffix='.h5ad')
    # 02. add z_data & data_unit & standardize the units of x,y,z.
    # add info to adata.
    adata_list = []
    for index, file in enumerate(sorted_file_names):
        with open(os.path.join(h5ad_path, file), 'r') as f:
            adata = ad.read(os.path.join(h5ad_path, file))
            if x_y_unit == '500nm':
                adata.obsm[spatial_mm_key] = np.zeros((adata.shape[0], 2))
                adata.obsm[spatial_mm_key][:, 0] = adata.obsm[spatial_key][:, 0] * 500 / 1000000
                adata.obsm[spatial_mm_key][:, 1] = adata.obsm[spatial_key][:, 1] * 500 / 1000000
            else:
                raise ValueError(f"Users need to do their own coordinate unit conversion")
            
            # add z-axis data from cutter transaction table to .spatial[:,2]
            if z_data == False:
                z_value = [A.data['Z_index'].tolist()[index]] * adata.obs.shape[0]
                z_value = np.array(z_value).reshape(-1, 1)
                adata.obsm[spatial_mm_key] = np.hstack([adata.obsm[spatial_mm_key], z_value])

            # standardize the units of all x,y,z. add data_type to .uns
            if bin_type == "square_bin":
                adata.uns['data_unit'] = {'binsize': binsize, 'z_size': z_thickness, 'unit': z_unit}
            elif bin_type == "cell_bin":
                adata.uns['data_unit'] = {'binsize': 'cellbin', 'z_size': z_thickness, 'unit': z_unit}
            else:
                raise ValueError(f"the type of bin_type should be provided, 'square_bin' or 'cell_bin'")

            # update adata
            # if not os.path.isdir(out_path):
            #     os.mkdir(out_path)
            adata.write(os.path.join(out_path, file))
            adata_list.append(adata)
            del adata
    return adata_list
            

def sort_file_names(file_path, suffix: Literal['.h5ad', '.gef', '.gem', '.tif']):
    files = os.listdir(file_path)
    h5ad_files = [file for file in files if file.endswith(suffix)]
    lengths = {len(f) for f in h5ad_files}
    all_equal = len(lengths) == 1  
    if all_equal:
        sorted_file_names = sorted(h5ad_files)
    else:
        pattern = '^[0-9]+'
        file_dict = {}
        for file_name in h5ad_files:
            old_file_path = os.path.join(file_path, file_name)
            match = re.match(pattern, file_name)
            if match:
                prefix = match.group(0)
            else:
                prefix = '0'
            prefix_num = int(prefix)
            file_dict[file_name] = prefix_num
        sorted_file_names = sorted(file_dict, key=file_dict.get)
    return sorted_file_names


class SliceRecordsParser:
    """ Analyze the cutter flow table """
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
        data.columns = ['SSDNA_SN', 'SSDNA_ChipNo', 'Z_index']
        data = data.dropna(how='any', axis=0)
        self.data = data

    def extract_file_from_sap(self, sap_input: str,
                              path_outdir: str,
                              file_suffixs='.tissue.gef'):
        chip_id_list = self.data.SSDNA_ChipNo
        files = os.listdir(sap_input)  # In normal use, ‘/data/input’
        absolute_path_list = open(os.path.join(path_outdir, 'absolute_path_list.txt'), 'w')
        for chip_id in chip_id_list:
            for root, dirs, files in os.walk(sap_input):
                files = [f for f in files if f.startswith(chip_id)] 
                result = [f for f in files if f.endswith(file_suffixs)]
                for f in result:
                    absolute_path_list.write(os.path.join(root, f) + '\n')
        absolute_path_list.close()
        list_path = os.path.join(path_outdir, 'absolute_path_list.txt')
        glog.info(f'Save path list to: {list_path}')
    
    def rename_h5adfile(self, h5ad_path):
        def _rename_files(file_list, relation):
            for file_name in file_list:
                first_part = file_name[:file_name.index('_')]
                for key in relation:
                    if relation[key] == first_part:
                        new_name = key + '_' + file_name
                        os.rename(h5ad_path + '/' + file_name, h5ad_path + '/' + new_name)
        data = self.data
        relation = dict(zip(data['SSDNA_SN'].tolist(), data['SSDNA_ChipNo'].tolist()))
        file_list = sort_file_names(h5ad_path, suffix='.h5ad')
        _rename_files(file_list, relation)


def read_and_parse_by_celltype(outdir: str,
                               spatial_regis: str,
                               anno: str,
                               celltype: str,
                               h5ad_path: Optional[str] = None,
                               adata_list: Optional[list] = None,
                               sc_xyz: Optional[list] = None):
    """
    Get x,y,z,anno columns as mesh input.

    Args:
        outdir:
        adata_list:
        celltype:
        h5ad_path: path of input of .h5ad files.
        spatial_regis: The column key in .obsm, default to be 'spatial_regis'. note that x,y,z 
        anno: The column key/name that identifies the grouping information(for example, clusters that correspond to different cell types)of spots.
        sc_xyz:The scale by which the spatial points in h5ad is scaled. when the `sc_xyz` is list, the model is scaled along the xyz
                axis at different scales. If `sc_xyz` is None, there will be by scale by defult parameter.
        

    Returns:
        a list of adata. and update adata to outpath which format that meets the requirements of 3D flow analysis.
    """
    xli = []
    yli = []
    zli = []
    tyli = []
    if h5ad_path:
        fnames = sort_file_names(file_path=h5ad_path, suffix='.h5ad')
        adata_list = []
        for fname in fnames:
            path = os.path.join(h5ad_path, fname)
            adata = anndata.read(path)
            adata_list.append(adata)
    elif adata_list:
        adata_list = adata_list
    else:  
        raise ValueError(f"h5ad_path and adata_list should have at least one that is not None.")
    for adata in adata_list:
        if adata.uns['data_unit']['binsize'] == 'cellbin':
            binsize = 10
        else:
            binsize = adata.uns['data_unit']['binsize']
        z_size = adata.uns['data_unit']['z_size']
        match = re.search(r'(\d+)um', z_size)
        z_size = int(match.group(1))
        if sc_xyz is None:
            sc_xyz = [None] * 3
            sc_xyz[0] = 1000/(binsize*0.5)
            sc_xyz[1] = 1000/(binsize*0.5)
            sc_xyz[2] = 1000/z_size
        x = (adata[adata.obs[anno] == celltype].obsm[spatial_regis][:, 0]*sc_xyz[0]).tolist()
        y = (adata[adata.obs[anno] == celltype].obsm[spatial_regis][:, 1]*sc_xyz[1]).tolist()
        z = (adata[adata.obs[anno] == celltype].obsm[spatial_regis][:, 2]*sc_xyz[2]).tolist()
        ty = adata[adata.obs[anno] == celltype].obs[anno].tolist()
        del adata
        gc.collect()

        xli = xli + x
        del x
        gc.collect()

        yli = yli + y
        del y
        gc.collect()

        zli = zli + z
        del z
        gc.collect()

        tyli = tyli + ty
        del ty
        gc.collect()
    
    # output
    outdir = './'
    data = np.column_stack((xli, yli, zli, tyli))
    with open(outdir+celltype+'_info.txt', 'w') as O:
        for row in data:
            O.write('\t'.join([str(elem) for elem in row]) + '\n')
