import os
import json
import shutil
import gzip
from turtle import width

import numpy as np
import pandas as pd
import h5py

from glob import glob
from tqdm import tqdm
from stereo3d.manual.align_parameters import apply_affine_deformation


def trans_points(x, y, offset=None, mat=None, map_x=None, map_y=None):
    """
    Args:
        x:
        y:
        offset: [int, int] - Here offset is the starting coordinate of cutting
        mat:
    Returns:
        coord: x, y
    """
    if map_x is not None and map_y is not None:
        x_array = x.values
        y_array = y.values
        
        new_x = map_x[y_array, x_array]  
        new_y = map_y[y_array, x_array]
        
        x = pd.Series(new_x, index=x.index, name=x.name)
        y = pd.Series(new_y, index=y.index, name=y.name)

    if offset:
        x = x - offset[0]
        y = y - offset[1]

    coord = np.array([x, y])
    coord = coord.transpose(1, 0)
    if mat and len(mat) > 3:
        mat = mat[2:]
    if mat:
        coord = np.concatenate([coord, np.ones((coord.shape[0], 1))], axis=1)
        cor_trans_val = np.transpose(coord)

        cor_pro = np.dot(mat, cor_trans_val)

        x_arr = cor_pro[0, :].transpose()
        y_arr = cor_pro[1, :].transpose()

        coord = np.concatenate([np.expand_dims(x_arr, axis=1),
                                np.expand_dims(y_arr, axis=1)], axis=1)

    return coord[:, 0], coord[:, 1]


def gem_read(
        gem_file
):
    """
    Args:
        gem_file:
    """
    suffix = os.path.splitext(gem_file)[1]
    if suffix == ".gz":
        fh = gzip.open(gem_file, "rb")
    else:
        fh = open(str(gem_file), "rb")  # pylint: disable=consider-using-with
    title = ""
    # Move pointer to the header of line
    eoh = 0
    header = ""
    for line in fh:
        line = line.decode("utf-8")
        if not line.startswith("#"):
            title = line
            break
        header += line
        eoh = fh.tell()
    fh.seek(eoh)
    # Initlise
    # title = title.strip("\n").split("\t")
    title = title.strip().split("\t")
    umi_count_name = [i for i in title if "ount" in i][0]
    title = ["x", "y", umi_count_name]

    type_dict = {"geneID": str}
    type_dict.update(dict(zip(title, [np.uint32] * len(title))))
    title.insert(0, "geneID")

    df = pd.read_csv(
        fh,
        sep="\t",
        header=0,
        usecols=title,
        dtype=type_dict,
    )

    return df


def read_gem_from_gef(gef_file):
    h = h5py.File(gef_file, 'r')
    gene = h['geneExp']['bin1']['gene'][:]
    expression = h['geneExp']['bin1']['expression'][:]
    df = pd.DataFrame(columns=('geneID', 'x', 'y', 'MIDCount'))
    df['x'] = expression['x']
    df['y'] = expression['y']
    df['MIDCount'] = expression['count']
    _ = np.zeros((expression.shape[0],), dtype='S64')

    gene_name = 'geneID' if 'geneID' in gene.dtype.names else 'gene'
    for i in range(gene.shape[0]):
        s, o = (gene[i]['offset'], gene[i]['count'])
        df['geneID'][s: s + o] = gene[i][gene_name]

    return df


def gef_trans(gef_file, offset, mat, shape, output_path):
    shutil.copy(gef_file, output_path)
    map_x = None
    map_y = None
    if mat and len(mat) > 3:
        p = mat[0]
        q = mat[1]
        map_x, map_y = apply_affine_deformation(shape[0], shape[1], p, q, alpha=1.0)
    with h5py.File(output_path, 'r+') as h:
        expression = h['geneExp']['bin1']['expression'][:]
        new_x, new_y = trans_points(expression['x'], expression['y'], offset, mat, map_x, map_y)
        h['geneExp']['bin1']['expression']['x'] = new_x
        h['geneExp']['bin1']['expression']['y'] = new_y


def anndata_trans(adata_file, offset, mat, shape, output_path):
    import scanpy as sc
    adata = sc.read_h5ad(adata_file)
    map_x = None
    map_y = None
    if mat and len(mat) > 3:
        p = mat[0]
        q = mat[1]
        map_x, map_y = apply_affine_deformation(shape[0], shape[1], p, q, alpha=1.0)
    if "spatial" in adata.obsm.keys():
        x, y = adata.obsm["spatial"][:, 0], adata.obsm["spatial"][:, 1]
        new_x, new_y = trans_points(x, y, offset, mat, map_x, map_y)
        adata.obsm["spatial"][:, 0] = new_x
        adata.obsm["spatial"][:, 1] = new_y
        adata.write_h5ad(output_path)


def gem_trans(gem_file, offset, mat, shape, output_path):
    """
    Args:
        gem_file:
        offset:
        mat:
        output_path: str - With file name
    """
    gem = gem_read(gem_file)
    
    map_x = None
    map_y = None
    if mat and len(mat) > 3:
        p = mat[0]
        q = mat[1]
        map_x, map_y = apply_affine_deformation(shape[0], shape[1], p, q, alpha=1.0)

    # gem['x'] = gem['x'] - min(gem['x'])
    # gem['y'] = gem['y'] - min(gem['y'])

    new_x, new_y = trans_points(gem['x'], gem['y'], offset, mat, map_x, map_y)

    gem['x'] = np.int_(np.round(new_x))
    gem['y'] = np.int_(np.round(new_y))
    gem.to_csv(output_path, sep='\t', index=False)


def trans_matrix_by_json(gem_path, cut_json_path, align_json_path, output_path):
    """
    Args:
        gem_path:
        cut_json_path:
        align_json_path:
        output_path:
    """
    if isinstance(gem_path, str):
        gem_list = glob(os.path.join(gem_path, '*.*'))
    elif isinstance(gem_path, list):
        gem_list = gem_path

    os.makedirs(output_path, exist_ok=True)
    with open(cut_json_path, 'r') as js:
        mask_cut_info = json.load(js)
    with open(align_json_path, 'r') as js:
        align_info = json.load(js)

    for matrix_file in tqdm(gem_list, desc='Gem', ncols=100):
        matrix_name = os.path.basename(matrix_file).split('.')[0]
        for key in mask_cut_info.keys():
            if matrix_name in key:
                mask_cut = mask_cut_info[key]
                break
        else:
            mask_cut = None

        for key in align_info.keys():
            if matrix_name in key:
                align = align_info[key]
                break
        else:
            align = None

        if mask_cut is not None or align is not None:
            # mask_cut = None
            mat = align['mat']
            shape = align['shape']
            if matrix_file.endswith('txt') or matrix_file.endswith('gem') or matrix_file.endswith('gem.gz'):
                gem_trans(
                    matrix_file, mask_cut, mat, shape, os.path.join(output_path, f"{matrix_name}.gem")
                )
            elif matrix_file.endswith('gef'):
                gef_trans(
                    matrix_file, mask_cut, mat, shape, os.path.join(output_path, f"{matrix_name}.gef")
                )
            elif matrix_file.endswith('.h5ad'):
                anndata_trans(matrix_file, mask_cut, mat, shape, os.path.join(output_path, f"{matrix_name}.h5ad"))


if __name__ == "__main__":
    # trans_matrix_by_json(gem_path=r"D:\02.data\E14-16h_a_bin1_image_gem",
    #                   cut_json_path=r"D:\02.data\E14-16h_a_bin1_image_regis\align_info.json",
    #                   align_json_path=r"D:\02.data\E14-16h_a_bin1_image_regis\align_info.json",
    #                   output_path=r"D:\02.data\E14-16h_a_bin1_image_gem\new_gem")
    # read_gem_from_gef(r"/media/Data1/user/szl/data/output/gem/SS200000122BL_B1_L1_x7649_y3592_w8139_h6537.gef")

    aaa = gem_read(r"D:\02.data\SS200000122BL_B1_L1_x7649_y3592_w8139_h6537.gem.gz")