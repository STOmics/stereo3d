import os
import json
import shutil
import gzip

import numpy as np
import pandas as pd
import h5py

from glob import glob
from tqdm import tqdm


def trans_points(x, y, offset=None, mat=None):
    """
    Args:
        x:
        y:
        offset: [int, int] - 此处offset为切割的起始坐标
        mat:
    Returns:
        coord: x, y
    """
    if offset:
        x = x - offset[0]
        y = y - offset[1]

    coord = np.array([x, y])
    coord = coord.transpose(1, 0)

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
        sep = "\t",
        header = 0,
        usecols = title,
        dtype = type_dict,
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


def gef_trans(gef_file, offset, mat, output_path):
    shutil.copy(gef_file, output_path)
    with h5py.File(output_path, 'r+') as h:
        expression = h['geneExp']['bin1']['expression'][:]
        new_x, new_y = trans_points(expression['x'], expression['y'], offset, mat)
        h['geneExp']['bin1']['expression']['x'] = new_x
        h['geneExp']['bin1']['expression']['y'] = new_y


def gem_trans(gem_file, offset, mat, output_path):
    """
    Args:
        gem_file:
        offset:
        mat:
        output_path: str - 带文件名
    """
    gem = gem_read(gem_file)

    # gem['x'] = gem['x'] - min(gem['x'])
    # gem['y'] = gem['y'] - min(gem['y'])

    new_x, new_y = trans_points(gem['x'], gem['y'], offset, mat)

    gem['x'] = np.int_(np.round(new_x))
    gem['y'] = np.int_(np.round(new_y))
    gem.to_csv(output_path, sep='\t', index=False)


def trans_gem_by_json(gem_path, cut_json_path, align_json_path, output_path):
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

    for gem_file in tqdm(gem_list, desc='Gem', ncols=100):
        gem_name = os.path.basename(gem_file).split('.')[0]
        for key in mask_cut_info.keys():
            if gem_name in key:
                mask_cut = mask_cut_info[key]
                break
        else:
            mask_cut = None

        for key in align_info.keys():
            if gem_name in key:
                align = align_info[key]
                break
        else:
            align = None

        if mask_cut is not None or align is not None:
            # mask_cut = None
            mat = align['mat']
            if gem_file.endswith('txt') or gem_file.endswith('gem') or gem_file.endswith('gem.gz'):
                gem_trans(
                    gem_file, mask_cut, mat, os.path.join(output_path, f"{gem_name}.gem")
                )
            elif gem_file.endswith('gef'):
                gef_trans(
                    gem_file, mask_cut, mat, os.path.join(output_path, f"{gem_name}.gef")
                )


if __name__ == "__main__":
    # trans_gem_by_json(gem_path=r"D:\02.data\luqin\E14-16h_a_bin1_image_gem",
    #                   cut_json_path=r"D:\02.data\luqin\E14-16h_a_bin1_image_regis\align_info.json",
    #                   align_json_path=r"D:\02.data\luqin\E14-16h_a_bin1_image_regis\align_info.json",
    #                   output_path=r"D:\02.data\luqin\E14-16h_a_bin1_image_gem\new_gem")
    # read_gem_from_gef(r"/media/Data1/user/szl/liyumei/data/output/gem/SS200000122BL_B1_L1_x7649_y3592_w8139_h6537.gef")

    aaa = gem_read(r"D:\02.data\liuhuanlin\SS200000122BL_B1_L1_x7649_y3592_w8139_h6537.gem.gz")