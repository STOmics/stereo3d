#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/7/22 5:39 PM
# @Author  : zhangchao
# @File    : txt2adata.py
# @Email   : zhangchao5@genomics.cn
import os.path

import pandas as pd
import os.path as osp
import scipy.sparse as sp
import pandas as pd
import tqdm
from anndata import AnnData
import scanpy as sc
import glob


def data_encapsulation(data, bin_size, save=False):
    # allowed_columns = ["geneID", "MIDCount", f"bin{bin_size}_label", "x", "y"]
    # for col in data.columns:
    #     assert col in allowed_columns, "Error, Got an invalid column name, which only support {}".format(allowed_columns)
    vals = data["MIDCount"].to_numpy()
    cell_list = data[f"bin{bin_size}_label"].astype("category")
    data["geneID"] = data["geneID"].fillna("Unkown")
    gene_list = data["geneID"].astype("category")
    row = cell_list.cat.codes.to_numpy()
    col = gene_list.cat.codes.to_numpy()
    coo = data.groupby(f"bin{bin_size}_label").mean(numeric_only=True)[["x", "y"]]
    obs = pd.DataFrame(index=(map(str, cell_list.cat.categories)))
    var = pd.DataFrame(index=(map(str, gene_list.cat.categories)))
    adata_x = sp.csr_matrix((vals, (row, col)), shape=(len(obs), len(var)))
    adata = AnnData(adata_x, obs=obs, var=var)
    adata.obsm['spatial'] = coo.to_numpy()
    if isinstance(save, str):
        adata.write_h5ad(save)
    return adata


def generate_binlabel(data, bin_size=50):
    assert isinstance(data, pd.DataFrame)
    columns = data.columns.values
    for col in ["x", "y"]:
        assert col in columns
    for col in ["new_x", "new_y", f"bin{bin_size}_label"]:
        assert col not in columns
    
    data["new_x"] = data.x // bin_size
    data["new_y"] = data.y // bin_size
    data[f"bin{bin_size}_label"] = data.new_x.map(str) + "-" + data.new_y.map(str)


def batch_cluster(matrix_dir: str, save_dir: str, bin_size=20):
    gem_list = glob.glob(osp.join(matrix_dir, "*.gem"))
    for it in tqdm.tqdm(gem_list, desc='Bin-{} Cluster'.format(bin_size), ncols=100):
        i = osp.basename(it)
        save_path = osp.join(save_dir, i.replace('.gem', '.h5ad'))  # 根据需求修改save_name
        df = pd.read_csv(it, comment='#', sep='\t')  # 根据 lasso 得到文件去读
        generate_binlabel(df, bin_size=bin_size)  # 根据需求修改bin_size
        data_encapsulation(df, bin_size=bin_size, save=save_path)


def batch_spatial_leiden(h5ad_path: str, save_path: str, spot_size=15):
    from matplotlib import pyplot as plt

    h5ad_list = glob.glob(osp.join(h5ad_path, "*.h5ad"))
    for it in tqdm.tqdm(h5ad_list, desc='Spatial Leiden', ncols=100):
        adata = sc.read_h5ad(it)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        with plt.rc_context():
            # sc.pl.umap(adata, color="leiden", show=False)
            sc.pl.spatial(adata, color="leiden", spot_size=spot_size, show=False)
            dst_path = osp.join(save_path, osp.basename(it).replace('.h5ad', '.png'))
            plt.savefig(dst_path, bbox_inches='tight')


if __name__ == '__main__':
    matrix_dir = r'C:\Users\BGI\Desktop\stereo3d-test\output1\03.gem'
    save_dir = r'C:\Users\BGI\Desktop\stereo3d-test\output1\05.cluster'

    batch_cluster(matrix_dir=matrix_dir, save_dir=save_dir)
    batch_spatial_leiden(h5ad_path=save_dir, save_path=save_dir)
    
