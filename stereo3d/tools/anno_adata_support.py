import scanpy as sc
import numpy as np
import argparse

import os
import sys

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_PATH))
sys.path.append(ROOT_PATH)
print(CURR_PATH)
print(ROOT_PATH)

import tqdm
import glog
import warnings
warnings.filterwarnings('ignore')
from stereo3d.gem.transform import trans_matrix_by_json
from stereo3d.h5ad.uniform_cluster_color_v2 import read_and_parse_by_celltype, organ_mesh


def create_3D_coord(h5ad_list, out_path, cluster_key="leiden"):
    adatas = []
    for file in tqdm.tqdm(h5ad_list, desc='SpecifiedColor-DataLoad', ncols=100):
        adata = sc.read_h5ad(file)
        adatas.append(adata)

    categories=[]

    for i, c in enumerate(tqdm.tqdm(range(len(adatas)), desc='SpecifiedColor', ncols=100)):
        sub_data = adatas[c]
        sub_data.obsm['spatial_mm'] = np.zeros((sub_data.shape[0], 2))
        sub_data.obsm['spatial_mm'][:, 0] = sub_data.obsm['spatial'][:, 0] * 500 / 1000000
        sub_data.obsm['spatial_mm'][:, 1] = sub_data.obsm['spatial'][:, 1] * 500 / 1000000
        categories=categories+sub_data.obs[cluster_key].tolist()


        z_value = [i * 0.008] * sub_data.obs.shape[0]
        z_value = np.array(z_value).reshape(-1, 1)

        sub_data.obsm['spatial_mm'] = np.hstack([sub_data.obsm['spatial_mm'], z_value])
        save_path = os.path.join(out_path, os.path.basename(h5ad_list[i]))
        sub_data.write_h5ad(save_path)
    return list(set(categories))

def adata_insert_organ(matrix_path, output_path,
                       cut_json_path: [str, None] = None,
                       align_json_path: [str, None] = None,
                       cluster_key: str = "leiden"):
    trans_adata_path = os.path.join(output_path, "11.tans_adata")
    os.makedirs(trans_adata_path, exist_ok=True)
    organ_path = os.path.join(output_path, "12.adata_organ")
    os.makedirs(organ_path, exist_ok=True)
    if isinstance(cut_json_path, str) and isinstance(align_json_path, str):

        trans_matrix_by_json(matrix_path, cut_json_path, align_json_path,trans_adata_path)
        glog.info('Completed the reuse of registration parameters to anndata file')

    else:
        for filename in os.listdir(matrix_path):
            import shutil
            if filename.endswith('.h5ad'):
                source_path = os.path.join(matrix_path, filename)
                target_path = os.path.join(trans_adata_path, filename)
                shutil.copy2(source_path, target_path)

    h5ad_list = [os.path.join(trans_adata_path, i) for i in os.listdir(trans_adata_path) if i.endswith(".h5ad")]
    categories = create_3D_coord(h5ad_list, trans_adata_path, cluster_key)
    for c in tqdm.tqdm(categories, desc='Organ', ncols=100):
        organ_path_ = read_and_parse_by_celltype(
            outdir=organ_path, spatial_regis='spatial_mm', anno=cluster_key, celltype=c,
            adata_list=None, h5ad_list=h5ad_list, sc_xyz=None)
        organ_mesh(organ_path_, organ_path_.replace('.txt', '.obj'))
    glog.info('Completed insert organ')


def main(args, para):
    glog.info('Embed custom clustering results into the 3D model"')

    adata_insert_organ(matrix_path=args.matrix_path,
                       output_path=args.output_path,
                       cut_json_path=args.cut_json_path,
                       align_json_path=args.align_json_path,
                       cluster_key=args.cluster_key)
    glog.info('Finished: custom clustering 3D result in 12.adata_organ')

usage = """ Submit data format verification """
PROG_VERSION = 'v0.0.1'
if __name__ == "__main__":
    #
    # matrix_path = r"D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output\ann_file"
    # output_path = r"D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output"
    # align_json_path = r"D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output\02.register\01.align_mask\align_info.json"
    # cut_json_path = r"D:\00.user\zhangying7\stereo3D\Drosophila_melanogaster_demo\output\02.register\00.crop_mask\mask_cut_info.json"json
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-m", "--matrix_path", action="store", dest="matrix_path", type=str, required=True,
                        help="Input matrix path.")
    parser.add_argument("-o", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="output path. ")
    parser.add_argument("-aj", "--align_json_path", action="store", dest="align_json_path", type=str,default=None,
                        help="align json path.")
    parser.add_argument("-cj", "--cut_json_path", action="store", dest="cut_json_path", type=str, default=None,
                        help="align json path.")
    parser.add_argument("-ck", "--cluster_key", action="store", dest="cluster_key", type=str, default="leiden",
                        help="the save cluster label ")
    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)
