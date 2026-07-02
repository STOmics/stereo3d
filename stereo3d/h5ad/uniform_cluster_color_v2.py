import logging
import glog
from typing import Optional
import anndata
import gc

import anndata as ad
import os
import scanpy as sc
from matplotlib import pyplot as plt
from anndata import AnnData
import tqdm
import numpy as np

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def _is_placeholder_var_names(var_names):
    """True if all var_names are placeholders gene_{i} (cellbin.gef geneName was empty)."""
    import re
    pat = re.compile(r'^gene_\d+$')
    return all(pat.match(str(v)) is not None for v in var_names)


def _check_var_names_alignment(adatas, h5ad_list=None):
    """
    Guard var_names consistency before multi-slice concat.
    """
    if len(adatas) <= 1:
        return
    import os
    ref = list(adatas[0].var_names)
    ref_placeholder = _is_placeholder_var_names(ref)
    identical = True
    for k, a in enumerate(adatas[1:], start=1):
        cur = list(a.var_names)
        if cur != ref:
            identical = False
        cur_placeholder = _is_placeholder_var_names(cur)
        if (ref_placeholder or cur_placeholder) and (cur != ref or cur_placeholder != ref_placeholder):
            tag = os.path.basename(h5ad_list[k]) if h5ad_list else f'slice {k}'
            raise RuntimeError(
                f'[cellbin multi-slice] var_names are placeholders (gene_i) but slice {tag} '
                f'differs from slice0 (n={len(cur)} vs {len(ref)}). Placeholder names cannot '
                f'guarantee cross-slice gene identity; merging would misalign genes. '
                f'Please fill geneName in cellbin.gef before multi-slice clustering.'
            )
    if not identical:
        # Real names + mismatch: inner intersection is safe
        sizes = [len(set(a.var_names)) for a in adatas]
        inter = len(set(ref).intersection(*[set(a.var_names) for a in adatas[1:]]))
        glog.warning(
            f'[multi-slice] var_names differ across slices; aligning via join=inner '
            f'(per-slice unique genes {sizes}, intersection {inter}).'
        )
    elif ref_placeholder:
        glog.warning(
            '[cellbin multi-slice] var_names are placeholders gene_i (cellbin.gef geneName '
            'was empty, fell back to row index). Multi-slice merge assumes identical gene '
            'order across slices (only safe within one SAW run / gene panel).'
        )


def uniform_cluster_color(h5ad_list: list, out_path:str, z_index_list: list):
    # h5ad_list = glob.glob(os.path.join(h5ad_path, "*.h5ad"))
    import harmonypy

    harmonypy.harmony.logger.setLevel(logging.WARN)  # Disable harmony's info log, harmony is a dependency library of sc
    adatas = []
    for file in tqdm.tqdm(h5ad_list, desc='SpecifiedColor-DataLoad', ncols=100):
        adata = ad.read(file)
        adatas.append(adata)
        del adata
    # Multi-slice gene-alignment guard: check var_names before concat 
    _check_var_names_alignment(adatas, h5ad_list=h5ad_list)
    # adata_all = ad.concat(adatas)
    # join='inner': align by var_names intersection (safe with real gene names)
    adata_all = AnnData.concatenate(*adatas, join='inner')
    del adatas

    sc.pp.normalize_total(adata_all)
    sc.pp.log1p(adata_all)
    sc.tl.pca(adata_all, svd_solver='arpack')

    sc.external.pp.harmony_integrate(adata_all, key='batch')
    sc.pp.neighbors(adata_all, n_neighbors=10, use_rep='X_pca_harmony')
    sc.tl.umap(adata_all)
    sc.tl.leiden(adata_all)

    for i, c in enumerate(tqdm.tqdm(adata_all.obs["batch"].cat.categories, desc='SpecifiedColor', ncols=100)):
        sub_data = adata_all[adata_all.obs["batch"] == c]
        sub_data.obsm['spatial_mm'] = np.zeros((sub_data.shape[0], 2))
        sub_data.obsm['spatial_mm'][:, 0] = sub_data.obsm['spatial'][:, 0] * 500 / 1000000
        sub_data.obsm['spatial_mm'][:, 1] = sub_data.obsm['spatial'][:, 1] * 500 / 1000000

        z_value = [z_index_list[i]] * sub_data.obs.shape[0]
        z_value = np.array(z_value).reshape(-1, 1)
        # print(sub_data.obsm['spatial_mm'], z_value)

        sub_data.obsm['spatial_mm'] = np.hstack([sub_data.obsm['spatial_mm'], z_value])
        save_path = os.path.join(out_path, os.path.basename(h5ad_list[i]))
        with plt.rc_context():
            # sc.pl.umap(adata, color="leiden", show=False)
            sc.pl.spatial(sub_data, color="leiden", spot_size=15, show=False)
            dst_path = save_path.replace('.h5ad', '.png')
            plt.savefig(dst_path, bbox_inches='tight')
        sub_data.write_h5ad(save_path)
    return adata_all.obs['leiden'].cat.categories.tolist()


def read_and_parse_by_celltype(outdir: str, spatial_regis: str, anno: str, celltype: str,
                               h5ad_list: Optional[list] = None, adata_list: Optional[list] = None,
                               sc_xyz: Optional[list] = None, z_index_list: list = []):
    """
    Get x,y,z,anno columns as mesh input.

    Args:
        outdir:
        h5ad_list: path of input of .h5ad files.
        adata_list:
        spatial_regis: The column key in .obsm, default to be 'spatial_regis'. note that x,y,z
        anno: The column key/name that identifies the grouping information(for example, clusters that correspond
              to different cell types)of spots.
        sc_xyz:The scale by which the spatial points in h5ad is scaled. when the `sc_xyz` is list, the model is
               scaled along the xyz axis at different scales. If `sc_xyz` is None, there will be by scale by
               defult parameter.

    Returns:
        a list of adata. and update adata to outpath which format that meets the requirements of 3D flow analysis.
    """
    xli = []
    yli = []
    zli = []
    tyli = []
    if h5ad_list:
        fnames = h5ad_list
        adata_list = []
        for fname in fnames:
            adata = anndata.read(fname)
            adata_list.append(adata)
    elif adata_list:
        adata_list = adata_list
    else:
        raise ValueError(f"h5ad_path and adata_list should have at least one that is not None.")
    for i, adata in enumerate(adata_list):
        if adata.uns.get('data_unit', {}).get('binsize') == 'cellbin':
            binsize = 10
        else:
            binsize = 20
        # z_size = adata.uns['data_unit']['z_size']
        # match = re.search(r'(\d+)um', z_size)
        # z_size = int(match.group(1))
        z_size = z_index_list[i] * 100
        if sc_xyz is None:
            sc_xyz = [None] * 3
            sc_xyz[0] = 1000 / (binsize * 0.5)
            sc_xyz[1] = 1000 / (binsize * 0.5)
            #sc_xyz[2] = 1000 / z_size
        x = (adata[adata.obs[anno] == celltype].obsm[spatial_regis][:, 0] * sc_xyz[0]).tolist()
        y = (adata[adata.obs[anno] == celltype].obsm[spatial_regis][:, 1] * sc_xyz[1]).tolist()
        #z = (adata[adata.obs[anno] == celltype].obsm[spatial_regis][:, 2] * sc_xyz[2]).tolist()
        z = [z_size] * adata[adata.obs[anno] == celltype].obsm[spatial_regis].shape[0]
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

    data = np.column_stack((xli, yli, zli, tyli))
    organ_path = os.path.join(outdir, '{}.txt'.format(celltype))
    with open(organ_path, 'w') as O:
        for row in data:
            O.write('\t'.join([str(elem) for elem in row]) + '\n')
    return organ_path


def organ_mesh(
        organ_path: str,
        mesh_output_path: str,
        z_interval=0.008,
        random = None,
):
    from stereo3d.mesh.create_mesh_3d import points_3d_to_mesh

    if organ_path.endswith('.txt'):
        points_3d = list()
        with open(organ_path, 'r') as f:
            for i in f.readlines():
                points = [float(j) / 100 for j in i.split('\t')[:3]]
                points_3d.append(points)
        points_3d = np.array(points_3d)

    else: 
        # 'ply'
        points_3d = organ_path
    
    output_path, name = os.path.split(mesh_output_path)

    if len(np.unique(points_3d[:, 2])) == 1:
        glog.warning(f'\n The z_interval of the points is only 1 dims.')
        return
    
    if random:
        random = 200

    points_3d_to_mesh(points_3d,
                      z_interval=z_interval,
                      mesh_scale=1,
                      output_path=output_path,
                      show_mesh=False,
                      name=name.replace('.obj', ''),
                      random=random)


if __name__ == '__main__':
    organ_path = r'C:\Users\BGI\Desktop\stereo3d-test\output\07.organ\1.txt'
    mesh_output_path = r'C:\Users\BGI\Desktop\stereo3d-test\output\07.organ\1.obj'
    organ_mesh(organ_path, mesh_output_path)
