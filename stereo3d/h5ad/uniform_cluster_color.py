import anndata as ad
from stereo.core.stereo_exp_data import AnnBasedStereoExpData
import numpy as np
from .data_process import sort_file_names
import os


def uniform_cluster_color(
    h5ad_path:str,
    out_path:str,
):  
    # 01. Select adatas. Here, select the first, second and third digit adata to construct
    # the reference h5ad. Users can select from multiple pieces of data.
    sorted_file_names = sort_file_names(h5ad_path,suffix='.h5ad')
    #  optional q1,q2,q3
    q1 = int(np.ceil(np.percentile(range(0,len(sorted_file_names)), 25)))
    q2 = int(np.ceil(np.percentile(range(0,len(sorted_file_names)), 50)))
    q3 = int(np.ceil(np.percentile(range(0,len(sorted_file_names)), 75)))
    adatas = []
    for file in [sorted_file_names[i] for i, _ in enumerate(sorted_file_names) if i in [q1,q2,q3]]:
        with open(os.path.join(h5ad_path, file), 'r') as f:
            adata = ad.read(os.path.join(h5ad_path, file))
            adatas.append(adata)
            del adata
    adatas[0].obs['batch'] = '0'
    adatas[1].obs['batch'] = '1'
    adatas[2].obs['batch'] = '2'
    adata_all = ad.concat(adatas,join='inner')
    del adatas
    #  02. merge adatas and bathch intergration
    data_all = AnnBasedStereoExpData(None, based_ann_data=adata_all)
    del adata_all
    # data_all.tl.raw_checkpoint()
    # data_all.tl.raw
    # data_all.tl.highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5,
    # res_key='highly_variable_genes', n_top_genes=2000)
    # data_all._ann_data = data_all._ann_data[:, data_all._ann_data.var.highly_variable]
    # data_all.exp_matrix = data_all._ann_data.X.A
    data_all.tl.normalize_total()
    data_all.tl.log1p()
    data_all.tl.pca(use_highly_genes=False, n_pcs=50, res_key='pca')
    #  batches integration
    data_all.tl.batches_integrate(pca_res_key='pca', res_key='pca_integrated')
    data_all.tl.neighbors(pca_res_key='pca_integrated', n_pcs=50, res_key='neighbors_integrated')
    data_all.tl.umap(pca_res_key='pca_integrated', neighbors_res_key='neighbors_integrated', res_key='umap_integrated')
    data_all.tl.leiden(neighbors_res_key='neighbors_integrated', res_key='leiden')
    # 03. uniform cluster color
    uniform_list = []  
    for string in set(data_all._ann_data.obs.leiden):  
        new_string = "uniform_" + string  
        uniform_list.append(new_string)  
    uniform_dict = {key: value for key, value in zip(set(data_all._ann_data.obs.leiden), uniform_list)}
    data_all._ann_data.obs['uniform_leiden'] = [uniform_dict[cl] for cl in data_all._ann_data.obs.leiden] 
    # 04. write adata
    data_all._ann_data.write(out_path+'st_ref.h5ad')

    del data_all