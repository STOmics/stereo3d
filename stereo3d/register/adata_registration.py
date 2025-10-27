# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/6/3 17:25
🌟 File  : adata_registration.py
🌟 Description  : 
🌟 Key Words  :
"""
import os
import glog
import copy

import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

from paste import pairwise_align as paste_align, \
    generalized_procrustes_analysis, \
    paste_align_accuracy

from stereo3d.h5ad.data_process import sort_file_names


class AlignSlicer:
    def __init__(self,
                 adata_list: list,
                 h5ad_path: str,
                 out_dir: str,
                 spatial_key: str,
                 spatial_regis_key: str,
                 anno: str,
                 anno_color: str):
        """
            Align coordinates of spatial transcriptomics and return the aligned coordinates in .obsm['spatial_regis'].

            Args:
                adata_list: list contain all adata,
                    which has been formatted to meet the formatting requirements of the 3d process.

                h5ad_path: path of input of .h5ad files.(format h5ad files)

                out_dir: path of save update registered .h5ad files.

                spatial_key: the column key in .obsm, default to be 'spatial'.

                spatial_regis_key: the column key in .obsm,
                    contain the new coordinates of the slice after alignment, default is 'spatial_regis'.

                anno: The column key/name that identifies the grouping information(for example,
                    clusters that correspond to different cell types)of spots.

                anno_color: The key in .uns, corresponds to a dictionary that map group names to group colors.

            Returns:
                update adata, the aligned coordinates are stored in .obsm[spatial_regis_key].
                pis, list,transfer matrix,
        """
        self.adata_list = copy.deepcopy(adata_list)
        self.h5ad_path = h5ad_path
        self.out_dir = out_dir
        self.spatial_key = spatial_key
        self.spatial_regis_key = spatial_regis_key
        self.anno = anno
        self.anno_color = anno_color

    def plot(self, adata_list, spatial):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        # plot
        for adata in adata_list:
            data = adata.obsm[spatial]
            labels = adata.obs[self.anno].tolist()
            cat_list = adata.obs[self.anno].cat.categories.tolist()

            colors = [adata.uns[self.anno_color][cat_list.index(label)] for label in labels]
            ax.scatter(data[:, 0], data[:, 1], c = colors, marker = 'o', s = 3, alpha = 0.8)
        # add axis label.
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        fig.savefig(os.path.join(self.out_dir, spatial + "_plot.png"))
        print("save plot in " + os.path.join(self.out_dir, spatial + "_plot.png"))
        plt.show()
        plt.close(fig)

    # x,y coordinates are taken for the function of registration
    def align_2d(self):
        adata_st_list = copy.deepcopy(self.adata_list)
        for i in range(len(adata_st_list)):
            adata_st_list[i].obsm[self.spatial_key] = np.delete(adata_st_list[i].obsm[self.spatial_key], 2, axis = 1)
        return adata_st_list

    # add z coordinates
    def align_3d(self, adata_st_list):
        sorted_file_names = sort_file_names(self.h5ad_path, suffix = '.h5ad')
        for i, (x, y) in enumerate(zip(sorted_file_names, adata_st_list)):
            adata_st_list[i].obsm[self.spatial_regis_key] = \
                np.insert(adata_st_list[i].obsm[self.spatial_regis_key],
                          2, self.adata_list[i].obsm[self.spatial_key][:, 2], axis = 1)
            adata_st_list[i].obsm[self.spatial_key] = \
                np.insert(adata_st_list[i].obsm[self.spatial_key],
                          2, self.adata_list[i].obsm[self.spatial_key][:, 2], axis = 1)
            adata_st_list[i].write(os.path.join(self.out_dir, x))
        return adata_st_list


class Paste(AlignSlicer):
    def __init__(self, adata_list, h5ad_path, out_dir, spatial_key, spatial_regis_key, anno, anno_color):
        super().__init__(adata_list, h5ad_path, out_dir, spatial_key, spatial_regis_key, anno, anno_color)

    def align(self):
        print("Before align...")
        self.plot(self.adata_list, self.spatial_key)
        print("Using PASTE algorithm for alignment.")
        adata_st_list = self.align_2d()
        # Align spots
        print("Aligning spots...")
        pis = []
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            pi = paste_align(adata_st_list[i], adata_st_list[i + 1], spatial_key = self.spatial_key)
            pis.append(pi)
        # Transform
        S1, S2 = generalized_procrustes_analysis(adata_st_list[0].obsm[self.spatial_key],
                                                 adata_st_list[1].obsm[self.spatial_key],
                                                 pis[0])
        adata_st_list[0].obsm[self.spatial_regis_key] = S1
        adata_st_list[1].obsm[self.spatial_regis_key] = S2
        for i in range(1, len(adata_st_list) - 1):
            S1, S2 = generalized_procrustes_analysis(adata_st_list[i].obsm[self.spatial_regis_key],
                                                     adata_st_list[i + 1].obsm[self.spatial_key],
                                                     pis[i])
            adata_st_list[i + 1].obsm[self.spatial_regis_key] = S2
        print("save align coordinates in .obsm" + '_' + self.spatial_regis_key)
        adata_st_list = self.align_3d(adata_st_list)
        print("After align...")
        self.plot(adata_st_list, self.spatial_regis_key)
        return adata_st_list, pis


def align(
        adata,
        key='spatial_mm',
        regis_key='spatial_regis',
        anno='auto_anno',
        anno_color='anno_color',
        method='paste',
        file_path='',
        output_path='',
):
    """
    Args:
        adata: str | list
        key: spatial_mm
        regis_key: spatial_regis | spatial_affine
        anno: auto_anno
        anno_color: anno_color
        method: paste | slat

    Return:
        adata_list:
    """
    if isinstance(adata, str):
        sorted_file_names = sort_file_names(adata, suffix = '.h5ad')
        adata_list = [ad.read(os.path.join(adata, i)) for i in sorted_file_names]
    elif isinstance(adata, list):
        adata_list = adata
    else:
        return

    if method == 'paste':
        paste = Paste(adata_list,
                      h5ad_path=file_path,
                      spatial_key=key,
                      out_dir = output_path,
                      spatial_regis_key=regis_key,
                      anno=anno,
                      anno_color=anno_color)

        adata_list, pis = paste.align()
        align_accur = paste_align_accuracy(adata_list, pis, anno = anno)
        glog.info(f"Align accuracy -- {regis_key}: {align_accur}")
    else:
        return

    # h5ad2img(adata_list, h5ad_path = os.path.join(self.output_path, self._h5ad_output),
    #          out_dir = os.path.join(self.output_path, self._align_qc_data, '01.before_align'),
    #          spatial_key = key, suffix = '.tif')
    #
    # h5ad2img(adata_list, h5ad_path = os.path.join(self.output_path, self._h5ad_output),
    #          out_dir = os.path.join(self.output_path, self._align_qc_data, '02.after_align'),
    #          spatial_key = regis_key, suffix = '.tif')
    #
    # # align accuracy
    # dd_value_before, pc_score_before = fft_score(tif_path = os.path.join(self.output_path,
    #                                                                      self._align_qc_data,
    #                                                                      '01.before_align'))
    # dd_value_regis, pc_score_regis = fft_score(tif_path = os.path.join(self.output_path,
    #                                                                    self._align_qc_data,
    #                                                                    '02.after_align'))
    #
    # plot_fft_score(dd_value_before, dd_value_regis, figsize = (6, 2), cutoff = 10,
    #                title = 'Displacement deviation value',
    #                ylabel = 'Dd value',
    #                savefig = os.path.join(self.output_path, self._align_qc_data, 'dd_value.png'))
    #
    # plot_fft_score(pc_score_before, pc_score_regis, figsize = (6, 2), cutoff = 30,
    #                title = 'Phase correlation score',
    #                ylabel = 'PC score',
    #                savefig = os.path.join(self.output_path, self._align_qc_data, 'pc_score.png'))


if __name__ == '__main__':
    adata_path = r"C:\Users\87393\Downloads\test_ad"
    output_path = r"C:\Users\87393\Downloads\test_ad1"
    sorted_file_names = sorted([os.path.join(adata_path, i) for i in os.listdir(adata_path)])

    adata_list = [ad.read(i) for i in sorted_file_names]
    align(
        adata=adata_list,
        key='spatial_mm',
        regis_key='spatial_regis',
        anno='leiden',
        anno_color='leiden_colors',
        method='paste',
        file_path=adata_path,
        output_path=output_path,
    )

    print(1)
