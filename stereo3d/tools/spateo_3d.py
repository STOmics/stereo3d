import argparse
import json
import os.path
from pathlib import Path
import numpy as np
import spateo as st
import dynamo as dyn
import tqdm
from dynamo.preprocessing import Preprocessor
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import scanpy as sc
import datetime


class Naming(object):
    """ Manage process output files """
    def __init__(self, output_path: str, tag: str = 'spateo'):
        """
        Args:
            output_path: The root path where the output results are to be saved
            tag: Prefix name of the file to be saved
        """
        self._output: str = output_path
        self.tag: str = tag

    @staticmethod
    def idir(dir_path: str) -> Path:
        """ If the path does not exist, create it
        Args:
            dir_path: Customized Path

        Returns: path with type - Path

        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

        return p

    @property
    def h5ad_dir(self, ):
        """ Directory where h5ad is stored
        """
        return self.idir(os.path.join(self._output, 'h5ad'))

    @property
    def trans_h5ad_dir(self, ):
        """ Directory where h5ad is stored
        """
        return self.idir(os.path.join(self._output, 'trans_h5ad'))

    @property
    def h5ad_file(self, ):
        return '{}_alignment.h5ad'.format(self.tag)

    @property
    def trans_param_file(self, ):
        return 'alignment.json'

    def trans_h5ad_file(self, prefix: str):
        return '{}_trans.h5ad'.format(prefix)

    @property
    def mesh_models_dir(self, ):
        return self.idir(os.path.join(self._output, 'mesh_models'))

    @property
    def pc_models_dir(self, ):
        return self.idir(os.path.join(self._output, 'pc_models'))

    @property
    def plot_dir(self, ):
        return self.idir(os.path.join(self._output, 'plot'))

    @property
    def original_slices(self, ): return self.tag

    @property
    def align_slices(self, ): return '{}_aligned'.format(self.tag)

    @property
    def align_overlay_slices(self, ): return '{}_overlay_aligned'.format(self.tag)

    @property
    def plotter_pt_filename(self, ): return '0_{}_aligned_pc_model.vtk'.format(self.tag)

    @property
    def td_pt_filename(self, ): return '{}_3d_pc.gif'.format(self.tag)

    @property
    def plotter_mesh_filename(self, ): return '0_{}_aligned_mesh_model.vtk'.format(self.tag)

    @property
    def td_mesh_filename(self, ): return '{}_3d_mesh.gif'.format(self.tag)

    @staticmethod
    def sub_mesh_model_file(idx: int, sub_tag: str):
        return '{}_{}_aligned_mesh_model.vtk'.format(idx, sub_tag)

    @staticmethod
    def sub_pt_model_file(idx: int, sub_tag: str):
        return '{}_{}_aligned_pc_model.vtk'.format(idx, sub_tag)


def simplify(data: sc.AnnData, keep_obs=['annotation'], keep_obsm=['spatial'],):
    """ Remove fields that are not used in calculations to reduce memory consumption
    Args:
        data: Complex raw data
        keep_obs: Fields that need to be retained
        keep_obsm: Fields that need to be retained

    Returns: Simplified data

    """
    _ = sc.AnnData(
        obs=data.obs[keep_obs],
        obsm={k: data.obsm[k] for k in keep_obsm},
        X=data.X
    )

    return _


def write_spateo_h5ad(data: sc.AnnData, file_path: Path):
    """ Handle types that cannot be written in custom fields
    Args:
        data: Anndata with custom fields
        file_path: save path

    Returns:

    """
    if 'ntr' in data.obs:
        del data.obs['ntr']
    if 'ntr' in data.var:
        del data.var['ntr']
    data.write_h5ad(file_path)


class Spateo3D(object):
    """
    3D reconstruction process based on the Sapteo framework:
        https://github.com/aristoteleo/spateo-release
    """
    def __init__(self,
                 z_step: int = 20,
                 cluster_pts: int = 1000,
                 spatial_key='spatial',
                 cluster_key='annotation',
                 key_slice='slice_ID',  # single h5ad
                 key_add='align_spatial',
                 key_3d='spatial_3D',
                 key_tissue='tissue'):
        self._z_step = z_step
        self._data_path: str = ''
        self._spatial_key = spatial_key
        self._cluster_key = cluster_key
        self._key_add = key_add
        self._key_3d = key_3d
        self._key_slice = key_slice
        self._key_tissue = key_tissue
        self._cluster_point = cluster_pts
        self._tag: str = ''
        self.device: str = 'cpu'
        self._ng = Naming('')
        self._trans: list = []
        self._slices: list = []

    @property
    def color_key(self, ): return '{}_colors'.format(self._cluster_key)

    def _prepare(self, data_path: str, show_fig: bool = True):
        """
        Args:
            data_path: The file path where h5ad is stored. Multiple slices can be in one file or in multiple files
            in the same directory.

        Returns:

        """
        # read
        if os.path.isfile(data_path):
            adata = st.read(data_path)
        else:
            files = sorted(os.listdir(data_path))[:2]
            slices0 = []
            for idx, f in enumerate(tqdm.tqdm(files, desc='Load data', colour='green', unit='h5ad', ncols=100)):
                raw_data = st.read(os.path.join(data_path, f))
                sim_data = simplify(raw_data)
                sim_data.obs[self._key_slice] = [idx] * sim_data.obsm[self._spatial_key].shape[0]
                sim_data.obs.index = '{}_'.format(idx) + sim_data.obs.index
                slices0.append(sim_data)
            adata = sc.concat(slices0, axis=0, join='inner', merge='unique')
            cgt = ['S{}'.format(i) for i in range(len(files))]
            adata.obs[self._key_slice] = adata.obs[self._key_slice].astype('category')
            adata.obs[self._key_slice] = adata.obs[self._key_slice].cat.rename_categories(cgt)

        # downsample
        if self._cluster_point > 0:
            sampled_ind = (adata.obs.groupby(self._cluster_key, group_keys=False).apply(
                lambda x: x.sample(n=self._cluster_point, random_state=0, replace=True),
                include_groups=True).index)  # Sampling with replacement
            adata_downs = adata[sampled_ind, :]
            adata = adata_downs.copy()

        # adata.write_h5ad(Path(r'D:\data\stereo3d\spateo\em1.h5ad'))
        # adata = st.read(Path(r'D:\data\stereo3d\spateo\em1.h5ad'))

        # pca
        prep = Preprocessor()
        prep.preprocess_adata(adata, recipe='monocle')
        dyn.tl.reduceDimension(adata, basis='pca')
        slice_label = adata.obs[self._key_slice].cat.categories
        self._slices = [adata[adata.obs[self._key_slice] == s] for s in slice_label]
        if show_fig:
            self._show_slices(self._slices, self._spatial_key, self._ng.original_slices)

    def _show_slices(self, slices: list, key: str, save_path: str):
        st.pl.slices_2d(slices=slices, label_key=self._cluster_key, spatial_key=key,
                        height=2, center_coordinate=True, show_legend=True,
                        legend_kwargs={'loc': 'upper center', 'bbox_to_anchor': (0.5, 0),
                                       'ncol': 5, 'borderaxespad': -6, 'frameon': False},
                        save_show_or_return='save',
                        save_kwargs={"path": self._ng.plot_dir / save_path,
                                     "prefix": 'spateo',
                                     "ext": "png"}
                        )

    def _show_overlay_slices(self, slices: list, key: str, save_path: str):
        st.pl.overlay_slices_2d(slices=slices, spatial_key=key, height=2, overlay_type='both',
                                save_show_or_return='save',
                                save_kwargs={"path": self._ng.plot_dir / save_path,
                                             "prefix": 'spateo',
                                             "ext": "png"}
                                )

    def export_trans_para(self, ):
        with open(self._ng.plot_dir / self._ng.trans_param_file, 'w') as fd:
            dct = {}
            for i, t in enumerate(self._trans):
                dct[i + 1] = {k: v.tolist() for k, v in t.items()}
            json.dump(dct, fd, indent=2)

    def export_trans_h5ads(self, data_path: str):
        if os.path.isfile(data_path):
            adata = st.read(data_path)
            slice_label = adata.obs[self._key_slice].cat.categories
            slices = [adata[adata.obs[self._key_slice] == s] for s in slice_label]
            for s in slices:
                s.obsm[self._spatial_key] = s.obsm[self._spatial_key][:, :2]
            aligned_slices = st.align.morpho_align_apply_transformation(
                models=slices, spatial_key=self._spatial_key, key_added=self._key_add, transformation=self._trans)
            aligned_adata = st.concat(aligned_slices, axis=0, join='inner', merge='unique')
            for k in [self._key_slice, self._cluster_key]:
                aligned_adata.obs[k] = aligned_adata.obs[k].astype('category')
            write_spateo_h5ad(
                aligned_adata, self._ng.trans_h5ad_dir / self._ng.trans_h5ad_file(prefix='{}_trans'.format(self._tag)))
        else:
            files = sorted(os.listdir(data_path))[:2]
            slices = []
            for idx, f in enumerate(tqdm.tqdm(files, desc='Load data', colour='green', unit='h5ad', ncols=100)):
                raw_data = st.read(os.path.join(data_path, f))
                sim_data = simplify(raw_data)
                sim_data.obs[self._key_slice] = [idx] * sim_data.obsm[self._spatial_key].shape[0]
                sim_data.obs.index = '{}_'.format(idx) + sim_data.obs.index
                slices.append(sim_data)
            aligned_slices = st.align.morpho_align_apply_transformation(
                models=slices, spatial_key=self._spatial_key, key_added=self._key_add, transformation=self._trans)
            for idx, ad in enumerate(aligned_slices):
                tag = os.path.basename(files[idx]).replace('.h5ad', '')
                for k in [self._key_slice, self._cluster_key]:
                    ad.obs[k] = ad.obs[k].astype('category')
                    write_spateo_h5ad(
                        ad, self._ng.trans_h5ad_dir / self._ng.trans_h5ad_file(prefix='{}_trans'.format(tag)))

    def _alignment(self, show: bool = True, save_h5ad: bool = True):
        for s in self._slices:
            s.obsm[self._spatial_key] = s.obsm[self._spatial_key][:, :2]

        self._trans = st.align.morpho_align_transformation(
            models=self._slices, spatial_key=self._spatial_key, key_added=self._key_add, device=self.device,
            verbose=False, rep_layer='X_pca', rep_field='obsm', dissimilarity='cos')
        aligned_slices = st.align.morpho_align_apply_transformation(
            models=self._slices, spatial_key=self._spatial_key, key_added=self._key_add, transformation=self._trans)

        if show:
            self._show_slices(aligned_slices, self._spatial_key, self._ng.align_slices)
            self._show_overlay_slices(aligned_slices, self._key_add, self._ng.align_overlay_slices)

        aligned_adata = st.concat(aligned_slices, axis=0, join='inner', merge='unique')
        for k in [self._key_slice, self._cluster_key]:
            aligned_adata.obs[k] = aligned_adata.obs[k].astype('category')

        # write_spateo_h5ad(aligned_adata, Path(r'D:\data\stereo3d\spateo\em2.h5ad'))
        # aligned_adata = st.read(Path(r'D:\data\stereo3d\spateo\em2.h5ad'))

        slices_idx = np.array([[self._z_step * aligned_adata.obs[self._key_slice].cat.categories.get_loc(i)
                                for i in aligned_adata.obs[self._key_slice]]]).T
        aligned_adata.obsm[self._key_3d] = np.concatenate([aligned_adata.obsm[self._key_add], slices_idx], axis=1)

        label = aligned_adata.obs[self._cluster_key].cat.categories.tolist()
        colors = ['#{:02x}{:02x}{:02x}'.format(
            int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in plt.cm.rainbow(np.linspace(0, 1, len(label)))]
        if self.color_key not in aligned_adata.uns:
            aligned_adata.uns[self.color_key] = colors

        if save_h5ad:
            write_spateo_h5ad(aligned_adata, self._ng.h5ad_dir / self._ng.h5ad_file)

    def _reconstruct_outer_3d(self, ):
        aligned_adata = st.read(self._ng.h5ad_dir / self._ng.h5ad_file)
        labels = aligned_adata.obs[self._cluster_key].cat.categories.tolist()
        self._reconstruct_outer(aligned_adata)
        self._reconstruct_inner(labels)

    def _show_three_d(self, model, key: str, colormap, save_path: str, model_style='points'):
        st.pl.three_d_plot(model=model, key=key, model_style=model_style, model_size=8, show_axes=True,
                           window_size=(1200, 1200), show_outline=True, off_screen=True, jupyter='static',
                           outline_kwargs={'show_labels': False, 'outline_width': 3}, colormap=colormap,
                           filename=save_path
                           )

    def _reconstruct_outer(self, adata: sc.AnnData):
        colormap = {adata.obs[self._cluster_key].cat.categories[idx]: c for idx, c in enumerate(adata.uns[self.color_key])}
        point_cloud, plot_cmap = st.tdr.construct_pc(
            adata=adata, spatial_key=self._key_3d,
            groupby=self._cluster_key, key_added=self._key_tissue, colormap=colormap)
        st.tdr.save_model(model=point_cloud, filename=str(self._ng.pc_models_dir / self._ng.plotter_pt_filename))

        self._show_three_d(point_cloud, self._key_tissue,
                           plot_cmap,
                           str(self._ng.plot_dir / self._ng.td_pt_filename))

        mesh, _, _ = st.tdr.construct_surface(
            pc=point_cloud, key_added=self._key_tissue, alpha=0.6, cs_method='pyvista',  # cs_args={'mc_scale_factor': 1},
            smooth=5000, scale_factor=1.)
        self._show_three_d(mesh, self._key_tissue,
                           plot_cmap, str(self._ng.plot_dir / self._ng.td_mesh_filename),
                           model_style='surface')
        st.tdr.save_model(model=mesh, filename=str(self._ng.mesh_models_dir / self._ng.plotter_mesh_filename))

    def _reconstruct_inner(self, labels: list):
        point_cloud = st.tdr.read_model(filename=str(self._ng.pc_models_dir / self._ng.plotter_pt_filename))
        for idx, sub_type in enumerate(labels):
            try:
                sub_type_rpc = st.tdr.three_d_pick(model=point_cloud, key=self._key_tissue, picked_groups=sub_type)[0]
                color_ = mplc.to_hex(c=sub_type_rpc['{}_rgba'.format(self._key_tissue)][0], keep_alpha=True)
                # sub_type_tpc = st.tdr.interactive_rectangle_clip(model=sub_type_rpc, key=key_3d, invert=True)[0]
                sub_type_mesh, sub_type_pc, _ = st.tdr.construct_surface(
                    pc=sub_type_rpc, key_added=self._key_add, label=sub_type, color=color_, alpha=0.6,
                    cs_method='pyvista', smooth=5000, scale_factor=1.
                )
                st.tdr.save_model(
                    model=sub_type_pc, filename=str(self._ng.pc_models_dir /
                                                    self._ng.sub_pt_model_file(idx + 1, sub_type)))
                st.tdr.save_model(
                    model=sub_type_mesh, filename=str(self._ng.mesh_models_dir /
                                                      self._ng.sub_mesh_model_file(idx + 1, sub_type)))
            except:
                pass

    def reconstruct(self, data_path: str, output_path: str, aligned: bool = False):
        """
        Args:
            data_path:
            output_path:
            aligned:

        Returns:

        """
        self._ng = Naming(output_path)
        if os.path.isfile(data_path):
            self._tag = os.path.basename(data_path).replace('.h5ad', '')
        else:
            self._tag = os.path.basename(data_path)
        self._ng.tag = self._tag
        self._prepare(data_path)
        self._alignment()
        self._reconstruct_outer_3d()
        self.export_trans_para()
        if self._cluster_point > 0:
            self.export_trans_h5ads(data_path)


def main(args, para):
    print('proc start in ', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    s3d = Spateo3D(z_step=args.z_step, cluster_pts=args.cluster_pts, cluster_key=args.cluster, key_slice=args.slice)
    s3d.reconstruct(args.input_path, args.output_path)


usage = """ 3D reconstruction with spateo """
PROG_VERSION = 'v0.0.1'

"""
python spateo_3d.py -input D:\data\stereo3d\spateo\gy\E14-16h_a_count_normal_stereoseq.h5ad --output D:\data\stereo3d\spateo\gy\E14-16h -z_step 1
python spateo_3d.py -input D:\data\stereo3d\spateo\mouseb\brain --output D:\data\stereo3d\spateo\mouseb\E115 -z_step 15 -cluster_pts 3000
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-input", "--input_path", action="store", dest="input_path", type=str, required=True,
                        help="path of h5ad")
    parser.add_argument("-output", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="output path")
    parser.add_argument("-cluster", "--cluster", action="store", dest="cluster", type=str, default='annotation',
                        help="cluster key")
    parser.add_argument("-slice", "--slice", action="store", dest="slice", type=str, default='slice_ID',
                        help="slice key")
    parser.add_argument("-z_step", "--z_step", action="store", dest="z_step", type=int, default=2,
                        help="Step in z")
    parser.add_argument("-cluster_pts", "--cluster_pts", action="store",
                        dest="cluster_pts", type=int, default=-1,
                        help="count of down sample point in each cluster")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)
