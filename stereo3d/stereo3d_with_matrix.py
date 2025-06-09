import glog
import argparse
import os
from glob import glob
import tqdm
import sys

import numpy as np
import warnings
warnings.filterwarnings('ignore')

CURR_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.dirname(CURR_PATH)
sys.path.append(ROOT_PATH)

from stereo3d.file.slice import SliceSequence
from stereo3d.register.mask_crop import cut_mask


class Stereo3DwithTissueMatrix(object):
    def __init__(self, ) -> None:
        self.matrix_path: str = None
        self.tissue_mask: str = None
        self.record_sheet: str = None
        self.output_path: str = None
        self._matrix: list = None
        self._tissue: list = None

        self._overwrite_flag: bool = True
        self._slice_seq = SliceSequence()

        # sn_name = ss.get_chip_seq()
        # z_interval = ss.get_z_interval(index='bf')

    def get_matrix_path(self, chip_name: str) -> str:
        for suffix in ['gef', 'gem.gz', 'gem', 'txt']:
            gem_matrix_p = os.path.join(self.matrix_path, '{}.{}'.format(chip_name, suffix))
            if os.path.exists(gem_matrix_p):
                return gem_matrix_p
            else:
                pass

        return ''
    
    def _input_check(self, ):
        for it in [self.matrix_path, self.tissue_mask, self.record_sheet]:
            assert os.path.exists(it), 'Path {} are not exists'.format(it)
        if not os.path.exists(self.output_path): 
            os.makedirs(self.output_path)
            glog.info('Output path not exists, we have created it')
        self._slice_seq.from_xlsx(file_path=self.record_sheet)

        self._matrix = list()
        self._tissue = list()
        for chip_name in self._slice_seq.get_chip_seq():
            tissue_p = os.path.join(self.tissue_mask, '{}.tif'.format(chip_name))
            if os.path.exists(tissue_p):
                self._tissue.append(tissue_p)

            p = self.get_matrix_path(chip_name)
            if p != '':
                self._matrix.append(p)

        # assert len(self._tissue) == len(self._matrix), 'List length of matrix != List length of mask'
        glog.info('A total of {} slices were identified'.format(len(self._tissue)))
        return 0

    def _crop_mask(self, ):
        crop_mask_path = os.path.join(self.output_path, "02.register", "00.crop_mask")
        if not os.path.exists(crop_mask_path): os.makedirs(crop_mask_path)
        if self._overwrite_flag:
            cut_mask(self._tissue, crop_mask_path)
            glog.info('Crop mask is overwrite the files.')
        else:
            files_num = len(os.listdir(crop_mask_path)) - 1

            if files_num != len(self._tissue):
                cut_mask(self._tissue, crop_mask_path)
                glog.info('Crop mask updated.')
            else:
                glog.info("Files all exist, skip crop mask.")

        return crop_mask_path
    
    def _register(self, crop_mask_path):
        from stereo3d.register.registration import align_slices
        # align mask
        crop_tissue_list = [os.path.join(crop_mask_path, os.path.basename(i)) for i in self._tissue]
        align_output_path = os.path.join(self.output_path, "02.register", "01.align_mask")
        if not os.path.exists(align_output_path): os.makedirs(align_output_path)

        if self._overwrite_flag:
            align_slices(crop_tissue_list, align_output_path)
            glog.info('Align mask is overwrite the files.')
        else:
            files_num = len(os.listdir(align_output_path)) - 1
            if files_num != len(crop_tissue_list):
                align_slices(crop_tissue_list, align_output_path)
                glog.info('Align mask updated.')
            else:
                glog.info("Files all exist, skip align mask.")

        return align_output_path
    
    def _transform_gem(self, ):
        from stereo3d.gem.transform import trans_gem_by_json

        # Gem path
        crop_json_path = os.path.join(self.output_path, "02.register", "00.crop_mask", "mask_cut_info.json")
        align_json_path = os.path.join(self.output_path, "02.register", "01.align_mask", "align_info.json")
        gem_save_path = os.path.join(self.output_path, "03.gem")
        if self._overwrite_flag:
            trans_gem_by_json(self._matrix, crop_json_path, align_json_path, gem_save_path)
            glog.info('Trans gem is overwrite the files.')
        else:
            files_num = len(os.listdir(gem_save_path))
            if files_num != len(self._matrix):
                trans_gem_by_json(self._matrix, crop_json_path, align_json_path, gem_save_path)
                glog.info('Trans gem updated.')
            else:
                glog.info("Files all exist, skip trans gem.")
        return gem_save_path
    
    def _create_outermost_surface(self, align_output_path, pixel4mm=0.0005):
        from stereo3d.mesh.create_mesh_3d import get_mask_3d_points
        from stereo3d.mesh.create_mesh_3d import points_3d_to_mesh
         
        z_interval = self._slice_seq.z_interval
        z_interval_dict = self._slice_seq.get_z_interval(index='short')
        crop_tissue_list = [os.path.join(align_output_path, os.path.basename(i)) for i in self._tissue]

        mask_z_interval = list()
        for mask in crop_tissue_list:
            ind = os.path.basename(mask).split('.')[0]
            for k, v in z_interval_dict.items():
                if k == ind: mask_z_interval.append(v)

        mesh_output_path = os.path.join(self.output_path, "04.mesh")
        if not os.path.exists(mesh_output_path): os.makedirs(mesh_output_path, exist_ok=True)
        points_3d = get_mask_3d_points(crop_tissue_list,
                                    mask_z_interval,
                                    z_interval=z_interval,
                                    pixel4mm=pixel4mm,
                                    output_path=mesh_output_path
                                    )
        try:
            points_3d
        except:
            points_3d = np.loadtxt(os.path.join(mesh_output_path, "mask_3d_points.txt"))

        points_3d_to_mesh(points_3d,
                        z_interval=z_interval,
                        mesh_scale=1,
                        output_path=mesh_output_path,
                        show_mesh=False)
        
    def _h5ad_list(self, ):
        lst = list()
        for i in self._matrix:
            chip = os.path.basename(i).split('.')[0]
            lst.append('{}.h5ad'.format(chip))
        return lst
    
    def _insert_organ(self, align_matrix: str):
        from stereo3d.h5ad.txt2adata import batch_cluster, batch_spatial_leiden
        from stereo3d.h5ad.uniform_cluster_color_v2 import uniform_cluster_color
        from stereo3d.h5ad.uniform_cluster_color_v2 import read_and_parse_by_celltype
        from stereo3d.h5ad.uniform_cluster_color_v2 import organ_mesh

        glog.info('Clustering & Color Alignment')
        color_h5ad = os.path.join(self.output_path, '06.color')
        transform_h5ad = os.path.join(self.output_path, '05.transform')
        organ = os.path.join(self.output_path, '07.organ')
        for i in [color_h5ad, transform_h5ad, organ]: 
            if not os.path.exists(i): os.makedirs(i)

        batch_cluster(matrix_dir=align_matrix, save_dir=transform_h5ad)
        batch_spatial_leiden(h5ad_path=transform_h5ad, save_path=transform_h5ad)
        h5ad_list = [os.path.join(transform_h5ad, i) for i in self._h5ad_list()]
        categories = uniform_cluster_color(h5ad_list, color_h5ad)
        glog.info('Cluster total categories are {}'.format(categories))
        color_h5ad_list = [os.path.join(color_h5ad, i) for i in self._h5ad_list()]

        for c in tqdm.tqdm(categories, desc='Organ', ncols=100):
            organ_path_ = read_and_parse_by_celltype(
                outdir=organ, spatial_regis='spatial_mm', anno='leiden', celltype = c,
                adata_list=None, h5ad_list=color_h5ad_list, sc_xyz=None)
            try:
                organ_mesh(organ_path_, organ_path_.replace('.txt', '.obj'))
            except Exception as e:
                glog.error(f"Organ {c}: {e}")
        glog.info('Completed insert organ')

    def reconstruction_3D(
            self,
            matrix_path: str,
            tissue_mask: str,
            record_sheet: str,
            output_path: str,
            overwrite: int = 1
    ):
        """

        Args:
            matrix_path:
            tissue_mask:
            record_sheet:
            output_path:
            overwrite:

        Returns:

        """
        self.matrix_path = matrix_path
        self.tissue_mask = tissue_mask
        self.record_sheet = record_sheet
        self.output_path = output_path

        self._overwrite_flag = True if overwrite else False
        glog.info("----------01.Extract data----------")
        flag = self._input_check()
        glog.info('Completed verification of input parameters.')

        glog.info("----------02.Crop Mask----------")
        crop_mask_path = self._crop_mask()
        glog.info('Completed crop tissue mask and save the result.')

        glog.info("----------03.Register Mask----------")
        align_output_path = self._register(crop_mask_path)
        glog.info('Completed adjacent slice alignment.')

        glog.info("----------04.Transform Gene----------")
        gem_save_path = self._transform_gem()
        glog.info('Completed the reuse of registration parameters to matrix files.')

        glog.info("----------05.Calculate mesh----------")
        self._create_outermost_surface(align_output_path)
        glog.info('The outermost envelope of the organ has been created.')

        glog.info("----------06.Insert organ----------")
        self._insert_organ(gem_save_path)
        glog.info('Internal organ mesh has been generated')


def main(args, para):
    glog.info('Stereo3D will use multiple slices of the same organ to achieve 3D reconstruction')
    swtm = Stereo3DwithTissueMatrix()
    swtm.reconstruction_3D(matrix_path=args.matrix_path, 
                           tissue_mask=args.tissue_mask,
                           record_sheet=args.record_sheet,
                           output_path=args.output_path,
                           overwrite = args.overwrite)
    glog.info('Welcome to cooperate again')


usage = """ Submit data format verification """
PROG_VERSION = 'v0.0.1'


if __name__ == '__main__':
    # E:\app\anaconda\setup\custom\envs\stereo3d\python stereo3d_with_matrix.py
    # --matrix_path C:\Users\BGI\Desktop\stereo3d-test\gy\00.data\01.gem
    # --tissue_mask C:\Users\BGI\Desktop\stereo3d-test\gy\00.data\00.mask
    # --record_sheet C:\Users\BGI\Desktop\stereo3d-test\gy\00.data\E-ST20220923002_slice_records_E14_16.xlsx
    # --output C:\Users\BGI\Desktop\stereo3d-test\output

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-matrix", "--matrix_path", action="store", dest="matrix_path", type=str, required=True,
                        help="Input matrix path.")
    parser.add_argument("-tissue", "--tissue_mask", action="store", dest="tissue_mask", type=str, required=True,
                        help="Input tissue mask path.")
    parser.add_argument("-record", "--record_sheet", action="store", dest="record_sheet", type=str, required=True,
                        help="Input record sheet path. ")
    parser.add_argument("-overwrite", "--overwrite", action="store", dest="overwrite", type=int, required=False,
                        default = 0, help="Overwrite old files, 0 is False, 1 is True. ")
    parser.add_argument("-output", "--output_path", action="store", dest="output_path", type=str, required=True,
                        help="Output path. ")
    parser.set_defaults(func=main)

    (para, args) = parser.parse_known_args()
    print(para, args)
    para.func(para, args)
