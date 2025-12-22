from construct.contrib.align import Align
from construct.contrib.roi import ROI
import nibabel
import glog
import os
import tifffile
import cv2 as cv
import numpy as np
import tqdm


class BlockFace():
    def __init__(self, ):
        # super(BlockFace, self).__init__()
        self._images = list()
        self._size_per_pixel = 0
        self._z_interval = 0
        self._output = ''
        self._dir = {'roi': '00.roi', 'registration': '01.registration', 'tissue': '02.tissue'}
    
    def get_tbf_name(self, ): return self._dir['tissue']
    
    def set_size_per_pixel(self, s): self._size_per_pixel = s

    def set_z_interval(self, i): self._z_interval = i

    def _tissue_crop(self, model_name):
        glog.info('1) => Cropped out the tissue region')
        r = ROI()

        roi_output = os.path.join(self._output, self._dir['roi'])
        dir_make(roi_output)
        r.dump(self._images, output=roi_output, model_name=model_name)

    def _seq_registration(self, ):
        glog.info('2) => Align neighbor slice in Z axis')
        align = Align()
        roi_output = os.path.join(self._output, self._dir['roi'])
        seq = [os.path.join(roi_output, os.path.basename(it)) for it in self._images]
        align_output = os.path.join(self._output, self._dir['registration'])
        dir_make(align_output)
        align.run(roi_output, seq, output=align_output)

    @staticmethod
    def dice(img0, img1):
        img0 = (img0 > 0).astype(np.uint8)
        img1 = (img1 > 0).astype(np.uint8)
        smooth = 1.
        img0_f = img0.flatten()
        img1_f = img1.flatten()

        intersection = np.sum(img0_f * img1_f)
        return (2. * intersection + smooth) / (np.sum(img0_f) + np.sum(img1_f) + smooth)

    def _tissue_mask(self, model_name):
        from construct.dnn.tseg.detector import TissueSegmentationBcdu
        # from skimage.metrics import structural_similarity as ssim
        glog.info('3) => Generate tissue mask for each slice')
        try: sg = TissueSegmentationBcdu(gpu='1', mode='onnx', num_threads=int(3))
        except: sg = TissueSegmentationBcdu(gpu='-1', mode='onnx', num_threads=int(3))

        abs_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(abs_path, f"../weights/{model_name}")
        sg.f_init_model(model_path=model_path)
        mask_output = os.path.join(self._output, self._dir['tissue'])
        dir_make(mask_output)
        seq = [os.path.basename(it) for it in self._images]
        align_output = os.path.join(self._output, self._dir['registration'])

        for it in tqdm.tqdm(seq, desc='Tissue mask'):
            image_file = os.path.join(align_output, it)
            output_file = os.path.join(mask_output, it)
            if os.path.exists(output_file): continue
            img = tifffile.imread(image_file)
            pred = sg.f_predict(img)
            pred = np.expand_dims(pred, axis=2)
            pred = np.concatenate((pred, pred, pred), axis=-1)
            mask = pred * img
            cv.imwrite(output_file, mask)

    def _create_stacked(self, ):
        glog.info('5) => Generate stacked image')
        mask_output = os.path.join(self._output, self._dir['tissue'])
        stacked_path = os.path.join(self._output, 'stackede.tif')
        
        if not os.path.exists(stacked_path):
            align_images = list()
            seq = [os.path.basename(it) for it in self._images]
            dtype = None
            for it in tqdm.tqdm(seq, desc='NRRD'):
                arr = cv.imread(os.path.join(mask_output, it), -1)
                align_images.append(arr)
            dtype = align_images[0].dtype
            align_images = np.array(align_images, dtype=dtype)
            tifffile.imwrite(stacked_path, align_images)
            glog.info('Dump stacked file to ({}) finished'.format(stacked_path))
        else: glog.info('File {} exists'.format(stacked_path))

    def _create_obj(self, ):
        glog.info('6) => Generate model file -> obj')
        #TODO: without slicer UI tool, but python scripts
        glog.warn('This function not work now, please waiting. You can export it with 3d-slicer: https://www.slicer.org/')

    def _create_nrrd(self, ):
        glog.info('4) => Generate NRRD')
        nrrd_path = os.path.join(self._output, 'block_face.nii.gz')
        mask_output = os.path.join(self._output, self._dir['tissue'])

        if not os.path.exists(nrrd_path):
            align_images = list()
            
            seq = [os.path.basename(it) for it in self._images]
            for it in tqdm.tqdm(seq, desc='NRRD'):
                arr = cv.imread(os.path.join(mask_output, it), -1)
                gray = cv.cvtColor(arr, cv.COLOR_RGB2GRAY)
                align_images.append(gray)
            align_images = np.array(align_images).transpose((2, 1, 0))
            affine = np.diag((self._size_per_pixel, self._size_per_pixel, self._z_interval, 0))
            nrrd_data = nibabel.Nifti1Image(dataobj=align_images, affine=affine)
            nibabel.save(nrrd_data, nrrd_path)
            glog.info('Dump nrrd file to {} finished'.format(nrrd_path))
        else:
            glog.info('File {} exists'.format(nrrd_path))

    def create_mesh(self, images, output, tissue_crop_model_name, tissue_mask_model_name):
        glog.info('Build the 3D surface through the block face image.')
        self._output = output
        self._images = images
        dir_make(output)
        self._tissue_crop(model_name=tissue_crop_model_name)
        self._seq_registration()
        self._tissue_mask(model_name=tissue_mask_model_name)
        self._create_nrrd()
        self._create_stacked()
        self._create_obj()


def dir_make(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        glog.info('Path {} not exists & make it'.format(fpath))


def main():
    bf = BlockFace()
    bf._leader_file = r"D:\data\E-ST20220923002_slice_records_20221110.xlsx"
    bf._saw_file = r""
    bf._block_face_file = r"D:\data\Blockface_20221110_mouse_embroyo_rename"
    bf._output_path = r"D:\data\res2"
    bf.block_face_mesh()


if __name__ == '__main__':
    main()
