import os
import math
import json
import shutil
import cv2 as cv
import numpy as np
import tifffile as tif

from glob import glob
from tqdm import tqdm
from tqdm.contrib import tzip
from .align import FFTMatcher, ColorMatcher


class RigidTrans:
    def __init__(self,
                 src_image,
                 dst_image,
                 scale_by_dst=1,
                 translate_only=False,
                 registration=True):
        """
        Args:
            src_image: str | array Image to be registered
            dst_image: str | array Target Image
            scale_by_dst: float | None The ratio of the image to be registered to the target image
            translate_only: True | False Is it only translation?
        """
        if isinstance(src_image, str):
            if os.path.isfile(src_image):
                self.src_image = cv.imread(src_image, -1)
        elif isinstance(src_image, np.ndarray):
            self.src_image = src_image

        if isinstance(dst_image, str):
            if os.path.isfile(dst_image):
                self.dst_image = cv.imread(dst_image, -1)
        elif isinstance(dst_image, np.ndarray):
            self.dst_image = dst_image

        self.scale2dst = scale_by_dst
        self.translate_only = translate_only
        self.registration = registration

    def find_rigid_trans(self, is_flip=False, down_sampling=2):
        """
        Args:
            is_flip: bool - Whether to mirror (horizontal mirroring is the default)
            down_sampling: int - Downsampling factor for registration
        Returns:
            new_image: array - Registering images
            trans_mat: array [3 * 3] - Transformation Matrix
            best_score: float - Registration score
        """
        if self.scale2dst is None:
            self.scale2dst = self.find_scale(self.src_image, self.dst_image)
        
        _src_image = self._scale_image(self.src_image, self.scale2dst)

        _src_image = self._scale_image(_src_image, 1 / down_sampling)
        _dst_image = self._scale_image(self.dst_image, 1 / down_sampling)

        # Relative scale transformation matrix
        scale_mat = np.eye(3)
        scale_mat[:2, :2] = scale_mat[:2, :2] * self.scale2dst

        # Downsampling scale transformation matrix
        down_scale_mat = np.eye(3)
        down_scale_mat[:2, :2] = down_scale_mat[:2, :2] / down_sampling
        down_scale_mat_inv = np.matrix(down_scale_mat).I

        # Subsequent rotation and displacement transformation matrix
        trans_mat = np.eye(3)

        if self.translate_only:
            matcher_fft = FFTMatcher()
            matcher_color = ColorMatcher()
            offset_ft_x, offset_ft_y, score_ft = matcher_fft.neighbor_match(
                src_img=self._image_channel(_src_image, hsv=1),
                dst_img=self._image_channel(_dst_image, hsv=1)
            )
            offset_x, offset_y, best_score = matcher_color.neighbor_match(
                _src_image,
                _dst_image,
                offset_ft_x,
                offset_ft_y
            )
            trans_mat[:2, 2] = [-offset_x, -offset_y]
            trans_mat = down_scale_mat_inv @ trans_mat @ down_scale_mat @ scale_mat
        else:
            if is_flip: flip_set = [False, True]
            else: flip_set = [False]
            rotate_set = [False, True]

            max_area = -1
            best_flip = False
            best_coarse_mat = None
            best_regis_image = None
            for flip in flip_set:
                for angle_180 in rotate_set:
                    regis_image, mutual_area, mat = self.coarse_regis(
                        _src_image, _dst_image, flip, angle_180
                    )

                    if mutual_area > max_area:
                        max_area = mutual_area
                        best_coarse_mat = mat
                        best_regis_image = regis_image
                        best_flip = flip

            fine_mat = None
            best_score = 0
            if best_coarse_mat is not None:
                fine_mat, best_score = self.find_best_regis(
                    best_regis_image, _dst_image, search_range=15
                )

            if best_flip:
                trans_mat[0, :] = [-1, 0, _src_image.shape[1] - 1]
            if self.registration == False: 
                print("Warning: Registration is turned off.")  
                trans_mat = np.array([[1., 0., 0], [0., 1., 0], [0, 0, 1.]])
                best_score = 1
            elif self.registration == True and fine_mat is not None:
                trans_mat = down_scale_mat_inv @ fine_mat @ best_coarse_mat @ \
                            trans_mat @ down_scale_mat @ scale_mat

        new_image = cv.warpPerspective(
            self.src_image, trans_mat, (self.dst_image.shape[1], self.dst_image.shape[0])
        )

        return new_image, trans_mat, best_score

    def coarse_regis(self, src_image, dst_image, flip, angle_180):
        """
        Args:
            src_image:
            dst_image:
            flip:
            angle_180:
        Returns:
            mutual_area:
            mat:
        """

        if flip:
            src_image = cv.flip(src_image, 1)

        src_coord, src_angle_rad = self.find_orien_and_center_from_moment(src_image, contour=False)
        dst_coord, dst_angle_rad = self.find_orien_and_center_from_moment(dst_image, contour=False)

        if angle_180:
            src_angle_rad += math.pi

        transl = np.array(dst_coord) - np.array(src_coord)
        ror_rad = src_angle_rad - dst_angle_rad

        regis_image, mat = self._transform_img(src_image, transl, dst_coord,
                                               ror_rad, dst_image.shape, scale_ratio=1)

        mutual_area = self.calculate_mutual_area(regis_image, dst_image)

        return regis_image, mutual_area, mat

    def find_best_regis(self,
                        trans_image,
                        dst_image,
                        down_scale = 10,
                        search_range = 10):
        """
        Args:
            trans_image:
            dst_image:
            down_scale:
            search_range:
        Returns:
            mat:
            best_score:
        """
        trans_image_s = self._scale_image(trans_image, 1 / down_scale)
        dst_image_s = self._scale_image(dst_image, 1 / down_scale)

        h, w = trans_image_s.shape[:2]
        best_rotate = None
        best_offset = None
        best_score = 0
        color_match = ColorMatcher()
        for rotate in range(-search_range, search_range):
            m_r = cv.getRotationMatrix2D((int(w / 2), int(h / 2)), rotate, scale=1)
            _trans_img = cv.warpAffine(trans_image_s, m_r, (w, h))
            offset_x, offset_y, score = color_match.neighbor_match(_trans_img, dst_image_s)
            if score > best_score:
                best_rotate = rotate
                best_score = score
                best_offset = (offset_x, offset_y)

        if best_rotate is None or best_score is None:
            return None, 0
        m_r = cv.getRotationMatrix2D((int(trans_image.shape[1] / 2),
                                      int(trans_image.shape[0] / 2)), best_rotate, scale=1)
        m_t = np.float32([[1, 0, -best_offset[0] * down_scale], [0, 1, -best_offset[1] * down_scale]])
        mat = np.concatenate((np.array(m_t), np.array([[0, 0, 1]])), axis=0) @ \
              np.concatenate((np.array(m_r), np.array([[0, 0, 1]])), axis=0)

        # _src = cv2.warpPerspective(trans_image_s, mat, (w, h))
        return mat, best_score

    @staticmethod
    def _transform_img(img, transl, ror_center, ror_rad, canvas_shape, scale_ratio = 1):
        """ Image Transformation """
        # transl: np.array([height, width])
        M_t = np.float32([[1, 0, transl[1]], [0, 1, transl[0]]])
        translated = cv.warpAffine(img, M_t, (canvas_shape[1], canvas_shape[0]))
        M_r = cv.getRotationMatrix2D((int(ror_center[1]), int(ror_center[0])), -ror_rad / math.pi * 180,
                                     scale=scale_ratio)
        transformed_img = cv.warpAffine(translated, M_r, (canvas_shape[1], canvas_shape[0]))
        mat = np.concatenate((np.array(M_r), np.array([[0, 0, 1]])), axis=0) @ \
              np.concatenate((np.array(M_t), np.array([[0, 0, 1]])), axis=0)
        return transformed_img, mat

    @staticmethod
    def _scale_image(image, scale):
        """ scale """
        if scale == 1:
            return image.copy()
        h, w = image.shape[:2]
        image_s = cv.resize(image, (int(w * scale), int(h * scale)))
        return image_s

    @staticmethod
    def _image_channel(img, gray = False, hsv = -1):
        if img.ndim == 3:
            if gray:
                return cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            if -1 < hsv < 3 and isinstance(hsv, int):
                return cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, hsv]
            return img[:, :, 2]
        else:
            return img

    @staticmethod
    def find_orien_and_center_from_moment(img, contour=True):
        """
        The input is a single-channel matrix. Note that the coordinate systems x and y of opencv and the matrix are swapped.
        """
        if contour:
            contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contour = sorted(contours, key=cv.contourArea, reverse=True)[0]
            M = cv.moments(contour)
        else:
            M = cv.moments(img)

        cx_cv = int(M['m10'] / M['m00'])
        cy_cv = int(M['m01'] / M['m00'])

        a = M['m20'] / M['m00'] - cx_cv ** 2
        b = 2 * (M['m11'] / M['m00'] - cx_cv * cy_cv)
        c = M['m02'] / M['m00'] - cy_cv ** 2

        angle_rad_cv = 1 / 2 * cv.fastAtan2(b, a - c) * math.pi / 180

        cx = cy_cv
        cy = cx_cv
        angle_rad = math.pi / 2 - angle_rad_cv
        return (cx, cy), angle_rad

    @staticmethod
    def calculate_mutual_area(img1, img2):
        """ Intersection area calculation """
        sh_0 = max([img1.shape[0], img2.shape[0]])
        sh_1 = max([img1.shape[1], img2.shape[1]])

        com = np.zeros(shape=(sh_0, sh_1))
        com1 = np.zeros(shape=(sh_0, sh_1))
        com2 = np.zeros(shape=(sh_0, sh_1))

        com1[:img1.shape[0], :img1.shape[1]] = img1

        com2[:img2.shape[0], :img2.shape[1]] = img2

        com[(com1 > 0) & (com2 > 0)] = 1
        return np.sum(com)

    @staticmethod
    def find_scale(src_image, dst_image):
        """
        Args:
            src_image:
            dst_image:
        Returns:

        """
        if isinstance(src_image, str) and os.path.exists(src_image):
            src_image = cv.imread(src_image, 0)
            src_image[src_image > 0] = 255

        if isinstance(dst_image, str) and os.path.exists(dst_image):
            dst_image = cv.imread(dst_image, 0)
            dst_image[dst_image > 0] = 255

        contours_src, _ = cv.findContours(src_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_dst, _ = cv.findContours(dst_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_src = sorted(contours_src, key=cv.contourArea, reverse=True)[0]
        contour_dst = sorted(contours_dst, key=cv.contourArea, reverse=True)[0]

        rect_src = cv.minAreaRect(contour_src)
        rect_dst = cv.minAreaRect(contour_dst)

        box_src = cv.boxPoints(rect_src)
        box_dst = cv.boxPoints(rect_dst)
        src_shape = [np.linalg.norm(box_src[0] - box_src[1]), np.linalg.norm(box_src[0] - box_src[2])]
        dst_shape = [np.linalg.norm(box_dst[0] - box_dst[1]), np.linalg.norm(box_dst[0] - box_dst[2])]

        scale = max(min(dst_shape) / min(src_shape), max(dst_shape) / max(src_shape))
        return scale


###################################
def json_write(info_dict, output_path):
    """
    Args:
        info_dict:
        output_path:
    """
    for k, v in info_dict.items():
        if info_dict[k]['mat'] is not None:
            if isinstance(info_dict[k]['mat'], np.ndarray):
                info_dict[k]['mat'] = info_dict[k]['mat'].tolist()
            # info_dict[k]['mat'] = info_dict[k]['mat'].tolist()
    with open(os.path.join(output_path, "align_info.json"), 'w') as f:
        json.dump(info_dict, f, indent=2)


def _align_slices_and_record(src_image, dst_image, scale, info_dict, registration=True):
    """
    Args:
        src_image:
        dst_image:
        scale:
        info_dict:
    """
    name = os.path.basename(src_image)
    if dst_image:
        rt = RigidTrans(src_image, dst_image, scale_by_dst=scale, registration = registration)
        new_image, trans_mat, best_score = rt.find_rigid_trans(down_sampling=4)
    else:
        new_image = None
        trans_mat = None
        best_score = None

    info_dict[name] = dict()
    info_dict[name]['mat'] = trans_mat
    info_dict[name]['shape'] = list(new_image.shape) if isinstance(new_image, np.ndarray) else None
    info_dict[name]['score'] = best_score
    return new_image, info_dict


def _align_slices_similar(images_list, output_path, registration=True):
    """
    Args:
        images_list:
        output_path:
    Returns:
        info_dict:
    """
    info_dict = dict()
    up_image = None
    for i, image in enumerate(tqdm(images_list, desc="Regis", ncols=100)):
        name = os.path.basename(image)
        if i == 0:
            try:
                shutil.copyfile(image, os.path.join(output_path, name))
            except shutil.SameFileError: pass
            up_image = os.path.join(output_path, name)
            _, info_dict = _align_slices_and_record(
                src_image=image, dst_image=None, scale=1, info_dict=info_dict, registration=registration
            )
            continue

        new_image, info_dict = _align_slices_and_record( #edit
            src_image=image, dst_image=up_image, scale=1, info_dict=info_dict, registration=registration
        )

        tif.imwrite(os.path.join(output_path, name), new_image)
        up_image = os.path.join(output_path, name)

    return info_dict


def _align_slices_no_similar(images_list, output_path, dst_list, scale2dst, registration= True):
    """
    Args:
        images_list:
        output_path:
        dst_list:
        scale2dst:
    Returns:
        info_dict:
    """
    info_dict = dict()

    if len(images_list) == len(dst_list):
        for image, dst in tzip(images_list, dst_list):
            name = os.path.basename(image)

            new_image, info_dict = _align_slices_and_record(
                src_image=image, dst_image=dst, scale=scale2dst, info_dict=info_dict, registration=registration
            )

            tif.imwrite(os.path.join(output_path, name), new_image)
    # The following two cases must have the same file name
    elif len(images_list) < len(dst_list):
        for image in tqdm(images_list, desc="Align"):
            name = os.path.basename(image)
            for dst in dst_list:
                dst_name = os.path.basename(dst)
                if name == dst_name:
                    break
            new_image, info_dict = _align_slices_and_record(
                src_image=image, dst_image=dst, scale=scale2dst, info_dict=info_dict, registration=registration
            )
            tif.imwrite(os.path.join(output_path, name), new_image)
    else:
        unpaired_images_list = list()
        for image in tqdm(images_list, desc="Align"):
            name = os.path.basename(image)
            for dst in dst_list:
                dst_name = os.path.basename(dst)
                if name == dst_name:
                    break
            else:
                unpaired_images_list.append(image)
                continue

            new_image, info_dict = _align_slices_and_record(
                src_image=image, dst_image=dst, scale=scale2dst, info_dict=info_dict, registration=registration
            )
            tif.imwrite(os.path.join(output_path, name), new_image)

        for _image in tqdm(unpaired_images_list, desc="Multiplex"):
            name = os.path.basename(_image)
            mat, shape = _search_mat(_image, images_list, info_dict)
            new_image = _affine_image(_image, mat, shape)
            tif.imwrite(os.path.join(output_path, name), new_image)

    return info_dict


def _search_mat(image, images_list, info_dict):
    """ Find the mat matrix with the nearest index of the file name """
    index_src = images_list.index(image)
    index_list = list()
    for ind, dst in enumerate(images_list):
        name = os.path.basename(dst)
        for k in info_dict.keys():
            if name == k:
                index_list.append(ind)
                break

    near_list = [np.abs(index_src - i) for i in index_list]
    _index = near_list.index(min(near_list))
    index_dst = index_list[_index]

    name_dst = os.path.basename(images_list[index_dst])
    mat = info_dict[name_dst]['mat']
    shape = info_dict[name_dst]['shape']

    return mat, shape


def _affine_image(image_src, matrix, shape):
    """

    """
    src = cv.imread(image_src, -1)
    _src = cv.warpPerspective(src, np.array(matrix), (shape[1], shape[0]))
    return _src


def manual_align(images_list, output_path, manual_path, crop_tissue_list):
    """

    Args:
        images_list:
        output_path:

    Returns:

    """
    from stereo3d.manual import parse_xml2mat

    align_json = os.path.join(output_path, "align_info.json")
    info_dict = json.load(open(align_json, 'r'))
    info_dict_or = info_dict.copy()

    for ind, img_path in enumerate(images_list):
        name = os.path.basename(img_path).split(".")[0]
        if "_m" in name or "_manual" in name:
            _name = name.replace("_manual", "")
            _name = _name.replace("_m", "")

            manual_file = glob(os.path.join(manual_path, f"*{_name}*" + ".xml"))
            if len(manual_file) > 0:
                manual_mat, shape = parse_xml2mat(manual_file[0],_name)
            else: continue

            name_list = []
            for k, v in info_dict_or.items():
                if _name in k:
                    name_list.append(k)
                    register_mat = v['mat']
                    break
            if len(manual_mat) == 1: # rigid registration
                combine_manual_mat = manual_mat[0]
                if register_mat is None:
                    _register_mat = combine_manual_mat
                    info_dict[name_list[0]]['shape'] = shape
                else:
                    _register_mat = combine_manual_mat @ np.array(register_mat)
                    _register_mat = _register_mat.tolist()
            else: #elastic registration
                _register_mat = manual_mat[-1]
                if register_mat is None:
                    info_dict[name_list[0]]['shape'] = shape
                else:
                    #_register_mat = combine_manual_mat @ np.array(register_mat)
                    if isinstance(register_mat, (np.matrix, np.ndarray)):
                        register_mat = register_mat.tolist()
                    _register_mat = _register_mat + [*register_mat]
            #_register_mat = np.array(register_mat)

            info_dict[name_list[0]]['mat'] = _register_mat
            img_tmp = tif.imread(img_path)
            tif.imwrite(os.path.join(os.path.dirname(img_path), _name + ".tif"), img_tmp)
            os.remove(img_path)
            images_list[ind] = os.path.join(os.path.dirname(img_path), _name + ".tif")
            #break
            _image_list = crop_tissue_list[ind + 1:]
            _image_list.insert(0, images_list[ind])
            _info_dict = _align_slices_similar(_image_list, output_path)
            for k, v in _info_dict.items():
                if _name in k:
                    continue
                if isinstance(v['mat'], (np.matrix, np.ndarray)):
                    v['mat'] = v['mat'].tolist()
                info_dict[k]['mat'] = v['mat']

    #os.rename(img_path, os.path.join(os.path.dirname(img_path), _name + ".tif"))
    json_write(info_dict, output_path)


def align_slices(slices_path, output_path, dst_path=None, scale2dst=None, registration=True):
    """
    Register a series of slice images or mask images
    Args:
        slices_path: str - It is best to arrange them in order
        output_path:
        dst_path: str - If yes, align with slices_path
        scale2dst: float | int | None - slices_path and dst_path. The scale between images
    """
    os.makedirs(output_path, exist_ok=True)

    if isinstance(slices_path, str):
        images_list = glob(os.path.join(slices_path, "*.tif"))
        try:
            images_list = sorted(images_list, key=lambda x: int(os.path.basename(x).split('.')[0]))
        except ValueError:
            images_list = sorted(images_list)
    elif isinstance(slices_path, list):
        images_list = slices_path

    if dst_path:
        if isinstance(dst_path, str):
            dst_list = glob(os.path.join(dst_path, "*.tif"))
            dst_list = sorted(dst_list, key=lambda x: int(os.path.basename(x).split('.')[0]))
        elif isinstance(dst_path, list):
            dst_list = dst_path
        info_dict = _align_slices_no_similar(images_list, output_path, dst_list, scale2dst)
    else:
        info_dict = _align_slices_similar(images_list, output_path, registration=registration)

    json_write(info_dict, output_path)


if __name__ == "__main__":
    slices_path = r"D:\02.data\E14-16h_a_bin1_image_mask"
    # dst_path = r"F:\ssdna_auto"
    output_path = r"D:\02.data\E14-16h_a_bin1_image_regis"
    align_slices(slices_path, output_path)  # , dst_path, scale2dst=0.0038/0.0005)

    # with open(r"C:\Users\87393\Downloads\manual_registration\rigid.xml", 'r') as file:
    #     xml_con = file.read()
    # aaa = xmltodict.parse(xml_con)
    # print(1)
