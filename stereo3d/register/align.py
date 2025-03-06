import os.path
import numpy as np
import itertools
import cv2 as cv


class Matcher(object):
    def __init__(self):
        self.overlap = 0.1
        self.horizontal = True

    def get_roi(self, arr0, arr1):
        """
        get overlap area
        :param arr0: fov 1
        :param arr1: fov 2
        :return: overlap area
        """
        h0, w0 = arr0.shape
        h1, w1 = arr1.shape
        w = int(min(w0, w1) * self.overlap)
        h = int(min(h0, h1) * self.overlap)

        if self.horizontal:
            x0, y0, x1, y1 = [w0 - w, 0, w0, h0]
            x0_, y0_, x1_, y1_ = [0, 0, w, h1]
        else:
            x0, y0, x1, y1 = [0, h0 - h, w0, h0]
            x0_, y0_, x1_, y1_ = [0, 0, w1, h]
        return arr0[y0: y1, x0: x1], arr1[y0_: y1_, x0_: x1_]

    @staticmethod
    def slice_image(image: np.ndarray, slice_width: int, slice_height: int,
                    overlap_height_ratio=0.2, overlap_width_ratio=0.2):
        """
        Image is cropped and stacked
        """
        if image.ndim == 3:
            image_height, image_width, _ = image.shape
        else:
            image_height, image_width = image.shape

        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)

        slice_bboxes = []
        y_max = y_min = 0
        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap

        n_ims = 0
        sliced_image_result = []
        assert len(slice_bboxes) > 0, "slice image must > 0"
        for slice_bbox in slice_bboxes:
            n_ims += 1
            # extract image
            tlx = slice_bbox[0]
            tly = slice_bbox[1]
            brx = slice_bbox[2]
            bry = slice_bbox[3]
            image_pil_slice = image[tly:bry, tlx:brx]
            sliced_image_result.append(image_pil_slice)
        return sliced_image_result


class ColorMatcher(Matcher):
    def __init__(self, ):
        super(ColorMatcher, self).__init__()
        self._serach_pixel = 20
        self.src_img_zo = None
        self.dst_img_zo = None

    def get_color_score(self, src_img, dst_img, offset_x, offset_y):
        src_img_m = self.get_s_channel_mask(src_img)
        dst_img_m = self.get_s_channel_mask(dst_img)

        src_img_zo = (src_img_m > 0).astype(np.uint8)
        dst_img_zo = (dst_img_m > 0).astype(np.uint8)

        offset_src, offset_dst = self.get_offset_image(src_img_zo, dst_img_zo, offset_x, offset_y)
        offset_image = offset_src + offset_dst
        two_pixel = np.sum(offset_image == 2)
        one_pixel = np.sum(offset_image == 1)
        score = two_pixel / (two_pixel + one_pixel)

        return score

    def neighbor_match(self, src_img, dst_img, offset_x=None, offset_y=None):
        """
        Must need 3 channel BGR image.
        Args:
            src_img:
            dst_img:
        """
        src_img_m = self.get_s_channel_mask(src_img)
        dst_img_m = self.get_s_channel_mask(dst_img)

        if offset_x is None or offset_y is None:
            src_mass_x, src_mass_y = self.get_mass(src_img_m)
            dst_mass_x, dst_mass_y = self.get_mass(dst_img_m)

            if np.isnan(src_mass_x) or np.isnan(dst_mass_x):
                return 0, 0, 0
            offset_x = round(src_mass_x - dst_mass_x)
            offset_y = round(src_mass_y - dst_mass_y)

        self.src_img_zo = (src_img_m > 0).astype(np.uint8)
        self.dst_img_zo = (dst_img_m > 0).astype(np.uint8)

        result = (offset_x, offset_y, 0)
        while result != None:
            offset_x, offset_y, best_score = self.search_color_offset(offset_x, offset_y)
            result = self.location_best(offset_x, offset_y, best_score)
            if result != None:
                offset_x, offset_y, _ = result

        return offset_x, offset_y, best_score

    def location_best(self, offset_x, offset_y, score, k=4):
        best_offset = None
        best_score = score
        for pixel_x in range(-k, k + 1):
            for pixel_y in range(-k, k + 1):
                _pixel_x = pixel_x + offset_x
                _pixel_y = pixel_y + offset_y
                offset_src, offset_dst = self.get_offset_image(self.src_img_zo, self.dst_img_zo, _pixel_x, _pixel_y)
                _score = self._score(offset_src, offset_dst)
                if _score > best_score:
                    best_offset = [_pixel_x, _pixel_y]
                    best_score = _score
        if best_score > score:
            return best_offset[0], best_offset[1], best_score
        else:
            return None

    def search_color_offset(self, offset_x, offset_y):
        offset_src, offset_dst = self.get_offset_image(self.src_img_zo, self.dst_img_zo, offset_x, offset_y)
        best_score = self._score(offset_src, offset_dst)
        search_key = {'left': 1, 'right': 1, 'up': 1, 'down': 1}
        offset_score_dict = self._key_to_score(offset_x, offset_y, search_key)
        while best_score < max(offset_score_dict.values()):
            best_score = max(offset_score_dict.values())
            for key in offset_score_dict.keys():
                if offset_score_dict[key] == best_score:
                    break
            if key == 'left':
                search_key = {'left': 1, 'right': 0, 'up': 1, 'down': 1}
                offset_x -= 1
                offset_score_dict = self._key_to_score(offset_x, offset_y, search_key)
                offset_score_dict['right'] = best_score
            elif key == 'right':
                search_key = {'left': 0, 'right': 1, 'up': 1, 'down': 1}
                offset_x += 1
                offset_score_dict = self._key_to_score(offset_x, offset_y, search_key)
                offset_score_dict['left'] = best_score
            elif key == 'up':
                search_key = {'left': 1, 'right': 1, 'up': 1, 'down': 0}
                offset_y -= 1
                offset_score_dict = self._key_to_score(offset_x, offset_y, search_key)
                offset_score_dict['down'] = best_score
            elif key == 'down':
                search_key = {'left': 1, 'right': 1, 'up': 0, 'down': 1}
                offset_y += 1
                offset_score_dict = self._key_to_score(offset_x, offset_y, search_key)
                offset_score_dict['up'] = best_score
        return offset_x, offset_y, best_score

    def _key_to_score(self, offset_x, offset_y, search_key: dict):
        offset_dict = {'left': 0, 'right': 0, 'up': 0, 'down': 0}
        for key in search_key.keys():
            if search_key[key] != 1:
                continue
            if key == 'left':
                offset_x -= 1
            elif key == 'right':
                offset_x += 1
            elif key == 'up':
                offset_y -= 1
            elif key == 'down':
                offset_y += 1
            offset_src, offset_dst = self.get_offset_image(self.src_img_zo, self.dst_img_zo, offset_x, offset_y)
            score = self._score(offset_src, offset_dst)
            offset_dict[key] = score
        return offset_dict

    @staticmethod
    def _score(offset_src, offset_dst):
        offset_image = offset_src + offset_dst
        two_pixel = np.sum(offset_image == 2)
        one_pixel = np.sum(offset_image == 1)
        score = two_pixel / (two_pixel + one_pixel)
        return score

    @staticmethod
    def get_offset_image(image_src, image_dst, offset_x, offset_y):
        h, w = image_src.shape[:2]

        offset_x = int(offset_x)
        offset_y = int(offset_y)

        if offset_x > 0:
            src_start_x = offset_x
            src_end_x = None
            dst_start_x = 0
            dst_end_x = w - offset_x
        elif offset_x < 0:
            src_start_x = 0
            src_end_x = offset_x
            dst_start_x = -offset_x
            dst_end_x = None
        else:
            src_start_x = 0
            src_end_x = None
            dst_start_x = 0
            dst_end_x = None

        if offset_y > 0:
            src_start_y = offset_y
            src_end_y = None
            dst_start_y = 0
            dst_end_y = h - offset_y
        elif offset_y < 0:
            src_start_y = 0
            src_end_y = offset_y
            dst_start_y = -offset_y
            dst_end_y = None
        else:
            src_start_y = 0
            src_end_y = None
            dst_start_y = 0
            dst_end_y = None

        return image_src[src_start_y:src_end_y, src_start_x:src_end_x], \
               image_dst[dst_start_y:dst_end_y, dst_start_x:dst_end_x]

    @staticmethod
    def get_s_channel_mask(image):
        if image.ndim == 2:
            return image
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        _, _image = cv.threshold(image_hsv[:, :, 1], 30, 255, cv.THRESH_BINARY)
        return _image

    @staticmethod
    def get_mass(image):
        image = image.astype(float)
        image_x = np.sum(image, 0)
        xx = np.array(range(len(image_x)))
        xx_cal = xx * image_x
        x_mass = np.sum(xx_cal) / np.sum(image_x)

        image_y = np.sum(image, 1)
        yy = np.array(range(len(image_y)))
        yy_cal = yy * image_y
        y_mass = np.sum(yy_cal) / np.sum(image_y)
        return np.array([x_mass, y_mass])


class SIFTMatcher(Matcher):
    def __init__(self, ):
        super(SIFTMatcher, self).__init__()
        self.sift = cv.SIFT_create()
        self.__matcher = self.matcher()

    def neighbor_match(self, src_img, dst_img):

        train_local = np.array(src_img, dtype=np.uint8)
        query_local = np.array(dst_img, dtype=np.uint8)
        return self.sift_match(train_local, query_local)

    def sift_match(self, train, query):
        psd_kp1, psd_des1 = self.sift.detectAndCompute(train, None)
        kps1 = np.float32([kp.pt for kp in psd_kp1])
        psd_kp2, psd_des2 = self.sift.detectAndCompute(query, None)
        kps2 = np.float32([kp.pt for kp in psd_kp2])
        if len(kps1) == 0 or len(kps2) == 0:
            return None
        try:
            matches = self.__matcher.knnMatch(psd_des1, psd_des2, k=2)
        except:
            return None
        local_match = list()

        for m, n in matches:
            if m.distance < 0.45 * n.distance: local_match.append((m.trainIdx, n.queryIdx))

        if len(local_match) == 0:
            return None
        else:
            pts_a = np.float32([kps1[i] for (_, i) in local_match])
            pts_b = np.float32([kps2[i] for (i, _) in local_match])
            offset = np.median(pts_a - pts_b, axis=0)
            return [offset[0], offset[1], 100]

    @staticmethod
    def matcher():
        flann_index_kd_tree = 1
        index_params = dict(algorithm=flann_index_kd_tree, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        return flann


class FFTMatcher(Matcher):
    def __init__(self):
        super(FFTMatcher, self).__init__()

    def pcm_peak(self, image1: np.ndarray, image2: np.ndarray):
        """
        Compute peak correlation matrix for two images and Interpret the translation to find the translation with heighest ncc
        :param image1:the first image (the dimension must be 2)
        :param image2:the second image (the dimension must be 2)
        :return:
        _ncc : the highest ncc
        x : the selected x position
        y : the selected y position
        sub_image1: the overlapping area of image1
        sub_image2: the overlapping area of image2
        """
        assert image1.ndim == 2, "the dimension must be 2"
        assert image2.ndim == 2, "the dimension must be 2"
        size_y, size_x = image1.shape[:2]
        pcm = self._pcm(image1, image2).real
        yins, xins, pcm_ = self._multi_peak_max(pcm)
        lims = np.array([[-size_y, size_y], [-size_x, size_x]])
        max_peak = self._interpret_translation(
            image1, image2, yins, xins, *lims[0], *lims[1]
        )  # 与输入进来的位置参数刚好相反
        ncc, offset_y, offset_x, sub_dst, sub_src = max_peak
        return ncc, offset_y, offset_x, sub_dst, sub_src

    def neighbor_match(self, src_img, dst_img, axis=0):
        """
        Calculate the relative stitch offsets of adjacent FOVs
        :param src_img: [NxM], image of source, A is number of image to calculate cross-power specturm
        :param dst_img:[NxM], image of destination
        :return: offset, confidence
        """
        if axis == 0:
            self.horizontal = True
        else: self.horizontal = False

        # ncc, offset_y, offset_x, sub_dst, sub_src = self.pcm_peak(src_img, dst_img)

        sub_dst_crop, sub_src_crop = self._slice_images(src_img, dst_img)
        # if len(sub_src_crop) > 0:
        (offset_y, offset_x), confidence0, fft_image = self._stack_peak_pc(sub_dst_crop, sub_src_crop)
        #     offset_x += x0
        #     offset_y += y0
        # else:
        #     confidence0 = ncc

        return [offset_x, offset_y,  round(confidence0 * 100, 2)]

    @staticmethod
    def ncc(image1, image2):
        """Compute the normalized cross correlation for two images.
        Parameters
        ---------
        image1 : the first image (the dimension must be 2)
        image2 : the second image (the dimension must be 2)
        Returns
        -------
        ncc : the normalized cross correlation
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        image1 = image1.flatten()
        image2 = image2.flatten()
        n = np.dot(image1 - np.mean(image1), image2 - np.mean(image2))
        d = np.linalg.norm(image1) * np.linalg.norm(image2)
        if d < 0.0001:
            return 0
        return n / d

    @staticmethod
    def extract_overlap_subregion(image, y: int, x: int):
        """Extract the overlapping subregion of the image.
        Parameters
        ---------
        image : the image (the dimension must be 2)
        y : the y (second last dim.) position
        x : the x (last dim.) position
        Returns
        subimage : the extracted subimage
        """
        sizeY = image.shape[0]
        sizeX = image.shape[1]
        assert (np.abs(y) < sizeY) and (np.abs(x) < sizeX)
        # clip x to (0, size_Y)
        h_start = int(max(0, min(y, sizeY, key=int), key=int))
        # clip x+sizeY to (0, size_Y)
        h_end = int(max(0, min(y + sizeY, sizeY, key=int), key=int))
        w_start = int(max(0, min(x, sizeX, key=int), key=int))
        w_end = int(max(0, min(x + sizeX, sizeX, key=int), key=int))
        return image[h_start:h_end, w_start:w_end]

    def _slice_images(self, src_img, dst_img):
        """
        crop and stack images
        :param src_img: template image
        :param dst_img: background image
        :return: flod images
        """
        ratio = 0.7
        overlap_size = min(dst_img.shape[:2])
        width = overlap_size if self.horizontal else int(overlap_size * ratio)
        height = int(overlap_size * ratio) if self.horizontal else overlap_size
        src_slice_img = self.slice_image(src_img, slice_width=width, slice_height=height,
                                         overlap_height_ratio=0.2, overlap_width_ratio=0.2, )
        dst_slice_img = self.slice_image(dst_img, slice_width=width, slice_height=height,
                                         overlap_height_ratio=0.2, overlap_width_ratio=0.2, )
        src_slice_img, dst_slice_img = self._fliter_null_information_image(src_slice_img, dst_slice_img)
        return np.array(src_slice_img), np.array(dst_slice_img)

    def _stack_peak_pc(self, stack_im0, stack_im1):
        """
        Computes phase correlation between stack im0 and stack im1 and Interpret the translation to find the translation
        Args:
            stack_im0: the first stack image, N*W*H
            stack_im1: the second stack image, N*W*H
        Returns:
            ret: location
            success: matcher degree
            scps_result: phase correlation image
        """
        # TODO: Implement some form of high-pass filtering of PHASE correlation
        stack_im0 = [self._apodize(im) for im in stack_im0]
        stack_im1 = [self._apodize(im) for im in stack_im1]
        f0, f1 = [np.fft.fft2(arr) for arr in (stack_im0, stack_im1)]
        # spectrum can be filtered (already),
        # so we have to take precaution against dividing by 0
        eps = abs(f1).max() * 1e-15
        cps = abs(np.fft.ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1) + eps)))
        # scps = shifted cps
        scps = np.fft.fftshift(cps)
        scps_result = np.zeros_like(scps[0])
        sum_weights = 0 + 0.00000001
        for i, scp in enumerate(scps):
            scp = cv.GaussianBlur(scp, (7, 7), 1)
            thread = np.percentile(scp, 95)
            scp[np.where(scp < thread)] = 0
            (_, _), success = self._argmax_translation(scp)
            scps_result += scp * success
            sum_weights += success
        scps_result = scps_result / sum_weights
        (t0, t1), success = self._argmax_translation(scps_result)  # (y,x)
        if success < 0.05:
            scps_result = cv.GaussianBlur(scps_result, (7, 7), 1)
            win = cv.createHanningWindow((7, 7), cv.CV_32F)
            scps_result = cv.filter2D(scps_result, -1, win)
            (t0, t1), _ = self._argmax_translation(scps_result)  # (y,x)

        ret = np.array((np.round(t0), np.round(t1)))

        # _compensate_fftshift is not appropriate here, this is OK.
        t0 -= scps[0].shape[0] // 2
        t1 -= scps[0].shape[1] // 2

        ret -= np.array(scps[0].shape, int) // 2
        return ret, success, scps_result

    def _interpret_translation(self,
                               image1: np.ndarray,
                               image2: np.ndarray,
                               yins: np.ndarray,
                               xins: np.ndarray,
                               ymin: int,
                               ymax: int,
                               xmin: int,
                               xmax: int,
                               n: int = 2,
                               ):
        """Interpret the translation to find the translation with heighest ncc.
        Parameters
        ---------
        image1 : the first image (the dimension must be 2)
        image2 : the second image (the dimension must be 2)
        yins : the y positions estimated by PCM
        xins : the x positions estimated by PCM
        ymin : the minimum value of y (second last dim.)
        ymax : the maximum value of y (second last dim.)
        xmin : the minimum value of x (last dim.)
        xmax : the maximum value of x (last dim.)
        n : the number of peaks to check, default is 2.
        Returns
        _ncc : the highest ncc
        x : the selected x position
        y : the selected y position
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        sizeY = image1.shape[0]
        sizeX = image1.shape[1]
        assert np.all(0 <= yins) and np.all(yins < sizeY)
        assert np.all(0 <= xins) and np.all(xins < sizeX)

        _ncc = -np.infty
        x, y = 0, 0
        overlap_img_1 = None
        overlap_img_2 = None

        ymagss = [yins, sizeY - yins]
        ymagss[1][ymagss[0] == 0] = 0
        xmagss = [xins, sizeX - xins]
        xmagss[1][xmagss[0] == 0] = 0

        # concatenate all the candidates
        _poss = []
        for ymags, xmags, ysign, xsign in itertools.product(
                ymagss, xmagss, [-1, +1], [-1, +1]
        ):
            yvals = ymags * ysign
            xvals = xmags * xsign
            _poss.append([yvals, xvals])
        poss = np.array(_poss)
        valid_ind = (
                (ymin <= poss[:, 0, :])
                & (poss[:, 0, :] <= ymax)
                & (xmin <= poss[:, 1, :])
                & (poss[:, 1, :] <= xmax)
        )
        assert np.any(valid_ind)
        valid_ind = np.any(valid_ind, axis=0)
        for pos in np.moveaxis(poss[:, :, valid_ind], -1, 0)[: int(n)]:
            for yval, xval in pos:
                if (ymin <= yval) and (yval <= ymax) and (xmin <= xval) and (xval <= xmax):
                    subI1 = self.extract_overlap_subregion(image1, yval, xval)
                    subI2 = self.extract_overlap_subregion(image2, -yval, -xval)
                    if subI1.size / (sizeX * sizeY) > 0.05 and min(subI1.shape) > 80:  # 在overlap区域, 最少10%的重叠区域
                        ncc_val = self.ncc(subI1, subI2)
                        if ncc_val > _ncc:
                            _ncc = float(ncc_val)
                            y = int(yval)
                            x = int(xval)
                            overlap_img_1 = subI1
                            overlap_img_2 = subI2
        return _ncc, y, x, overlap_img_1, overlap_img_2

    def _get_success(self, array, coord, radius=2):
        """
        Given a coord, examine the array around it and return a number signifying
        how good is the "match".

        Args:
            radius: Get the success as a sum of neighbor of coord of this radius
            coord: Coordinates of the maximum. Float numbers are allowed
                (and converted to int inside)

        Returns:
            Success as float between 0 and 1 (can get slightly higher than 1).
            The meaning of the number is loose, but the higher the better.
        """
        coord = np.round(coord).astype(int)
        coord = tuple(coord)

        subarr = self._get_subarr(array, coord, radius)

        theval = subarr.sum()
        theval2 = array[coord]
        # bigval = np.percentile(array, 97)
        # success = theval / bigval
        # TODO: Think this out
        success = np.sqrt(theval * theval2)
        return success

    def _argmax_translation(self, array):
        # We want to keep the original and here is obvious that
        # it won't get changed inadvertently
        array_orig = array.copy()
        ashape = np.array(array.shape, int)
        mask = np.ones(ashape, float)
        array *= mask

        # WE ARE FFTSHIFTED already.
        # ban translations that are too big
        aporad = (ashape // 6).min()
        mask2 = self._get_apofield(ashape, aporad, corner=True)
        array *= mask2
        # Find what we look for
        tvec = self._argmax_ext(array, 'inf')
        tvec = self._interpolate(array_orig, tvec)

        # If we use constraints or min filter,
        # array_orig[tvec] may not be the maximum
        success = self._get_success(array_orig, tuple(tvec), 2)

        return tvec, success

    def _interpolate(self, array, rough, rad=2):
        """
        Returns index that is in the array after being rounded.

        The result index tuple is in each of its components between zero and the
        array's shape.
        """
        rough = np.round(rough).astype(int)
        surroundings = self._get_subarr(array, rough, rad)
        com = self._argmax_ext(surroundings, 1)
        offset = com - rad
        ret = rough + offset
        # similar to win.wrap, so
        # -0.2 becomes 0.3 and then again -0.2, which is rounded to 0
        # -0.8 becomes - 0.3 -> len() - 0.3 and then len() - 0.8,
        # which is rounded to len() - 1. Yeah!
        ret += 0.5
        ret %= np.array(array.shape).astype(int)
        ret -= 0.5
        return ret

    def _apodize(self, what, aporad=None):
        """
        Given an image, it apodizes it (so it becomes quasi-seamless).
        When ``ratio`` is None, color near the edges will converge
        to the same colour, whereas when ratio is a float number, a blurred
        original image will serve as background.

        Args:
            what: The original image
            aporad (int): Radius [px], width of the band near the edges
                that will get modified
        Returns:
            The apodized image
        """
        if aporad is None:
            mindim = min(what.shape)
            aporad = int(mindim * 0.12)
        apofield = self._get_apofield(what.shape, aporad)
        res = what * apofield
        bg = self._get_borderval(what, aporad // 2)
        res += bg * (1 - apofield)
        return res

    @staticmethod
    def _get_borderval(img, radius=None):
        """
        Given an image and a radius, examine the average value of the image
        at most radius pixels from the edge
        """
        if radius is None:
            mindim = min(img.shape)
            radius = max(1, mindim // 20)
        mask = np.zeros_like(img, dtype=bool)
        mask[:, :radius] = True
        mask[:, -radius:] = True
        mask[radius, :] = True
        mask[-radius:, :] = True

        mean = np.median(img[mask])
        return mean

    @staticmethod
    def _get_apofield(shape, aporad, corner=False):
        """
        Returns an array between 0 and 1 that goes to zero close to the edges.
        """
        if aporad == 0:
            return np.ones(shape, dtype=float)
        apos = np.hanning(aporad * 2)
        vecs = []
        for dim in shape:
            assert dim > aporad * 2, \
                "Apodization radius %d too big for shape dim. %d" % (aporad, dim)
            toapp = np.ones(dim)
            toapp[:aporad] = apos[:aporad]
            toapp[-aporad:] = apos[-aporad:]
            vecs.append(toapp)
        apofield = np.outer(vecs[0], vecs[1])
        if corner:
            apofield[aporad:-aporad] = 1
            apofield[:, aporad:-aporad] = 1

        return apofield

    @staticmethod
    def _multi_peak_max(PCM):
        """Find the first to n th largest peaks in PCM.
        PCM : the peak correlation matrix
        Returns
        rows : the row indices for the peaks
        cols : the column indices for the peaks
        vals : the values of the peaks
        """
        row, col = np.unravel_index(np.argsort(PCM.ravel()), PCM.shape)
        vals = PCM[row[::-1], col[::-1]]
        return row[::-1], col[::-1], vals

    @staticmethod
    def _pcm(image1, image2):
        """Compute peak correlation matrix for two images.
        image1 :  the first image (the dimension must be 2)
        image2 : the second image (the dimension must be 2)
        Returns
        PCM : the peak correlation matrix
        """
        assert image1.ndim == 2
        assert image2.ndim == 2
        assert np.array_equal(image1.shape, image2.shape)
        F1 = np.fft.fft2(image1)
        F2 = np.fft.fft2(image2)
        FC = F1 * np.conjugate(F2)
        return np.fft.ifft2(FC / (np.abs(FC) + 0.0000000001)).real.astype(np.float32)

    @staticmethod
    def _fliter_null_information_image(src_imgs, dst_imgs):
        """
        Filter out sub img that has no information
        :param src_imgs: first crop and stack images
        :param dst_imgs: second crop and stack images
        :return: valid sub images
        """
        src_lists = []
        dst_lists = []
        fov_deepth = 8 if src_imgs[0].dtype == np.uint8 else 16
        img_grad_thread = 0.08 * (2 ** fov_deepth / 256) + 0.12
        for src, dst in zip(src_imgs, dst_imgs):
            src_grad = cv.Sobel(src, -1, 1, 1).std()
            dst_grad = cv.Sobel(dst, -1, 1, 1).std()
            if max(src_grad, dst_grad) > img_grad_thread:
                src_lists.append(src)
                dst_lists.append(dst)
        return src_lists, dst_lists

    @staticmethod
    def _argmax_ext(array, exponent):
        """
        Calculate coordinates of the COM (center of mass) of the provided array.
        Args:
            array (ndarray): The array to be examined.
            exponent (float or 'inf'): The exponent we power the array with. If the
                value 'inf' is given, the coordinage of the array maximum is taken.
        Returns:
            np.ndarray: The COM coordinate tuple, float values are allowed!
        """

        # When using an integer exponent for _argmax_ext, it is good to have the
        # neutral rotation/scale in the center rather near the edges
        if exponent == "inf":
            amax = np.argmax(array)
            ret = list(np.unravel_index(amax, array.shape))
        else:
            col = np.arange(array.shape[0])[:, np.newaxis]
            row = np.arange(array.shape[1])[np.newaxis, :]

            arr2 = array ** exponent
            arrsum = arr2.sum()
            if arrsum == 0:
                # We have to return SOMETHING, so let's go for (0, 0)
                return np.zeros(2)
            arrprody = np.sum(arr2 * col) / arrsum
            arrprodx = np.sum(arr2 * row) / arrsum
            ret = [arrprody, arrprodx]
            # We don't use it, but it still tells us about value distribution

        return np.array(ret)

    @staticmethod
    def _get_subarr(array, center, rad):
        """
        Args:
            array (ndarray): The array to search
            center (2-tuple): The point in the array to search around
            rad (int): Search radius, no radius (i.e. get the single point)
                implies rad == 0
        """
        dim = 1 + 2 * rad
        subarr = np.zeros((dim,) * 2)
        corner = np.array(center) - rad
        for ii in range(dim):
            yidx = corner[0] + ii
            yidx %= array.shape[0]
            for jj in range(dim):
                xidx = corner[1] + jj
                xidx %= array.shape[1]
                subarr[ii, jj] = array[yidx, xidx]
        return subarr


def main():
    src = r"D:\01.code\stereo3D\test_data\output\00.blockface\00.roi"
    # matcher_fft = FFTMatcher()
    # matcher_sift = SIFTMatcher()
    matcher_color = ColorMatcher()
    # for i in range(2, 154, 1):
    src_ = os.path.join(src, 'Image_{}.tif'.format('013'))
    dst_ = os.path.join(src, 'Image_{}.tif'.format('015'))
    # moving_image = tifffile.imread(src_)[:, :, 2]
    # fixed_image = tifffile.imread(dst_)[:, :, 2]
    fixed_image = cv.imread(src_, -1)
    moving_image = cv.imread(dst_, -1)
    offset_x, offset_y, score = matcher_color.neighbor_match(fixed_image, moving_image)
    # fixed_image_fft = cv.cvtColor(fixed_image, cv.COLOR_BGR2HSV)[:, :, 1]
    # moving_image_fft = cv.cvtColor(moving_image, cv.COLOR_BGR2HSV)[:, :, 1]

    # offset_fft_x, offset_fft_y, score_fft = matcher_fft.neighbor_match(fixed_image_fft, moving_image_fft)
    # offset_sift_x, offset_sift_y, score_sift = matcher_sift.neighbor_match(
    #     src_img=fixed_image,
    #     dst_img=moving_image
    # )
    #
    # color_score_fft = matcher_color.get_color_score(fixed_image, moving_image, offset_fft_x, offset_fft_y)
    # color_score_sift = matcher_color.get_color_score(fixed_image, moving_image, offset_sift_x, offset_sift_y)
    #
    # print(f"fft-({offset_fft_x}, {offset_fft_y}, {color_score_fft}) "
    #       f"sift-({offset_sift_x}, {offset_sift_y}, {color_score_sift})")


if __name__ == '__main__':
    main()
