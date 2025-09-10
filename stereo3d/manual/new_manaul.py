import xml.etree.ElementTree as ET
import numpy as np
import cv2

import tifffile as tif
import re

class trans():
    def __init__(self, img, pi):
        width, height = img.shape[:2]
        pcth = np.repeat(np.arange(height).reshape(height, 1), [width], axis=1)
        pctw = np.repeat(np.arange(width).reshape(width, 1), [height], axis=1).T

        self.img_coordinate = np.swapaxes(np.array([pcth, pctw]), 1, 2).T
        self.cita = compute_G(self.img_coordinate, pi, height, width)
        self.pi = pi
        self.W, self.A, self.Z = pre_compute_waz(self.pi, height, width, self.img_coordinate)
        self.height = height
        self.width = width

    def deformation(self, img, qi):
        qi = self.pi * 2 - qi
        mapxy = np.swapaxes(
            np.float32(compute_fv(qi, self.W, self.A, self.Z, self.height, self.width, self.cita, self.img_coordinate)),
            0, 1)
        img = cv2.remap(img, mapxy[:, :, 0], mapxy[:, :, 1], borderMode=cv2.BORDER_WRAP, interpolation=cv2.INTER_LINEAR)

        return img
    
def pre_compute_waz(pi, height, width, img_coordinate):
    '''

    :param pi:
    :param height:
    :param width:
    :param img_coordinate: 坐标信息矩阵
    :return:
    '''

    # height*width*控制点个数
    wi = np.reciprocal(
        np.power(np.linalg.norm(np.subtract(pi, img_coordinate.reshape(height, width, 1, 2)) + 0.000000001, axis=3), 2))

    # height*width*2
    pstar = np.divide(np.matmul(wi, pi), np.sum(wi, axis=2).reshape(height, width, 1))

    # height*width*控制点个数*2
    phat = np.subtract(pi, pstar.reshape(height, width, 1, 2))

    z1 = np.subtract(img_coordinate, pstar)
    z2 = np.repeat(np.swapaxes(np.array([z1[:, :, 1], -z1[:, :, 0]]), 1, 2).T.reshape(height, width, 1, 2, 1),
                   [pi.shape[0]], axis=2)

    # height*width*控制点个数*2*1
    z1 = np.repeat(z1.reshape(height, width, 1, 2, 1), [pi.shape[0]], axis=2)

    # height*width*控制点个数*1*2
    s1 = phat.reshape(height, width, pi.shape[0], 1, 2)
    s2 = np.concatenate((s1[:, :, :, :, 1], -s1[:, :, :, :, 0]), axis=3).reshape(height, width, pi.shape[0], 1, 2)

    a = np.matmul(s1, z1)
    b = np.matmul(s1, z2)
    c = np.matmul(s2, z1)
    d = np.matmul(s2, z2)

    # 重构wi形状
    ws = np.repeat(wi.reshape(height, width, pi.shape[0], 1), [4], axis=3)

    # height*width*控制点个数*2*2
    A = (ws * np.concatenate((a, b, c, d), axis=3).reshape(height, width, pi.shape[0], 4)).reshape(height, width,
                                                                                                   pi.shape[0], 2, 2)

    return wi, A, z1

def compute_fv(qi, W, A, Z, height, width, cita, img_coordinate):
    '''
    :param
    qi:
    :param
    W:
    :param
    A:
    :param
    Z:
    :param
    height:
    :param
    width:
    :param
    cita: 衰减系数，减少局部变形对整体的影响
    :param
    img_coordinate:
    :return:
    '''

    qstar = np.divide(np.matmul(W,qi), np.sum(W, axis=2).reshape(height,width,1))

    qhat = np.subtract(qi, qstar.reshape(height, width, 1, 2)).reshape(height, width, qi.shape[0], 1, 2)

    fv_ = np.sum(np.matmul(qhat, A),axis=2)

    fv = np.linalg.norm(Z[:,:,0,:,:],axis=2) / (np.linalg.norm(fv_,axis=3)+0.0000000001) * fv_[:,:,0,:] + qstar

    fv = (fv - img_coordinate) * cita.reshape(height, width, 1) + img_coordinate

    return fv

def compute_G(img_coordinate, pi, height, width, thre = 0.7):
    '''
    衰减系数计算
    :param img_coordinate:
    :param pi:
    :param height:
    :param width:
    :param thre: 影响系数，数值越大对控制区域外影响越大，反之亦然，取值范围0到无穷大
    :return:
    '''
    
    max = np.max(pi, 0)
    min = np.min(pi, 0)

    length = np.max(max - min)

    # 计算控制区域中心
    # p_ = (max + min) // 2
    p_ = np.sum(pi,axis=0) // 2

    # 计算控制区域
    minx, miny = min - length
    maxx, maxy = max + length
    minx = minx if minx > 0 else 0
    miny = miny if miny > 0 else 0
    maxx = maxx if maxx < height else height
    maxy = maxy if maxy < width else width

    k1 =(p_ - [0,0])[1] / (p_ - [0,0])[0]
    k2 =(p_ - [height,0])[1] / (p_ - [height,0])[0]
    k4 =(p_ - [0,width])[1] / (p_ - [0,width])[0]
    k3 =(p_ - [height, width])[1] / (p_ - [height, width])[0]
    k = (np.subtract(p_, img_coordinate)[:, :, 1] / (np.subtract(p_, img_coordinate)[:, :, 0] + 0.000000000001)).reshape(height, width, 1)
    k = np.concatenate((img_coordinate, k), axis=2)

    k[:,:p_[1],0][(k[:,:p_[1],2] > k1) | (k[:,:p_[1],2] < k2)] = (np.subtract(p_[1], k[:,:,1]) / p_[1]).reshape(height, width, 1)[:,:p_[1],0][(k[:,:p_[1],2] > k1) | (k[:,:p_[1],2] < k2)]
    k[:,p_[1]:,0][(k[:,p_[1]:,2] > k3) | (k[:,p_[1]:,2] < k4)] = (np.subtract(k[:,:,1], p_[1]) / (width - p_[1])).reshape(height, width, 1)[:,p_[1]:,0][(k[:,p_[1]:,2] > k3) | (k[:,p_[1]:,2] < k4)]
    k[:p_[0],:,0][(k1 >= k[:p_[0],:,2]) & (k[:p_[0],:,2] >= k4)] = (np.subtract(p_[0], k[:,:,0]) / p_[0]).reshape(height, width, 1)[:p_[0],:,0][(k1 >= k[:p_[0],:,2]) & (k[:p_[0],:,2] >= k4)]
    k[p_[0]:,:,0][(k3 >= k[p_[0]:,:,2]) & (k[p_[0]:,:,2] >= k2)] = (np.subtract(k[:,:,0], p_[0]) / (height - p_[0])).reshape(height, width, 1)[p_[0]:,:,0][(k3 >= k[p_[0]:,:,2]) & (k[p_[0]:,:,2] >= k2)]

    cita = np.exp(-np.power(k[:,:,0] / thre,2))
    #cita[minx:maxx,miny:maxy] = 1
    # 如果不需要局部变形，可以把cita的值全置为1
    # cita = 1

    return cita

def read_xml_transform_info(xml_file):
    """
    读取XML文件并返回变换信息字典
    
    Args:
        xml_file: XML文件路径
        
    Returns:
        dict: 包含变换信息的字典
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    info_dict = {
        'base_transform': None,
        'transform_sequence': []
    }
    
    # 查找包含变换序列的t2_patch
    patches = root.findall('.//t2_patch')
    if len(patches) >= 2:
        target_patch = patches[1]
        
        # 解析基础变换
        base_transform_str = target_patch.get('transform')
        if base_transform_str:
            info_dict['base_transform'] = parse_transform_matrix(base_transform_str)
        
        # 解析变换序列
        transform_list = target_patch.find('.//ict_transform_list')
        if transform_list is not None:
            for transform_elem in transform_list:
                transform_class = transform_elem.get('class')
                transform_data = transform_elem.get('data')
                
                info_dict['transform_sequence'].append({
                    'class': transform_class,
                    'data': transform_data
                })
    
    return info_dict
def parse_mls_data(data_str):
    """解析MLS变换数据"""
    parts = data_str.split()
    params = {
        'transform_type': parts[0],
        'dimension': int(parts[1]),
        'alpha': float(parts[2]),
        'src_points': [],
        'dst_points': []
    }
    
    # 解析控制点对
    n_points = (len(parts) - 3) // 5
    index = 3
    
    for i in range(n_points):
        src_x = float(parts[index])
        src_y = float(parts[index + 1])
        dst_x = float(parts[index + 2])
        dst_y = float(parts[index + 3])
        
        params['src_points'].append([src_x, src_y])
        params['dst_points'].append([dst_x, dst_y])
        index += 5
    
    return params



def apply_transforms_to_image(image, info_dict):
    """
    使用info_dict中的变换信息对图像进行变换
    
    Args:
        image: 输入图像(numpy数组)
        info_dict: 从read_xml_transform_info返回的字典
        
    Returns:
        numpy数组: 变换后的图像
    """
    if image is None:
        raise ValueError("输入图像为空")
    
    current_image = image.copy()
    print(sum(sum(current_image)))
    height, width = image.shape[:2]
    
    # 应用基础变换
    '''if info_dict['base_transform'] is not None:
        print(info_dict['base_transform'])
        mat = info_dict['base_transform'][:2,:]
        mat = [
      [
        0.8856592358742943,
        -0.4643357713620285,
        -1504.1468226057996
      ],
      [
        0.4643357713620285,
        0.8856592358742942,
        -1166.90896493057
      ]
    ]
        mat = np.array(mat)
        current_image = cv2.warpAffine(current_image, mat, (width, height))'''
        #apply_affine_transform(current_image, info_dict['base_transform'])
    
    # 应用变换序列
    for transform_info in info_dict['transform_sequence']:
        print(sum(sum(current_image)))
        if 'AffineModel2D' in transform_info['class']:
            continue
            values = list(map(float, transform_info['data'].split()))
            transform_matrix = np.array([[values[0], values[1], values[4]], [values[2], values[3], values[5]],[0, 0, 1]])
            transform_matrix = transform_matrix[:2,:]
            current_image = cv2.warpAffine(current_image, transform_matrix, (width, height))
            continue
            current_image = apply_affine_transform(current_image, transform_matrix)
        elif 'MovingLeastSquaresTransform2' in transform_info['class']:
            mls_params = parse_mls_data(transform_info['data'])
            pi = mls_params['src_points']
            src_pts = np.float32(mls_params['src_points'])
            dst_pts = np.float32(mls_params['dst_points'])

            # 计算变换并应用
            M = cv2.getAffineTransform(src_pts, dst_pts)
            print(M)
            current_image = cv2.warpAffine(current_image, M, (current_image.shape[1], current_image.shape[0]))
            '''pi =np.array(pi).astype(int)
            print(pi)
            qi = mls_params['dst_points']
            qi =np.array(qi).astype(int)
            print(qi)
            ddd = trans(current_image, pi)
            current_image = ddd.deformation(current_image, qi)'''
    
    return current_image

# 辅助函数
def parse_transform_matrix(transform_str):
    """解析transform属性中的矩阵"""
    match = re.search(r'matrix\(([^)]+)\)', transform_str)
    if match:
        values = list(map(float, match.group(1).split(',')))
        return np.array([
            [values[0], values[1], values[4]],
            [values[2], values[3], values[5]],
            [0, 0, 1]
        ])
    return np.eye(3)




# 使用示例
if __name__ == "__main__":
    # 读取XML信息
    xml_path = r"e:\03.users\wangaoli\data\raw_data\Drosophila_test\test_result\02.register\02.manual\A02183A2.xml"
    info_dict = read_xml_transform_info(xml_path)
    print("读取到变换信息:", info_dict)
    
    # 读取图像
    input_image_path = r"c:\Users\wangaoli\Desktop\remov\A02183A2.tif"
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    '''src_pts = np.float32([[715,617], [684,1013], [807,895]])
    dst_pts = np.float32([[740,609], [481,942], [719,956]])

    # 计算变换并应用
    M = cv2.getAffineTransform(src_pts, dst_pts)
    result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    output_path = r"e:\03.users\wangaoli\data\raw_data\Drosophila_test\test_result\02.register\01.align_mask\A02183A2_algo_ttest.tif"
    tif.imwrite(output_path, result)
    exit()
    '''
    # 应用变换
    transformed_image = apply_transforms_to_image(image, info_dict)
    
    # 保存结果
    output_path = r"e:\03.users\wangaoli\data\raw_data\Drosophila_test\test_result\02.register\01.align_mask\A02183A2_algo_ttest.tif"
    tif.imwrite(output_path, transformed_image)
    print("变换完成")