import numpy as np
import cv2
import tifffile
from scipy.linalg import solve
import numpy as np
import cv2
from scipy.linalg import solve
from scipy.spatial.distance import cdist

def compute_weights(p, v, alpha=1.0):
    """
    计算权重矩阵 w_ij = 1 / |p_j - v_i|^(2α)
    
    Parameters:
    p : numpy array, shape (np, 2)
        Source control points
    v : numpy array, shape (nv, 2)
        Target pixel positions (all pixels in image)
    alpha : float
        Weight decay parameter
    
    Returns:
    w : numpy array, shape (nv, np)
        Weight matrix
    """
    # 计算所有点对之间的距离
    distances = cdist(v, p, metric='euclidean')
    
    # 避免除零，添加小常数
    distances = np.maximum(distances, 1e-8)
    
    # 计算权重: w_ij = 1 / |p_j - v_i|^(2α)
    w = 1.0 / (distances ** (2 * alpha))
    
    return w

def Precompute_pstar(p, w):
    """
    计算加权质心 p*
    
    Parameters:
    p : numpy array, shape (np, 2)
        Source control points
    w : numpy array, shape (nv, np)  
        Weight matrix
    
    Returns:
    pstar : numpy array, shape (nv, 2)
        Weighted centroid for each target point
    """
    nv = w.shape[0]
    pstar = np.zeros((nv, 2))
    
    for i in range(nv):
        weights = w[i, :]
        total_weight = np.sum(weights)
        
        if total_weight > 1e-10:
            pstar[i, :] = np.dot(weights, p) / total_weight
        else:
            pstar[i, :] = np.mean(p, axis=0)
    
    return pstar

def Precompute_Affine(p, v, w):
    """
    修复版的预计算函数
    """
    # Computing pstar
    pstar = Precompute_pstar(p, w)
    
    np_points = p.shape[0]
    nv = v.shape[0]
    
    # 预计算 phat
    phat = p - pstar[:, np.newaxis, :]  # shape: (nv, np, 2)
    
    # 计算 M2 矩阵
    M2 = np.zeros((nv, 2, 2))
    for i in range(np_points):
        # w[:, i] shape: (nv,), phat[:, i] shape: (nv, 2)
        wi = w[:, i, np.newaxis, np.newaxis]  # shape: (nv, 1, 1)
        phi = phat[:, i, :, np.newaxis]  # shape: (nv, 2, 1)
        phi_T = phat[:, i, np.newaxis, :]  # shape: (nv, 1, 2)
        
        # 累加到 M2: w_i * phat_i^T * phat_i
        M2 += wi * (phi @ phi_T)
    
    # 计算逆矩阵
    IM2 = np.zeros((nv, 2, 2))
    for k in range(nv):
        try:
            IM2[k] = solve(M2[k], np.eye(2))
        except np.linalg.LinAlgError:
            IM2[k] = np.linalg.pinv(M2[k])
    
    # 计算 F1 = (v - pstar) * IM2
    M1 = v - pstar  # shape: (nv, 2)
    F1 = np.zeros((nv, 2))
    for k in range(nv):
        F1[k] = M1[k] @ IM2[k]
    
    # 计算 A 矩阵
    A = np.zeros((nv, np_points))
    for i in range(np_points):
        # A[:, i] = F1 * phat_i * w_i
        A[:, i] = np.sum(F1 * phat[:, i], axis=1) * w[:, i]
    return A


def apply_affine_deformation(height, width, p, q, alpha=1.0):
    """

    Parameters:
    image : numpy array
    p : source points, numpy array, shape (np, 2)
    q : dest points, numpy array, shape (np, 2)
    alpha : float

    Returns:
    deformed_image : numpy array
    """

    p = np.array(p).reshape(-1, 2)
    q = np.array(q).reshape(-1, 2)
    
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    v = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # 计算权重
    print("Computing weights...")
    w = compute_weights(p, v, alpha)
    

    A = Precompute_Affine(p, v, w)
    
    # 计算qstar
    nv = w.shape[0]
    qstar = np.zeros((nv, 2))
    for i in range(nv):
        weights = w[i, :]
        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            qstar[i, :] = np.dot(weights, q) / total_weight
    
    print("Computing deformed positions...")
    deformed_positions = A @ q + qstar
    print(deformed_positions.shape)
    
    deformed_positions[:, 0] = np.clip(deformed_positions[:, 0], 0, width - 1)
    deformed_positions[:, 1] = np.clip(deformed_positions[:, 1], 0, height - 1)
    
    map_x = deformed_positions[:, 0].reshape(height, width).astype(np.float32)
    map_y = deformed_positions[:, 1].reshape(height, width).astype(np.float32)

    return map_x, map_y


def apply_affine_deformation_faster(image, p, q, alpha=1.0):
    """

     Parameters:
    image : numpy array
    p : source points, numpy array, shape (np, 2)
    q : dest points, numpy array, shape (np, 2)
    alpha : float

    Returns:
    deformed_image : numpy array
    """
    p = np.array(p).reshape(-1, 2)
    q = np.array(q).reshape(-1, 2)
    height, width = image.shape[:2]
    
    # 创建像素坐标网格
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    v = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # 计算权重
    print("Computing weights...")
    w = compute_weights(p, v, alpha)
    
    # 预计算
    '''print("Precomputing transformation...")
    data = Precompute_Affine(p, v, w)
    A = data['A']'''
    A = Precompute_Affine(p, v, w)
    
    # 计算qstar
    nv = w.shape[0]
    qstar = np.zeros((nv, 2))
    for i in range(nv):
        weights = w[i, :]
        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            qstar[i, :] = np.dot(weights, q) / total_weight
    
    # 使用矩阵乘法计算变形位置
    print("Computing deformed positions...")
    deformed_positions = A @ q + qstar
    print(deformed_positions.shape)
    
    # 限制范围并创建输出图像
    deformed_positions[:, 0] = np.clip(deformed_positions[:, 0], 0, width - 1)
    deformed_positions[:, 1] = np.clip(deformed_positions[:, 1], 0, height - 1)
    
    map_x = deformed_positions[:, 0].reshape(height, width).astype(np.float32)
    map_y = deformed_positions[:, 1].reshape(height, width).astype(np.float32)
 
    print("Creating deformed image...")
    deformed_image = cv2.remap(image, map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    
    return deformed_image, map_x, map_y





# 使用示例
if __name__ == "__main__":
    # 加载图片
    image = tifffile.imread(r"D:\stereo3d_data\demo\Drosophila_melanogaster_demo\demoresult\02.register\01.align_mask\A02183A5.tif")
    # control point
    p = [[992., 774.], [876., 498.], [670., 808.], [874., 1164.]]
    q = [[1060., 762.], [850., 510.], [614., 808.], [900., 1136.]]
    
    deformed_image = apply_affine_deformation_faster(image, q, p, alpha=1.0)

    
    # 保存结果
    tifffile.imwrite(r"D:\stereo3d_data\demo\Drosophila_melanogaster_demo\demoresult\02.register\01.align_mask\A02183A5_test.tif", deformed_image)