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
    weight matrix w_ij = 1 / |p_j - v_i|^(2α)
    
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
    # distance between each target point and source control point
    distances = cdist(v, p, metric='euclidean')
    
    distances = np.maximum(distances, 1e-8)

    # weight: w_ij = 1 / |p_j - v_i|^(2α)
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
    
    """
    # Computing pstar
    pstar = Precompute_pstar(p, w)
    
    np_points = p.shape[0]
    nv = v.shape[0]
    
    # phat
    phat = p - pstar[:, np.newaxis, :]  # shape: (nv, np, 2)
    
    # M2 matrix
    M2 = np.zeros((nv, 2, 2))
    for i in range(np_points):
        # w[:, i] shape: (nv,), phat[:, i] shape: (nv, 2)
        wi = w[:, i, np.newaxis, np.newaxis]  # shape: (nv, 1, 1)
        phi = phat[:, i, :, np.newaxis]  # shape: (nv, 2, 1)
        phi_T = phat[:, i, np.newaxis, :]  # shape: (nv, 1, 2)

        M2 += wi * (phi @ phi_T)
    
    IM2 = np.zeros((nv, 2, 2))
    for k in range(nv):
        try:
            IM2[k] = solve(M2[k], np.eye(2))
        except np.linalg.LinAlgError:
            IM2[k] = np.linalg.pinv(M2[k])

    M1 = v - pstar  # shape: (nv, 2)
    F1 = np.zeros((nv, 2))
    for k in range(nv):
        F1[k] = M1[k] @ IM2[k]

    # A matrix
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
  
    w = compute_weights(p, v, alpha)
    

    A = Precompute_Affine(p, v, w)

    nv = w.shape[0]
    qstar = np.zeros((nv, 2))
    for i in range(nv):
        weights = w[i, :]
        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            qstar[i, :] = np.dot(weights, q) / total_weight
    
    print("Computing deformed positions...")
    deformed_positions = A @ q + qstar
    
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

    y_coords, x_coords = np.mgrid[0:height, 0:width]
    v = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    w = compute_weights(p, v, alpha)
    
    '''print("Precomputing transformation...")
    data = Precompute_Affine(p, v, w)
    A = data['A']'''
    A = Precompute_Affine(p, v, w)
    
    nv = w.shape[0]
    qstar = np.zeros((nv, 2))
    for i in range(nv):
        weights = w[i, :]
        total_weight = np.sum(weights)
        if total_weight > 1e-10:
            qstar[i, :] = np.dot(weights, q) / total_weight
    
    deformed_positions = A @ q + qstar
    
    deformed_positions[:, 0] = np.clip(deformed_positions[:, 0], 0, width - 1)
    deformed_positions[:, 1] = np.clip(deformed_positions[:, 1], 0, height - 1)
    
    map_x = deformed_positions[:, 0].reshape(height, width).astype(np.float32)
    map_y = deformed_positions[:, 1].reshape(height, width).astype(np.float32)
 
    print("Creating deformed image...")
    deformed_image = cv2.remap(image, map_x, map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    
    return deformed_image, map_x, map_y





if __name__ == "__main__":
    image = tifffile.imread(r"D:\stereo3d_data\demo\Drosophila_melanogaster_demo\demoresult\02.register\01.align_mask\A02183A5.tif")
    # control point
    p = [[992., 774.], [876., 498.], [670., 808.], [874., 1164.]]
    q = [[1060., 762.], [850., 510.], [614., 808.], [900., 1136.]]
    
    deformed_image = apply_affine_deformation_faster(image, q, p, alpha=1.0)

    tifffile.imwrite(r"D:\stereo3d_data\demo\Drosophila_melanogaster_demo\demoresult\02.register\01.align_mask\A02183A5_test.tif", deformed_image)