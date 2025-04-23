import os
import glog

import cv2 as cv
import numpy as np
import open3d as o3d

from tqdm import tqdm
from glob import glob


# ---------------points 3d create--------------- #
def contours_in(contours, shape):
    """
    没啥说的
    """
    p = np.zeros(shape=shape, dtype=np.uint8)
    cv.drawContours(p, contours, -1, 255, -1)
    a = np.where(p == 255)[0].reshape(-1, 1)
    b = np.where(p == 255)[1].reshape(-1, 1)
    coordinate = np.concatenate([b, a], axis=1).tolist()
    inside = [tuple(x) for x in coordinate]
    return np.array(inside)


def _contours2points(contours):
    """
    Image contours to points(x,y)
    Args:
        contours:
    Returns:
        contour_point:
    """

    contour_point = None
    for contour in contours:
        contour = contour.reshape(contour.shape[0], -1)
        if contour_point is None:
            contour_point = contour
        else:
            contour_point = np.concatenate((contour_point, contour), axis=0)

    return contour_point


def get_mask_3d_points(mask_path_list,
                       mask_z_interval,
                       z_interval,
                       pixel4mm,
                       output_path = None):
    """
    Args:
        mask_path_list: list 排序完成的mask路径
        mask_z_interval: list mask图像对应的z轴数值 - mm
        pixel4mm: float mask图像每个像素所代表的物理距离 - mm (一般是500nm)
        z_interval: 单位切片厚度 - mm
        output_path:
    Returns:
        points_3d: 所有的3d点
    """

    points_3d_edge = None
    points_3d_inside = None
    scale = int(z_interval / pixel4mm)

    for i, img_path in enumerate(tqdm(mask_path_list, desc="Mask", ncols=100)):
        image = cv.imread(img_path, -1)
        image = cv.resize(image, [int(image.shape[1] / scale), int(image.shape[0] / scale)])

        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contour_point = _contours2points(contours)

        edge_contour_point = contour_point
        inside_contour_point = contours_in(contours, image.shape)

        z_point = np.ones((edge_contour_point.shape[0], 1)) * mask_z_interval[i]
        edge_contour_point = edge_contour_point * pixel4mm
        edge_contour_point = np.concatenate((edge_contour_point, z_point), axis=1)

        z_point = np.ones((inside_contour_point.shape[0], 1)) * mask_z_interval[i]
        inside_contour_point = inside_contour_point * pixel4mm
        inside_contour_point = np.concatenate((inside_contour_point, z_point), axis=1)

        if points_3d_edge is None:
            points_3d_edge = edge_contour_point
        else:
            points_3d_edge = np.concatenate((points_3d_edge, edge_contour_point), axis=0)

        if points_3d_inside is None:
            points_3d_inside = inside_contour_point
        else:
            points_3d_inside = np.concatenate((points_3d_inside, inside_contour_point), axis=0)

    points_3d_edge[:, :2] = points_3d_edge[:, :2] * scale
    points_3d_inside[:, :2] = points_3d_inside[:, :2] * scale

    # points_3d_inside = _dense2sparse(points_3d_inside)
    points_3d_edge = np.concatenate((points_3d_edge, points_3d_inside), axis=0)
    points_3d_edge = np.unique(points_3d_edge, axis=0)
    # points_3d_edge = _dense2sparse(points_3d_edge)

    if output_path:
        np.savetxt(os.path.join(output_path, "mask_3d_points.txt"), points_3d_edge)

    return points_3d_edge


def _dense2sparse(points):
    x_points_list = list(set(points[:, 0]))
    x_points_list = sorted(x_points_list)
    y_points_list = list(set(points[:, 1]))
    y_points_list = sorted(y_points_list)

    new_points = None

    for x in tqdm(x_points_list, desc="To sparse"):
        for y in y_points_list:
            temp_points = points[(points[:, 0] == x) & (points[:, 1] == y)]

            if len(temp_points) == 0:
                continue

            temp_points = np.unique(temp_points, axis=0)

            if len(temp_points) > 2:
                z_points_list = list(set(temp_points[:, 2]))
                z_points_list = sorted(z_points_list)
                z_min = z_points_list[0]
                z_max = z_points_list[-1]

                temp_points = temp_points[(temp_points[:, 2] == z_min) | (temp_points[:, 2] == z_max)]

            if new_points is None:
                new_points = temp_points
            else:
                new_points = np.concatenate((new_points, temp_points), axis=0)

    return new_points


def _fix_points_3d(points_3d, z_interval):
    """
    对于切片时跳片的3d点做补全, 使用上一片填充
    Args:
        points_3d:
        z_interval: 切片间隔 - mm
    Return:
        points_3d：
    """
    z_points_list = list(set(points_3d[:, 2]))
    z_points_list = sorted(z_points_list)

    for index in range(len(z_points_list) - 1):
        z_up = z_points_list[index]
        z_down = z_points_list[index + 1]
        if z_down - z_up != z_interval:
            z_dis = int((z_down - z_up) / z_interval - 1)

            for i in range(z_dis):
                _z = z_interval * (i + 1) + z_up
                _points = points_3d[points_3d[:, 2] == z_up].copy()
                _points[:, 2] = _z
                points_3d = np.concatenate((points_3d, _points), axis=0)

    return points_3d


def _delete_outlier(points_3d, delta = 1.5):
    z_points_list = list(set(points_3d[:, 2]))
    z_points_list = sorted(z_points_list)
    new_points_3d = np.zeros([1, 3])

    x_std_list = list()
    y_std_list = list()
    points_flag = [False] * len(z_points_list)

    for z in z_points_list:
        points = points_3d[points_3d[:, 2] == z].copy()

        x_std = points[:, 0].std()
        x_std_list.append(x_std)

        y_std = points[:, 1].std()
        y_std_list.append(y_std)

    x_std_low = np.mean(x_std_list) - delta * np.std(x_std_list)
    x_std_high = np.mean(x_std_list) + delta * np.std(x_std_list)

    y_std_low = np.mean(y_std_list) - delta * np.std(y_std_list)
    y_std_high = np.mean(y_std_list) + delta * np.std(y_std_list)

    for index, (x_std, y_std) in enumerate(zip(x_std_list, y_std_list)):
        if x_std_low < x_std < x_std_high and y_std_low < y_std < y_std_high:
            points_flag[index] = True

    for flag, z in zip(points_flag, z_points_list):
        points = points_3d[points_3d[:, 2] == z].copy()
        if not flag:
            pass
        else:
            new_points_3d = np.concatenate((new_points_3d, points), axis=0)
    return new_points_3d


##-----------------mesh faction-----------------##
def _mesh_voxel_down_sample(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def _points_3d_read(points,
                    random=None,
                    z_interval=None):
    """
    Args:
        points:
        random: None | int  随机值比例
        z_interval: float z轴平均单位，若有则补全空余轴
    Returns:
        pcd: class - point cloud
    """
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
    elif isinstance(points, str):
        pcd = o3d.io.read_point_cloud(points)
        points = np.asarray(pcd.points)

    if isinstance(z_interval, (int, float)):
        points_3d = _fix_points_3d(points, z_interval)

    if isinstance(random, (int, float)):
        random_points = np.random.random(points_3d.shape) / random
        points_3d = points_3d + random_points

    pcd.points = o3d.utility.Vector3dVector(points_3d)

    return pcd


def _mesh_outlier_removal(pcd,
                          method='stat',
                          num_neighbors=20,
                          num_points = 20,
                          std_ratio=2.0,
                          radius=0.05):
    """
    Args:
        pcd:
        method: 统计滤波 statistical | 半径滤波 radius
        num_neighbors: statistical - K邻域点的个数
        num_points: radius - 领域半径内最少点数，低于该值为噪声点
        std_ratio: statistical - 标准差乘数
        radius: radius - 领域半径大小（与pcd的x, y取值范围关系较大）
    Returns:
        sor_pcd:
        outlier_pcd:
    """
    if method == "stat":
        sor_pcd, ind = pcd.remove_statistical_outlier(num_neighbors, std_ratio)
    elif method == "radius":
        sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
    else:
        glog.info("Wrong method name of outlier removal!")
        return

    sor_noise_pcd = pcd.select_by_index(ind, invert=True)

    return sor_pcd, sor_noise_pcd


def _mesh_create(pcd,
                 method='alpha_shape',
                 alpha=0.1,
                 radii=[0.005, 0.01, 0.02],
                 show_mesh=False):
    """
    Args:
        pcd:
        method: Alpha Shape | Ball pivoting
        alpha: Alpha - 参数控制
        radii: Ball - 参数控制
        show_mesh: 是否展示mesh
    Returns:
        mesh:
    """
    if method == "alpha_shape":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    elif method == "ball_pivoting":
        if len(np.asarray(pcd.normals)) == 0:
            pcd.estimate_normals()
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
    else:
        glog.info("Wrong method name of create mesh!")
        return

    mesh.compute_vertex_normals()
    if show_mesh:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    return mesh


def _mesh_filter_smooth(mesh,
                        method='simple',
                        simple_num=5,
                        lambda_filter=0.5,
                        show_mesh=False):
    """
    Args:
        mesh:
        method: simple | laplacian
        simple_num: simple - 平滑控制 迭代平滑
        lambda_filter: laplacian - 平滑控制
        show_mesh: 是否展示mesh
    Returns:
        mesh:
    """
    if method == "simple":
        mesh_filter = mesh.filter_smooth_simple(number_of_iterations=simple_num)
    elif method == "laplacian":
        mesh_filter = mesh.filter_smooth_laplacian(lambda_filter=lambda_filter)
    else:
        glog.info("Wrong method name of mesh filter smooth!")
        return

    mesh_filter.compute_vertex_normals()
    if show_mesh:
        o3d.visualization.draw_geometries([mesh_filter], mesh_show_back_face=True)

    return mesh_filter


def points_3d_to_mesh(points_3d,
                      z_interval,
                      mesh_scale = 1,
                      down_size = None,
                      output_path = None,
                      show_mesh=False,
                      name=None):
    """
    Args:
        points_3d: xyz点
        z_interval: 切片间隔尺寸 - mm
        mesh_scale: mesh放大的倍率
        down_size: 体素下采样倍率
        output_path:
        show_mesh:
        name:
    Return:

    """
    # points_3d = _delete_outlier(points_3d)
    pcd = _points_3d_read(points_3d, random=100, z_interval=z_interval)
    # pcd.estimate_normals()
    if show_mesh:
        o3d.visualization.draw_geometries([pcd], mesh_show_back_face=True)

    ################ 体素下采样
    if down_size is not None:
        pcd = _mesh_voxel_down_sample(pcd, z_interval * down_size)

    ###################### 滤波
    # pcd, _ = _mesh_outlier_removal(pcd, method="stat")

    ########################## Mesh构建
    alpha = np.pi * z_interval * (1 if not down_size else down_size)  # 半径控制 越小越棱角越少 但不能低于片与片间隔
    radii = [0.005, 0.01, 0.02]
    mesh = _mesh_create(pcd, method='alpha_shape', alpha=alpha, show_mesh=show_mesh)

    ###############Mesh 平滑
    mesh1 = _mesh_filter_smooth(mesh, method='simple', show_mesh=show_mesh)

    mesh1 = mesh1.scale(mesh_scale, mesh1.get_center())  # 尺度控制
    mesh1.triangle_normals = o3d.utility.Vector3dVector([])

    # mesh1.paint_uniform_color([1, 0.706, 0])  # 涂色

    name = name if name else "mask_mesh"
    o3d.io.write_triangle_mesh(os.path.join(output_path, f"{name}.obj"), mesh1)


if __name__ == "__main__":
    from stereo3d.file.slice import SliceSequence

    # xlsx = r"E:\lizepeng\m115\E-ST20220923002_slice_records_20221110.xlsx"
    # ss = SliceSequence()
    # ss.from_xlsx(file_path=xlsx)
    # z_interval_dict = ss.get_z_interval(index='bf')
    # z_interval = ss.z_interval
    #
    # pixel4mm = 0.0005
    #
    # mask_path = r"F:\lizepeng\1"
    # mask_path_list = glob(os.path.join(mask_path, "*.tif"))
    # mask_path_list = sorted(mask_path_list, key=lambda x: int(os.path.basename(x).split('.')[0]))
    #
    # mask_z_interval = list()
    # for mask in mask_path_list:
    #     ind = int(os.path.basename(mask).split('.')[0])
    #     for k, v in z_interval_dict.items():
    #         if int(float(k)) == ind:
    #             mask_z_interval.append(v)
    # ------------------ #
    mask_path = r"D:\02.data\luqin\E14-16h_a_bin1_image_regis"
    mask_path_list = glob(os.path.join(mask_path, "*.tif"))
    mask_path_list = sorted(mask_path_list)
    mask_z_interval = [i/1000 for i in range(0, 128, 8)]
    z_interval = 0.008
    pixel4mm = 0.0005

    points_3d = get_mask_3d_points(mask_path_list,
                                   mask_z_interval,
                                   z_interval=z_interval,
                                   pixel4mm=pixel4mm,
                                   output_path=r"D:\02.data\luqin\E14-16h_a_bin1_image_regis")

    # points_3d = np.loadtxt(r"E:\lizepeng\m115\m11.5\mask_3d_points_80.txt")

    points_3d_to_mesh(points_3d,
                      z_interval=z_interval,
                      mesh_scale=1,
                      output_path=r"D:\02.data\luqin\E14-16h_a_bin1_image_regis")

