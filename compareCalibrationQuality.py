import cv2
import glob
from natsort import natsorted
import os
import numpy as np

def compare_calibration_quality(factory_matrix, factory_dist, custom_matrix, custom_dist, images_folder, pattern_size, square_size):
    """
    对比出厂内参和自定义内参的质量
    """
    factory_errors = []
    custom_errors = []

    # 获取所有图像文件
    images = glob.glob(os.path.join(images_folder, "*.jpg")) + \
             glob.glob(os.path.join(images_folder, "*.png")) + \
             glob.glob(os.path.join(images_folder, "*.bmp"))
    images = natsorted(images) #重新排序

    obj_points = []
    rows = pattern_size[1]
    cols = pattern_size[0]
    for y in range(rows):
        for x in range(cols):
            obj_points.append([x * square_size, y * square_size, 0])
    obj_points = np.array(obj_points, dtype=np.float32)

    for img_path in images:
        # 读取图像并检测角点
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # 使用出厂参数计算重投影误差
            _, rvec_factory, tvec_factory = cv2.solvePnP(
                obj_points, corners, factory_matrix, factory_dist
            )
            corners_repro_factory, _ = cv2.projectPoints(
                obj_points, rvec_factory, tvec_factory, factory_matrix, factory_dist
            )
            error_factory = cv2.norm(corners, corners_repro_factory, cv2.NORM_L2) / len(corners)

            # 使用自定义参数计算重投影误差
            _, rvec_custom, tvec_custom = cv2.solvePnP(
                obj_points, corners, custom_matrix, custom_dist
            )
            corners_repro_custom, _ = cv2.projectPoints(
                obj_points, rvec_custom, tvec_custom, custom_matrix, custom_dist
            )
            error_custom = cv2.norm(corners, corners_repro_custom, cv2.NORM_L2) / len(corners)

            factory_errors.append(error_factory)
            custom_errors.append(error_custom)

    # 统计比较
    factory_mean = np.mean(factory_errors)
    custom_mean = np.mean(custom_errors)

    print(f"出厂内参平均重投影误差: {factory_mean:.4f} 像素")
    print(f"自定义内参平均重投影误差: {custom_mean:.4f} 像素")
    print(f"改进: {(factory_mean - custom_mean) / factory_mean * 100:.1f}%")

    return factory_errors, custom_errors

if __name__ == "__main__":

    images_folder = "F:/HiK_CAM/HiViewer/Data/MV-DB1300A(00DA3156477)/"  # F:\HiK_CAM\HiViewer\Data\MV-DB1300A(00DA3156477)

    # 相机内参（示例）。改成文件读取参数
    fx = 2415.3105468750
    fy = 2415.5842285156
    cx = 1573.9013671875
    cy = 1060.9655761719
    factory_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    factory_dist = np.zeros((5, 1))  # 畸变参数
    factory_dist[0, 0] = -0.1052031666  # k1
    factory_dist[1, 0] = 0.1446554661   # k2
    factory_dist[2, 0] = -0.0000056888  # p1
    factory_dist[3, 0] = -0.0001116273  # p2
    factory_dist[4, 0] = -0.0555365160  # k3

    # 相机内参（示例）
    fx = 2353.75075
    fy = 2355.24715
    cx = 1559.93418
    cy = 1057.23203
    custom_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)
    custom_dist = np.zeros((5, 1))  # 畸变参数
    custom_dist[0, 0] = -0.00893656  # k1
    custom_dist[1, 0] = 0.0031684    # k2
    custom_dist[2, 0] = -0.00072284  # p1
    custom_dist[3, 0] = -0.00048698  # p2
    custom_dist[4, 0] = -0.01147842  # k3

    pattern_size = (5, 6)
    square_size = 0.030
    factory_errors, custom_errors = compare_calibration_quality(factory_matrix, factory_dist, custom_matrix, custom_dist, images_folder, pattern_size, square_size)