import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from natsort import natsorted

def camera_calibration(images_folder, pattern_size, square_size):
    """
    相机内参标定函数
    参数:
        images_folder: 包含标定图像的文件夹路径
        pattern_size: 棋盘格内角点数量 (columns, rows)
        square_size: 每个方格的实际尺寸（米）
    返回:
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        rvecs: 旋转向量列表
        tvecs: 平移向量列表
        reprojection_error: 重投影误差
    """

    # 准备世界坐标系中的角点坐标 (Z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储对象点和图像点
    obj_points = []  # 3D点在世界坐标系中
    img_points = []  # 2D点在图像平面中

    # 获取所有图像文件
    images = glob.glob(os.path.join(images_folder, "*.jpg")) + \
             glob.glob(os.path.join(images_folder, "*.png")) + \
             glob.glob(os.path.join(images_folder, "*.bmp"))
    images = natsorted(images) #重新排序

    if not images:
        print("未找到图像文件")
        return None, None, None, None, None

    print(f"找到 {len(images)} 张图像")

    # 处理每张图像
    for i, fname in enumerate(images):
        print(f"处理图像 {i + 1}/{len(images)}: {os.path.basename(fname)}")

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # 亚像素级角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            obj_points.append(objp)
            img_points.append(corners_refined)

            # 可视化角点检测结果（可选）
            img_draw = img.copy()
            # draw_conrners_point_image(img_draw, pattern_size, corners_refined, ret)
            print(f"成功检测到角点")

        else:
            print(f"未检测到角点")

    print(f"\n成功处理 {len(obj_points)}/{len(images)} 张图像")

    if len(obj_points) < 5:
        print("有效图像数量不足，需要至少5张不同角度的图像")
        return None, None, None, None, None

    # 相机标定
    print("正在进行相机标定...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # 计算重投影误差
    total_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                           camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error

    mean_error = total_error / len(obj_points)

    print(f"\n标定完成!")
    print(f"重投影误差: {mean_error:.6f} 像素")

    return camera_matrix, dist_coeffs, rvecs, tvecs, mean_error


def save_calibration_results(camera_matrix, dist_coeffs, filename="camera_calibration.yml"):
    """
    保存标定结果到文件
    """
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("fx", camera_matrix[0, 0])
    fs.write("fy", camera_matrix[1, 1])
    fs.write("cx", camera_matrix[0, 2])
    fs.write("cy", camera_matrix[1, 2])

    fs.release()
    print(f"标定结果已保存到 {filename}")


def undistort_image(image_path, camera_matrix, dist_coeffs):
    """
    使用标定结果校正图像畸变
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 优化相机矩阵以获得更好的校正效果
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # 校正畸变
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像
    x, y, w, h = roi
    dst_cropped = dst[y:y + h, x:x + w]

    return dst, dst_cropped, new_camera_matrix

def draw_conrners_point_image(img_draw, pattern_size, corners_refined, ret):
    cv2.drawChessboardCorners(img_draw, pattern_size, corners_refined, ret)
    corners_array = np.array(corners_refined).reshape(-1, 2)
    for corner in corners_array:
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(img_draw, (x, y), 15, (0, 255, 0), -1)  # 添加绿色外圈
    # 按照corners_array顺序连接相邻点
    for i in range(len(corners_array) - 1):
        pt1 = tuple(corners_array[i].astype(int))
        pt2 = tuple(corners_array[i + 1].astype(int))
        cv2.line(img_draw, pt1, pt2, (255, 0, 0), 3)  # 蓝色实线，线宽3
    # 创建matplotlib图形
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.title('Chessboard Corners Detection', fontsize=16)
    plt.axis('on')
    # 显示图像
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 标定相机
    camera_matrix, dist_coeffs, rvecs, tvecs, error = camera_calibration(
        "F:/HiK_CAM/HiViewer/Data/MV-DB1300A(00DA3156477)",
        pattern_size=(6, 5), square_size=0.030
    )

    if camera_matrix is not None:
        print("\n相机内参矩阵:")
        print(camera_matrix)
        print("\n畸变系数:")
        print(dist_coeffs.ravel())

        # # 保存标定结果
        # save_calibration_results(camera_matrix, dist_coeffs)

        # # 测试畸变校正
        # test_image = "test_image.jpg"  # 替换为测试图像路径
        # if os.path.exists(test_image):
        #     undistorted, cropped, new_cam_matrix = undistort_image(
        #         test_image, camera_matrix, dist_coeffs
        #     )
        #
        #     # 保存校正后的图像
        #     cv2.imwrite("undistorted.jpg", undistorted)
        #     cv2.imwrite("undistorted_cropped.jpg", cropped)
        #     print("畸变校正完成，结果已保存")
