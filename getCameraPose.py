import cv2
import numpy as np
import matplotlib.pyplot as plt

#---------------- 参数区 ----------------#
# 棋盘格内角点数（patternSize）
rows = 6   # 行方向角点数 多少行？
cols = 5   # 列方向角点数

square_size = 30.0  # mm，格子实际边长

# 相机内参（示例）
fx = 2415.3105468750
fy = 2415.5842285156
cx = 1573.9013671875
cy = 1060.9655761719
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

dist_coeffs = np.zeros((5, 1))  # 畸变参数
dist_coeffs[0, 0] = -0.1052031666   # k1
dist_coeffs[1, 0] = 0.1446554661    # k2
dist_coeffs[2, 0] = -0.0000056888   # p1
dist_coeffs[3, 0] = -0.0001116273   # p2
dist_coeffs[4, 0] = -0.0555365160   # k3

#---------------- 生成obj_points ----------------#
# (rows, cols) = (高, 宽)
obj_points = []
for y in range(rows):
    for x in range(cols):
        obj_points.append([x * square_size, y * square_size, 0])
obj_points = np.array(obj_points, dtype=np.float32)

#---------------- 加载图像 ----------------#
filePath = "F:/HiK_CAM/HiViewer/Data/MV-DB1300A(00DA3156477)/" #F:\HiK_CAM\HiViewer\Data\MV-DB1300A(00DA3156477)
img = cv2.imread(filePath + "Image__Rgb_9.bmp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#---------------- 查找角点 ----------------#
pattern_size = (cols, rows)      # 必须为(cols, rows)！
ret, img_points = cv2.findChessboardCorners(gray, pattern_size)

if not ret:
    print("找不到棋盘格角点")
    exit()

# 亚像素优化（可选，强烈建议）
img_points = cv2.cornerSubPix(
    gray,
    img_points,
    winSize=(11,11),
    zeroZone=(-1,-1),
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

#---------------- solvePnP 求位姿 ----------------#
success, rvec, tvec = cv2.solvePnP(
    obj_points,
    img_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# rvec->R
R, _ = cv2.Rodrigues(rvec)

# 构造T_board2cam
T_board2cam = np.eye(4)
T_board2cam[:3,:3] = R
T_board2cam[:3, 3] = tvec.reshape(3)

print("T_board2cam=\n", T_board2cam)

# 求逆得到T_cam2board
T_cam2board = np.linalg.inv(T_board2cam)
print("T_cam2board=\n", T_cam2board)

#---------------- 绘制角点函数 ----------------#
# 直接绘制棋盘格角点
vis = img.copy()
cv2.drawChessboardCorners(vis, pattern_size, img_points, True)

# # 显示结果
# cv2.namedWindow("corners", cv2.WINDOW_NORMAL)
# cv2.imshow("corners", vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 简单增强：在原图基础上添加更大的角点标记
corners_array = np.array(img_points).reshape(-1, 2)
for corner in corners_array:
    x, y = int(corner[0]), int(corner[1])
    cv2.circle(vis, (x, y), 15, (0, 255, 0), -1)  # 添加绿色外圈
# 按照corners_array顺序连接相邻点
for i in range(len(corners_array) - 1):
    pt1 = tuple(corners_array[i].astype(int))
    pt2 = tuple(corners_array[i+1].astype(int))
    cv2.line(vis, pt1, pt2, (255, 0, 0), 3)  # 蓝色实线，线宽3
# 创建matplotlib图形
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title('Chessboard Corners Detection', fontsize=16)
plt.axis('on')
# 显示图像
plt.tight_layout()
plt.show()


