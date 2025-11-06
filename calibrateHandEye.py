import cv2
import numpy as np
import os
from mySolvePnP import solve_pnp
from eulerAnglesToPose import *

# -----------------------------------------
# 机器人末端相对基坐标位姿 (base -> end effector)
# 每个R是3x3，t是3x1
# -----------------------------------------
R_base2gripper = []  # EyeInHand时候是R_gripper2base
t_base2gripper = []

# # 示例：假设有3组机器人位姿（实际至少8组以上更稳定）
# R_base2gripper.append(np.eye(3))
# t_base2gripper.append(np.array([[0.0], [0.0], [0.0]]))

R_base2gripper, t_base2gripper = euler_to_pose_matrix('test_poses.txt')

# -----------------------------------------
# 棋盘格相对相机的位姿 (camera -> object)
# 通过 solvePnP 得到
# -----------------------------------------
R_target2cam = []
t_target2cam = []
# 从文件读取照片，然后用solvePnP求解T_base2gripper
idx = 1
filePath = "F:/HiK_CAM/HiViewer/Data/MV-DB1300A(00DA3156477)/"  # F:\HiK_CAM\HiViewer\Data\MV-DB1300A(00DA3156477)
while True:
    fileName = "Image__Rgb_" + str(idx) + ".bmp"
    file = filePath + fileName
    if not os.path.exists(file):
        break
    r, t = solve_pnp(file)
    # 将求得的T_base2gripper放入容器中
    R_target2cam.append(r)
    t_target2cam.append(t)
    idx = idx + 1

# -----------------------------------------
# 调用 OpenCV 手眼标定
# 这里使用 Tsai 方法 (可换内置其他方法)
# -----------------------------------------
# print("OpenCV version:", cv2.__version__)
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_base2gripper, t_base2gripper,  # Robot poses
    R_target2cam, t_target2cam,      # Calibration target poses
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("Rotation (cam -> gripper):\n", R_cam2gripper)
print("Translation (cam -> gripper):\n", t_cam2gripper)
