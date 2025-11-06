import numpy as np


def euler_to_pose_matrix(file_path):
    """
    从文本文件读取欧拉角数据并转换为位姿矩阵
    参数:
        file_path: 文本文件路径，每行包含X,Y,Z,RX,RY,RZ（角度单位：弧度）
    返回:
        R_list: 旋转矩阵列表，每个元素为3x3 numpy数组
        t_list: 平移向量列表，每个元素为3x1 numpy数组
    """
    R_list = []
    t_list = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            # 跳过空行
            if not line.strip():
                continue
            # 读取数据
            data = line.strip().split()
            if len(data) != 6:
                print(f"警告: 第{i + 1}行数据格式不正确，跳过该行")
                continue
            try:
                # 提取平移和欧拉角
                x, y, z, rx, ry, rz = map(float, data)
                # 创建平移向量 (3x1)
                t = np.array([[x], [y], [z]])
                # 计算旋转矩阵 (3x3)
                # 使用XYZ欧拉角顺序（也可以根据需要修改为其他顺序）
                R = euler_to_rotation_matrix(rx, ry, rz)
                # 添加到列表
                R_list.append(R)
                t_list.append(t)
            except ValueError:
                print(f"警告: 第{i + 1}行包含无效数字，跳过该行")

    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return [], []
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return [], []

    print(f"成功读取 {len(R_list)} 个位姿")
    return R_list, t_list


def euler_to_rotation_matrix(rx, ry, rz, order='XYZ'):
    """
    将欧拉角转换为旋转矩阵
    参数:
        rx, ry, rz: 绕X,Y,Z轴的旋转角度（弧度）
        order: 旋转顺序，默认为'XYZ'
    返回:
        R: 3x3旋转矩阵
    """
    # 计算每个轴的旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    # 根据指定的旋转顺序组合旋转矩阵
    if order == 'XYZ':
        R = Rz @ Ry @ Rx
    elif order == 'ZYX':
        R = Rx @ Ry @ Rz
    elif order == 'ZXY':
        R = Ry @ Rx @ Rz
    elif order == 'YZX':
        R = Rx @ Rz @ Ry
    elif order == 'YXZ':
        R = Rz @ Rx @ Ry
    elif order == 'XZY':
        R = Ry @ Rz @ Rx
    else:
        print(f"警告: 未知的旋转顺序 {order}，使用默认顺序 XYZ")
        R = Rz @ Ry @ Rx
    return R


def save_pose_matrices(R_list, t_list, output_file):
    """
    将旋转矩阵和平移向量保存到文件，用于验证
    参数:
        R_list: 旋转矩阵列表
        t_list: 平移向量列表
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w') as f:
            for i, (R, t) in enumerate(zip(R_list, t_list)):
                f.write(f"位姿 {i + 1}:\n")
                f.write("旋转矩阵 R:\n")
                for row in R:
                    f.write("  " + "  ".join(f"{val:10.6f}" for val in row) + "\n")
                f.write("平移向量 t:\n")
                f.write("  " + "  ".join(f"{val[0]:10.6f}" for val in t) + "\n")
                f.write("\n")
        print(f"位姿矩阵已保存到 {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == "__main__":

    R_list, t_list = euler_to_pose_matrix('test_poses.txt')

    # 打印结果
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        print(f"位姿 {i + 1}:")
        print("旋转矩阵 R:")
        print(R)
        print("平移向量 t:")
        print(t)
        print()

    # 可选：保存结果到文件
    # save_pose_matrices(R_list, t_list, 'pose_matrices.txt')