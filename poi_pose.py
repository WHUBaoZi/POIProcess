import os
import sys
import cv2
import numpy as np
from typing import NamedTuple
import json
current_dir_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_dir_path)
work_dir = os.path.dirname(current_dir)
sys.path.append(work_dir)

from utils.graphics_utils import getWorld2View2, focal2fov
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, read_intrinsics_binary


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    c2w_l = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE" or intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # print(f'FovX: {FovX}, FovY: {FovY}')
        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_path = os.path.join(images_folder, extr.name)
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        image = None

        # concate R and T to get world2view matrix
        W2C = getWorld2View2(R, T)
        C2W = np.linalg.inv(W2C)
        c2w_l.append(C2W)

        # print(f'image: {image.size}')

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, c2w_l


def get_video_info(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return f"Error: file '{file_path}' doesn't exist"

    # 打开视频文件
    video = cv2.VideoCapture(file_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        return f"Error: Can not open video file '{file_path}'"

    # 获取帧率
    fps = video.get(cv2.CAP_PROP_FPS)

    # 获取总帧数
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算视频时长（秒）
    duration = total_frames / fps

    # 释放视频对象
    video.release()
    return fps, duration


def colmap_matrix_rotation(poses, angle_list, axis='y'):
    """
    COLMAP坐标系下的矩阵旋转

    参数:
    angle_degrees : float
        旋转角度（度）
    axis : str
        旋转轴，可选 'x', 'y', 或 'z'

    返回:
    numpy.ndarray
        nx4x4矩阵
    """
    if angle_list == None:
        angle_list = [90] * len(poses)

    Rs = []
    for angle_degrees in angle_list:
        angle_radians = np.radians(angle_degrees)
        c, s = np.cos(angle_radians), np.sin(angle_radians)

        if axis == 'x':
            R = np.array([[1, 0, 0],
                          [0, c, -s],
                          [0, s, c]])
        elif axis == 'y':
            R = np.array([[c, 0, s],
                          [0, 1, 0],
                          [-s, 0, c]])
        elif axis == 'z':
            R = np.array([[c, -s, 0],
                          [s, c, 0],
                          [0, 0, 1]])
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
        Rs.append(R[None])
    Rs_array = np.concatenate(Rs)

    poses[:, :3, :3] = np.einsum('nij,nkj->nki', Rs_array, poses[:, :3, :3])

    return poses


# read all the name of jpgs in the folder
def aligh_poses(source_path, poi_results, main_video_path, iphone_video_path, main_extract_rate=10):
    main_fps, main_duration = get_video_info(main_video_path)
    iphone_fps, iphone_duration = get_video_info(iphone_video_path)
    rate = main_fps / iphone_fps

    main_idx_list = []
    iphone_idx_list = []
    iphone_dir_list = []
    iphone_txt_list = []
    iphone_pic_list = []

    for iphone_name in poi_results:

        iphone_txt = iphone_name[0]
        iphone_dir =  iphone_name[1]
        iphone_idx = iphone_name[2]
        iphone_pic = iphone_name[3]
        main_idx = iphone_idx * rate / main_extract_rate

        main_idx_list.append(round(main_idx))
        iphone_idx_list.append(iphone_idx)
        iphone_dir_list.append(iphone_dir)
        iphone_txt_list.append(iphone_txt)
        iphone_pic_list.append(iphone_pic)

    cameras_extrinsic_file = os.path.join(source_path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(source_path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    cam_infos_unsorted, c2w_l = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(source_path, "images"))
    camera_poses = np.stack(c2w_l, axis=0)

    poi_info_dict = {}
    for id, main_idx in enumerate(main_idx_list):
        poi_info_dict[id] = {}
        poi_info_dict[id]["poi"] = iphone_txt_list[id]
        poi_info_dict[id]["pic"] = iphone_pic_list[id]


    if not os.path.exists(os.path.join(source_path, "poi_results")):
        os.makedirs(os.path.join(source_path, "poi_results"))

    json_string = json.dumps(poi_info_dict, indent=4)
    with open(os.path.join(source_path, "poi_results", "main_list.json"), "w") as f:
        f.write(json_string)

    # save all selected 3x3 translations in one file
    translations = camera_poses[:, :3, 3:]
    main_t = translations[main_idx_list]
    np.save(os.path.join(source_path, "poi_results", 'trans.npy'), main_t)

    # save all selected 4x4 poses (after turn left / right) in one file
    main_p = camera_poses[main_idx_list]
    rotated_main_p = colmap_matrix_rotation(main_p, angle_list=iphone_dir_list, axis='y')
    np.save(os.path.join(source_path, "poi_results", 'poses.npy'), rotated_main_p)

