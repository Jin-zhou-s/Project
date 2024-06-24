import os.path
import urllib.request
import zipfile
import json
import numpy as np
import cv2
import glob
import pandas as pd
from PIL import Image


# The download callback function displays the download progress
def creat_reporthook(file_name):
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r{file_name} Downloading:{percent}%", end="")

    return reporthook


class Data_process:
    # Basic download address, save address, compressed package name
    def __init__(self):
        self.download_path = "https://drive.usercontent.google.com/download?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG&export=download&authuser=0&confirm=t&uuid=d5ff7265-8745-4ee9-a37a-1e79fd357271&at=APZUnTUhRt3uMykGrNWUQiZO3UmR:1716386844507"
        self.save_path = "Image_data"
        self.image_save_path = "Image_data/Image_dataset"
        self.zip_path = "nerf_synthetic"

    # Download the decompression function
    def download_and_unzipped_file(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.isdir(self.image_save_path):
            reporthook = creat_reporthook(self.zip_path)
            full_save_path = os.path.join(self.save_path, self.zip_path)
            urllib.request.urlretrieve(self.download_path, full_save_path, reporthook=reporthook)
            print("\ndownload finish")
            print("start unzipping")
            with zipfile.ZipFile(full_save_path, "r") as image_zip:
                image_zip.extractall(path=self.image_save_path)
            print("unzipping finish")
            os.remove(full_save_path)
        else:
            print("The dataset file is detected and the download is skipped")

    def read_directory(self, path=None):
        if not path:
            search_path = self.image_save_path
        else:
            search_path = os.path.join(self.image_save_path, path)

        items = os.listdir(search_path)
        folders = []
        for item in items:
            if os.path.isdir(os.path.join(search_path, item)):
                folders.append(item)
        print(folders)
        return folders

    def read_json(self, path, file_name):
        json_path = os.path.join(self.image_save_path, path, file_name)
        with open(json_path, 'r') as file:
            data = json.load(file)
        images = data['frames']
        camera_angle_x = data['camera_angle_x']
        file_paths = []
        rotation = []
        transform_matrix = []

        for img in images:
            transform = np.array(img['transform_matrix'], dtype=np.float32)
            transform_matrix.append(transform)
            rotation.append(img['rotation'])
            clean_path = img['file_path'].replace('./', '', 1)
            join_path = os.path.join(self.image_save_path, path, clean_path + ".png")
            image_path = join_path.replace('\\', '/')
            file_paths.append(image_path)

        df_image = pd.DataFrame({
            'Image_path': file_paths,
            'Image_rotation': rotation,
            'Image_transform': transform_matrix
        })

        return camera_angle_x, df_image


class Camera_calibration:
    def __init__(self):
        self.chessboard_size = (11, 8)
        self.square_size = 70
        self.obj_points = []
        self.img_points = []
        self.camera_matrix = None
        self.dist_coeffs = None

        self.objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        self.objp[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        self.objp *= self.square_size

    def draw_chessboard(self, image_path):
        # 获取图像尺寸
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        # 计算棋盘格的总尺寸
        board_width = self.chessboard_size[0] * self.square_size
        board_height = self.chessboard_size[1] * self.square_size

        # 确定棋盘格的左上角位置，使其居中显示
        start_x = (img_width - board_width) // 2
        start_y = (img_height - board_height) // 2
        # 创建棋盘格图案
        # 创建棋盘格图案
        chessboard = np.zeros_like(image)

        for i in range(self.chessboard_size[1]):
            for j in range(self.chessboard_size[0]):
                # 计算每个方格的顶点坐标
                top_left = (start_x + j * self.square_size, start_y + i * self.square_size)
                bottom_right = (start_x + (j + 1) * self.square_size, start_y + (i + 1) * self.square_size)
                # 绘制白色方格
                if (i + j) % 2 == 0:
                    cv2.rectangle(chessboard, top_left, bottom_right, (255, 255, 255), -1)

        # 将棋盘格图案叠加到原始图像上
        combined = cv2.addWeighted(image, 0.5, chessboard, 0.9, 0)
        return combined

    def add_image_calibration(self, image_path):
        image = self.draw_chessboard(image_path)
        if image is None:
            print(f"Failed to load image at {image_path}")
            return
        print(f"Loaded image at {image_path} with shape {image.shape}")
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(image_gray, (8, 11), None)
        print(ret)
        if ret:
            self.obj_points.append(self.objp.copy())
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
            self.img_points.append(corners2)
            img = cv2.drawChessboardCorners(image, self.chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        else:
            print(f"Chessboard corners not found in {image_path}")

