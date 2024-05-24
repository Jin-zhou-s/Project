import os.path
import urllib.request
import zipfile
import json
import numpy as np
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

    def read_json(self, path):
        json_path = os.path.join(self.image_save_path, path)
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
            file_paths.append(img['file_path'])

        return camera_angle_x, file_paths, rotation, transform_matrix

    def open_image(self, directory_path, image_path):
        imag = os.path.join(self.image_save_path, directory_path)
        cleaned_path = image_path.replace('./', '', 1)
        path = os.path.join(imag, cleaned_path + ".png")
        try:
            image = Image.open(path)
            image.show()
        except Exception as e:
            print(f"Error opening image: {e}")


class Camera_calibration:
    def __init__(self):
        self.chessboard_size = (9, 6)
        self.square_size = 1.0
        self.obj_points = []
        self.img_points = []
        self.camera_matrix = None
        self.dist_coffs = None

        self.objp = np.zeros((np.prod(self.chessboard_size), 3), np.float32)
        self.objp[:, :2] = np.indices(self.chessboard_size).T.reshape(-1, 2)
        self.objp *= self.square_size
