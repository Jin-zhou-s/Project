import os.path

from Py_Class import Data_process


def main():
    data_process = Data_process()
    data_process.download_and_unzipped_file()
    lin = data_process.read_directory()
    lin1 = data_process.read_directory(lin[0])
    path = os.path.join(lin[0], lin1[0])
    data_process.read_directory(path)
    path1 = os.path.join(path, "transforms_train.json")
    camera_angle_x, file_paths, rotation, transform_matrix = data_process.read_json(path1)
    print(camera_angle_x)
    print(file_paths[0])
    print(rotation)
    data_process.open_image(path, file_paths[2])


if __name__ == '__main__':
    main()
