import os.path

from Py_Class import Data_process, Camera_calibration


def main():
    data_process = Data_process()
    camera_calibration = Camera_calibration()
    data_process.download_and_unzipped_file()
    lin = data_process.read_directory()
    lin1 = data_process.read_directory(lin[0])
    path = os.path.join(lin[0], lin1[0])
    camera_angle_x, df_image = data_process.read_json(path, "transforms_train.json")
    print(camera_angle_x)
    for index, row in df_image.iterrows():
        file_path = row['Image_path']
        camera_calibration.add_image_calibration(file_path)


if __name__ == '__main__':
    main()
