import os.path
import torch.optim as optim

from Py_Class import Data_process, Camera_calibration, NeRFModel, train_nerf, load_data


def main():
    data_process = Data_process()
    camera_calibration = Camera_calibration()
    nerf_model = NeRFModel()
    data_process.download_and_unzipped_file()
    lin = data_process.read_directory()
    lin1 = data_process.read_directory(lin[0])
    path = os.path.join(lin[0], lin1[0])
    camera_angle_x, df_image = data_process.read_json(path, "transforms_train.json")
    print(camera_angle_x)
    image_array, poses_array = load_data(df_image)
    model = NeRFModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    train_nerf(image_array, poses_array, model, optimizer, num_epochs=10)


if __name__ == '__main__':
    main()
