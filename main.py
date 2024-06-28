import os.path
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from Py_Class import Data_process, Camera_calibration, NeRFModel, train_nerf, load_data, save_model, load_model, \
    generate_image, save_image


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

    # plt.imshow(image_array[0])
    # plt.title('Loaded Image with PIL')
    # plt.axis('off')
    # plt.show()
    model = NeRFModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    train_nerf(image_array, poses_array, model, optimizer, num_epochs=10000, check_interval=1000)
    save_model(model)
    model = NeRFModel()
    model = load_model(model)

    # Use the model to generate a picture
    pose = torch.tensor(poses_array[0], dtype=torch.float32).to(next(model.parameters()).device)  # Use the first camera pose
    image = generate_image(model, pose)
    save_image(image, "output.png")
    print("Image saved to output.png")


if __name__ == '__main__':
    main()
