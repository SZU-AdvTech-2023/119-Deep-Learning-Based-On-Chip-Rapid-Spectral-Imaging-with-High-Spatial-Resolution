import json
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.io import loadmat
from tqdm import trange

from ADMM_net import ADMM_net
from utils import img_resolve as ir
metadata = {'description': ' Wayho Tech ', 'wavelength units': 'Nanometers',
            'band names': '{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26}',
            'wavelength': '{450.00,460.00,470.00,480.00,490.00,500.00,510.00,520.00,530.00,540.00,550.00,'
                          '560.00,570.00,580.00,590.00,600.00,610.00,620.00,630.00,640.00,650.00,'
                          '660.00,670.00,680.00,690.00,700.00}',
            'reflectance scale factor': 1.000000}
def Config(Path = None):
    with open(Path, 'r') as f:
        config = json.load(f)
    return config
def run(real_path, out_path):
    path = os.path.join(real_path, 'valid.mat')
    real_imgs = loadmat(path)['data']

    config_path = os.path.join(out_path, "config", "config.json")
    config = Config(config_path)
    SNR = config["train"]["SNR"]
    alpha = config["train"]["alpha"]

    Phi_path = os.path.join(out_path, "Phi", "Phi.pt")
    Phi = torch.load(Phi_path)
    model = ADMM_net( 26, SNR, alpha)
    model_path_dir = os.path.join(out_path, "models")
    files_list = os.listdir(model_path_dir)
    model_path = os.path.join(model_path_dir, files_list[-1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_data = real_imgs.shape[0]
    random_indices = random.sample(range(total_data), 16)

    # 设置一个3x3的子图网格
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    count_f = 0
    count_r = 0
    with torch.no_grad():
        for i, ax in zip(random_indices, axs.flatten()):
            per_img = real_imgs[i, :]
            per_img_tensor = torch.from_numpy(per_img)
            per_img_tensor = per_img_tensor.unsqueeze(-1)
            per_img_tensor = per_img_tensor.unsqueeze(0)
            output = model(per_img_tensor, Phi)
            output = output.numpy()

            real_pixel = per_img.squeeze()
            output_pixel = output.squeeze()

            # 计算保真率（fidelity）
            fidelity_pixel = np.dot(real_pixel, output_pixel) / (
                        np.linalg.norm(real_pixel) * np.linalg.norm(output_pixel))
            print("site: ", i, ". fidelity: ", fidelity_pixel)

            # 计算均方根误差（RMSE）
            rmse = np.sqrt(np.mean((real_pixel - output_pixel) ** 2))
            print("site: ", i, ". RMSE: ", rmse)

            # 在同一子图上绘制实际像素和输出像素
            ax.plot(real_pixel, label='Real Pixel')
            ax.plot(output_pixel, label='Output Pixel')

            # 添加标签和子图标题
            ax.set_xlabel('Index')
            ax.set_ylabel('Pixel Value')
            ax.set_title(f'Site: {i}, Fidelity: {fidelity_pixel:.4f}, RMSE: {rmse:.4f}')

            # 添加子图图例
            ax.legend()
            count_f = count_f + fidelity_pixel
            count_r = count_r + rmse

    # 调整布局以获得更好的间距
    plt.tight_layout()
    # 显示整个图形
    plt.show()
    print(f"在SNR为{SNR}, alpha为{alpha}情况下：")
    print(f"平均保真率为：{count_f/16}")
    print(f"平均RMSE为：{count_r / 16}")

def run1(img_path, out_path):
    img = ir.read_img(img_path)[:,:,:]
    Phi_path = os.path.join(out_path, "Phi", "Phi.pt")
    config_path = os.path.join(out_path, "config", "config.json")
    config = Config(config_path)
    SNR = config["train"]["SNR"]
    alpha = config["train"]["alpha"]
    Phi = torch.load(Phi_path)
    model = ADMM_net(26, SNR, alpha)
    model_path_dir = os.path.join(out_path, "models")
    files_list = os.listdir(model_path_dir)
    model_path = os.path.join(model_path_dir, files_list[-1])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    output = np.zeros_like(img)
    with torch.no_grad():
        for i in trange(img.shape[0]):
            # for j in range(img.shape[1]):
            per_img = img[i,:,:]
            per_img_tensor = torch.from_numpy(per_img)
            per_img_tensor = per_img_tensor.unsqueeze(-1)
            per_output = model(per_img_tensor, Phi)
            per_output = per_output.numpy()
            output[i, :, :] = np.squeeze(per_output)
            # print(f"out{i},{j}")
    save_path = os.path.join(out_path, "out_img_60")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_path = os.path.join(save_path, "out_img_30")
    ir.save_img(output,metadata,save_img_path,dtype=np.float32)
    save_pri_path = os.path.join(save_path, "pri_img")
    ir.save_img(img, metadata, save_pri_path, dtype=np.float32)



if __name__ == '__main__':
    real_path = r"D:\learn\condata-init\dataset"
    out_path = r"D:\learn\condata-init\run\20240110_224555"
    real_img = r"D:\learn\condata-init\dataset_512\train_data\scene01_reflectance.img"
    run(real_path,out_path)
    # run1(real_img,out_path)