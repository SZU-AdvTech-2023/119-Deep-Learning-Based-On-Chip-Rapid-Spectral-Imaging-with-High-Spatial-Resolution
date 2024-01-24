import json
import os
import argparse
import numpy as np
import torch
from scipy.io import loadmat
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ADMM_net import ADMM_net
from dataset import MyData
from downsample import *


def load_Phi(degree=9, Nx=512, Ny=512, Nlambda=26):
    '''
    加载Phi矩阵
    :param degree: 降采样因子
    :param Nx: 图像宽度
    :param Ny: 图像高度
    :param Nlambda: 波段数
    :return: (sparse_matrix, random_array)
    '''
    mat_file_path = './Phi.mat'
    mat_data = loadmat(mat_file_path)
    data = mat_data['data']
    row_ind = []
    col_ind = []
    values = []
    # 外层循环：迭代26列
    for col_idx in range(Nlambda):
        # 内层循环：迭代64*64行
        column_values = data[:, col_idx]
        for row_idx in range(Nx * Ny):
            row_ind.append(row_idx)
            col_ind.append(row_idx + col_idx * Nx * Ny)
            values.append(column_values[row_idx % degree])

    # 将索引转换为 int64 数据类型
    indices = torch.tensor([row_ind, col_ind], dtype=torch.int64)
    sparse_matrix = torch.sparse_coo_tensor(
        indices,
        torch.FloatTensor(values),
        torch.Size([Nx * Ny, Nx * Ny * Nlambda])
    )
    random_array = torch.Tensor(data)
    return sparse_matrix, random_array


def Config(Read = True, time = None, Path = None):
    with open("config.json", 'r') as f:
        config = json.load(f)
    if Read:
        return config
    else:
        config["timestamp"] = time
        config_path = os.path.join(Path, "config.json")
        with open(config_path, "w") as out:
            json.dump(config, out, indent=2)

def run(data_path):

    config = Config()
    batch_size = config["train"]["batch_size"]
    learning_rate = config["train"]["learning_rate"]
    epoch = config["train"]["epoch"]
    in_channel = config["train"]["in_channel"]
    out_channel = config["train"]["out_channel"]
    degree = config["train"]["degree"]
    img_width = config["train"]["img_width"]
    SNR = config["train"]["SNR"]
    alpha = config["train"]["alpha"]
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    config_path = f"./run/{current_time}/config"
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    Config(Read=False, time=current_time, Path=config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MyData(data_path=data_path, Train=True)
    valid_dataset = MyData(data_path=data_path, Train=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                            shuffle=True,num_workers=0,drop_last=False)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                            shuffle=True,num_workers=0,drop_last=False)

    # 获取Phi
    # sPhi, Phi = random_Phi(degree=degree, Nx=img_width,Ny=img_width,Nlambda=in_channel)
    # sPhi, Phi = load_Phi(degree=degree, Nx=img_width, Ny=img_width, Nlambda=in_channel)
    mat_file_path = './Phi.mat'
    mat_data = loadmat(mat_file_path)
    Phi = mat_data['data']
    Phi = torch.from_numpy(Phi)

    Phi_path = f"./run/{current_time}/Phi"
    if not os.path.exists(Phi_path):
        os.makedirs(Phi_path)
    Phi_path = Phi_path + "/Phi.pt"
    torch.save(Phi, Phi_path)
    Phi = Phi.to(device)

    model = ADMM_net(in_channel, SNR, alpha).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # 初始学习率为0.001
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    total_train_step = 0
    total_valid_step = 0
    log_path = f"./run/{current_time}/logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)



    min_total_loss = 999
    for i in range(epoch):
        print(f"第{i}轮训练：")
        model.train()
        for data in train_dataloader:
            per_datas = data.unsqueeze(-1) # (batch,Nx, 1)
            per_datas = per_datas.to(device)
            outputs = model(per_datas, Phi)
            loss = torch.sqrt(criterion(outputs, per_datas))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_step = total_train_step + 1
            if(total_train_step % 100 == 0):
                print("训练次数：{},Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        # 测试
        total_valid_loss = 0
        model.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                per_datas = data.unsqueeze(-1)  # (batch,Nx, 1)
                per_datas = per_datas.to(device)
                outputs = model(per_datas, Phi)
                loss = torch.sqrt(criterion(outputs, per_datas))
                total_valid_loss = total_valid_loss + loss.item()
        print("整体测试集上的Loss：{}".format(total_valid_loss))
        writer.add_scalar("valid_loss",total_valid_loss, total_valid_step)
        total_valid_step = total_valid_step + 1
        print(min_total_loss, total_valid_loss)
        if (i % 25 == 0 and i != 0) or i==epoch-1:
            min_total_loss = total_valid_loss
            model_path = f"./run/{current_time}/models"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = model_path + f"/ADMM_{i}_{min_total_loss}.pth"
            torch.save(model.state_dict(), model_path)
            print("模型已保存")


# def main():
#     parser = argparse.ArgumentParser(description="ADMM网络训练脚本")
#     parser.add_argument("--data_path", type=str, required=True, default=r"D:\learn\condata\dataset_64",help="数据集路径")
#
#     args = parser.parse_args()
#     run(data_path=args.data_path)
#
# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    data_path = r"D:\learn\condata-init\dataset"
    run(data_path)
