import torch
import torch.nn as nn
import numpy as np
from NoiseLayer import NoiseLayer
class MyModel(nn.Module):
    def __init__(self,channel=26):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(channel, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, channel)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = x.view(x.size(0), x.size(1), 1)
        return x

class ADMM_net(nn.Module):
    def __init__(self, channel=26, SNR=30, alpha=0):
        super(ADMM_net, self).__init__()
        self.mymodel = MyModel(channel)
        self.noisemodel = NoiseLayer(SNR=SNR,alpha=alpha)
        # merge all gamma into one
        self.gamma = torch.nn.Parameter(torch.Tensor([10]*12))
    def forward(self, imgs, Phi): #, Train=True):
        # if Train:
        #     self.noisemodel.quantization=False
        #     self.noisemodel.add_noise=False
        # else:
        #     self.noisemodel.quantization=True
        #     self.noisemodel.add_noise=True

        size = imgs.size()
        batch, Nx, _ = size

        Phi = Phi.float()
        PhiT = torch.transpose(Phi.float(), dim0=0, dim1=1)
        PhiT_Phi = torch.matmul(PhiT, Phi)

        clear_y = torch.matmul(Phi, imgs)
        y = self.noisemodel(clear_y)
        v = torch.matmul(PhiT,y)
        u = torch.zeros_like(v)
        I = torch.eye(PhiT_Phi.size()[0]).to(PhiT_Phi.device)
        for i in range(1,13):
            # ************   方法一求x  直接法求逆   ******************
            gamma = self.gamma[i-1]
            qiu_ni = torch.inverse(PhiT_Phi+(gamma*I))
            vplusu = v+u
            x = torch.matmul(qiu_ni, (torch.matmul(PhiT, y) + vplusu))
            v = self.mymodel(x - u)
            u = u - (x - v)
            # ************   方法二求 x 通过公式替代求逆   ***************
            # rm = torch.diag(torch.matmul(Phi, PhiT))
            # up = y - torch.matmul(Phi, ((v+u)/gamma))
            # down = (gamma+rm).reshape(9,1)
            # at = up / down
            # xt = (v+u)/gamma + torch.matmul(PhiT, at)
            # v = self.mymodel(xt - u)
            # u = u - (xt - v)
            # ****************以下为算保真率代码****************************
            # x_p = x.squeeze()
            # xt_p = xt.squeeze()
            # x_p = x_p.detach().cpu().numpy()
            # xt_p = xt_p.detach().cpu().numpy()
            # fidelity_pixel = np.dot(x_p, xt_p) / (np.linalg.norm(x_p) * np.linalg.norm(xt_p))
            # print("site: ", i, ". fidelity: ", fidelity_pixel)
        return v

class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.dconv_down1 = double_conv(in_ch, 32, 3, 1)
        self.dconv_down2 = double_conv(32, 64, 3, 1)
        self.dconv_down3 = double_conv(64, 128, 3, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up2 = double_conv(64 + 32, 64, 3, 1)
        self.dconv_up1 = double_conv(32 + 32, 32, 3, 1)

        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()

    def forward(self, x):
        inputs = x
        conv1 = self.dconv_down1(x)
        maxpool1 = self.maxpool(conv1)

        conv2 = self.dconv_down2(maxpool1)
        maxpool2 = self.maxpool(conv2)

        conv3 = self.dconv_down3(maxpool2)

        x = self.upsample2(conv3)
        x = torch.cat([x, maxpool1], dim=0)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=0)

        x = self.dconv_up1(x)

        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs

        return out


if __name__ == '__main__':
    import numpy as np
    def random_Phi(degree=9, Nx=512, Ny=512, Nlambda=26):
        mean = 0
        std_dev = 1
        random_array = np.random.normal(mean, std_dev, (degree, Nlambda))
        row_ind = []
        col_ind = []
        values = []

        # 外层循环：迭代26列
        for col_idx in range(Nlambda):
            # 内层循环：迭代64*64行
            column_values = random_array[:, col_idx]
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
        random_array = torch.Tensor(random_array)
        return sparse_matrix, random_array


    x = torch.rand(26,16, 16, dtype=torch.float32)
    sPhi,Phi = random_Phi(9,16,16,26)
    print(Phi.shape)
    model = ADMM_net(Phi,26, 30)
    out = model(x)
    print(out.shape)



    # model = MyModel(26)
    # input_sample = torch.rand((8, 26, 1))
    # output = model(input_sample)
    # print(model)
    # print("Output shape:", output.shape)