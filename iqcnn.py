from torch.functional import F
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import os
import time
import numpy as np

from utils import read_gray_img_as_matrix, read_raw_img_as_matrix

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

IMG_SUFFIX = ["bmp", "jpg", "png", "raw", "jpeg"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def block_divide(img, K):
    W, H = img.shape
    assert W % K == 0 and H % K == 0
    r, c = W // K, H // K
    P = np.zeros((r * c, K, K))

    for i in range(r):
        for j in range(c):
            P[i * c + j, :, :] = img[K * i: K * (i + 1), K * j: K * (j + 1)]
    return P


def block_recon(array, K):
    H, _, _ = array.shape
    m = np.sqrt(H)
    R = int(m)
    C = int(m)
    I = np.zeros((R * K, C * K))

    k = 0
    for i in range(R):
        for j in range(C):
            t = array[k, :, :]
            I[i * K: (i + 1) * K, j * K: (j + 1) * K] = t
            k += 1

    return I


def block_divide_3D(img, K):
    W, H, C = img.shape
    assert W % K == 0 and H % K == 0 and C == 3
    r, c = W // K, H // K
    P = np.zeros((r * c * C, K, K))

    for k in range(C):
        for i in range(r):
            for j in range(c):
                P[k * r * c + i * c + j, :, :] = img[K * i: K * (i + 1), K * j: K * (j + 1), k]
    return P


def block_recon_3D(array, K, CH=3):
    H, _, _ = array.shape
    H = H // CH
    m = np.sqrt(H)
    R = int(m)
    C = int(m)
    I = np.zeros((R * K, C * K, CH))

    n = 0
    for k in range(CH):
        for i in range(R):
            for j in range(C):
                t = array[n, :, :]
                I[i * K: (i + 1) * K, j * K: (j + 1) * K, k] = t
                n += 1
    return I


class BlockDataset(Dataset):
    def __init__(self, img_path, block_size, gray=True):
        super(Dataset, self).__init__()
        self.img_path = img_path
        self.gray = gray
        if gray:
            self.raw_img_matrix = read_gray_img_as_matrix(img_path)
            self.W, self.H = self.raw_img_matrix.shape
            self.sample_num = self.W * self.H // (block_size * block_size)
            self.data = block_divide(self.raw_img_matrix, block_size)
        else:
            self.raw_img_matrix = read_raw_img_as_matrix(img_path)
            self.W, self.H, self.C = self.raw_img_matrix.shape
            self.sample_num = self.W * self.H * self.C // (block_size * block_size)
            self.data = block_divide_3D(self.raw_img_matrix, block_size)

    def __getitem__(self, index):
        data_item = self.data[index, :, :] / 255
        data_tensor = torch.Tensor(data_item)
        return data_tensor, index

    def __len__(self):
        return self.sample_num


class QIConv2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=1,
            bias: bool = True
    ):
        super(QIConv2D, self).__init__()

        self.cnn_weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_weight = nn.Parameter(torch.rand(out_channels))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias

    def forward(self, inputs):
        cos_inputs = torch.cos(inputs)
        sin_inputs = torch.sin(inputs)
        cos_theta = torch.cos(self.cnn_weight)
        sin_theta = torch.sin(self.cnn_weight)
        real_bias = torch.cos(self.bias_weight)
        imag_bias = torch.sin(self.bias_weight)

        real = F.conv2d(cos_inputs, cos_theta, bias=real_bias, padding=self.padding, stride=self.stride) - \
               F.conv2d(sin_inputs, sin_theta, bias=real_bias, padding=self.padding, stride=self.stride)
        imag = F.conv2d(sin_inputs, cos_theta, bias=imag_bias, padding=self.padding, stride=self.stride) + \
               F.conv2d(cos_inputs, sin_theta, bias=imag_bias, padding=self.padding, stride=self.stride)

        y = (np.pi / 2) - torch.atan2(imag, real)

        return y

    def extra_repr(self) -> str:
        return '{}, {}, kernel_size=({},{}), stride=({},{})'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.kernel_size, self.stride, self.stride
        )


class MixQICNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale=2
    ):
        super(MixQICNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.qiconv_1 = QIConv2D(in_channels, 1, 3, scale, 1)

        # deconv 尺寸计算：https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/113772349
        self.deconv_1 = nn.ConvTranspose2d(1, 1, 3, 2, 1, 1)
        self.conv_2 = nn.Conv2d(1, out_channels, 3, 1, 1)

    def encoder(self, x):
        x = self.qiconv_1(x)
        return x

    def decoder(self, x):
        x = self.deconv_1(x)
        x = self.conv_2(x)
        return x

    def forward(self, inputs):
        com_img = self.encoder(inputs)
        rec_img = self.decoder(com_img)
        return rec_img

    def __repr__(self):
        return "MixQICNN"


class QICNN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            scale=2
    ):
        super(QICNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.qiconv_1 = QIConv2D(in_channels, 1, 3, scale, 1)

        # deconv 尺寸计算：https://blog.csdn.net/hhhhhhhhhhwwwwwwwwww/article/details/113772349
        self.deconv_1 = nn.ConvTranspose2d(1, 1, scale + 1, scale, 1, 1)
        self.conv_2 = QIConv2D(1, out_channels, 3, 1, 1)

    def encoder(self, x):
        x = self.qiconv_1(x)
        return x

    def decoder(self, x):
        x = self.deconv_1(x)
        x = self.conv_2(x)
        x = torch.sin(x) * torch.sin(x)
        return x

    def forward(self, inputs):
        com_img = self.encoder(inputs)
        rec_img = self.decoder(com_img)
        return rec_img

    def __repr__(self):
        return "QICNN"


if __name__ == "__main__":
    # param
    is_train = True
    is_eval = True
    gray = False

    img_size = 512
    img_name = "butterfly.bmp"
    # data_path = "./data/Set5/Set5_size_" + str(img_size) + "/" + img_name
    data_path = "./data/Set5/size_" + str(img_size) + "/" + img_name
    save_model_path = "./model/" + img_name.split(".")[0] + "/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    block_size = 8
    batch_size = 1
    epochs = 200
    scale = 4
    loss_threshold = 1e-5

    # model = MixQICNN(1, 1, scale)
    model = QICNN(1, 1, scale)
    model.to(DEVICE)

    save_path = save_model_path + str(model) + "_" + str(epochs) + "_" + str(img_size) + "_" + str(
        block_size) + "_" + str(scale) + ".pth"

    raw_img = read_gray_img_as_matrix(data_path) if gray else read_raw_img_as_matrix(data_path)

    if is_train:
        train_dataset = BlockDataset(data_path, block_size, gray)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        loss_func = nn.MSELoss(size_average=False)
        # loss_func = nn.MSELoss()

        model.train()
        train_loss = 0
        best_loss = 1e20

        print("Begin Training ...")
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = Variable(data)
                data = data.to(DEVICE)
                optimizer.zero_grad()
                final = model(data)
                # print("dubug:", data, data.shape, final, final.shape)
                loss = loss_func(final, data)
                loss.backward()
                train_loss += loss.data
                optimizer.step()
            scheduler.step()
            avg_loss = train_loss / len(train_loader.dataset)
            print('====> Epoch: {} Average loss: {:.16f}'.format(epoch, avg_loss))

            # Save the best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_path)
                if avg_loss <= loss_threshold:
                    break

    if is_eval:
        print("Begin Predict ...")

        pred_path = "./result/" + str(model) + "_" + str(epochs) + "_" + str(img_size) + "_" + str(
            block_size) + "_" + str(scale) + "/"
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        test_dataset = BlockDataset(data_path, block_size, gray)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        model.eval()

        rec_img = torch.zeros((len(test_dataset), block_size, block_size))

        idx = 0
        with torch.no_grad():
            for _, (data, _) in enumerate(test_loader):
                data = Variable(data)
                data = data.to(DEVICE)
                final = model(data)
                rec_img[idx, :, :] = final
                idx += 1

        rec_img = rec_img.mul(255).clamp(0, 255).byte().cpu().numpy()
        rec_img = block_recon(rec_img, block_size) if gray else block_recon_3D(rec_img, block_size)

        rec_img = rec_img.astype(np.uint8)

        img = Image.fromarray(rec_img)
        img.show()
        img.save(pred_path + img_name + "_rec.bmp")

        mean_mse = compare_mse(raw_img, rec_img)
        mean_psnr = compare_psnr(raw_img, rec_img)
        mean_ssim = compare_ssim(raw_img, rec_img, multichannel=not gray)

        print("MSE: ", mean_mse)
        print("PSNR: ", mean_psnr)
        print("SSIM: ", mean_ssim)
        print("CR: ", scale)

        record_path = "./log/record_" + str(model) + ".txt"
        now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        line = str(now) + " -- " + img_name + " -- " + str(model) + " -- " + str(img_size) + " -- " + str(
            block_size) + " -- " + str(scale) + " -- " + str(mean_mse) + " -- " + str(mean_psnr) + " -- " + str(
            mean_ssim) + "\n"

        # 指标写入文件
        with open(record_path, mode="a+", encoding="utf-8") as f:
            f.write(line)
