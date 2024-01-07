#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum-Neural-Network 
@File    :mlp_block.py
@Author  :JackHCC
@Date    :2023/1/15 20:42 
@Desc    :

'''
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
    P = np.zeros((K * K, r * c))

    for i in range(r):
        for j in range(c):
            P[:, i * c + j] = img[K * i: K * (i + 1), K * j: K * (j + 1)].reshape((K * K,))
    return P


def block_recon(array, K):
    W, H = array.shape
    m = np.sqrt(H)
    R = int(m)
    C = int(m)
    I = np.zeros((R * K, C * K))

    k = 0
    for i in range(R):
        for j in range(C):
            t = array[:, k].reshape((K, K))
            I[i * K: (i + 1) * K, j * K: (j + 1) * K] = t
            k += 1

    return I


def block_divide_3D(img, K):
    W, H, C = img.shape
    assert W % K == 0 and H % K == 0 and C == 3
    r, c = W // K, H // K
    P = np.zeros((K * K, r * c * C))

    for k in range(C):
        for i in range(r):
            for j in range(c):
                P[:, k * r * c + i * c + j] = img[K * i: K * (i + 1), K * j: K * (j + 1), k].reshape((K * K,))
    return P


def block_recon_3D(array, K, CH=3):
    W, H = array.shape
    H = H // CH
    m = np.sqrt(H)
    R = int(m)
    C = int(m)
    I = np.zeros((R * K, C * K, CH))

    n = 0
    for k in range(CH):
        for i in range(R):
            for j in range(C):
                t = array[:, n].reshape((K, K))
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
        data_item = self.data[:, index] / 255
        data_tensor = torch.Tensor(data_item)
        return data_tensor, index

    def __len__(self):
        return self.sample_num


class MLP(nn.Module):
    def __init__(self, block_shape, scale):
        super(MLP, self).__init__()
        assert len(block_shape) == 2
        self.block_shape = block_shape
        self.scale = scale

        self.in_unit = block_shape[0] * block_shape[1]
        self.hidden_unit = self.in_unit // scale
        self.out_unit = self.in_unit

        self.model = nn.Sequential(
            nn.Linear(self.in_unit, self.hidden_unit),
            nn.ReLU(inplace=True),
            # nn.Linear(self.hidden_unit, self.hidden_unit),
            # nn.ReLU(inplace=True),
            nn.Linear(self.hidden_unit, self.out_unit),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        Y = self.model(X)
        return Y

    def __repr__(self):
        return "MLP"


if __name__ == "__main__":
    # param
    is_train = True
    is_eval = True
    gray = False

    img_size = 512
    img_name = "butterfly.bmp"
    # data_path = "./data/One_Shot/pix" + str(img_size) + "/" + img_name
    # data_path = "./data/Set5/Set5_size_" + str(img_size) + "/" + img_name
    data_path = "./data/Set5/size_" + str(img_size) + "/" + img_name
    save_model_path = "./model/" + img_name.split(".")[0] + "/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    block_size = 8
    batch_size = 4
    epochs = 200
    scale = 2
    loss_threshold = 1e-5

    model = MLP(block_shape=(block_size, block_size), scale=scale)
    model.to(DEVICE)

    save_path = save_model_path + str(model) + "_" + str(epochs) + "_" + str(img_size) + "_" + str(
        block_size) + "_" + str(scale) + ".pth"

    raw_img = read_gray_img_as_matrix(data_path) if gray else read_raw_img_as_matrix(data_path)

    if is_train:
        train_dataset = BlockDataset(data_path, block_size, gray)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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

        model.load_state_dict(torch.load(save_path))
        model.eval()

        rec_img = torch.zeros((block_size * block_size, len(test_dataset)))

        idx = 0
        with torch.no_grad():
            for _, (data, _) in enumerate(test_loader):
                data = Variable(data)
                data = data.to(DEVICE)
                final = model(data)
                rec_img[:, idx] = final
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
