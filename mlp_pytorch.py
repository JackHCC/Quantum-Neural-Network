#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum-Neural-Network 
@File    :mlp_pytorch.py
@Author  :JackHCC
@Date    :2023/1/2 12:18 
@Desc    :

'''
import torch
from torch import nn, optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import os
import time

from utils import cal_metrix_for_dir

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

IMG_SUFFIX = ["bmp", "jpg", "png", "raw", "jpeg"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_image(tensor, file_path):
    matrix = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
    matrix = matrix[0, 0, :, :]
    img = Image.fromarray(matrix)
    img.save(file_path)


class DataFactory(Dataset):
    def __init__(self, dataset_path):
        super(Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.all_file = os.listdir(self.dataset_path)
        self.img_file = [file for file in self.all_file if file.split(".")[-1] in IMG_SUFFIX]
        self.len = len(self.img_file)

    def __getitem__(self, index):
        img_name = self.img_file[index]
        img_path = self.dataset_path + img_name
        img = Image.open(img_path, mode="r")
        img_matrix = transforms.ToTensor()(img)
        return img_matrix, img_name, index

    def __len__(self):
        return self.len


class MLPCompress(nn.Module):
    def __init__(self, block_shape, scale):
        super(MLPCompress, self).__init__()
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
        assert len(X.shape) == 4  # (bs, c, w, h)
        bs = X.shape[0]
        XL = []
        for i in range(0, X.shape[2], self.block_shape[0]):
            for j in range(0, X.shape[3], self.block_shape[1]):
                res = self.model(torch.flatten(X[:, :, i:i + self.block_shape[0], j:j + self.block_shape[1]],
                                               start_dim=1))
                XL.append(res)

        X = torch.cat(XL, dim=1).view(bs, 1, X.shape[2], X.shape[3])  # .view(...)

        return X

    def __repr__(self):
        return "MLP-Compression"


if __name__ == "__main__":
    data_path = "./data/Set5/Set5_size_64/"
    save_model_path = "./model/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    block_size = 2
    batch_size = 1
    epochs = 500

    scale = 2

    train_dataset = DataFactory(data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = MLPCompress(block_shape=(block_size, block_size), scale=scale)
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=5e-2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    loss_func = nn.MSELoss(size_average=False)

    model.train()
    train_loss = 0
    best_loss = 1e20
    save_path = save_model_path + str(model) + ".pth"

    print("Begin Training ...")
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _, _) in enumerate(train_loader):
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
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

    print("Begin Predict ...")

    data_path = "./data/Set5/Set5_size_64/"

    pred_path = "./result/" + data_path.split("/")[-2] + "/"
    test_dataset = DataFactory(data_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.eval()

    with torch.no_grad():
        for _, (data, image_name, _) in enumerate(test_loader):
            img_name = image_name[0]
            data = Variable(data)
            data = data.to(DEVICE)
            final = model(data)

            save_image(final[:1].data, pred_path + img_name)

    mean_ssim = cal_metrix_for_dir(data_path, pred_path, compare_ssim)
    mean_psnr = cal_metrix_for_dir(data_path, pred_path, compare_psnr)
    mean_mse = cal_metrix_for_dir(data_path, pred_path, compare_mse)

    print("MSE: ", mean_mse)
    print("PSNR: ", mean_psnr)
    print("SSIM: ", mean_ssim)
    print("CR: ", scale)

    record_path = "./log/record_" + str(model) + ".txt"
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    line = str(now) + " -- " + str(model) + " -- " + str(
        scale) + " -- " + str(mean_mse) + " -- " + str(mean_psnr) + " -- " + str(
        mean_ssim) + "\n"

    # 指标写入文件
    with open(record_path, mode="a+", encoding="utf-8") as f:
        f.write(line)
