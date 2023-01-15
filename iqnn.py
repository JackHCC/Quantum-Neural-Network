#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum Holography 
@File    :iqnn.py
@Author  :JackHCC
@Date    :2022/10/29 23:08 
@Desc    :

'''

from torch.autograd import Function, gradcheck

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from PIL import Image
import os
import time
import numpy as np

from utils import read_gray_img_as_matrix
from mlp_block import block_recon, BlockDataset

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

IMG_SUFFIX = ["bmp", "jpg", "png", "raw", "jpeg"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IQMLP(Function):
    @staticmethod
    def forward(ctx, inputs, thetas, lambdas, deltas):
        # u = inputs.mm(thetas) + lambdas
        qinputs = IQMLP.quantumzied(inputs)
        qthetas = IQMLP.quantumzied(thetas)
        qlambdas = IQMLP.quantumzied(lambdas)

        u = torch.matmul(qinputs, qthetas) - qlambdas
        y = (np.pi / 2) * torch.sigmoid(deltas) - IQMLP.arg(u)

        ctx.save_for_backward(inputs, thetas, lambdas, deltas, u)

        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, thetas, lambdas, deltas, u = ctx.saved_tensors
        grad_inputs, grad_thetas, grad_lambdas, grad_deltas = [None] * 4

        du = -grad_outputs * IQMLP.arc_arg(u)

        du = du.t()
        u = u.t()
        tinputs = inputs.t()
        grad_thetas = torch.zeros(thetas.shape).t()

        for i, theta in enumerate(thetas.t()):
            real, imag = u[i].real, u[i].imag
            theta = theta.reshape(-1, 1)
            theta = theta + tinputs
            ans = du[i] * (torch.cos(theta) * real + torch.sin(theta) * imag) / real ** 2
            ans = torch.sum(ans, dim=1)
            grad_thetas[i] = ans

        grad_thetas = grad_thetas.t()

        lambdas = lambdas.reshape(-1, 1)
        grad_lambdas = du * (torch.cos(lambdas) * u.real + torch.sin(lambdas) * u.imag) / u.real ** 2
        grad_lambdas = torch.sum(grad_lambdas, dim=1)

        grad_deltas = grad_outputs * (np.pi / 2) * torch.sigmoid(deltas) * (1 - torch.sigmoid(deltas))

        return grad_inputs, grad_thetas, grad_lambdas, grad_deltas

    @staticmethod
    def arg(u):
        return torch.atan2(u.imag, u.real)

    @staticmethod
    def arc_arg(u):
        return 1 / (1 + torch.pow(u.imag / u.real, 2))

    @staticmethod
    def quantumzied(theta):
        return torch.complex(torch.cos(theta.clone()), torch.sin(theta.clone()))


class IQLinear(nn.Module):
    def __init__(self, input_gates, output_gates):
        super(IQLinear, self).__init__()

        self.theta = nn.Parameter(torch.Tensor(input_gates, output_gates))
        nn.init.uniform_(self.theta, -np.pi, np.pi)

        self.Lambdas = nn.Parameter(torch.Tensor(output_gates))
        nn.init.uniform_(self.Lambdas, -np.pi, np.pi)

        self.delta = nn.Parameter(torch.Tensor(output_gates))
        nn.init.uniform_(self.delta, -np.pi, np.pi)

    def forward(self, inputs):
        cos_inputs = torch.cos(inputs)
        sin_inputs = torch.sin(inputs)
        cos_theta = torch.cos(self.theta)
        sin_theta = torch.sin(self.theta)

        real = torch.matmul(cos_inputs, cos_theta) - \
               torch.matmul(sin_inputs, sin_theta) + torch.cos(self.Lambdas)

        imag = torch.matmul(sin_inputs, cos_theta) + \
               torch.matmul(cos_inputs, sin_theta) + torch.sin(self.Lambdas)

        y = (np.pi / 2) * torch.sigmoid(self.delta) - torch.atan2(imag, real)

        return y


class ComIQNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ComIQNN, self).__init__()
        output_size = input_size
        self.encoder = IQLinear(input_size, hidden_size)
        self.decoder = IQLinear(hidden_size, output_size)

    def forward(self, inputs):
        inputs = (np.pi / 2) * inputs
        comp_data = self.encoder(inputs)
        recon_data = self.decoder(comp_data)
        output = torch.sin(recon_data) * torch.sin(recon_data)
        return output

    def __repr__(self):
        return "ComIQNN"


if __name__ == "__main__":
    # param
    is_train = True
    is_eval = True

    data_path = "./data/One_Shot/pix256/butterfly.bmp"
    save_model_path = "./model/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    block_size = 8
    batch_size = 2
    epochs = 100
    scale = 2
    loss_threshold = 1e-5

    input_size = block_size * block_size
    hidden_size = input_size // scale

    model = ComIQNN(input_size, hidden_size)
    model.to(DEVICE)

    save_path = save_model_path + str(model) + ".pth"

    raw_img = read_gray_img_as_matrix(data_path)

    if is_train:
        train_dataset = BlockDataset(data_path, block_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        # loss_func = nn.MSELoss(size_average=False)
        loss_func = nn.MSELoss()

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
        pred_path = "./result/" + data_path.split("/")[-2] + "/"
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        test_dataset = BlockDataset(data_path, block_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

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
        rec_img = block_recon(rec_img, block_size)

        rec_img = rec_img.astype(np.uint8)

        img = Image.fromarray(rec_img)
        img.show()
        img.save(pred_path + str(model) + "_rec.bmp")

        mean_mse = compare_mse(raw_img, rec_img)
        mean_psnr = compare_psnr(raw_img, rec_img)
        mean_ssim = compare_ssim(raw_img, rec_img)

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
