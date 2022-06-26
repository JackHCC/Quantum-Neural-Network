#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum-Neural-Network 
@File    :heuristic_quantum_neural_network.py
@Author  :JackHCC
@Date    :2022/6/20 10:54 
@Desc    :Ab initio implementation of classical neural networks and quantum heuristic compressed neural networks

'''

import cv2
import numpy as np
from tqdm import tqdm
from utils import get_runtime, f, f_grad, arg, arg_grad, sigmoid, sigmoid_grad
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


np.random.seed(42)


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


def read_img(path):
    raw_img = cv2.imread(path)
    if len(raw_img.shape) > 2:
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2YCrCb)
        raw_img = raw_img[:, :, 0]
    return raw_img


# 经典压缩神经网络从头实现
class ClassicalMLPCompress:
    def __init__(self, image, init_obj=None, K=4, hidden_num=8, epochs=500, threshold=0.0005, lr=0.5):
        self.image = image
        self.init_obj = init_obj
        self.K = K
        self.hidden_num = hidden_num
        self.epochs = epochs
        self.threshold = threshold
        self.lr = lr
        self.train_x, self.train_y, self.test_x, self.test_y = self.build_dataset()

        self.B1, self.B3, self.sample_size = None, None, None

        # parameter
        self.W_21 = None
        self.W_32 = None
        self.b_2 = None
        self.b_3 = None
        self.E = []  # 迭代的loss记录
        self.iter_num = epochs

        self.output = None

        self.init_param()
        self.train()
        self.inference()

    def build_dataset(self):
        train_x = block_divide(self.image, self.K) / 255
        train_y = train_x
        test_x = train_x
        test_y = train_x
        return train_x, train_y, test_x, test_y

    def init_param(self):
        if not self.init_obj:
            self.B1, self.sample_size = self.train_x.shape
            self.B3, _ = self.train_y.shape

            self.W_21 = np.random.rand(self.hidden_num, self.B1)
            self.b_2 = np.random.rand(self.hidden_num, 1)
            self.W_32 = np.random.rand(self.B3, self.hidden_num)
            self.b_3 = np.random.rand(self.B3, 1)
        else:
            self.W_21 = self.init_obj.W_21
            self.b_2 = self.init_obj.b_2
            self.W_32 = self.init_obj.W_32
            self.b_3 = self.init_obj.b_3

    @get_runtime
    def train(self):
        for i in tqdm(range(self.epochs)):
            iter_error = 0
            for j in range(self.sample_size):
                input_data = self.train_x[:, j].reshape(self.B1, 1)
                output_data = self.train_y[:, j].reshape(self.B1, 1)
                # FP
                z2 = self.W_21 @ input_data + self.b_2

                a2 = 1 / (1 + np.exp(-z2))

                z3 = self.W_32 @ a2 + self.b_3
                a3 = 1 / (1 + np.exp(-z3))

                # BP
                e = (a3 - output_data)
                delta = (a3 * (1 - a3)) * e
                e = self.W_32.T @ delta
                self.W_32 = self.W_32 - self.lr * delta @ a2.T
                self.b_3 = self.b_3 - self.lr * delta

                delta = (a2 * (1 - a2)) * e
                self.W_21 = self.W_21 - self.lr * delta @ input_data.T
                self.b_2 = self.b_2 - self.lr * delta

                # error
                iter_error = iter_error + 0.5 * np.sum(np.power(a3 - output_data, 2))

            iter_error = iter_error / (self.sample_size * self.B3)
            self.E.append(iter_error)

            if iter_error < self.threshold:
                print("[BREAK] 迭代次数：", i, " error:", iter_error)
                self.iter_num = i
                break

    def inference(self):
        self.output = np.zeros(self.train_x.shape)
        for i in range(self.sample_size):
            input_data = self.train_x[:, i].reshape(self.B1, 1)

            z2 = self.W_21 @ input_data + self.b_2
            a2 = 1 / (1 + np.exp(-z2))

            z3 = self.W_32 @ a2 + self.b_3
            a3 = 1 / (1 + np.exp(-z3))

            self.output[:, i] = np.squeeze(a3)
        self.output = (block_recon(self.output, self.K) * 255).astype('uint8')


# 量子启发式压缩神经网络从头实现
class HeuristicQuantumMLPCompress:
    def __init__(self, image, init_obj=None, K=4, hidden_num=4, epochs=500, threshold=0.0005, lr=0.5, init_num=0.2):
        self.image = image
        self.init_obj = init_obj
        self.K = K
        self.hidden_num = hidden_num
        self.epochs = epochs
        self.threshold = threshold
        self.lr = lr
        self.init_num = init_num
        self.train_x, self.train_y, self.test_x, self.test_y = self.build_dataset()

        self.B1, self.B3, self.sample_size = None, None, None

        # parameter
        self.theta_kl = None
        self.lambda_k = None
        self.delta_k = None

        self.theta_nk = None
        self.lambda_n = None
        self.delta_n = None

        self.E = []  # 迭代的loss记录
        self.iter_num = epochs

        self.output = None

        self.init_param()
        self.train()
        self.inference()

    def build_dataset(self):
        train_x = block_divide(self.image, self.K) / 255
        train_y = train_x
        test_x = train_x
        test_y = train_x
        return train_x, train_y, test_x, test_y

    def init_param(self):
        if self.init_obj:
            self.theta_kl = self.init_obj.theta1
            self.lambda_k = self.init_obj.lambda1
            self.delta_k = self.init_obj.delta1

            self.theta_nk = self.init_obj.theta2
            self.lambda_n = self.init_obj.lambda2
            self.delta_n = self.init_obj.delta2
        else:
            self.B1, self.sample_size = self.train_x.shape
            self.B3, _ = self.train_y.shape

            self.theta_kl = np.random.rand(self.hidden_num, self.B1) * self.init_num
            self.lambda_k = np.random.rand(self.hidden_num, 1) * self.init_num
            self.delta_k = np.random.rand(self.hidden_num, 1)

            self.theta_nk = np.random.rand(self.B3, self.hidden_num) * self.init_num
            self.lambda_n = np.random.rand(self.B3, 1) * self.init_num
            self.delta_n = np.random.rand(self.B3, 1)

    @get_runtime
    def train(self):
        for i in tqdm(range(self.epochs)):
            iter_error = 0
            for j in range(self.sample_size):
                input_data = self.train_x[:, j].reshape(self.B1, 1)
                output_data = self.train_y[:, j].reshape(self.B1, 1)
                # FP
                y_l = input_data * np.pi / 2
                IO = f(y_l)

                u_k = f(self.theta_kl) @ IO - f(self.lambda_k)
                y_k = (np.pi / 2) * sigmoid(self.delta_k) - arg(u_k)
                HO = f(y_k)

                u_n = f(self.theta_nk) @ HO - f(self.lambda_n)
                y_n = (np.pi / 2) * sigmoid(self.delta_n) - arg(u_n)
                OP = f(y_n)

                output = np.imag(OP) * np.imag(OP)

                # BP
                # hidden - output layer
                d_n = -1 * (output_data - output) * np.sin(2 * y_n) * arg_grad(u_n)

                rep_y_k = np.repeat(y_k.T, self.B3, axis=0)
                rep_u_n_real = np.repeat(np.real(u_n), self.hidden_num, axis=1)
                rep_u_n_image = np.repeat(np.imag(u_n), self.hidden_num, axis=1)
                rep_d_n = np.repeat(d_n, self.hidden_num, axis=1)
                m_n = (np.cos(self.theta_nk + rep_y_k) * rep_u_n_real + np.sin(
                    self.theta_nk + rep_y_k) * rep_u_n_image) / np.power(rep_u_n_real, 2)

                s_n = (np.cos(self.lambda_n) * np.real(u_n) + np.sin(self.lambda_n) * np.imag(u_n)) / np.power(
                    np.real(u_n), 2)

                # E 对 delta_n 求梯度
                delta_delta2 = - (np.pi / 2) * (output_data - output) * np.sin(2 * y_n) * sigmoid_grad(self.delta_n)
                # 对 theta_n 求梯度
                delta_theta2 = -1 * rep_d_n * m_n
                # 对 lambda_n 求梯度
                delta_lambda2 = d_n * s_n

                # hidden layer - input layer
                e_k = np.sum(-1 * rep_d_n * m_n, axis=0).T.reshape(self.hidden_num, 1)

                d_k = e_k * arg_grad(u_k)

                rep_y_l = np.repeat(y_l.T, self.hidden_num, axis=0)
                rep_u_k_real = np.repeat(np.real(u_k), self.B1, axis=1)
                rep_u_k_image = np.repeat(np.imag(u_k), self.B1, axis=1)
                rep_d_k = np.repeat(d_k, self.B1, axis=1)

                m_k = (np.cos(self.theta_kl + rep_y_l) * rep_u_k_real + np.sin(
                    self.theta_kl + rep_y_l) * rep_u_k_image) / np.power(rep_u_k_real, 2)
                s_k = (np.cos(self.lambda_k) * np.real(u_k) + np.sin(self.lambda_k) * np.imag(u_k)) / np.power(
                    np.real(u_k), 2)

                # 对 delta_k 求梯度
                delta_delta1 = (np.pi / 2) * e_k * sigmoid_grad(self.delta_k)
                # 对 theta_k 求梯度
                delta_theta1 = -1 * rep_d_k * m_k
                # 对 lambda_k 求梯度
                delta_lambda1 = d_k * s_k

                # update parameter
                self.delta_n = self.delta_n - self.lr * delta_delta2
                self.theta_nk = self.theta_nk - self.lr * delta_theta2
                self.lambda_n = self.lambda_n - self.lr * delta_lambda2

                self.delta_k = self.delta_k - self.lr * delta_delta1
                self.theta_kl = self.theta_kl - self.lr * delta_theta1
                self.lambda_k = self.lambda_k - self.lr * delta_lambda1

                # all output error
                iter_error = iter_error + 0.5 * np.sum(np.power(output_data - output, 2))

            iter_error = iter_error / (self.sample_size * self.B3)
            self.E.append(iter_error)

            if iter_error < self.threshold:
                print("[BREAK] 迭代次数：", i, " error:", iter_error)
                self.iter_num = i
                break

    def inference(self):
        self.output = np.zeros(self.train_x.shape)
        for i in range(self.sample_size):
            input_data = self.train_x[:, i].reshape(self.B1, 1)

            y_l = input_data * np.pi / 2
            IO = f(y_l)

            u_k = f(self.theta_kl) @ IO - f(self.lambda_k)
            y_k = (np.pi / 2) * sigmoid(self.delta_k) - arg(u_k)
            HO = f(y_k)

            u_n = f(self.theta_nk) @ HO - f(self.lambda_n)
            y_n = (np.pi / 2) * sigmoid(self.delta_n) - arg(u_n)
            OP = f(y_n)

            Yi = np.imag(OP) * np.imag(OP)
            self.output[:, i] = np.squeeze(Yi)

        self.output = (block_recon(self.output, self.K) * 255).astype('uint8')


if __name__ == "__main__":
    # img_path = "./data/Set5/GTmod12/butterfly.png"
    img_path = "./data/Set5/GTmod12/bird.png"
    # img_path = "./data/lena.bmp"
    raw_img = read_img(img_path)
    H, W = raw_img.shape
    cv2.imwrite("./result/gray_bird.png", raw_img)

    # print(raw_img)
    print(raw_img.shape)

    K = 8
    hide_num = 16
    block_img = block_divide(raw_img, K)
    print(block_img.shape)

    img = block_recon(block_img, K)
    print(img.shape)

    # MLP = ClassicalMLPCompress(raw_img, K=K, hidden_num=hide_num)
    # out_img = MLP.output

    QuantumMLP = HeuristicQuantumMLPCompress(raw_img, K=K, hidden_num=hide_num)
    out_img = QuantumMLP.output

    p = compare_psnr(raw_img, out_img)
    # 对于多通道图像(RGB、HSV等)关键词multichannel要设置为True
    s = compare_ssim(raw_img, out_img, multichannel=False)
    m = compare_mse(raw_img, out_img)
    cr = H * W / (H * W * hide_num / (K * K) + hide_num * K * K)

    print('CR：{}，PSNR：{}，SSIM：{}，MSE：{}'.format(cr, p, s, m))

    cv2.imwrite("./result/bird-" + str(K) + "-" + str(hide_num) + ".png", out_img)

    cv2.namedWindow("Output", cv2.WINDOW_FREERATIO)
    cv2.imshow("Output", out_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
