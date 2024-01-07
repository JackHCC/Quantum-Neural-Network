#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum-Neural-Network 
@File    :utils.py
@Author  :JackHCC
@Date    :2022/6/20 16:02 
@Desc    :some tools functions

'''

import time
import numpy as np
import os
from PIL import Image


def f(input_data):
    output_data = np.cos(input_data) + 1j * np.sin(input_data)
    return output_data


def f_grad(input_data):
    output_data = -1 * np.sin(input_data) + 1j * np.cos(input_data)
    return output_data


def sigmoid(input_data):
    output_data = 1 / (1 + np.exp(-input_data))
    return output_data


def sigmoid_grad(input_data):
    output = sigmoid(input_data) * (1 - sigmoid(input_data))
    return output


def arg(input_data):
    output = np.angle(input_data)
    return output


def arg_grad(input_data):
    output = 1 / (1 + np.power((np.imag(input_data) / np.real(input_data)), 2))
    return output


def get_runtime(fn):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = fn(*args, **kwargs)
        end_time = time.time()
        print("程序运行时间: {} s".format(end_time - start_time))
        return results
    return wrapper


def read_gray_img(path):
    g = Image.open(path, mode="r")
    return g


def read_raw_img(path):
    g = Image.open(path)
    return g


def read_gray_img_as_matrix(path):
    g = Image.open(path, mode="r")
    G = np.array(g)
    return G


def read_raw_img_as_matrix(path):
    g = Image.open(path)
    G = np.array(g)
    return G


def cal_metrix_for_dir(ori_dir, pred_dir, method, img_type="metrix"):
    """
    ori_dir: 原图像目录
    pred_dir: 预测图像目录
    method: 指标计算函数
    """
    ori_dir_list = os.listdir(ori_dir)
    ori_dir_list = [file for file in ori_dir_list if file.split(".")[-1] in ["bmp", "jpg", "png", "raw", "jpeg"]]
    pred_dir_list = os.listdir(pred_dir)
    pred_dir_list = [file for file in pred_dir_list if file.split(".")[-1] in ["bmp", "jpg", "png", "raw", "jpeg"]]
    l = len(ori_dir_list)
    metrix = 0
    for ori_img, pred_img in zip(ori_dir_list, pred_dir_list):
        if img_type == "metrix":
            ori = read_gray_img_as_matrix(ori_dir + ori_img)
            pred = read_gray_img_as_matrix(pred_dir + pred_img)
        else:
            ori = read_gray_img(ori_dir + ori_img)
            pred = read_gray_img(pred_dir + pred_img)
        value = method(ori, pred)
        metrix += value
    return metrix / l



if __name__ == "__main__":
    path = "./data/Set5/size_64/butterfly.bmp"
    img = read_raw_img(path)
    img_arr = read_raw_img_as_matrix(path)
    print(img_arr.shape)
    img.show()