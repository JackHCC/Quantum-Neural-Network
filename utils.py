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

