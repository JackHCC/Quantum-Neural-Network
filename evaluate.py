#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum-Neural-Network 
@File    :evaluate.py
@Author  :JackHCC
@Date    :2022/6/26 12:07 
@Desc    :

'''
import numpy as np
import math


def mse(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return mse


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


