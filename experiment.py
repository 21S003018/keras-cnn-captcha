#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/2/25 11:30
@Author  : cbz
@Site    : https://github.com/1173710224/brain-computing/blob/cbz
@File    : analyse.py
@Software: PyCharm
@Descripe:
"""
from keras import layers
from keras.models import Model
from utils import load_data, APPEARED_LETTERS


def get_model():
    inputs = layers.Input((40, 40, 3)) # 定义输入层
    x = layers.Conv2D(32, 9, activation='relu')(inputs)
    x = layers.Conv2D(32, 9, activation='relu')(x)
    x = layers.MaxPool2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(640)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(len(APPEARED_LETTERS), activation='softmax')(x) # 定义输出层
    model = Model(inputs=inputs, outputs=out)
    return model
