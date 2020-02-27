#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import keras
from keras import layers
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from utils import load_data, APPEARED_LETTERS,generate_pic,parse_answer,pic_folder,weight_folder
import preprocess

# 定义文件路径
def train(epochs = 100):
    # 数据预处理--the 1st core step
    letter_num = len(APPEARED_LETTERS)
    data, label = load_data(pic_folder) # 图片数据预处理
    data_train, data_test, label_train, label_test = \
        train_test_split(data, label, test_size=0.1, random_state=0)
    label_categories_train = to_categorical(label_train, letter_num) # one-hot编码
    label_categories_test = to_categorical(label_test, letter_num) # one-hot编码
    # x_train,y_train,x_test,y_test 是表示神经网络的输入输出的常用表示
    x_train, y_train, x_test, y_test = data_train, label_categories_train,data_test, label_categories_test
    # 定义神经网络结构--the 2nd core step
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
    # 编译模型--the 3rd core step
    model.compile(
        optimizer='adadelta',
        loss=['categorical_crossentropy'],
        metrics=['accuracy'],
    )
    # 定义训练中间结果存储
    check_point = ModelCheckpoint(
        os.path.join(weight_folder, '{epoch:02d}.hdf5'))

    # 训练神经网络--the 4th core step
    his = model.fit(
            x_train, y_train, batch_size=128, epochs=epochs,
            validation_split=0.1, callbacks=[check_point],
        )
    return his

def predict():
    # 基于模型的预测--the 5th core step
    model_path = 'model/11.hdf5'
    pic_path = 'samples/HCMYT.jpg'
    model = keras.models.load_model(model_path)
    data = np.empty((5, 40, 40, 3), dtype="uint8")
    raw_img = preprocess.load_img(pic_path)
    sub_imgs = preprocess.gen_sub_img(raw_img)
    for sub_index, img in enumerate(sub_imgs):
        data[sub_index, :, :, :] = img / 255
    out = model.predict(data)
    return out


if __name__ == '__main__':
    input('友情提示，在开始运行之前，最好确保项目所在路径为全英文路径！如果已经确认无误，输入任意键+Enter即开始执行程序')
    generate_pic(num=1000) # 如果是第一次运行，需要生成一些训练数据,使用这行代码即可
    his = train(epochs=100) # 如果需要训练数据，使用这行代码即可,可以指定训练的轮数
    ans = predict() # 预测的结果比较抽象，需要处理一下，调用parse_answer函数即可
    # 数据后处理--the 6th core step
    # print(his.history) # metrics结果，配合train函数使用
    print(parse_answer(ans)) # 预测结果
    # visualize tensorboard的使用方法在后面会讲到
