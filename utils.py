import os
import cv2
import numpy as np
import preprocess
from generator import Captcha
from keras import layers
from keras.models import Model

APPEARED_LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]
CAT2CHR = dict(zip(range(len(APPEARED_LETTERS)), APPEARED_LETTERS))
CHR2CAT = dict(zip(APPEARED_LETTERS, range(len(APPEARED_LETTERS))))
pic_folder = r'data/'
weight_folder = r'model/'

def distinct_char(folder):
    chars = set()
    for fn in os.listdir(folder):
        if fn.endswith('.jpg'):
            for letter in fn.split('.')[0]:
                chars.add(letter)
    return sorted(list(chars))


def load_data(folder):
    img_list = [i for i in os.listdir(folder) if i.endswith('jpg')]
    letters_num = len(img_list) * 5
    print('total letters:', letters_num)
    data = np.empty((letters_num, 40, 40, 3), dtype="uint8")  # channel last
    label = np.empty((letters_num,))
    for index, img_name in enumerate(img_list):
        raw_img = preprocess.load_img(os.path.join(folder, img_name))
        sub_imgs = preprocess.gen_sub_img(raw_img)
        for sub_index, img in enumerate(sub_imgs):
            data[index*5+sub_index, :, :, :] = img / 255
            label[index*5+sub_index] = CHR2CAT[img_name[sub_index]]
        if index % 100 == 0:
            print('{} letters loads'.format(index*5))
    return data, label

def generate_pic(num):
    '''
    图片生成结果保存在data文件夹下
    :return:
    '''
    ans = input('本次运行将会新产生1000组随机样例\n确认请输入1，不生成请输入0')
    if int(ans) == 0:
        return
    letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z']
    c = Captcha(150, 40, letters, lc=5,folder='data/',fs=['FONT_ITALIC'], debug=False)
    c.batch_create_img(num)
    return

def parse_answer(lis):
    ans = ''
    for tmp in lis:
        mmax = -100000000000
        char = None
        i = 0
        for value in tmp:
            if value > mmax:
                mmax = value
                char = APPEARED_LETTERS[i]
            i += 1
        ans += char
    return ans

def backup_model():
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