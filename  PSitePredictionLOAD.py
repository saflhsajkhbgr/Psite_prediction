# -*- coding:utf-8 -*-
"""
@author: Felix Z
"""
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Reshape, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPool2D, Concatenate
import pELMparse_F as p
import PrimaryBinPrediction as tr
import matplotlib.pyplot as plt
from pylab import *


def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
        title('feature_map_{}'.format(i))

    plt.savefig('feature_map.png')
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")


if __name__ == "__main__":
    # replaced by your model name
    net = tr.PrimaryBinPrediction(num_classes=5)
    net.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005, momentum=0.5),
                loss=tf.losses.binary_crossentropy, metrics=[tf.metrics.binary_accuracy, tr.prediction_accuracy])
    net.build(input_shape=(None, 101, 20, 10))
    net.load_weights('/Users/felixzeng/Desktop/temp/model_weights_partI.h5')
    x_train, y_train, x_test, y_test, x_valid, y_valid = tr.load()
    data = tf.reshape(x_train[0], shape=[1, 100, 20, 10])

    f1 = net.call(data)[0]  # 只修改inpu_image
    # 第一层卷积后的特征图展示，输出是（1,149,149,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    # for _ in range(32):
    #     show_img = f1[:, :, :, _]
    #     show_img.shape = [149, 149]
    #     plt.subplot(4, 8, _ + 1)
    #     plt.imshow(show_img, cmap='gray')
    #     plt.axis('off')
    # plt.show()

    visualize_feature_map(f1)


