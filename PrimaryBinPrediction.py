# -*- coding:utf-8 -*-
"""
@author: Felix Z
"""
import keras.preprocessing.image
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, models, Sequential, Model, callbacks
from tensorflow.keras.layers import Conv2D, Dense, Reshape, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import GlobalMaxPool2D, Concatenate
import pELMparse_F as p
import graphviz
import matplotlib.pyplot as plt
import requests
import pELMparse_T as T
import numpy as np
import os
import sys
import graphviz


# 在第一层对channels和平面作Attention加权处理：https://arxiv.org/pdf/1807.06521.pdf
# Convolutional Block Attention Module

class AttentionBlk(Model):
    def __init__(self, filter_num, reduction_ratio, stride=1):
        """
        :param filter_num: [filter1, filter2, filter3]
        :param reduction_ratio: scalar
        :param stride: scalar
        """
        super().__init__()
        self.filter_num = filter_num
        self.reduction_ratio = reduction_ratio
        self.stride = stride
        # layers
        self.conv1 = Conv2D(self.filter_num[0], (1, 1), strides=self.stride, padding='same')
        self.bn1 = BatchNormalization(axis=3)
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(self.filter_num[1], (3, 3), strides=1, padding='same')
        self.bn2 = BatchNormalization(axis=3)
        self.relu2 = Activation('relu')
        self.conv3 = Conv2D(self.filter_num[2], (1, 1), strides=1, padding='same')
        self.bn3 = BatchNormalization(axis=3)
        self.channel_avgpool = GlobalAveragePooling2D()
        self.channel_maxpool = GlobalMaxPool2D()
        self.channel_fc1 = Dense(self.filter_num[2] // self.reduction_ratio, activation='relu')
        self.channel_fc2 = Dense(self.filter_num[2], activation='relu')
        self.channel_sigmoid = Activation('sigmoid')
        self.channel_reshape = Reshape((1, 1, self.filter_num[2]))
        self.spatial_conv2d = Conv2D(1, (7, 7), strides=1, padding='same')
        self.spatial_sigmoid = Activation('sigmoid')

    def call(self, inputs, **kwargs):

        x = inputs

        # 卷积层： [b, 100, 20, 10] => [b, 100, 20, 128]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # [b, 100, 20, 10] => [b, 100, 20, 10]
        # Channel Attention
        avgpool = self.channel_avgpool(x)
        maxpool = self.channel_maxpool(x)

        Dense_layer1 = self.channel_fc1
        Dense_layer2 = self.channel_fc2
        avg_out = Dense_layer2(Dense_layer1(avgpool))
        max_out = Dense_layer2(Dense_layer1(maxpool))

        channel = layers.add([avg_out, max_out])
        channel = self.channel_sigmoid(channel)
        channel = self.channel_reshape(channel)
        channel_out = tf.multiply(x, channel)

        # Spatial Attention
        avgpool = tf.reduce_mean(channel_out, axis=3, keepdims=True, name='spatial_avgpool')
        maxpool = tf.reduce_max(channel_out, axis=3, keepdims=True, name='spatial_maxpool')
        spatial = Concatenate(axis=3)([avgpool, maxpool])

        spatial = self.spatial_conv2d(spatial)
        spatial_out = self.spatial_sigmoid(spatial)
        CBAM_out = tf.multiply(channel_out, spatial_out)

        return CBAM_out


def attention_test():
    input = tf.ones(shape=[10, 500, 500, 3])
    blk = AttentionBlk(filter_num=[64, 128, 10], reduction_ratio=16)
    out = blk.call(input)
    print(out.shape)


# attention_test()

# 利用inception网络多个卷积核的特点构造主网络：
# https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43022.pdf


class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            layers.Conv2D(ch, kernelsz, strides=strides, padding=padding,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            layers.BatchNormalization(),
            layers.Activation('relu')
        ])

    def call(self, x, *args, **kwargs):
        x = self.model(x, training=False)
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch[0], kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch[1], kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch[2], kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch[3], kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch[4], kernelsz=5, strides=1)
        self.p4_1 = layers.MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch[5], kernelsz=1, strides=strides)

    def call(self, x, *args, **kwargs):
        # print('x_shape:', x.shape)
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        # print('x_shape:', x.shape)
        # [b, 101, 20, 128] => [b, 101, 20, 256]
        return x


def blk_test():
    blk = Sequential([
        InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=1),
        # [b, 101, 20, 128] => [b, 101, 20, 256]
        InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=1),
        # [b, 101, 20, 256] => [b, 101, 20, 256]
        InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=1)
    ])
    blk.build([200, 101, 20, 128])
    input = tf.ones(shape=[200, 101, 20, 128])
    out = blk.call(input)
    print(out.shape)


# blk_test()

class PrimaryBinPrediction(Model):
    def __init__(self, num_classes, **kwargs):
        super(PrimaryBinPrediction, self).__init__(**kwargs)

        # 第一层： 注意力模型,
        self.CBAM = AttentionBlk(filter_num=[64, 128, 10], reduction_ratio=16)

        # 隐藏层： Inception模型
        container = tf.keras.Sequential([
            InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2),
            InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2),
            InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2),
            InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2),
            InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2)
        ])
        self.net = container
        # [b, 101, 20, 256] => [b, 101, 20]
        self.pooling = layers.GlobalAveragePooling2D()

        # 分类层
        self.logit = layers.Dense(num_classes, activation='sigmoid')

    def call(self, x, *args, **kwargs):
        x = self.CBAM.call(inputs=x)
        x = self.net(x)
        x = self.pooling(x)
        # print('x_shape:', x.shape)
        y = self.logit(x)
        return y


def y_onehot(y):
    y_out = []
    for i in range(len(y)):
        l = []
        for j in range(101):
            j = j+1
            if j in y[i]:
                l.append(1)
            else:
                l.append(0)
        y_out.append(l)
    y_out = tf.constant(y_out, dtype=tf.float32)
    return y_out


def y_rectrangularize(y, max_num):
    ys = []
    for i in range(len(y)):
        l = []
        for j in range(max_num):
            try:
                l.append(y[i][j])
            except Exception:
                l.append(0)
        ys.append(l)
    return ys


def load():
    # max_num（最多p-sites）暂时不使用
    xtrain, ytrain, xtest, ytest, xvalid, yvalid, ztrain, ztest, zvalid = p.load_data()
    print('----------------------------------------------------------------------------------')

    # for i in range(len(ytrain)):
    #     for j in range(len(ytrain[i])):
    #         ytrain[i][j] -= 1
    #
    # for i in range(len(ytest)):
    #     for j in range(len(ytest[i])):
    #         ytest[i][j] -= 1
    #
    # for i in range(len(yvalid)):
    #     for j in range(len(yvalid[i])):
    #         yvalid[i][j] -= 1

    # 对y进行one-hot编码
    # y_train = y_onehot(ytrain)
    # y_test = y_onehot(ytest)
    # y_valid = y_onehot(yvalid)

    # 对y进行矩阵化，格式为元素索引
    # y_train = tf.constant(y_rectrangularize(ytrain, max_num), dtype=tf.float32)
    # y_test = tf.constant(y_rectrangularize(ytest, max_num), dtype=tf.float32)
    # y_valid = tf.constant(y_rectrangularize(yvalid, max_num), dtype=tf.float32)


    # 对x进行embedding
    x_train = p.onehot(xtrain)
    x_test = p.onehot(xtest)
    x_valid = p.onehot(xvalid)

    # 带structure的数据加载
    # x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid = T.main()
    #
    # x_train = tf.constant(T.onehot_seq_input(x_train), dtype=tf.float32)
    # x_test = tf.constant(T.onehot_seq_input(x_test), dtype=tf.float32)
    # x_valid = tf.constant(T.onehot_seq_input(x_valid), dtype=tf.float32)
    #
    # y_train, y_test, y_valid = tf.constant(y_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32), tf.constant(y_train, dtype=tf.float32)

    # # 将properties放到channel维度
    x_train = tf.transpose(tf.Variable(x_train, dtype=tf.float32), [0, 1, 3, 2])
    x_test = tf.transpose(tf.Variable(x_test, dtype=tf.float32), [0, 1, 3, 2])
    x_valid = tf.transpose(tf.Variable(x_valid, dtype=tf.float32), [0, 1, 3, 2])

    # z_train = tf.constant(z_train, dtype=tf.int32)
    # z_test = tf.constant(z_test, dtype=tf.int32)
    # z_valid = tf.constant(z_valid, dtype=tf.int32)
    #
    # print('x_train slice:', x_train[0])
    # print('y_train slice:', y_train[0])
    # print('x_test slice:', x_test[0])
    # print('y_test slice:', y_test[0])
    # print('x_valid slice:', x_valid[0])
    # print('y_valid slice:', y_valid[0])
    # print('---------------------------------------------------------------------')
    # print('x_train:', x_train.shape)
    # print('x_test:', x_test.shape)
    # print('x_valid:', x_valid.shape)
    # print('y_train:', y_train.shape)
    # print('y_test:', x_test.shape)
    # print('y_valid:', x_valid.shape)

    # 将y放在长度为len(bin)的小区域内索引
    # for sample in ytrain:
    #     for y in sample:
    #         if y<20:
    #             pass
    #         if 20 <= y < 30:


    # 将bins转换成tensor
    z_train = tf.constant(ztrain, dtype=tf.int32)
    z_test = tf.constant(ztest, dtype=tf.int32)
    z_valid = tf.constant(zvalid, dtype=tf.int32)

    print('x_train slice:', x_train[0][0])
    # print('y_train:', y_train[0])
    print('z_train slice:', z_train[0])
    print('x_test slice:', x_test[0][0])
    # print('y_test:', y_test[0])
    print('z_test slice:', z_test[0])
    print('x_valid slice:', x_valid[0][0])
    # print('y_valid:', y_valid[0])
    print('z_valid slice:', z_valid[0])
    print('---------------------------------------------------------------------')
    print('x_train:', x_train.shape)
    # print('y_train:', y_train.shape)
    print('z_train:', z_train.shape)
    print('x_test:', x_test.shape)
    # print('y_test:', y_test.shape)
    print('z_test:', z_test.shape)
    print('x_valid:', x_valid.shape)
    # print('y_valid:', y_valid.shape)
    print('z_valid:', z_valid.shape)
    #  y_train, y_test, y_valid

    # 不使用所有数据训练，切片
    # x_train = x_train[0:10000]
    # z_train = z_train[0:10000]
    return x_train, x_test, x_valid, z_train, z_test, z_valid


# x_train, y_train, x_test, y_test, x_valid, y_valid = load()

# Define custom loss

def de_loss(y_true, y_pred):
    """
    :param y_predict: a list of probabilities
    :param y_true: a list of category indices that has no fixed length
    :return: loss
    """
    # print(y_pred)
    # print(y_true)
    batch_loss = tf.Variable(0.0)
    for i in range(y_pred.shape[0]):
        yt = y_true[i]
        # print(yt)
        yp = y_pred[i]
        # print(yp)

        mask = yt > 0
        indices = tf.where(mask)
        loss = tf.Variable(0.0)

        for index in indices:
            # print(yt[index[0]])
            try:
                l = tf.losses.sparse_categorical_crossentropy(y_true=yt[index[0]], y_pred=yp)
            except Exception:
                print('error happened', yt[index[0]])
            loss = loss + l
        loss = loss / indices.shape[0]
        # print('loss:', loss)
        batch_loss = batch_loss + loss
        # print(batch_loss)
    batch_loss = batch_loss / y_true.shape[0]
    # print(y_true.shape[0])
    # print('------------------------------------------------------------------------')
    return batch_loss


def loss_test():
    # y_pred = tf.Variable([0.7, 0.2, 0.9, 0.9, 0.1, 0.2])
    # y_pred = tf.Variable([0.2, 0.9, 0.1, 0.1, 0.6, 0.6])
    y_pred = tf.Variable([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    y_true = tf.Variable([2, 3, 6, 7])
    loss = de_loss(y_pred=y_pred, y_true=y_true)
    # loss = tf.losses.sparse_categorical_crossentropy(y_true, y_pred)
    print(loss)

# loss_test()


def prediction_accuracy(y_true, y_pred):

    correct = tf.Variable(0)
    print('------------------------------------')
    print(y_true.shape)
    print('------------------------------------')
    for i in range(y_true.shape[0]):

        indices = tf.where(y_pred[i] > 0.5)
        l = []
        for j in range(5):
            if j in indices:
                l.append(1)
            else:
                l.append(0)
        true = tf.cast(y_true[i], dtype=tf.float32)
        pred = tf.constant(l, dtype=tf.float32)
        cor = tf.reduce_sum(tf.cast(tf.equal(true, pred), dtype=tf.int32))
        # print(cor)
        correct = correct + cor
    accuracy = correct / (y_true.shape[0] * 5)
    return accuracy


def metrics_test():
    accuracy = prediction_accuracy(y_true=tf.constant([[1, 0, 1, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 1]],
                                                      dtype=tf.int32),
                                   y_pred=tf.constant([[0.2, 0.1, 0.4, 0.2, 0.3], [0.2, 0.3, 0.1, 0.2, 0.3],
                                                       [0.4, 0.1, 0.2, 0.2, 0.3]]))
    print(accuracy)


# metrics_test()


def main():
    # shape test
    # data = tf.ones(shape=[1, 101, 20, 10])
    # blk = AttentionBlk(filter_num=[64, 64, 128], reduction_ratio=16)
    # blk.call(data)

    # 加载数据
    x_train, x_test, x_valid, z_train, z_test, z_valid = load()

    # 训练模型
    for i in range(1):
        print('iteration:', i)
        net = PrimaryBinPrediction(num_classes=5)

        tf.config.experimental_run_functions_eagerly(True)

        # tensorboard 可视化
        tbcallback = callbacks.TensorBoard(update_freq='batch', write_graph=True, write_images=True)

        net.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.5),
                    # 这里为了不 run eagerly取消了 metrics: prediction_accuracy
                    loss=tf.losses.binary_crossentropy, metrics=[tf.metrics.binary_accuracy, prediction_accuracy])

        # plot_model(net, to_file='model.png', show_shapes=True)

        net.build(input_shape=(None, 100, 20, 10))
        net.load_weights('/Users/felixzeng/Desktop/temp/model_weights_partI.h5')
        print('Testing Accuracy')
        net.evaluate(x_test, z_test)
        # print('weights:', net.trainable_variables)
        history = net.fit(x=x_train, y=z_train, batch_size=500, epochs=1, validation_data=(x_valid, z_valid), validation_freq=1, callbacks=[tbcallback])
        # print(history.history.keys())


        # 可视化3
        ig1, ax_acc = plt.subplots()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.plot(history.history['binary_accuracy'])
        print('binary accurcacy:', history.history['binary_accuracy'])
        plt.plot(history.history['prediction_accuracy'])
        print('prediction_accuracy:', history.history['prediction_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        print('val_binary_accuracy:', history.history['val_binary_accuracy'])
        plt.plot(history.history['val_prediction_accuracy'])
        print('val_prediction_accuracy:', history.history['val_prediction_accuracy'])
        plt.legend(['binary_accuracy', 'prediction_accuracy', 'val_binary_accuracy', 'val_prediction_accuracy'],
                   loc='upper right')
        plt.show()

        fig2, ax_loss = plt.subplots()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model- Loss')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.show()


        # 测试集
        # print('Testing Accuracy')
        # net.evaluate(x_test, z_test)

        # 保存模型
        # net.save_weights('/Users/felixzeng/Desktop/temp/model_weights_partI_visualization.h5', save_format='h5')

    # 每个里面只要一个site
    return None


# main()

