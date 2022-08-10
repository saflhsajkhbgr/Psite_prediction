# -*- coding:utf-8 -*-
"""
@author: Felix Z
"""
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Reshape, BatchNormalization, Activation, GlobalAveragePooling2D, GlobalMaxPool2D
import pELMparse as load
import pELMparse_T as T
import PrimaryBinPrediction as tr
import matplotlib.pyplot as plt


class SecondaryBinPrediction(Model):
    def __init__(self, num_classes, **kwargs):
        super(SecondaryBinPrediction, self).__init__(**kwargs)

        self.ChAttention = tr.AttentionBlk(filter_num=[64, 128, 10], reduction_ratio=16)
        self.hidden = Sequential([tr.InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2),
                                  tr.InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2),
                                  tr.InceptionBlk(ch=[64, 96, 128, 16, 32, 32], strides=2)])
        self.pooling = GlobalAveragePooling2D()
        self.logit = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs,  *args, **kwargs):
        x = inputs
        x = self.ChAttention(x)
        x = self.hidden(x)
        x = self.pooling(x)
        y = self.logit(x)
        return y


def preprocessing_20():
    x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid = T.main()
    x_train = tf.constant(T.onehot_seq_input(x_train), dtype=tf.float32)
    x_test = tf.constant(T.onehot_seq_input(x_test), dtype=tf.float32)
    x_valid = tf.constant(T.onehot_seq_input(x_valid), dtype=tf.float32)

    x_train = tf.gather(tf.transpose(x_train, [0, 1, 3, 2]), axis=3, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    x_test = tf.gather(tf.transpose(x_test, [0, 1, 3, 2]), axis=3, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    x_valid = tf.gather(tf.transpose(x_valid, [0, 1, 3, 2]), axis=3, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    y_train = tf.constant(y_train, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)
    y_valid = tf.constant(y_valid, dtype=tf.float32)

    z_train = tf.constant(z_train, dtype=tf.int32)
    z_test = tf.constant(z_test, dtype=tf.int32)
    z_valid = tf.constant(z_valid, dtype=tf.int32)

    print('x_train slice:', x_train[0][0])
    # print('y_train slice:', y_train[0])
    # print('z_train slice:', z_train[0])
    print('x_test slice:', x_test[0][0])
    # print('y_test slice:', y_test[0])
    # print('z_test slice:', z_test[0])
    print('x_valid slice:', x_valid[0][0])
    # print('y_valid slice:', y_valid[0])
    # print('z_valid slice:', z_valid[0])
    print('---------------------------------------------------------------------')
    print('x_train:', x_train.shape)
    print('x_test:', x_test.shape)
    print('x_valid:', x_valid.shape)
    print('y_train:', y_train.shape)
    print('y_test:', y_test.shape)
    print('y_valid:', y_valid.shape)
    print('z_train:', z_train.shape)
    print('z_test:', z_test.shape)
    print('z_valid:', z_valid.shape)
    return x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid


def preprocessing_5():
    # tf.config.experimental_run_functions_eagerly(True)

    x_train, x_test, x_valid, y_train, y_test, y_valid, z_train, z_test, z_valid = load.load()
    x_train = tf.transpose(tf.constant(x_train, dtype=tf.float32), [0, 1, 3, 2])
    x_test = tf.transpose(tf.constant(x_test, dtype=tf.float32), [0, 1, 3, 2])
    x_valid = tf.transpose(tf.constant(x_valid, dtype=tf.float32), [0, 1, 3, 2])

    # y_train = tf.constant(y_train, dtype=tf.int32)
    # y_test = tf.constant(y_test, dtype=tf.int32)
    # y_valid = tf.constant(y_valid, dtype=tf.int32)

    ytrain = []
    for i in range(len(y_train)):
        ytrain.append([])
        for j in range(len(y_train[i])):
            ytrain[i].append([])
            for k in range(len(y_train[j])):
                if all(y_train[i][j][k] == [0, 0, 0, 0]):
                    ytrain[i][j].append(0)
                else:
                    ytrain[i][j].append(1)

    y_train = tf.constant(ytrain, dtype=tf.int32)

    ytest = []
    for i in range(len(y_test)):
        ytest.append([])
        for j in range(len(y_test[i])):
            ytest[i].append([])
            for k in range(len(y_test[j])):
                if all(y_test[i][j][k] == [0, 0, 0, 0]):
                    ytest[i][j].append(0)
                else:
                    ytest[i][j].append(1)
    y_test = tf.constant(ytest, dtype=tf.int32)

    yvalid = []
    for i in range(len(y_valid)):
        yvalid.append([])
        for j in range(len(y_valid[i])):
            yvalid[i].append([])
            for k in range(len(y_valid[j])):
                if all(y_valid[i][j][k] == [0, 0, 0, 0]):
                    yvalid[i][j].append(0)
                else:
                    yvalid[i][j].append(1)
    y_valid = tf.constant(yvalid, dtype=tf.int32)

    z_train = tf.constant(z_train, dtype=tf.int32)
    z_test = tf.constant(z_test, dtype=tf.int32)
    z_valid = tf.constant(z_valid, dtype=tf.int32)

    print('x_train slice:', x_train[0])
    print('y_train slice:', y_train[0])
    print('z_train slice:', z_train[0])
    print('x_test slice:', x_test[0])
    print('y_test slice:', y_test[0])
    print('z_test slice:', z_test[0])
    print('x_valid slice:', x_valid[0])
    print('y_valid slice:', y_valid[0])
    print('z_valid slice:', z_valid[0])
    print('---------------------------------------------------------------------')
    print('x_train:', x_train.shape)
    print('x_test:', x_test.shape)
    print('x_valid:', x_valid.shape)
    print('y_train:', y_train.shape)
    print('y_test:', y_test.shape)
    print('y_valid:', y_valid.shape)
    print('z_train:', z_train.shape)
    print('z_test:', z_test.shape)
    print('z_valid:', z_valid.shape)
    return x_train[0:1000], y_train[0:1000], z_train[0:150], x_test, y_test, z_test, x_valid, y_valid, z_valid


# x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid = preprocessing_5()


class OverallNetwork(Model):
    def __init__(self, num_classes1, num_classes2, z_train, **kargs):
        super(OverallNetwork, self).__init__(**kargs)
        self.z_train = z_train
        # self.p1 = tr.PrimaryBinPrediction(num_classes=num_classes1)
        # self.p1.build(input_shape=(None, 100, 20, 10))
        # self.p1.load_weights('/Users/felixzeng/Desktop/temp/model_weights_partI.h5')
        self.p2 = SecondaryBinPrediction(num_classes=num_classes2)

    def call(self, x_train, *args, **kargs):
        x1 = x_train
        first_bin_pred = self.p1.call(x1)[0]
        print('first bin prediction:', first_bin_pred)
        l1 = tf.losses.binary_crossentropy(self.z_train, first_bin_pred)
        print('l1:', l1)
        indices = tf.where(first_bin_pred > 0.5)
        print('indices:', indices)
        x2 = []
        for index in range(indices.shape[0]):
            if indices[index][0] == 0:
                # print('prediction belongs to bin1')
                x2.append(x1[0:20])
            if indices[index][0] == 1:
                # print('prediction belongs to bin2')
                x2.append(x1[21:40])
            if indices[index][0] == 2:
                # print('prediction belongs to bin3')
                x2.append(x1[41:60])
            if indices[index][0] == 3:
                # print('prediction belongs to bin4')
                x2.append(x1[61:80])
            if indices[index][0] == 4:
                # print('prediction belongs to bin5')
                x2.append(x1[81:100])
        print('x2:', len(x2))
        y = []
        for x in x2:
            y.append(self.p2.call(x))
        return y, indices, l1


def custom_loss(y_pred, indices, y_true, l1, alpha):
    """
    :param y_pred: 预测的20个概率(针对预测的bin中的20个位置)
    :param indices: 5个bin中间预测有p-site的概率
    :param y_true: 预测的bin对应的20个标签（1或0）
    :param alpha: 超参数 ——— 决定第二个loss的权重
    :return: 总损失
    """
    loss = tf.Variable(0.0)
    for i in range(len(y_true)):
        true = tf.gather_nd(y_true[i], indices)
        print('true:', true)
        l2 = tf.losses.binary_crossentropy(true, y_pred[i])
        l = alpha * l1 + l2
        loss = loss + l
    return loss


def train1():
    # 使用SGD训练，保持第一个大loss的权重够大，否则第二个模型的崩塌概率会变高
    tf.config.experimental_run_functions_eagerly(True)

    x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid = preprocessing_20()
    model = OverallNetwork(num_classes1=5, num_classes2=20, z_train=z_train)

    variables = OverallNetwork.trainable_variables
    # print(variables)
    for batch in range(x_train.shape[0]):
        model.z_train = z_train[batch]
        print('batch:', batch)
        with tf.GradientTape() as tape:
            x = tf.reshape(x_train[batch], [1, 100, 20, 10])
            y, indices, l1 = model.call(x)
            print('y_pred:', y)
            loss = custom_loss(y_pred=y, y_true=y_train[batch], indices=indices, l1=l1, alpha=0.5)
            print('total loss:', loss)
        grads = tape.gradient(target=loss, sources=variables)
        tf.keras.optimizer.apply_gradients(zip(grads, variables))
        print('loss:', loss)
    model.save('visualization.h5')
    return None


def train2():
    tf.config.experimental_run_functions_eagerly(True)

    x_train, y_train, z_train, x_test, y_test, z_test, x_valid, y_valid, z_valid = preprocessing_5()

    p1 = tr.PrimaryBinPrediction(num_classes=5)
    p1.build(input_shape=(None, 100, 20, 10))
    p1.load_weights('/Users/felixzeng/Desktop/temp/model_weights_partI.h5')

    p2 = SecondaryBinPrediction(num_classes=5)
    p2.build(input_shape=(None, 20, 20, 10))
    losses = []
    for batch in range(len(x_train)):
        print('-------------------------------------------------------------------------------------')
        print('batch num:', batch)
        # first_pred = p1.call(tf.reshape(x_train[batch], shape=[1, 100, 20, 10]))[0]
        # print('predicted large bin:', first_pred)
        # print('real large bin:', z_train[batch])
        # l1 = tf.losses.binary_crossentropy(z_train[batch], first_pred)
        # print('loss 1:', l1)
        # indices = tf.where(first_pred > 0.5)
        # print('indices:', indices)

        print('y_train:', y_train[batch])
        indices = []
        for i in range(len(y_train[batch])):
            if not(all(y_train[batch][i] == [0, 0, 0, 0, 0])):
                indices.append([i])
        print('indices:', indices)
        indices = tf.constant(indices, dtype=tf.int32)
        print('indices:', indices)
        y = tf.gather(y_train[batch], indices)
        # y = tf.gather_nd(y_train[batch], indices)
        # print('y_train:', y_train[batch])
        # print('mini bin:', y)
        variables = p2.trainable_variables
        x2 = []
        for index in range(indices.shape[0]):
            if indices[index][0] == 0:
                # print('prediction belongs to bin1')
                x2.append(x_train[batch][0:20])
            if indices[index][0] == 1:
                # print('prediction belongs to bin2')
                x2.append(x_train[batch][20:40])
            if indices[index][0] == 2:
                # print('prediction belongs to bin3')
                x2.append(x_train[batch][40:60])
            if indices[index][0] == 3:
                # print('prediction belongs to bin4')
                x2.append(x_train[batch][60:80])
            if indices[index][0] == 4:
                # print('prediction belongs to bin5')
                x2.append(x_train[batch][80:100])
        # print('x2 length:', len(x2))
        # print('-------------------------------------------------------------------------------------')
        for x in range(len(x2)):
            with tf.GradientTape() as tape:
                pred = p2.call(tf.reshape(x2[x], shape=[1, 20, 20, 10]))
                # print('pred:', pred)
                loss = tf.losses.binary_crossentropy(y[x], pred)
                print('loss:', loss)
            losses.append(loss)
            grads = tape.gradient(loss, variables)
            optimizer = tf.keras.optimizers.RMSprop(lr=1e-4, momentum=0.5)
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

    fig2, ax_loss = plt.subplots()
    plt.xlabel('batch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.plot(losses)
    plt.show()


train2()