# coding: utf8
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.layers import Dropout, Bidirectional, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from keras import backend as K #转换为张量

import numpy as np
import time
import os



# from config import *

os.environ["PATH"] += os.pathsep + '/data/'


def CNN_multi_model(input_length, vocab_length, output_dim, neurons_num, classfile):
    input = Input((input_length,))
    embedding = Embedding(vocab_length, output_dim, input_length=input_length)(input)
    convs = []
    for kernel_size in [2, 3, 4, 5, 6]:
        c = Conv1D(neurons_num, kernel_size, activation='relu')(embedding)
        c = GlobalMaxPooling1D()(c)
        convs.append(c)
    x = Concatenate()(convs)
    dropout = Dropout(rate=0.5)(x)
    dense = Dense(neurons_num, activation='relu')(dropout)
    output = Dense(classfile, activation='softmax')(dense)
    model = Model(inputs=input, outputs=output)
    # plot_model(model, show_shapes=True, show_layer_names=True, to_file=png_file)
    return model


# def CNN_stack_model(input_length, vocab_length, output_dim, neurons_num, classfile):
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_length,
#                         output_dim=output_dim,
#                         input_length=input_length))
#     for i in [3, 2]:
#         model.add(Conv1D(neurons_num, i, padding='same'))
#         model.add(MaxPooling1D(4, 1, padding='same'))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(classfile, activation='softmax'))
#     return model


# def BILSTM_model(input_length, vocab_length, output_dim, neurons_num, classfile):
#     model = Sequential()
#     # 设置模型第一层：embedding层
#     model.add(Embedding(input_dim=vocab_length,
#                         output_dim=output_dim,
#                         input_length=input_length))
#     # 设置模型第二层：
#     model.add(Bidirectional(LSTM(neurons_num, kernel_initializer='random_normal'), merge_mode='sum'))
#     # model.add(Dropout(0.5))
#     # 设置全连接层
#     # model.add(Dense(100, activation='relu'))
#     model.add(Dense(128, activation='relu', kernel_initializer='random_normal',
#                     kernel_regularizer=tf.keras.regularizers.l2(0.03)))
#     model.add(Dropout(0.5))
#     model.add(Dense(classfile, activation='softmax'))
#     return model


def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, model_path):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    # 训练模型
    # 设置被监测的数据不再提升，则停止训练(如果3轮训练，val_loss值提升小于0.01，则停止训练)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3)
    # 设置自动降低学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.0001)
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[earlystopping, reduce_lr])
    # 预测结果
    print('训练集结果...')
    y_train_pre = model.predict(x_train)
    print(classification_report(np.argmax(y_train, axis=1), np.argmax(y_train_pre, axis=1)))

    print('验证集结果...')
    y_test_pre = model.predict(x_test)
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_test_pre, axis=1)))
    # 保存模型
    model.save(filepath=model_path)


if __name__ == '__main__':
    print('加载数据中...')
    # # 加载中性数据
    # X_train = np.load(zx_x + '.npy')
    # Y_train = np.load(zx_y + '.npy')

    # 加载正负数据
    X_train = np.load('./data/sentenceVec.npy',allow_pickle=True)
    Y_train = np.load('./data/labels.npy',allow_pickle=True)

    print(type(X_train))
    print(type(Y_train))

    print(X_train.shape)
    print(Y_train.shape)
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

    for i in range(5):
        # 划分数据集（x是数据，y是标签）
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3)

        # 配置模型参数
        # # 是否中性
        # input_length = 877
        # vocab_length = 2439
        # model_dir = zx_cnn_model

        # # 是否正负
        # input_length = 814
        # vocab_length = 1394
        # model_dir = zf_cnn_model

        # # 一级预防
        # input_length = 1091
        # vocab_length = 2504
        # model_dir = yj_cnn_model

        # 二级预防
        input_length = 33
        vocab_length = 39963
        model_dir = './cnnmulti.h5'  #模型保存路径


        output_dim = 300
        neurons_num = 128
        classfile = 3

        # 配置训练参数
        epochs = 20
        batch_size = 32

        model_time = time.time()
        print('构建模型...')
        model = CNN_multi_model(input_length, vocab_length, output_dim, neurons_num, classfile)
        # model = CNN_stack_model(input_length, vocab_length, output_dim, neurons_num, classfile)
        print('训练模型...')
        train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, model_dir)

        print('训练模型消耗的时间是{}秒。'.format(time.time() - model_time))






