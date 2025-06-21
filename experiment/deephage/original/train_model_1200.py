#!/usr/bin/env python

from sklearn.metrics import classification_report

from keras.layers import *
from keras.models import Model
import h5py
import sys
from keras import regularizers
import keras.callbacks

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras.callbacks
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

KTF.set_session(sess)
import pandas as pd

def classifaction_report_csv(report,path_save,c):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()

        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    if c==0:
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(path_save + 'classification_report.csv', index = False)
    else:
        print('train')
    return report_data

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, accuracy, viru_acc, temp_acc, train_viru_acc, train_temp_acc,
                  path_save,max_length,lr_rate,b_size):
        iters = range(len(self.losses[loss_type]))
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.ylim((0, 2))
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.title('%s test_acc: %s \n test--- viru: %s temp: %s \ntrain---viru: %s temp: %s'
                  % (str(max_length), accuracy, viru_acc, temp_acc,
                     train_viru_acc, train_temp_acc))
        plt.legend(loc="upper right")
        plt.show()
        plt.savefig(path_save + str(max_length) + '_' + str(lr_rate) +'_'+ str(b_size) +'.png')

max_len = [1200]
lr_rate = 0.0001
b_size = 32
def main():
    for max_length in max_len:
        for r in range(5):
            j = r + 1
            argv0_list = sys.argv[0].split("\\")
            script_name = argv0_list[len(argv0_list) - 1]

            path_save = '/ldata3/sfwu/pycharm_run/delet_2/fold_5/800_1200_20000/' + str(j) + '/' ##############################
            predict_save_path = path_save + str(max_length) + '_' + str(lr_rate)+'_'+ str(b_size)  +'_prediction.csv'  # argv[5]
            model_save_path = path_save + str(max_length) + '_' + str(lr_rate)+'_'+ str(b_size) + '_model.h5'  # argv[6]
            path_mat = '/ldata3/sfwu/pycharm_run/delet_2/fold_5/800_1200_20000/'+ str(j) + '/' ################################

            train_matrix = h5py.File(path_mat+'P_train_ds.mat')
            print('1')
            train_label = h5py.File(path_mat+'T_train_ds.mat')
            print('2')
            test_matrix = h5py.File(path_mat+'P_test.mat')
            print('3')
            test_label = h5py.File(path_mat+'T_test.mat')
            print('4')

            train_matrix = train_matrix['P_train_ds'][:]
            train_label = train_label['T_train_ds'][:]
            test_matrix = test_matrix['P_test'][:]
            test_label = test_label['T_test'][:]

            train_matrix = train_matrix.transpose()
            train_label = train_label.transpose()
            test_matrix = test_matrix.transpose()
            test_label = test_label.transpose()

            train_num = train_label.shape[0]
            test_num = test_label.shape[0]

            train_matrix = train_matrix.reshape(train_num, max_length, 4)
            test_matrix = test_matrix.reshape(test_num, max_length, 4)

            imp=Input(shape=(max_length,4))
            x=Conv1D(64,6,activation='relu',padding='same')(imp)
            x=MaxPooling1D(3)(x)
            x=BatchNormalization()(x)
            x=Dropout(0.3)(x)
            x=GlobalAveragePooling1D()(x)
            x=Dense(64,activation='relu')(x)
            x=BatchNormalization()(x)
            predictions=Dense(1,activation='sigmoid')(x)
            adam = keras.optimizers.Adam(lr=lr_rate)
            model=Model(inputs=imp,outputs=predictions)
            model.compile(loss='binary_crossentropy',
                          optimizer=adam,
                          metrics=['accuracy'])
            history = LossHistory()
            model.fit(train_matrix, train_label,
                      epochs=100,
                      batch_size=b_size,shuffle=True,validation_data=(test_matrix,test_label),callbacks=[history])
            loss, accuracy = model.evaluate(test_matrix, test_label)
            print('test loss: ', loss)
            print('test accuracy: ', accuracy)
            predict = model.predict(test_matrix)
            np.savetxt(predict_save_path, predict)
            model.save(model_save_path)

            report_test = classification_report(test_label, (predict>0.5))
            print('test')
            print(report_test)
            report_dic_test = classifaction_report_csv(report_test,path_save,0)
            [temp_acc, viru_acc] = [report_dic_test[0].get('recall'),report_dic_test[1].get('recall')]

            predict_train = model.predict(train_matrix)
            report_train = classification_report(train_label, (predict_train>0.5))
            print('train')
            print(report_train)
            report_dic_train = classifaction_report_csv(report_train,path_save,1)
            [train_temp_acc, train_viru_acc] =[report_dic_train[0].get('recall'),report_dic_train[1].get('recall')]

            history.loss_plot('epoch', accuracy, viru_acc, temp_acc,train_viru_acc,train_temp_acc,path_save,max_length,lr_rate,b_size)

if __name__ == "__main__":
    main()
