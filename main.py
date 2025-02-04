import time

import h5py
import keras
import matplotlib as mpl
import numpy as np
from keras import Model
from keras.src.layers import *
from sklearn.metrics import classification_report

mpl.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

KTF.set_session(sess)


def classifaction_report_csv(report, path_save, c):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        if len(row_data) == 0:
            continue

        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3]) if len(row_data) > 3 else -1
        row['support'] = float(row_data[4]) if len(row_data) > 3 else -1
        report_data.append(row)
    if c == 0:
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(path_save + 'classification_report.csv', index=False)
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
                  path_save, max_length, lr_rate, b_size):
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
        plt.savefig(path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '.png')


def save_run_time(file, t):
    with open(file, mode='a') as f:
        f.write(str(t) + '\n')


max_len = [400]
lr_rate = 0.0001
b_size = 32
num_epoch = 1


def main():
    group = "100_400"
    root_data_dir = f"data/{group}"
    root_result_dir = f"results/{group}"
    train_dir = os.path.join(root_data_dir, "train")
    test_dir = os.path.join(root_data_dir, "test")
    runtime_result_file = os.path.join(root_result_dir, "time.txt")
    if os.path.exists(runtime_result_file):
        os.remove(runtime_result_file)

    start = time.time()
    for max_length in max_len:
        for r in range(5):
            j = r + 1
            path_save = f'{root_result_dir}/fold_5/100_400/' + str(j) + '/'  ##############################
            os.makedirs(path_save, exist_ok=True)
            predict_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(
                b_size) + '_prediction.csv'  # argv[5]
            model_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(
                b_size) + '_model.keras'  # argv[6]

            train_sequence_file = [f for f in os.listdir(os.path.join(train_dir, "sequences")) if f'_{j}.mat' in f][0]
            train_label_file = [f for f in os.listdir(os.path.join(train_dir, "labels")) if f'_{j}.mat' in f][0]
            test_sequence_file = [f for f in os.listdir(os.path.join(test_dir, "sequences")) if f'_{j}.mat' in f][0]
            test_label_file = [f for f in os.listdir(os.path.join(test_dir, "labels")) if f'_{j}.mat' in f][0]
            print(train_sequence_file)
            print(train_label_file)
            print(test_sequence_file)
            print(test_label_file)

            train_matrix = h5py.File(os.path.join(train_dir, f'sequences/{train_sequence_file}'), 'r')
            print('1')
            train_label = h5py.File(os.path.join(train_dir, f'labels/{train_label_file}'), 'r')
            print('2')
            test_matrix = h5py.File(os.path.join(test_dir, f'sequences/{test_sequence_file}'), 'r')
            print('3')
            test_label = h5py.File(os.path.join(test_dir, f'labels/{test_label_file}'), 'r')
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

            imp = Input(shape=(max_length, 4))
            x = Conv1D(64, 6, activation='relu', padding='same')(imp)
            x = MaxPooling1D(3)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            predictions = Dense(1, activation='sigmoid')(x)
            adam = keras.optimizers.Adam(learning_rate=lr_rate)
            model = Model(inputs=imp, outputs=predictions)
            model.compile(loss='binary_crossentropy',
                          optimizer=adam,
                          metrics=['accuracy'])
            history = LossHistory()
            start_train = time.time()
            model.fit(train_matrix, train_label,
                      epochs=num_epoch,
                      batch_size=b_size, shuffle=True, validation_data=(test_matrix, test_label), callbacks=[history])
            end_train = time.time() - start_train
            save_run_time(runtime_result_file, end_train)
            loss, accuracy = model.evaluate(test_matrix, test_label)
            print('test loss: ', loss)
            print('test accuracy: ', accuracy)
            predict = model.predict(test_matrix)
            np.savetxt(predict_save_path, predict)
            model.save(model_save_path)

            report_test = classification_report(test_label, (predict > 0.5))
            print('test')
            print(report_test)
            report_dic_test = classifaction_report_csv(report_test, path_save, 0)
            [temp_acc, viru_acc] = [report_dic_test[0].get('recall'), report_dic_test[1].get('recall')]

            predict_train = model.predict(train_matrix)
            report_train = classification_report(train_label, (predict_train > 0.5))
            print('train')
            print(report_train)
            report_dic_train = classifaction_report_csv(report_train, path_save, 1)
            [train_temp_acc, train_viru_acc] = [report_dic_train[0].get('recall'), report_dic_train[1].get('recall')]

            history.loss_plot('epoch', accuracy, viru_acc, temp_acc, train_viru_acc, train_temp_acc, path_save,
                              max_length,
                              lr_rate, b_size)
    end = time.time() - start
    save_run_time(runtime_result_file, end)


if __name__ == "__main__":
    main()
