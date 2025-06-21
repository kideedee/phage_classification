import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import BatchNormalization
from sklearn.metrics import classification_report
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense

# Set GPU memory growth to prevent OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print(f"GPU available: {physical_devices}")
else:
    print("No GPU found. Using CPU.")


def classification_report_csv(report, path_save, c):
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
    if c == 0:
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(path_save + 'classification_report.csv', index=False)
    else:
        print('train')
    return report_data


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))  # Changed from 'acc' to 'accuracy'
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))  # Changed from 'val_acc' to 'val_accuracy'

    def on_epoch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))  # Changed from 'acc' to 'accuracy'
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))  # Changed from 'val_acc' to 'val_accuracy'

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
        plt.savefig(path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '.png')


max_len = [400]
lr_rate = 0.0001
b_size = 32


def main():
    for max_length in max_len:
        for r in range(5):
            j = r + 1

            # Path setup
            path_save = '/ldata3/sfwu/pycharm_run/delet_2/fold_5/100_400_20000/' + str(j) + '/'
            predict_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '_prediction.csv'
            model_save_path = path_save + str(max_length) + '_' + str(lr_rate) + '_' + str(b_size) + '_model.h5'
            path_mat = '/ldata3/sfwu/pycharm_run/delet_2/fold_5/100_400_20000/' + str(j) + '/'

            # Create directory if it doesn't exist
            os.makedirs(path_save, exist_ok=True)

            # Load data
            with h5py.File(path_mat + 'P_train_ds.mat', 'r') as f:
                train_matrix = f['P_train_ds'][:]
            print('1')

            with h5py.File(path_mat + 'T_train_ds.mat', 'r') as f:
                train_label = f['T_train_ds'][:]
            print('2')

            with h5py.File(path_mat + 'P_test.mat', 'r') as f:
                test_matrix = f['P_test'][:]
            print('3')

            with h5py.File(path_mat + 'T_test.mat', 'r') as f:
                test_label = f['T_test'][:]
            print('4')

            # Transpose data
            train_matrix = train_matrix.transpose()
            train_label = train_label.transpose()
            test_matrix = test_matrix.transpose()
            test_label = test_label.transpose()

            train_num = train_label.shape[0]
            test_num = test_label.shape[0]

            # Reshape data
            train_matrix = train_matrix.reshape(train_num, max_length, 4)
            test_matrix = test_matrix.reshape(test_num, max_length, 4)

            # Build model with Functional API
            inputs = Input(shape=(max_length, 4))
            x = Conv1D(64, 6, activation='relu', padding='same')(inputs)
            x = MaxPooling1D(3)(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            predictions = Dense(1, activation='sigmoid')(x)

            model = Model(inputs=inputs, outputs=predictions)

            # Use Adam optimizer with learning rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)

            # Compile model
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            # Setup callback
            history = LossHistory()

            # Train model
            model.fit(
                train_matrix, train_label,
                epochs=100,
                batch_size=b_size,
                shuffle=True,
                validation_data=(test_matrix, test_label),
                callbacks=[history]
            )

            # Evaluate model
            loss, accuracy = model.evaluate(test_matrix, test_label)
            print('test loss: ', loss)
            print('test accuracy: ', accuracy)

            # Generate predictions
            predict = model.predict(test_matrix)
            np.savetxt(predict_save_path, predict)

            # Save model
            model.save(model_save_path)

            # Classification report for test data
            report_test = classification_report(test_label, (predict > 0.5))
            print('test')
            print(report_test)
            report_dic_test = classification_report_csv(report_test, path_save, 0)
            [temp_acc, viru_acc] = [report_dic_test[0].get('recall'), report_dic_test[1].get('recall')]

            # Classification report for training data
            predict_train = model.predict(train_matrix)
            report_train = classification_report(train_label, (predict_train > 0.5))
            print('train')
            print(report_train)
            report_dic_train = classification_report_csv(report_train, path_save, 1)
            [train_temp_acc, train_viru_acc] = [report_dic_train[0].get('recall'), report_dic_train[1].get('recall')]

            # Plot results
            history.loss_plot('epoch', accuracy, viru_acc, temp_acc, train_viru_acc, train_temp_acc, path_save,
                              max_length, lr_rate, b_size)


if __name__ == "__main__":
    main()
