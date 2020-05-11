#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python >= 2.7

import numpy as np
import pandas as pd
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers,models,optimizers,regularizers
import keras.backend as K

def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def createModel():
    model = models.Sequential()
    model.add(layers.Conv1D(filters=32, kernel_size=3, strides=1,activation='relu',input_shape=(14,1),kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same',activation='relu',kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same',activation='relu',kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same',activation='relu',kernel_regularizer=regularizers.l1(0.001)))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024,activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc', precision,recall, f1])
    return model

def main():
    ##Load data
    df = pd.read_csv('./train_data.txt', sep='\t', header=None)
    df.columns = ['case', 'gene', 'upstream_snp', 'downstream_snp', 'intron_snp', 'synonymous_snp', 'nonsynonymous_snp', 'upstream_indel',
                  'downstream_indel', 'intron_indel', 'synonymous_indel', 'nonsynonymous_indel', 'module', 'expression', 'QTN', 'QTL', 'QTG', 'type', 'label']
    selected_columns = ['upstream_snp', 'downstream_snp', 'intron_snp', 'synonymous_snp', 'nonsynonymous_snp', 'upstream_indel',
                        'downstream_indel', 'intron_indel', 'synonymous_indel', 'nonsynonymous_indel', 'module', 'expression', 'QTN', 'QTL']
    X = df[selected_columns].astype(np.float64)
    y = np.array(df[['label']]).reshape((-1))
    print('Positive(N): %d' % df['label'].sum())
    print('Negative(N): %d' % (len(y) - df['label'].sum()))
    ##Scale
    X = preprocessing.scale(X)
    X = np.reshape(X, [len(X), 14, 1])
    y = to_categorical(y)
    ##Train set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    ## Create model
    model = createModel()
    history = model.fit(X_train, y_train, batch_size=10, epochs=60, verbose=0)
    loss,accuracy,precision,recall,f1 = model.evaluate(X_test, y_test , verbose=0)
    print("Loss:"+str(loss))
    print("Accuracy:"+str(accuracy))
    print("Precision:"+str(precision))
    print("Recall:"+str(recall))
    print("F1:"+str(f1))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write('User interrupt me! ;-) See you!\n')
        sys.exit(0)
