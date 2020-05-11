#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python >= 2.7

import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    ##Load data
    df = pd.read_csv('./train_data.txt', sep='\t', header=None)
    df.columns = ['case', 'gene', 'upstream_snp', 'downstream_snp', 'intron_snp', 'synonymous_snp', 'nonsynonymous_snp', 'upstream_indel',
                  'downstream_indel', 'intron_indel', 'synonymous_indel', 'nonsynonymous_indel', 'module', 'expression', 'QTN', 'QTL', 'QTG', 'type', 'label']
    selected_columns = ['upstream_snp', 'downstream_snp', 'intron_snp', 'synonymous_snp', 'nonsynonymous_snp', 'upstream_indel',
                        'downstream_indel', 'intron_indel', 'synonymous_indel', 'nonsynonymous_indel', 'module', 'expression', 'QTN', 'QTL']
    X = df[selected_columns]
    y = df['label']
    print('Positive(N): %d' % df['label'].sum())
    print('Negative(N): %d' % (len(y) - df['label'].sum()))
    ##Scale
    X = preprocessing.scale(X)
    ##Train set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    ##Grid search
    model = MLPClassifier(early_stopping=True)
    param_grid = [
        {'activation': ['identity', 'logistic','tanh','relu'], 'solver': ['sgd', 'adam'], 'hidden_layer_sizes': [(64,64),(32,32),(16,16)], 'tol': [1e-4,1e-5], 'max_iter': [50, 100, 200],},
        {'activation': ['identity', 'logistic','tanh','relu'], 'solver': ['lbfgs'], 'hidden_layer_sizes': [(8,8),(8,4),(4,4)], 'tol': [1e-4,1e-5], 'max_iter': [10,20, 40, 60],},
        ]
    gridsearch = GridSearchCV(model, param_grid, scoring='f1', n_jobs=-1, cv=4)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_
    ##Print the best parameters
    best_parameters = model.get_params()
    for param_name in sorted(best_parameters.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))
    ##Print the model scores
    preds = model.predict(X_test)
    print('Accuracy: %.4f' % accuracy_score(y_test, preds))
    print('Precision: %.4f' % precision_score(y_test, preds))
    print('Recall: %.4f' % recall_score(y_test, preds))
    print('F1: %.4f' % f1_score(y_test, preds))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write('User interrupt me! ;-) See you!\n')
        sys.exit(0)
