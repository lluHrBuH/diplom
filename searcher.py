from algos import algosProd
import numpy as np
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Searcher():
    def __init__(self, X_train, y_train, algo='hnsw(nmslib)'):
        '''
        Init approximate nearest neighbors algorithm
        :param X_train: embedding for build indices/build graph. Vector (n,m)
        :param y_train: corresponding class for X_train. Vector (1,m)
        :param algo: name of algo to use. Look ann.py
        '''
        print('Start init', algo)
        self.ann = algosProd[algo]
        self.ann.fitClasses(X_train, y_train)
        self.ann.fit(X_train)
        self.name = algo
        print('init done', algo)

    def predict(self, q, n=10):
        '''
        Predict class of vector q using approximate n-nearest neighbors. Return predicted class
        :param q: vector to classification
        :param n: number of neighbors to use
        :return: predicted class of vector q
        '''
        return self.ann.predict(q, n)


