from algos import algosProd
import numpy as np
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from searcher import Searcher

def getNeighborFreq( searcher, X_test, y_test, minN = 2, maxN = 10 ):
    '''
        get Precision, Recall and frequency for one algo
        :param searcher: Searched object
        :param X_test: embedding for test. Vector (k,m)
        :param y_test: corresponding class for X_test. Vector (k, )
        :param minN: min number of nearest neighbor
        :param maxN: max number of nearest neighbor
        :return: precision, recall, frequency
        '''
    n = np.arange(minN, maxN, 1)
    frequency = np.zeros(maxN-minN)
    precision = np.zeros(maxN - minN)
    recall = np.zeros(maxN - minN)
    for n_i in n:
        print(n_i-minN)
        y_pred = np.zeros(y_test.shape)
        start  = time.time()
        for idx in range(0, len(X_test) - 1):
            y_pred[idx] = searcher.predict(X_test[idx], n=n_i)
        precision[n_i - minN] = metrics.precision_score(y_test, y_pred, average='macro')
        recall[n_i - minN] = metrics.recall_score(y_test, y_pred, average='micro')
        frequency[n_i-minN] = float(len(X_test)) / float(time.time() - start)
    return (precision, recall, frequency)

def plotAllFreq(X_train, y_train, X_test, y_test, min_n = 2, max_n = 15):
    '''
    Plot Precision, Recall / frequency for all algo in algos.py/algoProd
    :param X_train: embedding for build indices/build graph. Vector (n,m)
    :param y_train: corresponding class for X_train. Vector (n, )
    :param X_test:  embedding for test. Vector (k,m)
    :param y_test:  corresponding class for X_test. Vector (k, )
    :param min_n: min number of nearest neighbor
    :param max_n: max number of nearest neighbor
    :return: nothing
    '''
    fig_precision, ax_precision = plt.subplots()
    ax_precision.set(xlabel='Item per second', ylabel='Precision',
                    title='Precision / Item per second(1/sec)')
    ax_precision.grid()
    fig_recall, ax_recall = plt.subplots()
    ax_recall.set(xlabel='Item per second', ylabel='Recall',
                  title='Recall / Item per second (1/sec)')
    ax_recall.grid()

    algos = []
    for algo in algosProd:
        try:
            algos.append(Searcher(X_train, y_train, algo))
        except:
            print('Algo init failed :(', algo)
            pass

    for searcher in algos:
        print('Start algo test:', searcher.name)
        (precision, recall, freq) = getNeighborFreq(searcher, X_test, y_test, min_n, max_n)
        ax_precision.plot(freq, precision,alpha=0.6,label=searcher.name,  linewidth=2)
        ax_recall.plot(freq, recall,alpha=0.6,label=searcher.name,  linewidth=2)
    ax_recall.legend(loc='best')
    ax_precision.legend(loc='best')

    fig_precision.savefig('precision_time.png', dpi=600)
    fig_recall.savefig('recall_time.png', dpi=600)
    #plt.show()



def getNeighborMetric( searcher, X_test, y_test, minN = 2, maxN = 10):
    '''
    get Precision, Recall for one algo
    :param searcher: Searched object
    :param X_test: embedding for test. Vector (k,m)
    :param y_test: corresponding class for X_test. Vector (k, )
    :param minN: min number of nearest neighbor
    :param maxN: max number of nearest neighbor
    :return: precision, recall
    '''
    start = time.time()
    print('Start plotNeighborMetric')
    n = np.arange(minN, maxN, 1)
    precision = np.zeros(maxN-minN)
    recall = np.zeros(maxN - minN)
    for n_i in n:
        print(n_i-minN)
        y_pred = np.zeros(y_test.shape)
        for idx in range(0, len(X_test) - 1):
            y_pred[idx] = searcher.predict(X_test[idx], n=n_i)
        precision[n_i-minN] = metrics.precision_score(y_test, y_pred, average='macro')
        recall[n_i-minN] = metrics.recall_score(y_test, y_pred, average='micro')
    print(searcher.name, (time.time()-start)/float(len(n)*len(X_test)))
    return(precision, recall)



def plotAllMetric(X_train, y_train, X_test, y_test, min_n = 2, max_n = 15):
    '''
    Plot Precision, Recall / Number of nearest neighbor for all algo in algos.py/algoProd
    :param X_train: embedding for build indices/build graph. Vector (n,m)
    :param y_train: corresponding class for X_train. Vector (n, )
    :param X_test:  embedding for test. Vector (k,m)
    :param y_test:  corresponding class for X_test. Vector (k, )
    :param min_n: min number of nearest neighbor
    :param max_n: max number of nearest neighbor
    :return: nothing
    '''
    fig_precision, ax_precision = plt.subplots()
    ax_precision.set(xlabel='Nearest neighbor', ylabel='Precision',
                    title='Precision / Nearest neighbor')
    ax_precision.grid()
    fig_recall, ax_recall = plt.subplots()
    ax_recall.set(xlabel='Nearest neighbor', ylabel='Recall',
                  title='Recall / Nearest neighbor')
    ax_recall.grid()

    algos = []
    for algo in algosProd:
        try:
            algos.append(Searcher(X_train, y_train, algo))
        except:
            print('Algo init failed :(', algo)
            pass

    n = np.arange(min_n, max_n, 1)
    for searcher in algos:
        print('Start algo test:', searcher.name)
        (precision, recall) = getNeighborMetric(searcher, X_test, y_test, min_n, max_n)
        ax_precision.plot(n, precision,alpha=0.6,label=searcher.name,  linewidth=2)
        ax_recall.plot(n, recall,alpha=0.6,label=searcher.name,  linewidth=2)
    ax_recall.legend(loc='best')
    ax_precision.legend(loc='best')

    fig_precision.savefig('precision.png', dpi=600)
    fig_recall.savefig('recall.png', dpi=600)
    #plt.show()

