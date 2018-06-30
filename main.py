from algos import algosProd, algosTest
import numpy as np
import time
from sklearn import metrics

def initProd(X, y, algo='hnsw(nmslib)'):
    print('Start init')
    ann = algosProd[algo]
    ann.fitClasses(X, y)
    ann.fit(X)
    print('init done')
    return ann


X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
ann = initProd(X_train, y_train)

start=time.time()
y_pred = np.zeros(y_test.shape)
for idx in range(0,len(X_test)-1):
    y_pred[idx] = ann.predict(X_test[idx], n=30)

print('Precision:',
      metrics.precision_score(y_test, y_pred, average='macro'),
      ' Recall:',
      metrics.recall_score(y_test, y_pred, average='micro'))

print('Time per iter', (time.time() - start) / float(len(X_test)))
