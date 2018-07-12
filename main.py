import numpy as np
from sklearn.model_selection import train_test_split
from searcher import Searcher
from plot import plotAllMetric, plotAllFreq



#X_train = np.load('X_train.npy')
#y_train = np.load('y_train.npy')
#X_test = np.load('X_test.npy')
#y_test = np.load('y_test.npy')


X = np.load('./data_vggface/embeddings.npy')
y = np.load('./data_vggface/labels.npy')

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)

#plotAllMetric(X_train, y_train, X_test, y_test)
plotAllFreq(X_train, y_train, X_test, y_test)
