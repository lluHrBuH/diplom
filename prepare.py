#import h5py
#file = h5py.File('sift-128-euclidean.hdf5')
#print(file['train'][0])
#print(file['test'][0])
import numpy as np

classNumber = 1000
classExample = 100
shape = 128
eps = 32
trainSize = 20

XTest = np.zeros((classNumber*classExample, shape))
yTest = np.zeros(classNumber*classExample)


XTrain = np.zeros((classNumber*classExample*trainSize, shape))
yTrain = np.zeros(classNumber*classExample*trainSize)

for i in range(0, classNumber):
    base = np.random.random_integers(10000, size=(shape,))
    for j in range(0, classExample):
        delta = np.random.random_integers(-int(eps/shape**0.5),int(eps/shape**0.5), size=(shape,))
        XTest[i*classExample+j] = base + delta
        yTest[i*classExample+j] = i
    for j in range(0, classExample*trainSize):
        delta = np.random.random_integers(-int(eps / shape ** 0.5), int(eps / shape ** 0.5), size=(shape,))
        XTrain[i * classExample*trainSize + j] = base + delta
        yTrain[i * classExample*trainSize + j] = float(i)

    print(i)
np.save('X_train.npy', XTrain)
np.save('y_train.npy', yTrain)
np.save('X_test.npy', XTest)
np.save('y_test.npy', yTest)
