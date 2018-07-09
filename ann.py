import pyflann
import numpy
import annoy
import sklearn.neighbors
import sklearn.preprocessing
import hdidx
import os
#import pykgraph
from scipy.spatial.distance import pdist as scipy_pdist
from lopq import LOPQModel, LOPQSearcher
from n2 import HnswIndex
import nmslib
from scipy import stats

INDEX_DIR = 'indices'

class BaseANN(object):
    def fitClasses(self,X, y):
        y = y.reshape((-1, 1))
        self.classes_ = []
        self._y = numpy.empty(y.shape, dtype=numpy.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = numpy.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

    def fit(self, X):
        pass
    def query(self, q, n):
        return []
    def predict(self, q, n=20):
        neigh_ind = self.query(q, n)
        classes_ = self.classes_
        _y = self._y
        _y = self._y.reshape((-1, 1))

        n_outputs = len(classes_)
        n_samples = 1

        y_pred = numpy.empty((n_samples, n_outputs))
        for k, classes_k in enumerate(classes_):
            mode, _ = stats.mode(_y[neigh_ind, k])
            mode = numpy.asarray(mode.ravel(), dtype=numpy.intp)
            y_pred[:, k] = classes_k.take(mode)
        y_pred = y_pred.ravel()
        return y_pred

class Annoy(BaseANN):
    def __init__(self, n_trees, search_k):
        self._n_trees = n_trees
        self._search_k = search_k
        self.name = 'Annoy(n_trees=%d, search_k=%d)' % (n_trees, search_k)

    def fit(self, X):
        self._annoy = annoy.AnnoyIndex(X.shape[1], metric='euclidean')
        for i, x in enumerate(X):
            self._annoy.add_item(i, x.tolist())
        self._annoy.build(self._n_trees)

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)

class BruteForce(BaseANN):
    def __init__(self):
        self.name = 'BruteForce()'

    def fit(self, X):
        self._nbrs = sklearn.neighbors.NearestNeighbors(algorithm='brute', metric='euclidean')
        self._nbrs.fit(X)

    def query(self, v, n):
        return list(self._nbrs.kneighbors([v],
            return_distance = False, n_neighbors = n)[0])

    def query_with_distances(self, v, n):
        (distances, positions) = self._nbrs.kneighbors([v],
            return_distance = True, n_neighbors = n)
        return zip(list(positions[0]), list(distances[0]))

class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, precision=numpy.float32):
        self._precision = precision
        self.name = 'BruteForceBLAS()'

    def fit(self, X):
        lens = (X ** 2).sum(-1)  # precompute (squared) length of each vector
        self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)

    def query(self, v, n):
        return map(lambda index, _: index, self.query_with_distances(v, n))
    popcount = []
    for i in range(256):
      popcount.append(bin(i).count("1"))
    def query_with_distances(self, v, n):
        dists = self.lengths - 2 * numpy.dot(self.index, v)
        nearest_indices = numpy.argpartition(dists, n)[:n]  # partition-sort by distance, get `n` closest
        indices = [idx for idx in nearest_indices ]
        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, scipy_pdist([ep, ev], metric="euclidean")[0])
        return map(fix, indices)

class FLANN(BaseANN):
    def __init__(self,  target_precision):
        self._target_precision = target_precision
        self.name = 'FLANN(target_precision=%f)' % target_precision

    def fit(self, X):
        self._flann = pyflann.FLANN(target_precision=self._target_precision, algorithm='autotuned', log_level='info')
        self._flann.build_index(X)

    def query(self, v, n):
        return self._flann.nn_index(v, n)[0][0]


class Hdidx(BaseANN):
    def __init__(self, nsubq):
        self.name = 'Hdidx(nsubq={})'.format(nsubq)
        self._nsubq = nsubq
        self._index = None
    def fit(self, X):
        X = numpy.array(X)
        X = X.astype(numpy.float32)
        self._index = hdidx.indexer.IVFPQIndexer()
        self._index.build({'vals': X, 'nsubq': self._nsubq})
        self._index.add(X)

    def query(self, v, n):
        v = v.astype(numpy.float32)
        return self._index.search(v, n)


class KDTree(BaseANN):
    def __init__(self, leaf_size=20):
        self.name = 'KDTree(leaf_size=%d)' % leaf_size
        self._leaf_size = leaf_size
    def fit(self, X):
        self._tree = sklearn.neighbors.KDTree(X, leaf_size=self._leaf_size)
    def query(self, v, n):
        dist, ind = self._tree.query([v], k=n)
        return ind[0]
#
# class KGraph(BaseANN):
#     def __init__(self, P, index_params):
#         self.name = 'KGraph(P=%d)' % (P)
#         self._P = P
#         self._index_params = index_params
#     def fit(self, X):
#         if X.dtype != numpy.float32:
#             X = X.astype(numpy.float32)
#         self._kgraph = pykgraph.KGraph(X, "euclidean")
#         path = os.path.join(INDEX_DIR, 'kgraph-index-euclidean')
#         if os.path.exists(path):
#             self._kgraph.load(path)
#         else:
#             self._kgraph.build(**self._index_params)
#             if not os.path.exists(INDEX_DIR):
#               os.makedirs(INDEX_DIR)
#             self._kgraph.save(path)
#
#     def query(self, v, n):
#         if v.dtype != numpy.float32:
#             v = v.astype(numpy.float32)
#         result = self._kgraph.search(numpy.array([v]), K=n, threads=1, P=self._P)
#         return result[0]


class LOPQ(BaseANN):
    def __init__(self, v):
        m = 4
        self.name = 'LOPQ(v={}, m={})'.format(v, m)
        self._m = m
        self._model = LOPQModel(V=v, M=m)
        self._searcher = None

    def fit(self, X):
        X = numpy.array(X)
        X = X.astype(numpy.float32)
        self._model.fit(X)
        self._searcher = LOPQSearcher(self._model)
        self._searcher.add_data(X)

    def query(self, v, n):
        v = v.astype(numpy.float32)
        nns = self._searcher.search(v, quota=100)
        return nns

class N2(BaseANN):
    def __init__(self, m):
        threads = 8
        self.name = 'N2(m={}, threads={})'.format(m, threads)
        self._m = m
        self._threads = threads
        self._index = None

    def fit(self, X):
        X = numpy.array(X)
        X = X.astype(numpy.float32)
        self._index = HnswIndex(X.shape[1], "L2")
        for el in X:
            self._index.add_data(el)
        self._index.build(m=self._m, n_threads=self._threads)

    def query(self, v, n):
        v = v.astype(numpy.float32)
        nns = self._index.search_by_vector(v, n)
        return nns

class NmslibReuseIndex(BaseANN):
    @staticmethod
    def encode(d):
        return ["%s=%s" % (a, b) for (a, b) in d.iteritems()]

    def __init__(self, method_name, index_param, query_param):
        self._method_name = method_name
        self._index_param = NmslibReuseIndex.encode(index_param)
        self._query_param = NmslibReuseIndex.encode(query_param)
        self.name = 'Nmslib(method_name=%s, index_param=%s, query_param=%s)' % (
        self._method_name, self._index_param, self._query_param)
        self._index_name = os.path.join(INDEX_DIR,
                                        "nmslib_%s_%s_%s" % (self._method_name, "euclidean" , '_'.join(self._index_param)))

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        if self._method_name == 'vptree':
            # To avoid this issue:
            # terminate called after throwing an instance of 'std::runtime_error'
            # what():  The data_old size is too small or the bucket size is too big. Select the parameters so that <total # of records> is NOT less than <bucket size> * 1000
            # Aborted (core dumped)
            self._index_param.append('bucketSize=%d' % min(int(X.shape[0] * 0.0005), 1000))

        self._index = nmslib.init('l2', [], self._method_name, nmslib.DataType.DENSE_VECTOR,
                                  nmslib.DistType.FLOAT)

        for i, x in enumerate(X):
            nmslib.addDataPoint(self._index, i, x.tolist())

        if os.path.exists(self._index_name):
            print('Loading index from file')
            nmslib.loadIndex(self._index, self._index_name)
        else:
            nmslib.createIndex(self._index, self._index_param)
            nmslib.saveIndex(self._index, self._index_name)

        nmslib.setQueryTimeParams(self._index, self._query_param)

    def query(self, v, n):
        return nmslib.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        nmslib.freeIndex(self._index)


class NmslibNewIndex(BaseANN):
    def __init__(self, method_name, method_param):
        self._method_name = method_name
        self._method_param = NmslibReuseIndex.encode(method_param)
        self.name = 'Nmslib(method_name=%s, method_param=%s)' % (self._method_name, self._method_param)

    def fit(self, X):
        if self._method_name == 'vptree':
            self._method_param.append('bucketSize=%d' % min(int(X.shape[0] * 0.0005), 1000))

        self._index = nmslib.init('l2', [], self._method_name, nmslib.DataType.DENSE_VECTOR,
                                  nmslib.DistType.FLOAT)
        for i, x in enumerate(X):
            nmslib.addDataPoint(self._index, i, x.tolist())

        nmslib.createIndex(self._index, self._method_param)

    def query(self, v, n):
        return nmslib.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        nmslib.freeIndex(self._index)
