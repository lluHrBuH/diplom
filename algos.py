from ann import *

algosProd = {
    'Annoy':Annoy(200, 1000),
    'BruteForce':BruteForce(),
    'BruteForceBLAS':BruteForceBLAS(),
    'FLANN':FLANN(0.95),
    'Hdidx':Hdidx(16),
    'KDTree':KDTree(200),
    #'KGraph':KGraph(32,{'reverse': -1}),
    'LOPQ':LOPQ(64),
    'N2':N2(64),
    'hnsw(nmslib)':NmslibReuseIndex(
        'hnsw',
        {"M": 20, "post": 2, "efConstruction": 400, 'skip_optimized_index' : 1 },
        {"ef": 15 },
    ),
    'MP-lsh(lshkit)':NmslibNewIndex('lsh_multiprobe', {"desiredRecall": 0.9, "H": 1200001, "T": 10, "L": 50, "tuneK": 10}),
}

algosTest = {
    'Annoy':{
        'constructor':Annoy,
        'args':[[100, 200, 400], [20000, 40000, 100000, 200000, 400000]]
    },
    'BruteForce':{
        'constructor':BruteForce,
        'args':[]
    },
    'BruteForceBLAS':{
        'constructor':BruteForceBLAS,
        'args':[]
    },
    'FLANN':{
        'constructor':FLANN,
        'args': [[0.97, 0.98, 0.99, 0.995]]
    },
    'Hdidx':{
        'constructor':Hdidx,
        'args':[[2, 4, 8]]
    },
    'KDTree':{
        'constructor':KDTree,
        'args':[[10, 20, 40, 50]]
    },
 #   'KGraph':{
 #       'constructor':KGraph,
 #       'args':[[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], {'reverse': -1}]
 #   },
    'LOPQ':{
        'constructor':LOPQ,
        'args':[[4, 8, 16]]
    },
    'N2':{
        'constructor':N2,
        'args':[[2, 4, 8, 10, 12, 16, 20, 32, 60, 64]]
    },
    'hnsw(nmslib)':{
        'constructor':NmslibReuseIndex,
        'args':[
            ['hnsw'],
            [
                {"M": 20, "post": 2, "efConstruction": 400},
                {"M": 32, "post": 2, "efConstruction": 400},
                {"M": 12, "post": 0, "efConstruction": 400},
                {"M": 4, "post": 0, "efConstruction": 400},
                {"M": 8, "post": 0, "efConstruction": 400}
            ],
            [
                {"ef": [ 90, 100, 120, 140, 160, 200,300, 400]},
                {"ef": [40, 50, 70, 80, 120, 200, 400]},
                {"ef": [ 40, 50, 70, 80, 120]},
                {"ef": [ 50, 70, 90, 120]},
                {"ef": [70, 90, 120, 160]}
            ]
        ]
    },
    'MP-lsh(lshkit)':{
        'constructor':NmslibNewIndex,
        'args':[['lsh_multiprobe'], [{"desiredRecall": [ 0.7, 0.6, 0.5,0.4, 0.3, 0.2, 0.1], "H": 1200001, "T": 10, "L": 50, "tuneK": 10}]]
    }
}