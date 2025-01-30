from itertools import combinations
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise as pw
from sklearn.metrics.pairwise import cosine_similarity
from config.env import SEED


def cosine_dissimilarity(X, Y=None):
    return (1. - pw.cosine_similarity(X, Y)) / 2.


def kl_distances(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist[i][j] = kl_div(X[i], X[j]).mean()
    return dist


def js_distances(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist[i][j] = jensenshannon(X[i], X[j])
    return dist


def wasserstein_distances(X):
    n = X.shape[0]
    dist = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        dist[i][j] = wasserstein_distance(X[i], X[j])
    return dist


def edc(X, n_components=2, seed=SEED):
    decomposed_X = TruncatedSVD(
        n_components,
        algorithm='arpack',
        random_state=seed
    ).fit_transform(X.T)
    return cosine_similarity(X, decomposed_X.T)


def madc(X, correction=None):
    ''' calculate the data-driven measure such as MADD'''
    # Input: pm-> proximity matrix; Output: dm-> data-driven distance matrix
    # pm.shape=(n_clients, n_dims), dm.shape=(n_clients, n_clients)
    pm = cosine_similarity(X)
    n_clients, n_dims = pm.shape[0], pm.shape[1]
    """ Too Slow, and misunderstanding MADD. Deprecated
    for i in range(n_clients):
        for j in range(i+1, n_clients):
            for k in range(n_clients):
                if k !=i and k != j:
                    dm[i,j] = dm[j,i] = abs(np.sum((pm[i]-pm[k])**2)**0.5 - \
                        np.sum((pm[j]-pm[k])**2)**0.5)
    """
    # Fast version
    '''1, Get the repeated proximity matrix.
        We write Row1 = d11, d12, d13, ... ; and Row2 = d21, d22, d23, ...
        [   Row1    ]   [   Row2    ]       [   Rown    ]
        |   Row1    |   |   Row2    |       |   Rown    |
        |   ...     |   |   ...     |       |   ...     |
        [   Row1    ],  [   Row2    ], ..., [   Rown    ]
    '''
    row_pm_matrix = np.repeat(pm[:, np.newaxis, :], n_clients, axis=1)
    # print('row_pm', row_pm_matrix[0][0][:5], row_pm_matrix[0][1][:5])

    # Get the repeated colum proximity matrix
    '''
        [   Row1    ]   [   Row1    ]       [   Row1    ]
        |   Row2    |   |   Row2    |       |   Row2    |
        |   ...     |   |   ...     |       |   ...     |
        [   Rown    ],  [   Rown    ], ..., [   Rown    ]
    '''
    col_pm_matrix = np.tile(pm, (n_clients, 1, 1))
    # print('col_pm', col_pm_matrix[0][0][:5], col_pm_matrix[0][1][:5])

    # Calculate the absolute difference of two disstance matrix, It is 'abs(||u-z|| - ||v-z||)' in MADD.
    # d(1,2) = ||w1-z|| - ||w2-z||, shape=(n_clients,); d(x,x) always equal 0
    '''
        [   d(1,1)  ]   [   d(1,2)  ]       [   d(1,n)  ]
        |   d(2,1)  |   |   d(2,2)  |       |   d(2,n)  |
        |   ...     |   |   ...     |       |   ...     |
        [   d(n,1)  ],  [   d(n,2)  ], ..., [   d(n,n)  ]
    '''
    absdiff_pm_matrix = np.abs(col_pm_matrix - row_pm_matrix)  # shape=(n_clients, n_clients, n_clients)
    # Calculate the sum of absolute differences
    if correction is True:
        # We should mask these values like sim(1,2), sim(2,1) in d(1,2)
        mask = np.zeros(shape=(n_clients, n_clients))
        np.fill_diagonal(mask, 1)  # Mask all diag
        mask = np.repeat(mask[np.newaxis, :, :], n_clients, axis=0)
        for idx in range(mask.shape[-1]):
            # mask[idx,idx,:] = 1 # Mask all row d(1,1), d(2,2)...; Actually d(1,1)=d(2,2)=0
            mask[idx, :, idx] = 1  # Mask all 0->n colum for 0->n diff matrix,
        dm = np.sum(np.ma.array(absdiff_pm_matrix, mask=mask), axis=-1) / (n_dims - 2.0)
    else:
        dm = np.sum(absdiff_pm_matrix, axis=-1) / n_dims
    # print('absdiff_pm_matrix', absdiff_pm_matrix[0][0][:5])
    return dm  # shape=(n_clients, n_clients)


