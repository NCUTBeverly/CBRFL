from k_means_constrained.k_means_constrained_ import k_means_constrained
from sklearn.cluster import AgglomerativeClustering


def k_means(M, group_num, seed=None, max_size=None, min_size=None):
    _, labels, _ = k_means_constrained(
        M, n_clusters=group_num,
        size_min=min_size,
        size_max=max_size,
        random_state=seed
    )
    return labels


def agglomerative_clustering(M, group_num=2):
    clustering = AgglomerativeClustering(
        n_clusters=group_num,
        metric="precomputed",
        linkage="complete",
    )
    clustering.fit(M)
    return clustering.labels_

