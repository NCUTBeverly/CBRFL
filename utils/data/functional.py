import warnings
import numpy as np


# 切分序列舍弃最后一段
# {0:np[1,2,3],1:np[3,4,5]}
def split_indices(num, indices):
    return {cid: idxs for cid, idxs in enumerate(np.split(indices, num)[:-1])}


# 平均切分样本，舍弃最后一段
# (101,3) np[33,33,33]
def balance_split(client_num, sample_num):
    return (np.ones(client_num) * int(sample_num / client_num)).astype(int)


def unbalance_lognormal_split(client_num, sample_num, sgm):
    if sgm != 0.0:
        client_sample_ratios = np.random.lognormal(
            mean=np.log(int(sample_num / client_num)),
            sigma=sgm,
            size=client_num
        )
        client_sample_nums = (client_sample_ratios / np.sum(client_sample_ratios) * sample_num).astype(int)
        # 修正下样本数,从第一开始找到可用于修正项进行修正
        diff = np.sum(client_sample_nums) - sample_num
        if diff > 0:
            client_sample_nums[np.argmax(client_sample_nums)] -= diff
        else:
            client_sample_nums[np.argmin(client_sample_nums)] += diff
    else:
        client_sample_nums = balance_split(client_num, sample_num)
    return client_sample_nums


def unbalance_dirichlet_split(client_num, sample_num, alpha, min_require_size=10):
    min_size, proportions = 0, None
    while min_size < min_require_size:
        proportions = np.random.dirichlet(np.repeat(alpha, client_num))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * sample_num)
    client_sample_nums = (proportions * sample_num).astype(int)
    return client_sample_nums


# iid
def iid_partition(client_sample_nums, sample_num):
    """Partition data indices in IID way given sample numbers for each client.
    Args:
        client_sample_nums : Sample numbers for each client.
        sample_num (int): Number of samples.
    Returns:
        dict: ``{ client_id: indices}``.
    """
    sample_indices = np.random.permutation(sample_num)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, sample_indices)
    return client_dict


# non_iid
def noniid_dirichlet_partition(targets, client_num, class_num, dirichlet_alpha, min_require_size=None):
    """

    Non-iid partition based on Dirichlet distribution. The method is from "hetero-dir" partition of
    `Bayesian Nonparametric Federated Learning of Neural Networks <https://arxiv.org/abs/1905.12022>`_
    and `Federated Learning with Matched Averaging <https://arxiv.org/abs/2002.06440>`_.

    This method simulates heterogeneous partition for which number of data points and class
    proportions are unbalanced. Samples will be partitioned into :math:`J` clients by sampling
    :math:`p_k \sim \text{Dir}_{J}(\alpha)` and allocating a :math:`p_{p,j}` proportion of the
    samples of class :math:`k` to local client :math:`j`.

    Sample number for each client is decided in this function.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        client_num (int): Number of clients for partition.
        class_num (int): Number of classes in samples.
        dirichlet_alpha (float): Parameter alpha for Dirichlet distribution.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    if min_require_size is None:
        min_require_size = class_num
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    min_size, sample_num, idx_batch = 0, targets.shape[0], []
    while min_size < min_require_size:
        idx_batch.clear()
        for _ in range(client_num):
            idx_batch.append([])
        # for each class in the dataset
        for k in range(class_num):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(
                np.repeat(dirichlet_alpha, client_num)
            )
            # Balance
            proportions = np.array([
                p * (len(idx_j) < sample_num / client_num)
                for p, idx_j in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    client_dict = {}
    for cid in range(client_num):
        np.random.shuffle(idx_batch[cid])
        client_dict[cid] = np.array(idx_batch[cid])
    return client_dict


def noniid_shard_partition(targets, client_num, shard_num):
    """Non-iid partition used in FedAvg `paper <https://arxiv.org/abs/1602.05629>`_.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        client_num (int): Number of clients for partition.
        shard_num (int): Number of shards in partition.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    sample_num = targets.shape[0]
    shard_size = int(sample_num / shard_num)
    if sample_num % shard_num != 0:
        warnings.warn("warning: length of dataset isn'v divided exactly by shard_num. "
                      "Some samples will be dropped.")
    shards_per_client = int(shard_num / client_num)
    if shard_num % client_num != 0:
        warnings.warn("warning: shard_num isn'v divided exactly by client_num. "
                      "Some shards will be dropped.")
    indices = np.arange(sample_num)
    # sort sample indices according to labels
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    # corresponding labels after sorting are [0, .., 0, 1, ..., 1, ...]
    sorted_indices = indices_targets[0, :]
    # permute shards idx, and slice shards_per_client shards for each client
    rand_perm = np.random.permutation(shard_num)
    num_client_shards = np.ones(client_num) * shards_per_client
    # sample index must be int
    num_cumsum = np.cumsum(num_client_shards).astype(int)
    # shard indices for each client
    client_shards_dict = split_indices(num_cumsum, rand_perm)
    # map shard idx to sample idx for each client
    client_dict = dict()
    for cid in range(client_num):
        shards_set = client_shards_dict[cid]
        current_indices = [
            sorted_indices[shard_id * shard_size: (shard_id + 1) * shard_size]
            for shard_id in shards_set
        ]
        client_dict[cid] = np.concatenate(current_indices, axis=0)
    return client_dict


def noniid_client_dirichlet_partition(client_sample_nums, targets, class_num, dirichlet_alpha, verbose=True):
    """Non-iid Dirichlet partition.

    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients ``client_sample_nums``.
    It's different from :func:`noniid_dirichlet_partition`.

    Args:
        targets (list or numpy.ndarray): Sample targets.
        class_num (int): Number of classes in samples.
        dirichlet_alpha (float): Parameter alpha for Dirichlet distribution.
        client_sample_nums: A numpy array consisting ``num_clients`` integer elements, each represents sample number of corresponding clients.
        verbose (bool, optional): Whether to print partition process. Default as ``True``.

    Returns:
        dict: ``{ client_id: indices}``.

    """
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    rand_perm = np.random.permutation(targets.shape[0])
    targets = targets[rand_perm]
    client_num = len(client_sample_nums)
    class_priors = np.random.dirichlet(alpha=[dirichlet_alpha] * class_num,
                                       size=client_num)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(class_num)]
    class_amount = [len(idx_list[i]) for i in range(class_num)]

    client_indices = [
        np.zeros(client_sample_nums[cid]).astype(np.int64)
        for cid in range(client_num)
    ]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(client_num)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = \
                idx_list[curr_class][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(client_num)}
    return client_dict


def noniid_label_skew_quantity_based_partition(targets, client_num, class_num, major_class_num):
    """
    Args:
        targets (np.ndarray): Labels od dataset.
        client_num (int): Number of clients.
        class_num (int): Number of unique classes.
        major_class_num (int): Number of classes for each client, should be less than ``num_classes``.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(client_num)]
    # only for major_classes_num < num_classes.
    # if major_classes_num = num_classes, it equals to IID partition
    times = [0 for _ in range(class_num)]
    contain = []
    for cid in range(client_num):
        current = [cid % class_num]
        times[cid % class_num] += 1
        j = 1
        while j < major_class_num:
            ind = np.random.randint(class_num)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)

    for k in range(class_num):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k, times[k])
        ids = 0
        for cid in range(client_num):
            if k in contain[cid]:
                idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                ids += 1
    client_dict = {cid: idx_batch[cid] for cid in range(client_num)}
    return client_dict


def noniid_fcube_synthetic_partition(data):
    """Feature-distribution-skew:synthetic partition.

    Synthetic partition for FCUBE dataset. This partition is from `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_.

    Args:
        data (np.ndarray): Data of dataset :class:`FCUBE`.

    Returns:
        dict: ``{ client_id: indices}``.
    """
    client_indices = [[] for _ in range(4)]
    for idx, sample in enumerate(data):
        p1, p2, p3 = sample
        if (p1 > 0 and p2 > 0 and p3 > 0) or (p1 < 0 and p2 < 0 and p3 < 0):
            client_indices[0].append(idx)
        elif (p1 > 0 and p2 > 0 and p3 < 0) or (p1 < 0 and p2 < 0 and p3 > 0):
            client_indices[1].append(idx)
        elif (p1 > 0 and p2 < 0 and p3 > 0) or (p1 < 0 and p2 > 0 and p3 < 0):
            client_indices[2].append(idx)
        else:
            client_indices[3].append(idx)
    client_dict = {cid: np.array(client_indices[cid]).astype(int) for cid in range(4)}
    return client_dict
