
from torch.utils.data import ConcatDataset
from torchvision.datasets import MNIST, CelebA, CIFAR10, EMNIST, FashionMNIST
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
import benchmark.data.dataset.transformer as tf
from benchmark.data.dataset.fcube import FCUBE
from benchmark.data.dataset.leaf import Shakespeare, Sent140
from benchmark.data.fed.inc_fds import split
from benchmark.data.partitioner.fcube import FCUBEPartitioner
from config.env import DATASET, LEAF, SEED
from utils.data.fed import DPFederatedDataset
from utils.data.partitioner import BasicPartitioner, get_target, get_data
from utils.nlp import ToFixedSeq
from utils.tool import func_name


def shakespeare(args: dict):
    root = f'{LEAF}/{func_name()}/data/'
    fds = Shakespeare(root)
    return fds


# mnist
def mnist(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        MNIST(root, transform=ToTensor(), train=True),
        MNIST(root, transform=ToTensor(), train=False),
    ])
    # 划分器
    dp = BasicPartitioner(get_target(ds), args)
    fds = DPFederatedDataset(ds, dp)
    return fds


def femnist(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    split = args.get('split', 'letters')
    ds = ConcatDataset([
        EMNIST(root, split=split, train=True, transform=ToTensor(), target_transform=tf.ToEMnistTarget(split)),
        EMNIST(root, split=split, train=False, transform=ToTensor(), target_transform=tf.ToEMnistTarget(split))
    ])
    dp = BasicPartitioner(get_target(ds), args)
    fds = DPFederatedDataset(ds, dp)
    return fds


def fmnist(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = ConcatDataset([
        FashionMNIST(root, train=True, transform=ToTensor()),
        FashionMNIST(root, train=False, transform=ToTensor())
    ])
    dp = BasicPartitioner(get_target(ds), args)
    fds = DPFederatedDataset(ds, dp)
    return fds


def fcube(args: dict):
    train_size = args.get('train_size', 50000)
    test_size = args.get('test_size', 10000)
    seed = args.get('seed', SEED)
    ds = ConcatDataset([
        FCUBE(train_size=train_size, seed=seed),
        FCUBE(test_size=test_size, seed=seed)
    ])
    fds = FCUBEPartitioner(get_data(ds), args)
    return fds

def sent140(args: dict):
    root = f'{LEAF}/{func_name()}/data/'
    fds = Sent140(root)
    return fds


def cifar10(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    transform = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = ConcatDataset([
        CIFAR10(root, train=True, transform=transform),
        CIFAR10(root, train=False, transform=transform),
    ])
    dp = BasicPartitioner(get_target(ds), args)
    fds = DPFederatedDataset(ds, dp)
    return fds


def celeba(args: dict):
    root = f'{DATASET}/{func_name()}/raw/'
    ds = CelebA(
        root, split='all',
        target_type='attr',
        transform=Compose([Resize(32), ToTensor()]),
        target_transform=tf.ToCelebaAttrTarget(1)
    )
    dp = BasicPartitioner(get_target(ds), args)
    fds = DPFederatedDataset(ds, dp)
    return fds
