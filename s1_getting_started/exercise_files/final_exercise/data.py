import torch
import numpy as np
import os


def mnist():
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784) 

    path = os.path.abspath(os.path.join(os.getcwd(),'data', 'corruptmnist'))

    train_0 = np.load(os.path.join(path, 'train_0.npz'))
    train_1 = np.load(os.path.join(path, 'train_1.npz'))
    train_2 = np.load(os.path.join(path, 'train_2.npz'))
    train_3 = np.load(os.path.join(path, 'train_3.npz'))
    train_4 = np.load(os.path.join(path, 'train_4.npz'))
    train = train_0

    test = np.load(os.path.join(path, 'test.npz'))
    return train, test

