import torch
from torch.utils.data import DataLoader

import utils.dataset as dataset
from net.vgg16 import VGG16


mnist = dataset.MNIST('data/train.csv')
dataloader = DataLoader(mnist, batch_size = 4)


for data in dataloader:
    print(data)

