import time
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


import utils.dataset as dataset
from net.vgg16 import VGG16

warnings.filterwarnings('ignore')



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/train.csv")

    args = parser.parse_args()
    return args.data_path

def main():

    data_path = arg_parser()
    mnist = dataset.MNIST(data_path)
    dataloader = DataLoader(mnist, batch_size=8, num_workers=4)


    vgg16 = VGG16()
    vgg16.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg16.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

    for epcho in range(30):
        print('training epcho {}'.format(epcho))
        for labels, images in dataloader:
            images = Variable(images.contiguous().view(-1, 1, 228, 228)).cuda()
            labels = Variable(labels.squeeze()).cuda()
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            print("epcho {}, loss {}".format(epcho, loss.data[0]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), "log")


#97

if __name__ == '__main__':
    main()