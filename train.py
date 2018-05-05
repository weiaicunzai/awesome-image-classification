import time
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


import utils.dataset as dataset
#from net.vgg16 import VGG16
from net.vgg5 import VGG5

warnings.filterwarnings('ignore')



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="data/train.csv")

    args = parser.parse_args()
    return args.data_path

def main():

    data_path = arg_parser()
    mnist = dataset.MNIST(data_path)
    dataloader = DataLoader(mnist, batch_size=32, num_workers=4)


    vgg = VGG5()
    vgg.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg.parameters(), lr=1e-2, momentum=0.9)

    loss = 0
    for epcho in range(10):
        print('training epcho {}'.format(epcho))
        for labels, images in dataloader:
            images = Variable(images.contiguous().view(-1, 1, 28, 28)).cuda()
            labels = Variable(labels.squeeze()).cuda()
            outputs = vgg(images)
            loss = criterion(outputs, labels)
            #print("epcho {}, loss {}".format(epcho, loss.data[0]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss.data[0])

    torch.save(vgg.state_dict(), "log/vgg.pth")


#97

if __name__ == '__main__':
    main()