import time
<<<<<<< HEAD
=======
import argparse
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


import utils.dataset as dataset
<<<<<<< HEAD
#from net.vgg16 import VGG16
from net.vgg5 import VGG5
from utils.parser import arg_parser
=======
from net.vgg16 import VGG16
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a

warnings.filterwarnings('ignore')



<<<<<<< HEAD

def main():

    args = arg_parser()
    data_path = args.data_path
    batch_size = args.batch_size
    mnist = dataset.MnistTrain(data_path)
    dataloader = DataLoader(mnist, batch_size=batch_size, num_workers=4)


    vgg = VGG5()
    vgg.cuda()
    criterion = nn.CrossEntropyLoss()

    loss = 0
    
    para = list(vgg.parameters())
    for epcho in range(10):
        print('training epcho {}'.format(epcho))
        optimizer = optim.SGD(vgg.parameters(), lr=1e-2, momentum=0.9)
        for labels, images in dataloader:
            images = Variable(images.contiguous().view(-1, 1, 28, 28)).cuda()
            labels = Variable(labels.squeeze()).cuda()
            outputs = vgg(images)
            loss = criterion(outputs, labels)
            #print("epcho {}, loss {}".format(epcho, loss.data[0]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss.data[0] / batch_size)

    torch.save(vgg.state_dict(), "log/vgg.pth")
=======
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
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a


#97

if __name__ == '__main__':
    main()