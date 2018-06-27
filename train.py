import os
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable


import utils.dataset as dataset
#from net.vgg16 import VGG16
from net.vgg5 import VGG5
from utils.parser import arg_parser
from utils.dataset import MnistTrain

warnings.filterwarnings('ignore')




def main():

    args = arg_parser()
    data_path = args.data_path
    model_path = args.model_path
    batch_size = args.batch_size
    mnist = MnistTrain(os.path.join(data_path, 'train.csv'))
    dataloader = DataLoader(mnist, batch_size=batch_size, num_workers=4)


    vgg = VGG5()
    vgg.cuda()
    criterion = nn.CrossEntropyLoss()

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
            print(loss.data[0])

    torch.save(vgg.state_dict(), os.path.join(model_path, 'vgg.pth'))


#97

if __name__ == '__main__':
    main()