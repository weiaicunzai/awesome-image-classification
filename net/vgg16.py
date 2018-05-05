import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16




class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        #vgg16 configuration D
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(), #read dropout paper
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

        #self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = nn.Softmax()(x)
        return x

   # def _initialize_weights(self):
   #     for m in self.modules():
   #         if isinstance(m, nn.Conv2d):
   #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
   #             m.weight.data.normal_(0, math.sqrt(2. / n))
   #             if m.bias is not None:
   #                 m.bias.data.zero_()
   #         elif isinstance(m, nn.BatchNorm2d):
   #             m.weight.data.fill_(1)
   #             m.bias.data.zero_()
   #         elif isinstance(m, nn.Linear):
   #             m.weight.data.normal_(0, 0.01)
   #             m.bias.data.zero_()

#print(VGG16())
#class Test(nn.Module):
#    def __init__(self):
#        super().__init__()
        #self.features = nn.S


#test = torch.zeros(2, 1, 228, 228)
##print(test)
###print(test)
#test = Variable(test)
#vgg16 = VGG16()
##print(vgg16.forward(test))
#print(nn.LogSoftmax()(Variable(torch.Tensor([0,10,0,0]))))

#test = torch.Tensor([1, 2, 3])
#print(test)
#print(nn.LogSoftmax()(Variable(test)))
#print(nn.Conv2d(1, 512, kernel_size=3, padding=1)(test))
#print(vgg16(pretrained=False))

#w = nn.Conv2d(1, 1, kernel_size=3, padding=1)
#print(list(w.parameters()))
#print(torch.Tensor(3, 3))