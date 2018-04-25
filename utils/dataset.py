import pandas as pd
import numpy as np
from skimage import 

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable




class MNIST(Dataset):

    def __init__(self, csv_path):
        self.dataset = pd.read_csv(csv_path)

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset.iloc[index].values
        print(data[1:].dtype)
        print(type(data[1:]))
        label, image = np.split(data, [1])
        label = Variable(torch.Tensor(label))
        image = Variable(torch.Tensor(image).contiguous().view(-1, 1, 228, 228))
        return label, image

