import pandas as pd
import numpy as np
<<<<<<< HEAD
#from skimage import io
#from skimage.transform import resize
=======
from skimage import io
from skimage.transform import resize
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a

import torch
from torch.utils.data import Dataset




<<<<<<< HEAD
class MnistTrain(Dataset):
=======
class MNIST(Dataset):
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a

   # def _vectorized(self, label):
   #     vector = np.zeros((1,10), dtype=np.int64)
   #     vector[0, label] = 1
   #     return vector

<<<<<<< HEAD
    def __init__(self, csv_train_path):
        self.dataset = pd.read_csv(csv_train_path)
=======
    def __init__(self, csv_path):
        self.dataset = pd.read_csv(csv_path)
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a

    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset.iloc[index].values
        label, image = np.split(data, [1])
        #label = self._vectorized(label)
<<<<<<< HEAD
        image = np.reshape(image, (28, 28)).astype(np.float32)
        #image = resize(image, (228, 228)).astype(np.float32)
        return label, image


class MnistTest(Dataset):

    def __init__(self, csv_test_path):
        self.dataset = pd.read_csv(csv_test_path)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image = self.dataset.iloc[index].values
        image = np.reshape(image, (28, 28)).astype(np.float32)
        return image
=======
        image = np.reshape(image, (28, 28))
        image = resize(image, (228, 228)).astype(np.float32)
        return label, image

>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a
