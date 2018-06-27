import os

import torch
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import DataLoader

from net.vgg5 import VGG5
from utils.parser import arg_parser
from utils.dataset import MnistTest




def predict(batches, model):
    indexes, images = batches
    images = Variable(images.contiguous().view(-1, 1, 28, 28))
    labels = model(images)
    labels = labels.max(1)[1]
    return indexes.numpy(), labels.data.numpy()

def main():
    args = arg_parser()
    model_path = args.model_path
    data_path = args.data_path
    vgg = VGG5()
    vgg.load_state_dict(torch.load(os.path.join(model_path, 'vgg.pth')))

    test_dataset = MnistTest(os.path.join(data_path, 'test.csv'))
    data_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)

    column_list = ['ImageId', 'Label']
    results = pd.DataFrame(columns=column_list)
    for batch in data_loader:
        result = predict(batch, vgg)
        result = pd.DataFrame(list(zip(result[0], result[1])), columns=column_list)
        results = results.append(result)
    
    results.to_csv('result.csv', index=False)

if __name__ == '__main__':
    main()