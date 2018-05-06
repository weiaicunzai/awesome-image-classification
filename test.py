<<<<<<< HEAD



from net.vgg5 import VGG5
from utils.parser import arg_parser




def predict()
=======
import pandas
import numpy as np
import torch
from torch.autograd import Variable

# Constants
TRAIN_RATIO = 0.9
BATCH_SIZE = 8


# Define models
class CNN_Model(torch.nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        # Define conv layers
        # self.conv_layers = []
        self.conv1 = torch.nn.Conv2d(1, 8, 3) # img: 26*26
        self.conv2 = torch.nn.Conv2d(8, 16, 2, stride=2) # img: 13*13
        self.conv3 = torch.nn.Conv2d(16, 32, 3) # img: 11*11
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=2) # img: 5*5

        self.fully_conected = torch.nn.Linear(25 * 64, 10)
        self.logsoftmax = torch.nn.LogSoftmax()
        self.relu = torch.nn.ReLU()

    def forward(self, image, dropout_p=0.2):
        dropout = torch.nn.Dropout(dropout_p)

        image = image.float()
        image = self.conv1(image)
        image = self.relu(image)
        image = dropout(image)

        image = self.conv2(image)
        image = self.relu(image)
        image = dropout(image)

        image = self.conv3(image)
        image = self.relu(image)
        image = dropout(image)

        image = self.conv4(image)
        image = self.relu(image)
        image = dropout(image)

        logits = self.fully_conected(image.view(-1, 25 * 64))
        return self.logsoftmax(logits)

def prepare_examples(batch, labled=True):
    assert len(batch.shape) == 2

    if labled:
        assert batch.shape[1] == 785
        labels, images = np.split(batch, [1], axis=1)
        labels = Variable(torch.LongTensor(labels).squeeze())
        images = Variable(torch.from_numpy(images).contiguous().view(-1, 1, 28, 28))
        return labels, images
    else:
        assert batch.shape[1] == 784
        images = Variable(torch.from_numpy(batch).contiguous().view(-1, 1, 28, 28))
        return images


def train_batch(batch, model, criterion, optimizer):
    labels, images = prepare_examples(batch)
    log_prob_classes = model(images)
    loss = criterion(log_prob_classes, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.data[0] / len(batch)

def evaluate(batch, model):
    predictions = predict(batch.values, model).reshape(-1)
    labels = batch["label"].values

    assert labels.shape[0] == BATCH_SIZE
    assert predictions.shape[0] == BATCH_SIZE

    num_correct = np.equal(labels, predictions).sum()
    return num_correct

def predict(batch, model):
    assert batch.shape[0] == BATCH_SIZE
    if batch.shape[-1] == 784:
        images = prepare_examples(batch, labled=False)
    elif batch.shape[-1] == 785:
        _, images = prepare_examples(batch)

    log_probs = model(images, dropout_p=0)
    max_indicies = log_probs.max(1)[1]
    return max_indicies.data.numpy()

if __name__ == "__main__":
    # Load data into a pandas dataframe
    digit_data = pandas.read_csv("data/train.csv")
    break_point = int(len(digit_data) * TRAIN_RATIO)
    train_data = digit_data[:break_point]
    validation_data = digit_data[break_point:]

    # Start Training
    model = CNN_Model()
    learning_rate = 0.01
    criterion = torch.nn.NLLLoss()
    losses = []

    for epoch in range(5):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i:i+BATCH_SIZE].values
            losses.append(train_batch(batch, model, criterion, optimizer))

            if (i / BATCH_SIZE + 1) % 1000 == 0:
                print("Average Trainging Loss is {}".format(sum(losses) / len(losses)))
                losses = []
        learning_rate *= 0.7

    # Evaluated
    losses = []
    for i in range(0, len(validation_data), BATCH_SIZE):
        img = validation_data[i:i+BATCH_SIZE]
        losses.append(evaluate(img, model))
    print("Validation Loss is {:.2%}".format(sum(losses) / len(validation_data)))

torch.save(model.state_dict(), "model_params/model")
>>>>>>> 7539374dd08dff8a8c588a36a95d5606e5b38d6a
