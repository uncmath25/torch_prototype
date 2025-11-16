import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from data_loader import DataLoader


class TorchPrototype:
    '''
    Motivated by PyTorch Training example:
    https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    '''
    @staticmethod
    def train():
        model = TorchPrototypeNN()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        # running_loss = 0.
        # last_loss = 0.
        train_loader = DataLoader.build(is_train=True)
        for i, data in enumerate(train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # print(inputs.shape)
            # print(labels.shape)
            # print(labels)
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # print(outputs.shape)
            # print(outputs)
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            optimizer.step()
            # # Gather data and report
            # running_loss += loss.item()
            # if i % 1000 == 999:
            #     last_loss = running_loss / 1000 # loss per batch
            #     print('  batch {} loss: {}'.format(i + 1, last_loss))
            #     tb_x = epoch_index * len(training_loader) + i + 1
            #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            #     running_loss = 0.
        print(i)
        print(loss)

    @staticmethod
    def _show_image(idx):
        # 60000, 28, 28
        images = DataLoader.get_images(is_train=True)
        for row in images[idx]:
            print(row)
        Image.fromarray(images[idx]).show()
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        # 28, 60000, 28
        images = trans(images)
        image = (images[:, idx, :].T.numpy()*255).astype(np.uint8)
        for row in image:
            print(row)
        Image.fromarray(image).show()


class TorchPrototypeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(28, 28, 100)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        # x = self.conv(x)
        # x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x

def main():
    # _debug()
    TorchPrototype.train()


def _debug():
    # train_labels = DataLoader.get_labels(is_train=True)
    # print(train_labels.shape)
    # print(train_labels[:5])
    # test_labels = DataLoader.get_labels(is_train=False)
    # print(test_labels.shape)
    # print(test_labels[:5])
    # TorchPrototype._show_image(1)
    train_loader = DataLoader.build(is_train=True)
    print(train_loader.dataset)
    print(train_loader.dataset.tensors)
    print(train_loader.dataset.tensors[0].shape)
    print(train_loader.dataset.tensors[1].shape)


if __name__ == '__main__':
    main()
