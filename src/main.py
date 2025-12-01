import torch

from data_loader import DataLoader
from model import TorchPrototypeNN
# from utils import Utils


class TorchPrototype:
    '''
    Motivated by PyTorch Training example:
    https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
    '''
    def setup(self):
        self._train_loader = DataLoader.build(is_train=True)
        self._model = TorchPrototypeNN()
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.001)

    def train(self):
        self._train_epoch()

    def _train_epoch(self):
        # running_loss = 0.
        # last_loss = 0.
        for i, data in enumerate(self._train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            # print(inputs.shape)
            # print(labels.shape)
            # print(labels)
            # Zero your gradients for every batch!
            self._optimizer.zero_grad()
            # Make predictions for this batch
            outputs = self._model(inputs)
            # print(outputs.shape)
            # print(outputs)
            # Compute the loss and its gradients
            loss = self._loss_fn(outputs, labels)
            loss.backward()
            # Adjust learning weights
            self._optimizer.step()
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


def main():
    # _debug()
    trainer = TorchPrototype()
    trainer.setup()
    trainer.train()


def _debug():
    # train_labels = DataLoader.get_labels(is_train=True)
    # print(train_labels.shape)
    # print(train_labels[:5])
    # test_labels = DataLoader.get_labels(is_train=False)
    # print(test_labels.shape)
    # print(test_labels[:5])
    # Utils._show_image(1)
    train_loader = DataLoader.build(is_train=True)
    print(train_loader.dataset)
    print(train_loader.dataset.tensors)
    print(train_loader.dataset.tensors[0].shape)
    print(train_loader.dataset.tensors[1].shape)


if __name__ == '__main__':
    main()
