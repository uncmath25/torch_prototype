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
        self._val_loader = DataLoader.build(is_train=False)
        self._model = TorchPrototypeNN()
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.001)

    def train(self):
        EPOCHS = 2
        best_vloss = 1_000_000.
        for epoch in range(EPOCHS):
            print(f'EPOCH {epoch + 1}:')
            # Make sure gradient tracking is on, and do a pass over the data
            self._model.train(True)
            last_loss = self._train_epoch()
            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self._model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self._val_loader):
                    vinputs, vlabels = vdata
                    voutputs = self._model(vinputs)
                    vloss = self._loss_fn(voutputs, vlabels)
                    running_vloss += vloss
            last_vloss = float(running_vloss / (i + 1))
            print(f'LOSS train {round(last_loss, 2)} valid {round(last_vloss, 2)}')
            # Track best performance, and save the model's state
            if last_vloss < best_vloss:
                best_vloss = last_vloss
                model_name = 'prototype_temp'
                torch.save(self._model.state_dict(), model_name)

    def _train_epoch(self):
        running_loss = 0.
        last_loss = 0.
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
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print(f'batch {i+1} loss: {round(last_loss, 2)}')
                running_loss = 0.
        return last_loss


def main():
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


def _eval():
    model = TorchPrototypeNN()
    model.load_state_dict(torch.load('prototype_temp'))
    model.eval()
    val_loader = DataLoader.build(is_train=False)
    idx = 0
    val_input = val_loader.dataset.tensors[0][idx:(idx + 1)]
    val_output = val_loader.dataset.tensors[1][idx]
    pred_output = model(val_input)
    print(f'actual output: {val_output}')
    print(f'pred output: {pred_output}')
    import torch.nn as nn
    print(f'pred output: {nn.Softmax(dim=1)(pred_output)}')


if __name__ == '__main__':
    # main()
    # _debug()
    _eval()
