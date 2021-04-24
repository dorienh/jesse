import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
)


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )  # lstm
        self.fc_1 = nn.Linear(hidden_size * 2, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # hidden state

        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size
        )  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        hn = hn.view(x.size(0), -1)  # reshaping the data for Dense layer next
        out = self.relu(hn)

        out = self.fc_1(out)  # first Dense

        out = self.relu(out)  # relu

        out = self.fc(out)  # Final Output
        out = self.sigmoid(out)

        return out


class Model_n_to_1(object):
    def __init__(
        self, num_classes, input_size, hidden_size, num_layers, seq_length, device
    ):
        self.model = LSTM1(num_classes, input_size, hidden_size, num_layers, seq_length)
        self.device = device

    def train(
        self, epochs, data, optimizer, criterion, checkpointdir="./lstm_model.pth"
    ):
        valid_loss_min = np.Inf
        self.model = self.model.to(self.device)
        self.model.float()
        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            # train the model
            self.model.train()

            for batch_idx, (data, target) in tqdm(enumerate(data["train"])):
                data = data.to(self.device)
                target = target.to(self.device)
                data = data.float()
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target.long())
                loss.backward()
                optimizer.step()
                train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)

            self.model.eval()
            for batch_idx, (data, target) in tqdm(enumerate(data["valid"])):
                data = data.to(self.device)
                target = target.to(self.device)
                data = data.float()
                output = self.model(data)  # .double())
                loss = criterion(output, target.long())
                valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            # print training/validation statistics
            print(
                "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                    epoch, train_loss, valid_loss
                )
            )
            if valid_loss < valid_loss_min:
                print(
                    "validation loss decreased from {:.6f} to {:.6f}. Model is being saved to {}.".format(
                        valid_loss_min, valid_loss, checkpointdir
                    )
                )

                valid_loss_min = valid_loss
                torch.save(self.model, checkpointdir)

    def load(self, checkpointdir):
        self.model.load_state_dict(torch.load(checkpointdir))
        self.model = self.model.to(device=self.device)

    def predict(self, testdata, target=None):
        self.model.eval()
        predictions = self.model(testdata.float()).detach().numpy()
        idxs = np.unique(np.where(np.isnan(predictions))[0]).tolist()
        preds = np.delete(predictions.copy(), idxs, axis=0)
        if target:
            y_test = np.delete(torch.clone(target), idxs, axis=0)
            print(f"Accuracy:  {accuracy_score(y_test, np.round(preds[:,1]))}")
            print(
                f"Confusion Matrix: \n{confusion_matrix(y_test, np.round(preds[:,1]))}"
            )
            print(f"Precision: {precision_score(y_test, np.round(preds[:,1]))}")
            print(f"AOC Curve: {roc_auc_score(y_test, np.round(preds[:,1]))}")

