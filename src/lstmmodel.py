import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
    roc_curve,
    recall_score
)

from utils import custom_print
import matplotlib.pyplot as plt


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
        self.bn1 = nn.BatchNorm1d(hidden_size * num_layers)
        self.fc_1 = nn.Linear(hidden_size * num_layers, 128)  # fully connected 1
        self.bn2 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,h_0, c_0):
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(
            x, (h_0, c_0)
        )  # lstm with input, hidden, and internal state
        hn = hn.view(x.size(0), -1)  # reshaping the data for Dense layer next
        out = self.bn1(self.relu(hn))

        out = self.bn2(self.fc_1(out))  # first Dense

        out = self.relu(out)  # relu

        out = self.fc(out)  # Final Output
        out = self.sigmoid(out)

        return out


class LstmModel_n_to_1(object):
    def __init__(
        self, num_classes, input_size, hidden_size, num_layers, seq_length, device
    ):
        self.model = LSTM1(num_classes, input_size, hidden_size, num_layers, seq_length)
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    def train(
        self, epochs, dataloader, optimizer, criterion, checkpointdir="./lstm_model.pth"
    ):
        valid_loss_min = np.Inf
        self.model.to(self.device)
        self.model.float()
        for epoch in range(1, epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            # train the model
            self.model.train()

            for batch_idx, (data, target) in tqdm(
                enumerate(dataloader["train"]), total=len(dataloader["train"])
            ):

                data = data.to(self.device)
                h_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(self.device)  # hidden state
                c_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(self.device)
                target = target.to(self.device)
                data = data.float()
                optimizer.zero_grad()
                output = self.model(data,h_0,c_0)
                loss = criterion(output, target.float())
                loss.backward()
                optimizer.step()
                train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)
            self.model.eval()
            for batch_idx, (data, target) in tqdm(
                enumerate(dataloader["valid"]), total=len(dataloader["valid"])
            ):
                data = data.to(self.device)
                h_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(self.device)  # hidden state
                c_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size).to(self.device)
                target = target.to(self.device)
                data = data.float()
                output = self.model(data,h_0,c_0)  
                loss = criterion(output, target.float())
                valid_loss += (1 / (batch_idx + 1)) * (loss.data - valid_loss)

            custom_print(
                "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                    epoch, train_loss, valid_loss
                ),
                True,
            )
            if valid_loss < valid_loss_min:
                print(
                    "validation loss decreased from {:.6f} to {:.6f}. Model is being saved to {}.".format(
                        valid_loss_min, valid_loss, checkpointdir
                    ),
                    True,
                )

                valid_loss_min = valid_loss
                torch.save(self.model.state_dict(), checkpointdir)

    def load(self, checkpointdir):
        self.model.load_state_dict(torch.load(checkpointdir))
        self.model = self.model.to(device=self.device)

    def predict(self, testdata, target=None):
        self.model.eval()
        testdata = testdata.to(self.device)
        predictions = self.model(testdata.float()).detach().numpy()
        idxs = np.unique(np.where(np.isnan(predictions))[0]).tolist()
        preds = np.delete(predictions.copy(), idxs, axis=0)
        
        if target != None:
            y_test = np.delete(torch.clone(target), idxs, axis=0)
            y_test = y_test.max(dim=1)[1]
            custom_print(
                f"Accuracy:  {accuracy_score(y_test, np.round(preds[:,1]))}",
                header=True,
            )
            custom_print(
                f"Confusion Matrix: \n{confusion_matrix(y_test, np.round(preds[:,1]))}",
                True,
            )
            custom_print(
                f"Precision: {precision_score(y_test, np.round(preds[:,1]))}",
                header=True,
            )
            custom_print(
                f"Recall: {recall_score(y_test, np.round(preds[:,1]))}",
                header=True,
            )
            custom_print(
                f"AOC Curve: {roc_auc_score(y_test, np.round(preds[:,1]))}", header=True
            )
        fpr, tpr, thresholds = roc_curve(y_test, np.round(preds[:,1]),pos_label=2)
        roc_auc = roc_auc_score(y_test, np.round(preds[:,1]))
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
