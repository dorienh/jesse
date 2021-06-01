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
class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
    
class WaveNet(nn.Module):
    def __init__(self, last_x_days=14, input_size=192, kernel_size=3, next_x_days=1,num_classes=1):
        super().__init__()
        #self.LSTM = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.wave_block1 = Wave_Block(last_x_days, 16, 12, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 8, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 4, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.wave_block5 = Wave_Block(128, 64, 4, kernel_size)
        self.wave_block6 = Wave_Block(64, next_x_days, 1, kernel_size)
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x = self.wave_block5(x)
        x = self.wave_block6(x)
        x = x.squeeze()
        x = self.relu(self.bn(x))  # relu
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


class WaveNet_n_to_1(object):
    def __init__(
        self, num_classes, input_size, seq_length, device
    ):
        self.model = WaveNet(num_classes=num_classes,last_x_days=seq_length,input_size=input_size,next_x_days=1)
        self.num_classes = num_classes
        self.device = device
    def train(
        self, epochs, dataloader, optimizer, criterion, checkpointdir="./wavenet_model.pth"
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
                target = target.to(self.device)
            
                data = data.float()
                target = target.float()
                optimizer.zero_grad()
                output = self.model(data)
                # print(output)
                # print(target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += (1 / (batch_idx + 1)) * (loss.data - train_loss)
            self.model.eval()
            for batch_idx, (data, target) in tqdm(
                enumerate(dataloader["valid"]), total=len(dataloader["valid"])
            ):
                
                data = data.to(self.device)
                target = target.to(self.device)
            
                data = data.float()
                target = target.float()
                output = self.model(data)  
                ind = torch.any(output.isnan(),dim=1)
                # print(torch.any(target.isnan(),dim=1))
                loss = criterion(output[~ind], target[~ind])
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

            try:
                custom_print(
                    f"AOC Curve: {roc_auc_score(y_test, np.round(preds[:,1]))}", header=True
                )
            except:
                custom_print(
                    f"Only one class present in y_true. ROC AUC score is not defined in that case.", header=True
                )

        fpr, tpr, thresholds = roc_curve(y_test, np.round(preds[:,1]),pos_label=2)
        try:
            roc_auc = roc_auc_score(y_test, np.round(preds[:,1]))
        except: roc_auc = 0
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
        
if __name__ == "__main__":
    model = WaveNet()
    b = model(torch.randn(64,14,192))