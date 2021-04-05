import torch
import torch.nn as nn

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout = 0.2)  # lstm
        self.fc_1 = nn.Linear(hidden_size*2, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)  # hidden state

        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(x.size(0),-1)  # reshaping the data for Dense layer next
        out = self.relu(hn)

        out = self.fc_1(out)  # first Dense

        out = self.relu(out)  # relu

        out = self.fc(out)  # Final Output
        out = self.sigmoid(out)

        return out