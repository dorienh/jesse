
import numpy as np
import pandas as pd
# df = pd.read_csv('../datasource/cc/histo_minute_daily_e_CCCAGG_f_BTC_t_USD_d_2019-07-21.csv', index_col = 'time', parse_dates=True)
# % histo_minute_daily_e_CCCAGG_f_BTC_t_USD_d_2019-07-21

#todo: load summarydf/
# f = '../analyses/dataset/hourly_cc/pivots_BCH-USD.csv'
f = '../analyses/dataset/daily/pivots_BTC-USD.csv'

# f = '../analyses/predictions/daily/pred_pivots_BTC-USD.csv'
df = pd.read_csv(f, sep=",", index_col=0)
print(df.head(5))

print(df.columns)

# X = df.iloc[:, :-1]
y = df.iloc[:, 10:12]

y = y.astype(int)
# y = df['Close']
X = df.iloc[:, np.r_[0:6]]

X_old = X
y_old = y
print(X.columns)
## todo - load all other features percentage and indicators
# load other currencies
# add batches
##todo change y variable to multiple 1-0
#todo change loss

print(y.head(5))
print(X.head(5))

from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()


X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y)

y_mm = y.to_numpy() #added

#first 200 for training

X_train = X_ss[:1800, :]
X_test = X_ss[1800:, :]

# y_train = y_mm[:1800, :]
# y_test = y_mm[1800:, :]
y_train = y_mm[:1800, :]
y_test = y_mm[1800:, :]


print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape)


import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable


X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))

a = X_train_tensors.shape[0]
# Their API takes a tensor of shape (sequence length, batch size, dim) if batch_first=False (default) and (batch size, sequence length, dim) if batch_first=True.

#reshaping to rows, timestamps, features

X_train_tensors_final = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))


X_test_tensors_final = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))  # hidden state

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))  # internal state

        # Propagate input through LSTM

        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        out = self.relu(hn)

        out = self.fc_1(out)  # first Dense

        out = self.relu(out)  # relu

        out = self.fc(out)  # Final Output
        out = self.sigmoid(out)

        return out


num_epochs = 100000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 6 #number of features
hidden_size = 15 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 2 #number of output classes


lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class


criterion = torch.nn.BCELoss() #MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm1.forward(X_train_tensors_final)  # forward pass
    optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)

    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, i.e backprop
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))




df_X_ss = ss.transform(X_old) #old transformers
df_y_mm = y_mm #.mm #.transform(y_old) #old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))



train_predict = lstm1(df_X_ss) #forward pass
data_predict = train_predict.data.numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()


import matplotlib.pyplot as plt
data_predict = mm.inverse_transform(data_predict) #reverse transformation
# dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=200, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actual Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show()

buffer = []
# a = len(df.index)
# while (len(buffer) + len(data_predict)) < a:
#     buffer.append(0)

# new_pred = buffer + data_predict

#todo note this include predictions for training data
df['Buy_pred'] = data_predict[:,0]
df['Sell_pred'] = data_predict[:,1]


f = '../analyses/predictions/daily/pred_pivots_BTC-USD.csv'

df.to_csv(f, index = True)

torch.save(lstm1.state_dict(), './model')