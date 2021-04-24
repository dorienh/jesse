from preprocess import Preprocess
from model import Model_n_to_1
from loss import FocalLoss

import torch

y_cols = [
    "Top_p15_a4",
    "Btm_p15_a4",
    "Buy_p15_a4",
    "Sell_p15_a4",
    "Top_p40_a1",
    "Btm_p40_a1",
    "Buy_p40_a1",
    "Sell_p40_a1",
    "ODR",
    "Top",
    "Btm",
    "Trend",
    "WM",
    "last_pivot",
]

cols_to_pred = ["Buy_p15_a4"]

datadir = "../dataset/daily/"

process = Preprocess(datadir, y_cols, cols_to_pred)

# NOTE Uncomment BELOW and see how this work
# X_train,y_train,X_test,y_test,X_val,y_val = process.process()
# print(f"X_train Shape {X_train.shape}")
# print(f"X_test Shape {X_test.shape}")
# print(f"X_val Shape {X_val.shape}")
# print(f"y_train Shape {y_train.shape}")
# print(f"y_val Shape {y_val.shape}")
# print(f"y_test Shape {y_test.shape}")
# X_train,y_train,X_test,y_test,X_val,y_val = process.prepare_tensors()
# print(f"X_train Shape {X_train.shape}")
# print(f"X_test Shape {X_test.shape}")
# print(f"X_val Shape {X_val.shape}")
# print(f"y_train Shape {y_train.shape}")
# print(f"y_val Shape {y_val.shape}")
# print(f"y_test Shape {y_test.shape}")

dataloaders = process.prepare_data_loaders(batch_size=64)

num_epochs = 10  # 100 epochs
learning_rate = 0.003  # 0.001 lr

input_size = 14  # number of features (this does not include date)
hidden_size = 8  # number of features in hidden state
num_layers = 2  # number of stacked lstm layers
num_classes = 2  # number of output classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model_n_to_1(
    num_classes=num_classes,
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    seq_length=process.inputshape,
    device=device,
)

criterion = FocalLoss(gamma=2, alpha=0.25)
optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
checkpointdir = "./lstm_model.pth"
model.train(
    epochs=5,
    data=dataloaders,
    optimizer=optimizer,
    criterion=criterion,
    checkpointdir=checkpointdir,
)

model.load(checkpointdir=checkpointdir)

model.predict(testdata=process.X_test, target=process.y_test)
