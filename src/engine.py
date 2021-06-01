from datapreprocess import Preprocess
from wavenetmodel import WaveNet_n_to_1
from lstmmodel import LstmModel_n_to_1
from loss import FocalLoss
from utils import custom_print

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

cols_to_pred = ["Sell_p40_a1"]

datadir = "../dataset/hourly_demo/"
last_x_days = 10
batch_size = 200
model_type = "Wavenet" # "Lstm" or "Wavenet"

process = Preprocess(datadir, y_cols, cols_to_pred,last_x_days=last_x_days)

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


dataloaders = process.prepare_data_loaders(batch_size)

# import _pickle as pickle
# with open('dataloaders.pickle', 'wb') as handle:
#     pickle.dump(dataloaders, handle, -1) #, protocol=pickle.HIGHEST_PROTOCOL
#     handle.close


num_epochs = 1  # 100 epochs
learning_rate = 0.003  # 0.001 lr

hidden_size = 8  # number of features in hidden state
num_layers = 3  # number of stacked lstm layers
num_classes = 2  # number of output classes

device = torch.device("cpu")

if model_type == "Wavenet":
    model = WaveNet_n_to_1(
        num_classes=num_classes,
        input_size=process.inputshape[2],
        seq_length=process.inputshape[1],
        device=device,
    )
elif model_type == 'Lstm':
    model = LstmModel_n_to_1(
        num_classes=num_classes,
        input_size=process.inputshape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=process.inputshape[1],
        device=device,
    )

# criterion = FocalLoss(gamma=2, alpha=0.25)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
checkpointdir = f"./{model_type}.pth"
custom_print(f"Training With {process.inputshape[2]} Features", header=True)
model.train(
    epochs=num_epochs,
    dataloader=dataloaders,
    optimizer=optimizer,
    criterion=criterion,
    checkpointdir=checkpointdir,
)

model.load(checkpointdir=checkpointdir)

model.predict(testdata=process.X_test, target=process.y_test)

