from model import LSTM1
from loss import FocalLoss
import os
import torch
import ml_tools as tools
import numpy as np
import pandas as pd

dr = '../dataset/daily/'
y_cols=['Top_p15_a4',	'Btm_p15_a4',	'Buy_p15_a4',	'Sell_p15_a4',	'Top_p40_a1',	
            'Btm_p40_a1',	'Buy_p40_a1',	'Sell_p40_a1',	'ODR',	'Top',	'Btm',	
            'Trend','WM','last_pivot']
#set the cols to be predicted here
cols_to_pred=['Buy_p15_a4']
#whether to add the crypto label as a feature or not
crypto_feature = True
pd.set_option('mode.chained_assignment', None)
df_list, crypto_ids = tools.preprocess_data(datadir=dr)

X_list, y_list = tools.separate_input_output(df_list,y_cols, cols_to_pred, crypto_ids, add_crypto_id = crypto_feature)
#list of all the train,val and test inputs and outputs 

#filter_data is an optional step that excludes if there are very few rows in a df (e.g. <1000)
filter_data = True
X_train_list, y_train_list, X_test_list, y_test_list, X_val_list, y_val_list = \
tools.split_data(X_list,y_list, filter_data, test_ratio = 0.2, val_ratio = 0.1)

X_train_mm, scalers = tools.normalize(X_train_list, crypto_id=crypto_feature) #normalization is done only on the training data
X_train_reduced, pcas = tools.dimensionality_reduction(X_train_mm, factor = 10) 
X_test_reduced, X_val_reduced = tools.prepare_test_val_set(X_test_list, X_val_list, scalers, pcas, crypto_id = crypto_feature)

X_train_tensors, X_test_tensors, X_val_tensors, y_train_tensors, y_test_tensors, y_val_tensors = \
tools.prepare_tensors(X_train_reduced, X_test_reduced, X_val_reduced, y_train_list, y_test_list, y_val_list)

loaders = tools.prepare_data_loaders(X_train_tensors, y_train_tensors, X_val_tensors, y_val_tensors, X_test_tensors, y_test_tensors, batch_size = 64, shuffle = True)

num_epochs = 10 #100 epochs
learning_rate = 0.003 #0.001 lr

input_size = 15 #number of features (this does not include date)
hidden_size = 8 #number of features in hidden state
num_layers = 2  #number of stacked lstm layers
num_classes = 2 #number of output classes

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors.shape[1]) #our lstm class

criterion = FocalLoss(gamma = 2,alpha = 0.25) 
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

print('Starting Training')

lstm1 = tools.train(num_epochs,lstm1, loaders, optimizer, criterion)

print('Finished Training!')

#lstm1 = torch.load(PATH)

lstm1.eval()
predictions=lstm1(X_test_tensors.float()).detach().numpy()

#There are 2 null predictions that I couldn't resolve, so excluding them.
idxs = np.unique(np.where(np.isnan(predictions))[0]).tolist()
preds=np.delete(predictions.copy(),idxs, axis = 0)
y_test=np.delete(torch.clone(y_test_tensors),idxs, axis = 0)
  
#Calling this function will print all the performance metrics
tools.evaluate(preds, y_test)