#!pip install tsfresh scipy>=1.5 		##Install these libraries if not there

import os
import pandas as pd
import numpy as np
import datetime
import tsfresh.utilities.dataframe_functions as df_utilities
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

#read all the files into a list of dataframes
def preprocess_data(datadir='./jesse/dataset/daily/'):
    #datadir: where all the data files are
    #add_crypto_id: whether to use crypto_ids as a feature
    files = [os.path.join(datadir,f) for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]
        
    df_files=[]
    df_files=[pd.read_csv(f) for f in files]
    crypto_ids = []
    for file in files:
        z= file.split('/')
        z =z[len(z)-1].split('.')[0].split('_')[1]
        crypto_ids.append(z)
    #fill all NaN and Inf values
    for df in df_files:
        df.fillna(method='ffill',inplace=True)
        dates=df.date
        df.drop(columns=['date'],inplace=True)
        df*=1.0                     #This indirectly maps all BOOL values to 1's and 0's of type float64
        df_utilities.impute(df)     #This function maps inf->max and -inf->min for each column
        df.insert(0,column='date',value=dates)
        
    return df_files, crypto_ids
  
def separate_input_output(df_list,y_cols, y, crypto_ids, add_crypto_id = True):
    outputs=[]  #This will contain all the output columns for each df
    inputs=[]   #This will contain all the input columns for each df

    for i,df in enumerate(df_list):
        if add_crypto_id:
            le=LabelEncoder()
            le.fit(crypto_ids)
    
            if 'CryptoID' in df.columns:
                del df['CryptoID']
    
            df.insert(1,'CryptoID',pd.Series(le.transform([crypto_ids[i]]*len(df))))    #adding unique id's for cryptos as an additional feature 
                                                                        #to make sure our model learns the uniqueness of each crypto
    #This step is to take care of any change in the sequence of columns in different files
        if i==0:
            cols=df.columns
            df = df.reindex(columns=cols)

        outputs.append(df[y])
        inputs.append(df[[col for col in df.columns if col not in y_cols]])
    return inputs, outputs   
    

def split_data(X_list,y_list, filter_data = True, test_ratio = 0.2, val_ratio = 0.1):
    #This is an additional step to filter out any crypto df with less than a significant amount of data (atleast 1000 rows).
    #Completely optional
    if filter_data:
        X_above_1k = []
        y_above_1k = []
    
        for i in range(len(X_list)):
            if len(X_list[i]) >= 1000:
                X_above_1k.append(X_list[i])
                y_above_1k.append(y_list[i])
        X_list = X_above_1k
        y_list = y_above_1k
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    X_val_list = []
    y_val_list = []

    for i in range(len(X_list)):
        X_train, X_test, y_train, y_test = train_test_split(
            X_list[i], y_list[i], test_size = test_ratio, random_state = 42)      #20% for test
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size = val_ratio/(1-test_ratio), random_state = 42)    #0.1/0.8 = 0.125 for val, since 0.8*0.125=10% 
        X_train_list.append(X_train)
        y_train_list.append(y_train)
        X_test_list.append(X_test)
        y_test_list.append(y_test)
        X_val_list.append(X_val)
        y_val_list.append(y_val)
    return X_train_list, y_train_list, X_test_list, y_test_list, X_val_list, y_val_list
  

def normalize(inputs, crypto_id = True):
    scalers=[]   #list of scalers
    inputs_mm=[]
    for i in range(len(inputs)):
        scaler = MinMaxScaler()
        if crypto_id:
            input_scaled=scaler.fit_transform(inputs[i].copy().iloc[:,2:]) #leaving out date & cryptoid column
            inputs_mm.append(inputs[i])
            inputs_mm[i].iloc[:,2:]=input_scaled
        else:
            input_scaled=scaler.fit_transform(inputs[i].copy().iloc[:,1:]) #leaving out date column
            inputs_mm.append(inputs[i])
            inputs_mm[i].iloc[:,1:]=input_scaled
        scalers.append(scaler)
    return inputs_mm,scalers

def dimensionality_reduction(inputs, factor = 10, crypto_id=True):
    #inputs is the scaled training set
    #factor is the final number of momentum indicators we want (i.e. date, cryptoID, excluding open,high, low and close)
    X_pca = [] 
    reduced_data = []
    for data in inputs:
        if 'date' in data.columns:
            data.index=data.date
            del data['date']
        pca = PCA(n_components=factor)
        if crypto_id:
            non_reduced = np.array(data.iloc[:,:5])     #saving columns cryptoID, open,high, low and close
            pca.fit(data.iloc[:,5:])
            reduced = pca.transform(data.iloc[:,5:])
            reduced = np.append(non_reduced,reduced,axis=1)
        else:
            non_reduced = np.array(data.iloc[:,:4])     #saving columns open,high, low and close
            pca.fit(data.iloc[:,4:])
            reduced = pca.transform(data.iloc[:,4:])
            reduced = np.append(non_reduced,reduced,axis=1)
        X_pca.append(pca)
        reduced_data.append(reduced)
    return reduced_data, X_pca
  
def prepare_test_val_set(inputs_test, inputs_val, scalers, pcas,crypto_id = True):
    X_test_mm = []
    X_test_processed = []
    X_val_mm = []
    X_val_processed = []
    for i,df in enumerate(inputs_test):
        if 'date' in df.columns:
            df.index=df.date
            del df['date']
        if crypto_id:
            test_mm = scalers[i].transform(df.copy().iloc[:,1:])     #leaving out cryptoID column
            X_test_mm.append(df)
            X_test_mm[i].iloc[:,1:]=test_mm
            non_reduced = np.array(X_test_mm[i].iloc[:,:5])     #saving columns cryptoID, open,high, low and close
            reduced = pcas[i].transform(X_test_mm[i].iloc[:,5:])
            reduced = np.append(non_reduced,reduced,axis=1)
        else:
            test_mm = scalers[i].transform(df.copy())
            X_test_mm.append(df)
            X_test_mm[i]=test_mm
            non_reduced = np.array(X_test_mm[i].iloc[:,:4])     #saving columns open,high, low and close
            reduced = pcas[i].transform(X_test_mm[i].iloc[:,4:])
            reduced = np.append(non_reduced,reduced,axis=1)
    
        X_test_processed.append(reduced)
  
    for i,df in enumerate(inputs_val):
        if 'date' in df.columns:
            df.index=df.date
            del df['date']
        if crypto_id:
            val_mm = scalers[i].transform(df.copy().iloc[:,1:])     #leaving out cryptoID column
            X_val_mm.append(df)
            X_val_mm[i].iloc[:,1:]=val_mm
            non_reduced = np.array(X_val_mm[i].iloc[:,:5])     #saving columns cryptoID, open,high, low and close
            reduced = pcas[i].transform(X_val_mm[i].iloc[:,5:])
            reduced = np.append(non_reduced,reduced,axis=1)
        else:
            val_mm = scalers[i].transform(df.copy())
            X_val_mm.append(df)
            X_val_mm[i]=val_mm
            non_reduced = np.array(X_val_mm[i].iloc[:,:4])     #saving columns open,high, low and close
            reduced = pcas[i].transform(X_val_mm[i].iloc[:,4:])
            reduced = np.append(non_reduced,reduced,axis=1)
    
        X_val_processed.append(reduced)
  
    return X_test_processed, X_val_processed
  
 
def prepare_tensors(X_train_reduced, X_test_reduced, X_val_reduced, y_train_list, y_test_list, y_val_list):
    #stacking all the data 
    X_train = X_train_reduced[0]
    y_train = np.array(y_train_list[0])
    X_test = X_test_reduced[0]
    y_test = np.array(y_test_list[0])
    X_val = X_val_reduced[0]
    y_val = np.array(y_val_list[0])
    #now we stack all the data 
    for i in range(1,len(X_train_reduced)):
        X_train = np.row_stack((X_train,X_train_reduced[i]))
        y_train = np.row_stack((y_train,np.array(y_train_list[i])))
    for i in range(1,len(X_test_reduced)):
        X_test = np.row_stack((X_test,X_test_reduced[i]))
        y_test = np.row_stack((y_test,np.array(y_test_list[i])))
    for i in range(1,len(X_val_reduced)):
        X_val = np.row_stack((X_val,X_val_reduced[i]))
        y_val = np.row_stack((y_val,np.array(y_val_list[i])))
  
    X_train_tensors = torch.tensor(X_train,dtype=torch.double)
    X_test_tensors = torch.tensor(X_test,dtype=torch.double)
    X_val_tensors = torch.tensor(X_val,dtype=torch.double)

    y_train_tensors = torch.tensor(y_train,dtype=torch.int64)
    y_test_tensors = torch.tensor(y_test,dtype=torch.int64)
    y_val_tensors = torch.tensor(y_val,dtype=torch.int64)
  
    # Their API takes a tensor of shape (sequence length, batch size, dim) if batch_first=False (default) and (batch size, sequence length, dim) if batch_first=True.
    #reshaping to rows, timestamps, features

    X_train_tensors = torch.reshape(X_train_tensors,   (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors = torch.reshape(X_test_tensors,  (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))
    X_val_tensors = torch.reshape(X_val_tensors,  (X_val_tensors.shape[0], 1, X_val_tensors.shape[1]))

    return X_train_tensors, X_test_tensors, X_val_tensors, y_train_tensors, y_test_tensors, y_val_tensors
  
  
def prepare_data_loaders(X_train_tensors, y_train_tensors, X_val_tensors, y_val_tensors, X_test_tensors, y_test_tensors, batch_size = 64, shuffle = True):
    """Returns a dictionary of loader trains"""
    
    train_data = []
    val_data = []
    test_data = []
    for i in range(len(X_train_tensors)):
       train_data.append([X_train_tensors[i], y_train_tensors[i]])

    for i in range(len(X_val_tensors)):
       val_data.append([X_val_tensors[i], y_val_tensors[i]])

    for i in range(len(X_test_tensors)):
       test_data.append([X_test_tensors[i], y_test_tensors[i]])

    batch_size = 64
    loader_train=DataLoader(train_data,
                           batch_size=batch_size,
                           shuffle=True)
    loader_val=DataLoader(val_data,
                          batch_size=batch_size,
                          shuffle=False)
    loader_test=DataLoader(test_data,
                           batch_size=batch_size,
                           shuffle=False)

    loaders=  {'train': loader_train,
               'valid': loader_val,
               'test': loader_test}
               
               
    return loaders

def train(n_epochs, model, loaders,  optimizer, criterion, path = './Models/LSTM1/'):
    """returns trained model"""
    valid_loss_min = np.Inf 
    model.float()    
    path = path+'LSTM1.pt'
    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        valid_loss = 0.0
        
        # train the model 
        model.train()

        for batch_idx, (data, target) in enumerate(loaders['train']):
            #if use_cuda:
            #    data, target = data.cuda(), target.cuda()
            data = data.float()
            optimizer.zero_grad()    
            output=model(data)#.double())
            loss=criterion(output,target.long())
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * (loss.data - train_loss))    

        # validate the model 
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            #if use_cuda:
            #    data, target = data.cuda(), target.cuda()
            data = data.float()
            output=model(data)#.double())
            loss=criterion(output,target.long())
            valid_loss+=((1/(batch_idx + 1))*(loss.data - valid_loss))
            model.eval()

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        if valid_loss < valid_loss_min:
            print('validation loss decreased from {:.6f} to {:.6f}. Model is being saved to {}.'.format(
                valid_loss_min,valid_loss, path))
            
            valid_loss_min=valid_loss
            torch.save(model, path)
            
        
        # return trained model
    return model



def save_metrics(save_path, train_loss_list, valid_loss_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list}
    
    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path)
    print(f'Metrics loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list']
    

def evaluate(predictions,y_test_tensors):
    print(f'Accuracy:  {accuracy_score(y_test_tensors, np.round(predictions[:,1]))}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test_tensors, np.round(predictions[:,1]))}')
    print(f'Precision: {precision_score(y_test_tensors, np.round(predictions[:,1]))}')
    print(f'AOC Curve: {roc_auc_score(y_test_tensors, np.round(predictions[:,1]))}')
