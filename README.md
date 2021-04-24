# jesse

I am a professor in AI and need a well documented and best practices used PyTorch script for the following task (I would do it myself, but no time to code):



1. Preprocessing [most already implemented, but needs error checking + PEP8 and improvements to file/class structure)]
- load all of the .csv files included as data (there will be more in the future, so load all files in specified folder (either daily or hourly))
- the files should be normalized per file
- currently a filename token is given as input to the model, this can probably be remooved. 
- do a 20-10-70% test validation training split: 20% last rows of each of the files is for test. 
- provide a production option whereby all data is used for training
- custum data loader with stratified sampling


2. Model

- allow user to specify the y column to predict. (e.g. Sell_p40_a4)
- allow a list of columns to be removed from the model input (i.e., Top_p15_a4,Btm_p15_a4,Buy_p15_a4,Sell_p15_a4,Top_p40_a1,Btm_p40_a1,Buy_p40_a1,Sell_p40_a1,Top,Btm,last_pivot)
- input of the model is of dimensions: all_features n-sequence length (n can be set and could be for instance 14 days)
- Implement the following model architectures as classes: 
  - Transformer
  - 2 layer LSTM
  - 2 layer LSTM with self-attention
  - Simple 2 layer FC
  - Wavenet (will need bigger input window, please allow easy changing of all model parameteres when calling the class)
- Add training/evaluation function with loss/accuracy plot for training, validaton, and test set
- Output confusion matrix for test set
- function to save / load model and predict based on small input dataframe (m rows)
- batchnorm for training optimization

A. Variant: n-to-1

- predicts the next 1 element of the specified column. 


B. Variant: n-to-m

- predcts the last m elements of column y. 
- it can use the previous t=0 until t=t-m elements of column y as input (this is not the case for Variant n-to-1)
- it can use the other x columns as input until t=t as usual. 

Notes:
- the model can be trained either on the hourly folder, or the daily folder. 
- Document code very well please and use PEP8 standard, it's ok to create many files/classes etc. 


Please push regularly to the repository. 


I will want to have easy functions to create/train/predict on new data (from other sources), e.g.: 

```python
  my_model = Model_n-to-1(n=14, layers=3,...)
  results = my_model.train(epochs=6,device=gpu_1,data='datafolder',production=False, save='filename')
  my_model.load('filename')
  predictions = my_model.predict(test_data=my_dataframe)
```


or a slightly better syntax if you can suggest it. 
