# jesse

I am a professor in AI and need a well documented and best practices used PyTorch script for the following task (I would do it myself, but no time to code):

- load all of the .csv files included as data (there will be more int he future, so perhaps load all files in folder)
- the files should be normalized per file
- do a 20-10-70% test validation training split: 20% last rows of each of the files is for test. 
- model input:
 - t_i-m until t_i rows/timesteps (with m customisable, tell me where I can change it, say m is 16 as an example) of columns:Volume,Open,High,Low,Close,ATR,CC,ODR,Trend,WM,Band, [please remove Buy/Sell columns/last_pivot]
 - Btm and Top column can be given as input until time t_i-s (not s is not m, but is smaller then m)
- model output (prediction):
 - Btm / Top column at timestep t_i-s until t_i [probably about 5 last time steps but it can be customisable again]
- model is many to many (you predict Btm/Top for the last s timesteps (s can be set to 1 too). Switch between the following architectures:
- Transformer
  - 2 layer LSTM
  - 2 layer LSTM with self-attention
  - Simple 2 layer FC
  - Wavenet (will need bigger input window, please document how to do this, this will be for minute data when I have it)

- Add training/evaluation function with plot
- Normalize data (min max scaler) per file
- batchnorm for training
- Provide function to get prediction with easy instruction on how the input looks like (last m rows of the csv), output is Btm/Top with probability percentage or CE value.

Notes:
- the model can be trained either on the hourly folder, or the daily, or minute folder. 
- Document code very well please

Example input folders in the data folder.

Please push regularly to the repository. 
