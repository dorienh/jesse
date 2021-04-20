# jesse

I am a professor in AI and need a well documented and best practices used PyTorch script for the following task (I would do it myself, but no time to code):



Preprocessing [most already implemented, but needs error checking + PEP8 and improvements to file/class structure)]
- load all of the .csv files included as data (there will be more int he future, so perhaps load all files in folder)
- the files should be normalized per file
- do a 20-10-70% test validation training split: 20% last rows of each of the files is for test. 
- model input:
  - I'm still working on generating the training data, the idea is that we can set the data to be a folder, e.g. daily, or hourly. Then all the files will be processed. Ideally they should be normalised per file. 
  - Use only the files that have all columns (there are a number which due to a bug on my side only have like 6 columns, ignore those for now): date,open,high,low,close,volume,ATR,CC,Top_p15_a4,Btm_p15_a4,Buy_p15_a4,Sell_p15_a4,Top_p40_a1,Btm_p40_a1,Buy_p40_a1,Sell_p40_a1,ODR,Top,Btm,Trend,WM,Band,last_pivot,ma7,ma21,26ema,12ema,MACD,ema,APO_12_26,AROOND_14,AROONU_14,AROONOSC_14,ATRr_14,BBL_5_2.0,BBM_5_2.0,BBU_5_2.0,BBB_5_2.0,BIAS_SMA_26,BOP,AR_26,BR_26,CCI_14_0.015,COPC_11_14_10,LDECAY_5,DEC_1,DEMA_10,DCL_20_20,DCM_20_20,DCU_20_20,EFI_13,EMA_10,ENTP_10,ABER_ZG_5_15,ABER_SG_5_15,ABER_XG_5_15,ABER_ATR_5_15,ACCBL_20,ACCBM_20,ACCBU_20,AD,ADOSC_3_10,ADX_14,DMP_14,DMN_14,AMATe_LR_2,AMATe_SR_2,AO_5_34,OBV,OBV_min_2,OBV_max_2,OBVe_4,OBVe_12,AOBV_LR_2,AOBV_SR_2,CDL_DOJI_10_0.1,CDL_INSIDE,CFO_9,CG_10,CHOP_14_1_100,CKSPl_10_1_9,CKSPs_10_1_9,CMF_20,CMO_14,EOM_14_100000000,ER_10,BULLP_13,BEARP_13,FISHERT_9_1,FISHERTs_9_1,FWMA_10,HA_open,HA_high,HA_low,HA_close,HILO_13_21,HILOl_13_21,HILOs_13_21,HL2,MFI_14,MIDPOINT_2,MIDPRICE_2,MOM_10,NATR_14,NVI_1,OHLC4,PDIST,PCTRET_1,PGO_14,PPO_12_26_9,PPOh_12_26_9,PPOs_12_26_9,PSARl_0.02_0.2,PSARs_0.02_0.2,PSARaf_0.02_0.2,PSARr_0.02_0.2,PSL_12,PVI_1,PVO_12_26_9,PVOh_12_26_9,PVOs_12_26_9,KURT_30,LR_14,LOGRET_1,MACD_12_26_9,MACDh_12_26_9,MACDs_12_26_9,MAD_30,MASSI_9_25,MCGD_10,MEDIAN_30,HLC3,HMA_10,INC_1,INERTIA_20_14,KAMA_10_2_30,KCLe_20_2,KCBe_20_2,KCUe_20_2,K_9_3,D_9_3,J_9_3,KST_10_15_20_30_10_10_10_15,KSTs_9,PVOL,PVR,PVT,PWMA_10,QQE_14_5_4.236,QQE_14_5_4.236_RSIMA,QQEl_14_5_4.236,QQEs_14_5_4.236,QS_10,QTL_30_0.5,RMA_10,ROC_10,RSI_14,RSX_14,RVGI_14_4,RVGIs_14_4,RVI_14,SINWMA_14,SKEW_30,SLOPE_1,WILLR_14,WMA_10,ZL_EMA_10,Z_30,SWMA_10,T3_10_0.7,TEMA_10,THERMO_20_2_0.5,THERMOma_20_2_0.5,THERMOl_20_2_0.5,THERMOs_20_2_0.5,TRIX_30_9,TRIXs_30_9,TRUERANGE_1,TSI_13_25,TTM_TRND_6,UI_14,UO_7_14_28,VAR_30,VIDYA_14,VTXP_14,VTXM_14,VWAP_D,VWMA_10,WCP,SMA_10,SMI_5_20_5,SMIs_5_20_5,SMIo_5_20_5,SQZ_20_2.0_20_1.5,SQZ_ON,SQZ_OFF,SQZ_NO,SSF_10_2,STDEV_30,STOCHk_14_3_3,STOCHd_14_3_3,STOCHRSIk_14_14_3_3,STOCHRSId_14_14_3_3,SUPERT_7_3.0,SUPERTd_7_3.0,SUPERTl_7_3.0,SUPERTs_7_3.0
  - you can use all columns for input except for: Top_p15_a4,Btm_p15_a4,Buy_p15_a4,Sell_p15_a4,Top_p40_a1,Btm_p40_a1,Buy_p40_a1,Sell_p40_a1,Top,Btm,last_pivot,
  - I will need to put these above mentioned columns as my predictive output (I'll train a model for one at a time)
  - input is t_i-m until t_i rows/timesteps (with m customisable, tell me where I can change it, say m is 16 as an example) of columns:Volume,Open,High,Low,Close,ATR,CC,ODR,Trend,WM,Band, 
  - Normalize data (min max scaler) per file
  - Optional switch (include yes or no): output column y can be given as input until time t_i-s (s is not m, but is smaller then m). So I want to predict either the last element of y, or the last m elements (give the previous ones as input)
- model output (prediction):
  - One column that can be set in the beginning, for instance: Btm / Top column at timestep t_i-s until t_i [probably about 5 last time steps but it can be customisable again]
- model is many to many (you predict Btm/Top for the last s timesteps (s can be set to 1 too). Switch between the following architectures:
  - Transformer
  - 2 layer LSTM
  - 2 layer LSTM with self-attention
  - Simple 2 layer FC
  - Wavenet (will need bigger input window, please document how to do this, this will be for minute data when I have it)

- Add training/evaluation function with loss/accuracy plot for training, validaton, and test set
- Output confusion matrix for test set
- function to save / load model and predict based on small input dataframe (m rows)

- batchnorm for training
- Provide function to get prediction with easy instruction on how the input looks like (last m rows of the csv), output is Btm/Top with probability percentage or CE value.

Notes:
- the model can be trained either on the hourly folder, or the daily, or minute folder. 
- Document code very well please

Example input folders in the data folder. (Sorry some don't have enough columns but will fix that soon just ignore these.)

Please push regularly to the repository. 
