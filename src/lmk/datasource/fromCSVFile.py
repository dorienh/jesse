from pandas_datareader.data import DataReader

from ..utils import Singleton
from .DataSource import DataSource
from .Yahoo import Yahoo
import yfinance as yf
from pandas_datareader import data as pdr
import pandas as pd
from datetime import datetime, date, timedelta
import pytz
import time
from pandas import Series, Timestamp, to_datetime



@Singleton
class fromCSVFile(DataSource):
    def __init__(self):
        self.hist = pd.DataFrame()
        self.crypto = False #random init


    def retrieve_history(self, symbol, _start, _end, crypto = False, max_range = True):
        self.crypto = crypto
        # f= '/Users/dorienherremans/Dropbox/DoBrain/AC/Projects/AIFi/code/Jesse code/datasource/coinmarketcap/all_currencies.csv'
        f = '/Users/dorienherremans/Dropbox/DoBrain/AC/Projects/AIFi/code/Jesse code/datasource/ml_candle/new/hourly/Binance_BTTUSDT_1h.csv_r' #gemini_ETHUSD_1hr.csv'
        hist = pd.read_csv(f, index_col=1, parse_dates=True) #skiprows=[0],
        #todo allow other symbols
        # is_symbol = hist['Symbol']=='BTC'
        # hist = hist[is_symbol]

        print(hist.columns)
        # hist = hist.rename(columns={"Market Cap": "Adj Close"})
        hist.columns = map(str.capitalize, hist.columns)
        hist['Adj Close'] = hist['Close']
        hist['Volume'] = hist.iloc[7]
        hist = hist[['Open', 'High', 'Low', 'Close', 'Volume','Adj Close']]
        # beforestart = _start - timedelta(days=1)
        # beforestartf = beforestart.strftime('%Y-%m-%d')
        if not max_range:
            hist = hist.loc[_end:_start]
        # hist = hist.iloc[::-1]
        # print(hist.head())
        # print(hist.tail())
        # print(hist.Symbol.unique())

        self.hist = hist


        return hist

    def get_symbol_name(self, symbol):
        return symbol

    def get_quote_today(self, symbol):
        #todo: hacked to just give latest in the dataframe
        # today = datetime.now(pytz.timezone('UTC')).date().strftime('%Y-%m-%d')
        # print("CAREFUL SHOULD NOT BE CALLED")
        # print(self.hist.iloc[-1]['Close'])
        # pdr.get_data_yahoo(symbol, start=(today - timedelta(days=1)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
        return self.hist.iloc[-1]['Close']


