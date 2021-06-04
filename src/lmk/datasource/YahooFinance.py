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
class YahooFinance(DataSource):
    def __init__(self):
        self.hist = pd.DataFrame()
        self.crypto = False #random init


    def retrieve_history(self, symbol, _start, _end, max_range = False, crypto = False):
        self.crypto = crypto

        # print("RANGE")
        # print(str(max_range ))

        # overriding default pandas dataloader
        yf.pdr_override()  # <== that's all it takes :-)
        _end = (datetime.strptime(_end, '%Y-%m-%d') + timedelta(days = 1)).strftime('%Y-%m-%d')
        # download dataframe

        # print("getting data")

        if max_range:
            hist = pdr.get_data_yahoo(symbol, period="max") #, interval = '1h') #last 730 days
        else:
            hist = pdr.get_data_yahoo(symbol, start=_start, end=_end)
        z=len(hist)

        # what is this command about"??? TODO not reversed?
        # hist["Adj Close"] = hist["Close"]
        # result:  Open      High       Low     Close  Adj Close  Volume

        # Correct for last day
        # minute data, tomorrow needs to be end day
        # pricedata = pdr.get_data_yahoo(stock, start="2020-12-20", end="2020-12-21", interval = "1d")

        # print(hist.tail(2))

        if self.crypto:

            # check if current utc date - 1 has an entry, if not:
            correct = False

            if correct:
                print("Correcting for 2 hour data delay in crypto")
                yesterday = datetime.strptime(_end, '%Y-%m-%d').date() - timedelta(days=1)
                yesterdayf = yesterday.strftime('%Y-%m-%d')

                # check if between 8-10:15am UTC
                # hist = hist.drop(yesterdayf)

                if yesterdayf not in hist.index:

                    print("yfinance correction: ")

                    finedata = pdr.get_data_yahoo(symbol, start=yesterdayf, end=_end, interval="1d")
                    c = finedata['Close'][-1]
                    o = finedata['Close'][0]
                    h = finedata['High'].max()
                    l = finedata['Low'].min()
                    # print(finedata.head(20))
                    yticker = yf.Ticker(symbol)
                    v = yticker.info['volume24Hr']
                    print('adding data for ' + yesterdayf)
                    # create dataframe
                    # datetime.now().date()
                    add = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Adj Close': c, 'Volume': v},
                                       index=[yesterday])


                    # drop last row
                    # hist.drop(hist.tail(1).index, inplace=True)
                    # concatenate
                    hist = pd.concat([hist, add])
        # print(hist.head(4))


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


