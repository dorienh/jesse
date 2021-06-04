# https://pypi.org/project/cryptocompare/

import cryptocompare

# cryptocompare.get_price('BTC',curr='USD',full=True)
# cryptocompare.get_historical_price_day('BTC', curr='EUR')


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
class CC(DataSource):
    def __init__(self):
        self.hist = pd.DataFrame()
        self.crypto = True #random init


    def retrieve_history(self, symbol, _start, _end, crypto = False, max_range = False):

        coin = symbol.split('-')[0]
        market = symbol.split('-')[1]


        ## note, to get full historical data this may work:
        # import requests
        # url = 'https://min-api.cryptocompare.com/data/histominute' + \
        #       '?fsym=ETH' + \
        #       '&tsym=USD' + \
        #       '&limit=2000' + \
        #       '&aggregate=1'
        # response = requests.get(url)
        # data = response.json()['Data']
        #
        # import pandas as pd
        # df = pd.DataFrame(data)
        # print(df)

        hist = cryptocompare.get_historical_price_day(coin, curr=market) #limit = 24
        # hist = cryptocompare.get_historical_price_hour(coin, curr=market)  # limit = 24
        # df = DataFrame(People_List, columns=['First_Name'])
        # data from json is in array of dictionaries
        df = pd.DataFrame.from_dict(hist)

        # df['time'] = pd.to_datetime(df['time']).astype(str)
        # df['time'] = datetime.utcfromtimestamp(df['time']) #.strftime('%Y-%m-%d'))


        # time is stored as an epoch, we need normal dates
        df['time'] = pd.to_datetime(df['time'], unit='s')
        # df['time']=df['time'].dt.date
        # print(df.columns.values)
        # ['time' 'high' 'low' 'open' 'volumefrom' 'volumeto' 'close'
        #  'conversionType' 'conversionSymbol']
        df.rename(columns={"high": "High", "low": "Low", "time": "date", "open":"Open","volumeto": "Volume",
                             "close": "Close", "volumefrom":'Adj Close'}, inplace=True)
        df['Adj Close']=df['Close']
        df.set_index('date', inplace=True)
        data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]

        # beforestart = datetime.strptime(_start, '%Y-%m-%d').date() - timedelta(days=1)
        # beforestartf = beforestart.strftime('%Y-%m-%d')

        if not max_range:
            data = data.loc[_start:_end]
        # print(data.head())
        # print(data.tail())

        # print(hist.head(10))
#         print(cryptocompare.get_pairs())

        # what is this command about"??? TODO not reversed?
        # hist["Adj Close"] = hist["Close"]
        # result:  Open      High       Low     Close  Adj Close  Volume

        # Correct for last day
        # minute data, tomorrow needs to be end day
        # pricedata = pdr.get_data_yahoo(stock, start="2020-12-20", end="2020-12-21", interval = "1d")

        # print(hist.tail(2))

#         if self.crypto:
#
#             # check if current utc date - 1 has an entry, if not:
#             correct = False
#
#             if correct:
#                 print("Correcting for 2 hour data delay in crypto")
#                 yesterday = datetime.strptime(_end, '%Y-%m-%d').date() - timedelta(days=1)
#                 yesterdayf = yesterday.strftime('%Y-%m-%d')
#
#                 # check if between 8-10:15am UTC
#                 # hist = hist.drop(yesterdayf)
#
#                 if yesterdayf not in hist.index:
#
#                     print("yfinance correction: ")
#
#                     finedata = pdr.get_data_yahoo(symbol, start=yesterdayf, end=_end, interval="1m")
#                     c = finedata['Close'][-1]
#                     o = finedata['Close'][0]
#                     h = finedata['High'].max()
#                     l = finedata['Low'].min()
#                     # print(finedata.head(20))
#                     yticker = yf.Ticker(symbol)
#                     v = yticker.info['volume24Hr']
#                     print('adding data for ' + yesterdayf)
#                     # create dataframe
#                     # datetime.now().date()
#                     add = pd.DataFrame({'Open': o, 'High': h, 'Low': l, 'Close': c, 'Adj Close': c, 'Volume': v},
#                                        index=[yesterday])
#
#
#                     # drop last row
#                     hist.drop(hist.tail(1).index, inplace=True)
#                     # concatenate
#                     hist = pd.concat([hist, add])
#                     print(hist.tail(4))


        self.hist = data


        return data

    def get_symbol_name(self, symbol):
        return symbol

    def get_quote_today(self, symbol):
        #todo: hacked to just give latest in the dataframe
        # today = datetime.now(pytz.timezone('UTC')).date().strftime('%Y-%m-%d')
        # print("CAREFUL SHOULD NOT BE CALLED")
        # print(self.hist.iloc[-1]['Close'])
        # pdr.get_data_yahoo(symbol, start=(today - timedelta(days=1)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
        return self.hist.iloc[-1]['Close']


