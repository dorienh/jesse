from pandas_datareader.data import DataReader

from ..utils import Singleton
from .DataSource import DataSource
import pandas as pd
from datetime import datetime, date, timedelta
import pytz
from datetime import datetime
import krakenex

import decimal
import time
import pandas as pd

# https://github.com/RomelTorres/alpha_vantage

@Singleton
class KrakenData(DataSource):
    def __init__(self):

        self.hist = pd.DataFrame()
        self.crypto = False #random initialization

    def unixToString(self, unixt):
        return datetime.utcfromtimestamp(int(unixt)).strftime('%Y-%m-%d')

    def stringToUnix(self, timestr):
        return time.mktime(datetime.strptime(timestr, "%Y-%m-%d").timetuple())

    def retrieve_history(self, symbol, _start, _end, crypto = False, max_range = False):
        # print(self.crypto)


        k = krakenex.API()
        ret = k.query_public('OHLC', data={'pair': symbol, 'interval': '1440', 'since': self.stringToUnix(_start)})  # , 'since': since
        # print (ret)

        df = pd.DataFrame(ret['result'][symbol])

        df[0] = pd.to_datetime(df[0], unit='s')
        df.rename(columns={0: 'date', 1: 'Open', 2: 'High', 3: 'Low', 4: 'Close', 5: 'Adj Close', 6: 'Volume'},
                  inplace=True)
        df['Adj Close'] = df['Close']
        df = df.set_index('date')
        data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].astype(float)
        # data = pd.to_numeric(data)
        # print(data.dtypes)

        self.hist = data
        return data

    def get_symbol_name(self, symbol):
        return symbol

    def get_quote_today(self, symbol):
        #todo: hacked to just give latest in the datafram
        # today = datetime.now(pytz.timezone('UTC')).date().strftime('%Y-%m-%d')
        # print("CAREFUL SHOULD NOT BE CALLED")
        # print(self.hist.iloc[-1]['Close'])
        # pdr.get_data_yahoo(symbol, start=(today - timedelta(days=1)).strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
        return self.hist.iloc[-1]['Close']


