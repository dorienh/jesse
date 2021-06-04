from pandas_datareader.data import DataReader

from ..utils import Singleton
from .DataSource import DataSource
import pandas as pd
from datetime import datetime, date, timedelta
import pytz
import alpha_vantage
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.cryptocurrencies import CryptoCurrencies

# https://github.com/RomelTorres/alpha_vantage

@Singleton
class AlphaVantage(DataSource):
    def __init__(self):
        a = 1

        self.hist = pd.DataFrame()
        self.key = '0TGTJNJUQGATSZMW'
        self.crypto = False #random initialization
        self.cc = CryptoCurrencies(key=self.key, output_format='pandas')
        self.ts = TimeSeries(key=self.key, output_format='pandas', indexing_type='date')

    def retrieve_history(self, symbol, _start, _end, crypto = False, max_range = False):
        self.crypto = crypto
        # print(self.crypto)
        if self.crypto:


            # assuming coin is 3 characters:
            coin = symbol.split('-')[0]
            market = symbol.split('-')[1]

            data, meta_data = self.cc.get_digital_currency_daily(symbol=coin, market=market)
            data.columns = ['Open', '1', 'High', '2', 'Low', '3', 'Close', '4', 'Volume', 'cap']
            data['4'] = data['Close']
            data.columns = ['Open', '1', 'High', '2', 'Low', '3', 'Close', 'Adj Close', 'Volume', 'cap']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume','Adj Close']]
            beforestart = datetime.strptime(_start, '%Y-%m-%d').date() - timedelta(days=1)
            beforestartf = beforestart.strftime('%Y-%m-%d')
            if not max_range:
                data = data.loc[_end :beforestartf]
            data = data.iloc[::-1]
            # print('Earliest data:')
            print(data.tail(3))
            # # print(data.columns.values)

        # stocks
        else:
            data, meta_data = self.ts.get_daily_adjusted(symbol=symbol, outputsize='full')
            # print(data.columns.values)
            data.rename(columns={"1. open": "Open", "2. high": "High", "3. low":"Low", "4. close":"Close", "5. adjusted close":"Adj Close", "6. volume":"Volume"}, inplace=True)
            data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
            beforestart = datetime.strptime(_start, '%Y-%m-%d').date() - timedelta(days=1)
            beforestartf = beforestart.strftime('%Y-%m-%d')
            data = data.loc[_end :beforestartf]
            data = data.iloc[::-1]
            # print('Earliest data:')
            # print(data.head(1))


        # data, meta_data = ts.get_daily(symbol='MSFT', outputsize='full')
        # print(data.head(2))
        # print(data.columns.values)


        # download dataframe
        # hist = pdr.get_data_yahoo(symbol, start=_start, end=_end)
        # z=len(hist)

        # what is this command about"??? TODO
        # hist["Adj Close"] = hist["Close"]
        # result:  Open      High       Low     Close  Adj Close  Volume
        #
        # self.hist = hist
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


