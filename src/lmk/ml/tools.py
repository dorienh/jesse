from datetime import datetime

import pandas as pd
# import technical_indicators_lib as ti
from cryptocompare import cryptocompare
from ta import add_all_ta_features
from ta.utils import dropna
import pandas_ta as ta
from lmk.calculator import EntryPointCalculator, ATRCalculator, PivotCalculator
from lmk.ticker import ensure_columns_exist
from datetime import timezone


from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor


import datetime
import time


def preprocessFile(f):
    # 2017-07-14 01-PM
    # custom_date_parser = lambda x: x.split(".")[0]
    #lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %I-%p")

    # print(datetime.datetime.fromtimestamp(1607320800000.0/1e3))
    def date_parser(string_list):
        dates=[]
        for item in string_list:
            # dates.append(datetime.datetime.fromtimestamp(float(item)))
            try:
                dates.append(datetime.datetime.fromtimestamp(float(item), timezone.utc))
                # print('ok')
            except:
                dates.append(datetime.datetime.fromtimestamp(float(item) / 1e3, timezone.utc))
                # print('error converting unix')

        return dates
        #[time.ctime(float(x)) for x in string_list]


    # df = pd.read_csv(io.StringIO(t), parse_dates=[0], sep=';',
    #                  date_parser=date_parser,
    #                  index_col='DateTime',
    #                  names=['DateTime', 'X'], header=None)

    # date_parser = pd.to_numeric.to_datetime #pd.datetools.to_datetime

    # https://stackoverflow.com/questions/21269399/datetime-dtypes-in-pandas-read-csv
    h = pd.read_csv(f, sep=",", index_col=0, parse_dates=[0], date_parser=date_parser) #, parse_dates=[0], date_parser=date_parser) # parse_dates=[0], date_parser=date_parser) #parse_dates=True)


    # h.index = h.index.split('.')[0]


    h.columns = map(str.capitalize, h.columns)
    print(h.columns)

    if "Volume" not in h.columns:
        # print ("fixing missing volume")
        vname = 'Volume ' + h['Symbol'].iloc[0][0:3].lower()

        try: h['Volume'] = h[vname]
        except:
            try:
                h['Volume'] = h['Volume ' + h['Symbol'].iloc[0][0:3].upper()]
            except:
                try:
                    h['Volume'] = h['Volume ' + h['Symbol'].iloc[0][0:4].upper()]
                except:
                    try:
                        h['Volume'] = h['Volume ' + h['Symbol'].iloc[0][0:4].lower()]
                    except:
                        try: h['Volume'] = h['Volume ' + h['Symbol'].iloc[0][0:5].upper()]
                        except:
                            h['Volume'] = h['Volume ' + h['Symbol'].iloc[0][0:5].lower()]


        # if "vname" not in h.columns:


    h = h[['Open', 'High', 'Low', 'Close', 'Volume']]   #crypt no need for 'Adj Close'

    # remove duplicate index values
    h = h[~h.index.duplicated(keep='last')]

    h = h.sort_index()

    return h

def reverseFile(f):

    # load the csv and user row 0 as headers
    try:
        df = pd.read_csv(f, sep=",", index_col=0, dtype='unicode')
    except:
        df = pd.read_csv(f, sep=",", index_col=0, skiprows=1, dtype='unicode')

    # reverse the data
    df = df.iloc[::-1]
    # df = df.sort_index()

    #sort by index

    # print(df.head(5))
    df.to_csv(f + "_r")
    print('Reversed ' + f)



def addEntries(h):

    c = ATRCalculator.ATRCalculator(window_size=20)
    h["ATR"] = h.apply(c, axis=1)
    h.fillna(method="backfill", axis=0, inplace=True)

    h = ensure_columns_exist(h, ["Top", "Btm", "CC"], pivot_window_size = 15, atr_factor = 4.0)
    h = ensure_columns_exist(h, ["Buy", "Sell"], pivot_window_size=15, atr_factor=4.0)
    h.rename(columns={'Top': 'Top_p15_a4', 'Buy': 'Buy_p15_a4', 'Sell': 'Sell_p15_a4', 'Btm': 'Btm_p15_a4'}, inplace=True)

    h = ensure_columns_exist(h, ["Top", "Btm", "CC"], pivot_window_size = 40, atr_factor = 1.0)
    h = ensure_columns_exist(h, ["Buy", "Sell"], pivot_window_size=40, atr_factor=1.0)
    h.rename(columns={'Top': 'Top_p40_a1', 'Buy': 'Buy_p40_a1', 'Sell': 'Sell_p40_a1', 'Btm': 'Btm_p40_a1'}, inplace=True)

    # h = ensure_columns_exist(h, ["Top", "Btm", "CC"], pivot_window_size = 300, atr_factor = 1.0)
    # h = ensure_columns_exist(h, ["Buy", "Sell"], pivot_window_size=300, atr_factor=1.0)
    # h.rename(columns={'Top': 'Top_p300_a1', 'Buy': 'Buy_p300_a1', 'Sell': 'Sell_p300_a1', 'Btm': 'Btm_p300_a1'}, inplace=True)


    h = ensure_columns_exist(h, ["WM", "Band", "Trend", "ODR"], pivot_window_size=15, atr_factor=4.0)
    # print(h.head(20))

    return h



def addTechnical_indicators(h):

    #todo check what about the first items?

    # Create 7 and 21 days Moving Average
    h['ma7'] = h['Close'].rolling(window=7).mean()
    h['ma21'] = h['Close'].rolling(window=21).mean()

    # Create MACD
    h['26ema'] = h['Close'].ewm(span=26).mean()
    h['12ema'] = h['Close'].ewm(span=12).mean()
    h['MACD'] = (h['12ema'] - h['26ema'])

    # def bollinger_strat(data, window, no_of_std):
    #     rolling_mean = data['Close'].rolling(window).mean()
    #     rolling_std = data['Close'].rolling(window).std()
    #
    #     df['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
    #     df['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)
    #
    # bollinger_strat(data, 20, 2)
    # Create Bollinger Bands
    # h['20sd'] = pd.stats.moments.rolling_std(h['Close'], 20)
    # h['upper_bband'] = h['ma21'] + (h['20sd'] * 2)
    # h['lower_bband'] = h['ma21'] - (h['20sd'] * 2)

    # Create Exponential moving average
    h['ema'] = h['Close'].ewm(com=0.5).mean()

    # Create Momentum
    # h['momentum'] = h['Close'] - 1

    # h.set_index('Date', drop=True, append=False, inplace=True, verify_integrity=False)
    # h = h.sort_index()

    # h.columns = map(str.lower, h.columns)
    # Method 1: get the data by sending a dataframe
    # obv = ti.OBV()
    # h['obv'] = obv.get_value_df(h)
    # Method 2: get the data by sending series values
    # obv_values = ti.obv.get_value_list(h["close"], h["volume"])

    # excellent to add a bunch of indicators:
    # h = add_all_ta_features(h, open="Open", high="High", low="Low", close="Close", volume="Volume")
    # mom_data.columns

    # add golden cross

    # more indicators:
    # https://twopirllc.github.io/pandas-ta/
    # add patterns from here https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html (and more indicators


    print(h.columns)
    # VWAP requires the DataFrame index to be a DatetimeIndex.
    # Replace "datetime" with the appropriate column from your DataFrame
    # h.set_index(pd.DatetimeIndex(h["Date"]), inplace=True)

    # Runs and appends all indicators to the current DataFrame by default
    # The resultant DataFrame will be large.
    h.ta.cores = 8
    # h.ta.strategy()
    # Or the string "all"
    error = False
    # h = h.sort_index()
    # h.to_csv("test.csv")
    # h.ta.strategy(exclude=["ichimoku", "dpo", "trima"], verbose=True)
    try:
        h.ta.strategy(exclude=["ichimoku", "dpo", "trima"], verbose=True) #
    except:
        print("ERROR with strategy function")
        error = True

    #data leaks:ichimoku and dpo

    # h.to_csv("test.csv")
    # h.fillna(method="ffill", axis=0, inplace=True)
    # Or the ta.AllStrategy
    # h.ta.strategy(ta.AllStrategy)

    # Use verbose if you want to make sure it is running.
    # h.ta.strategy(verbose=True)

    # Use timed if you want to see how long it takes to run.
    # h.ta.strategy(timed=True)

    # Choose the number of cores to use. Default is all available cores.


    # Maybe you do not want certain indicators.
    # Just exclude (a list of) them.
    # h.ta.strategy(exclude=["bop", "mom", "percent_return", "wcp", "pvi"], verbose=True)

    # Perhaps you want to use different values for indicators.
    # This will run ALL indicators that have fast or slow as parameters.
    # Check your results and exclude as necessary.
    # h.ta.strategy(fast=10, slow=50, verbose=True)



    #even more complete 200 indicators:
    # https://github.com/mrjbq7/ta-lib
    # Sanity check. Make sure all the columns are there
    # print(h.columns)

    #TODO: replace empty columns by NA? or zero or cut them off

    # h.to_csv("test2.csv")
    # print(h.head(5))

    h.fillna(method="backfill", axis=0, inplace=True)

    return h, error





def addPercentageChange(h):

    h['Close_pct'] = h['Close'].pct_change()
    h['Open_pct'] = h['Open'].pct_change()
    h['High_pct'] = h['High'].pct_change()
    h['Low_pct'] = h['Low'].pct_change()
    h['Volume_pct'] = h['Volume'].pct_change()
    #todo did I forget something?

    return h






def setSignals(h, Topname = "Top", Btmname = "Btm"):
    #todo test
    #todo drop any Buy or Sell column if it exists

    h['Top'] = h[Topname]
    h['Btm'] = h[Btmname]
    h = ensure_columns_exist(h, ["Buy", "Sell"], pivot_window_size=300, atr_factor=1.0)
    return h

#backtest using Buy and Sell from h
def backtestEntries(h, Buyname = 'Buy', Sellname = 'Sell'):

    # date,last_rec,rec_ago,rec_price,Volume,Open,High,Low,Close,ATR,Top,Btm,Buy,Sell,ODR,Trend,WM,Band,last_pivot
    h['Buy'] = h[Buyname]
    h['Sell'] = h[Sellname]
    StartCurrency = h.Close.iloc[0]
    CurrencyBalance = StartCurrency
    StockBalance = 0
    started = False
    pcost = 0.0026
    cost = 0.
    prev_rec = ''
    firstStarted = False


    worth = []

    # summarydf["Buy"] = getEntry(summarydf, trade_type=BUY, atr_factor=4)
    # summarydf["Sell"] = getEntry(summarydf, trade_type=SELL, atr_factor=4)

    for row, rec in h.iterrows():
        # first time rec happens on the day
        if started == False and rec.Buy:
            started = True

        if started:
            transprice = rec.Close
            if rec.Buy:  # or gain < 0.95:
                if CurrencyBalance > 0.00001:
                    CurrencyBalance = (CurrencyBalance - cost) - (CurrencyBalance * pcost)
                    # price =  dfbuy.at[mydatef]
                    # print("comparing "+ str(price) + " and "+ str(transprice))
                    StockBalance = (CurrencyBalance) / transprice
                    CurrencyBalance = 0
                    # print(str(row) + ' Buy at: ' + str(transprice) + ' resulting stock balance: ' + str(StockBalance))
                    last_rec_price = transprice

            elif rec.Sell:
                if StockBalance > 0.000001:
                    CurrencyBalance = (StockBalance * transprice) - cost
                    CurrencyBalance = CurrencyBalance - (CurrencyBalance * pcost)
                    StockBalance = 0
                    # print(str(row) + ' Sell at: ' + str(transprice) + ' resulting balance: ' + str(CurrencyBalance))

        worth_now = CurrencyBalance
        if CurrencyBalance < 0.0001:
            worth_now = StockBalance * rec.Close
        worth.append(worth_now)

    # print('Stocks: ' + str(StockBalance))

    if StockBalance != 0.:
        stockValue = StockBalance * float(h.Close.iloc[-1])
        # print('Value: ' + str(stockValue))
    else:
        stockValue = CurrencyBalance
    # print('Value: ' + str(stockValue))

    worth_otherwise = (StartCurrency / h.Close.iloc[0]) * h.Close.iloc[-1]
    # print('Without action worth: ' + str(worth_otherwise))
    # days = (datetime.strptime(h.index[-1], '%Y-%m-%d') - datetime.strptime(h.index[0],'%Y-%m-%d')).days
    # print (days)
    # gain ratio over just keep strategy
    print(str(stockValue / worth_otherwise))  # 'Gain ratio: ' +
    # profit percentage
    print(str((stockValue / StartCurrency)))
    # withouot action profit percentage
    print(str(worth_otherwise / StartCurrency))
    # print(str((summarydf.Close[-1]/summarydf.Close[0])*365/days))

    return worth


def get_cc_data(coin, market):

        hist = cryptocompare.get_historical_price_day(coin, curr=market)  # limit = 24
        df = pd.DataFrame.from_dict(hist)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={"high": "High", "low": "Low", "time": "date", "open": "Open", "volumeto": "Volume",
                           "close": "Close", "volumefrom": 'Adj Close'}, inplace=True)
        df['Adj Close'] = df['Close']
        df.set_index('date', inplace=True)
        data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        return data

def get_binance_data():
    # valid intervals - 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M


    df = pd.DataFrame()
    startDate = end
    limit = 5000
    while startDate > start:
        url = 'https://api.binance.com/api/v3/klines?symbol=' + \
              symbol + '&interval=' + interval + '&limit=' + str(iteration)
        if startDate is not None:
            url += '&endTime=' + str(startDate)

        df2 = pd.read_json(url)
        df2.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Closetime', 'Quote asset volume',
                       'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore']
        df = pd.concat([df2, df], axis=0, ignore_index=True, keys=None)
        startDate = df.Opentime[0]
    df.reset_index(drop=True, inplace=True)
    return df