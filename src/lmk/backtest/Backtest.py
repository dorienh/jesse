from lmk.ticker import Ticker
from datetime import datetime, date, timedelta
import csv
import pandas as pd
from lmk.backtest.Writer import Writer
from lmk.calculator.EntryPointCalculator import EntryPointCalculator, BUY, SELL
from lmk.backtest.Backtest import Backtest, getWorth, getWorthML, getWorthATR
from lmk.backtest.Writer import Writer
from lmk.backtest.Finplot import Finplot



"""EntryPointCalculator

Entry/Buy: price > btm + atr/2
Exit/Sell: price < top - atr/2
"""



def getEntry(s, trade_type=BUY, atr_factor=1.0):

    BUY=1
    SELL=2
    tradelist = []

    pivot = None
    wait_for_trade = False

    first = True
    idx = 0
    period = 30

    for col, row in s.iterrows():

        if first:
            trade = False


        if not first:
            trade = False

            atr = row["ATR"] * atr_factor

            if trade_type == BUY:

                period = s.Close.iloc[idx-period:idx]
                # min = period.min()

                if row.last_pivot < row.Close: # if is btm:
                    pivot = row["Close"]
                    wait_for_trade = True

                if pivot and wait_for_trade and row["Close"] >= pivot + atr/2.0:
                    wait_for_trade = False
                    trade = True

            elif trade_type == SELL:
                if row["last_pivot"] > row["Close"]:
                    pivot = row["Close"]
                    wait_for_trade = True

                if pivot and wait_for_trade and row["Close"] <= pivot - atr/2.0:
                    wait_for_trade = False
                    trade = True

        if idx > 30:
            first = False
        tradelist.append(trade)
        idx += 1

    return tradelist


def getWorthATR(summarydf):
    # date,last_rec,rec_ago,rec_price,Volume,Open,High,Low,Close,ATR,Top,Btm,Buy,Sell,ODR,Trend,WM,Band,last_pivot

    StartCurrency = summarydf.Close.iloc[0]
    CurrencyBalance = StartCurrency
    StockBalance = 0
    started = False
    pcost = 0.0026
    cost = 0.
    prev_rec = ''
    firstStarted = False

    method1 = True
    method2 = False
    worth = []

    summarydf["Buy"] = getEntry(summarydf, trade_type=BUY, atr_factor=4)
    summarydf["Sell"] = getEntry(summarydf, trade_type=SELL, atr_factor=4)


    if method1:
        # print('Method 1')
        # Based on Buy/Sell
        # old: 'Pair', 'Date', 'Price', 'Rec', 'Ago','ODR']
        for row, rec in summarydf.iterrows():
            # first time rec happens on the day
            if started == False and rec.Buy:
                started = True
                # print("started")
                firstStarted = False
                last_rec_price = rec.Close

            if started:
                transprice = rec.Close
                # print((transprice / last_rec_price))
                # print (transprice)
                # print(last_rec_price)
                gain = transprice / last_rec_price
                if (rec.last_rec != prev_rec) or firstStarted == False:
                    # transprice = float(rec[2])
                    firstStarted = True
                    if rec.Buy: # or gain < 0.95:
                        started = True
                        if CurrencyBalance > 0.00001:
                            CurrencyBalance = (CurrencyBalance - cost) - (CurrencyBalance * pcost)
                            # price =  dfbuy.at[mydatef]
                            # print("comparing "+ str(price) + " and "+ str(transprice))

                            StockBalance = (CurrencyBalance) / transprice
                            CurrencyBalance = 0
                            # print(str(row) + ' Buy at: ' + str(transprice) + ' resulting stock balance: ' + str(StockBalance))
                            last_rec_price = transprice

                    elif  rec.ODR or rec.Sell:
                        # if (rec.ODR):

                            # print('ODR')
                        if StockBalance > 0.000001:
                            CurrencyBalance = (StockBalance * transprice) - cost
                            CurrencyBalance = CurrencyBalance - (CurrencyBalance * pcost)
                            StockBalance = 0
                            # print(str(row) + ' Sell at: ' + str(transprice) + ' resulting balance: ' + str(CurrencyBalance))

            prev_rec = rec.last_rec
            worth_now = CurrencyBalance
            if CurrencyBalance < 0.0001:
                worth_now = StockBalance * rec.Close
            worth.append(worth_now)



        # print('Stocks: ' + str(StockBalance))


        if StockBalance != 0.:
            stockValue = StockBalance * float(summarydf.Close.iloc[-1])
            # print('Value: ' + str(stockValue))
        else: stockValue = CurrencyBalance
        # print('Value: ' + str(stockValue))

        worth_otherwise = (StartCurrency/summarydf.Close.iloc[0])*summarydf.Close.iloc[-1]
        # print('Without action worth: ' + str(worth_otherwise))
        days = (datetime.strptime(summarydf.index[-1], '%Y-%m-%d') - datetime.strptime(summarydf.index[0], '%Y-%m-%d')).days
        # print (days)
        # gain ratio over just keep strategy
        print(str(stockValue/worth_otherwise)) #'Gain ratio: ' +
        # profit percentage
        print(str((stockValue / StartCurrency )))
        # withouot actiioon profit percentage
        print(str(worth_otherwise/StartCurrency))


        # print(str((summarydf.Close[-1]/summarydf.Close[0])*365/days))




def getWorth(summarydf):
    # date,last_rec,rec_ago,rec_price,Volume,Open,High,Low,Close,ATR,Top,Btm,Buy,Sell,ODR,Trend,WM,Band,last_pivot

    StartCurrency = summarydf.Close.iloc[0]
    CurrencyBalance = StartCurrency
    StockBalance = 0
    started = False
    pcost = 0.0026
    cost = 0.
    prev_rec = ''
    firstStarted = False

    method1 = True
    method2 = False
    worth = []



    #TODO add switch
    if method1:
        # print('Method 1')
        # Based on Buy/Sell
        # old: 'Pair', 'Date', 'Price', 'Rec', 'Ago','ODR']
        for row, rec in summarydf.iterrows():
            # first time rec happens on the day
            if started == False and int(rec.rec_ago) == 0 and rec.last_rec == 'Buy':
                started = True
                # print("started")
                firstStarted = False
                last_rec_price = rec.Close

            if started:
                transprice = rec.Close
                # print((transprice / last_rec_price))
                # print (transprice)
                # print(last_rec_price)
                gain = transprice / last_rec_price
                if (rec.last_rec != prev_rec) or firstStarted == False:
                    # transprice = float(rec[2])
                    firstStarted = True
                    if rec.last_rec == 'Buy': # or gain < 0.95:
                        started = True
                        if CurrencyBalance > 0.00001:
                            CurrencyBalance = (CurrencyBalance - cost) - (CurrencyBalance * pcost)
                            # price =  dfbuy.at[mydatef]
                            # print("comparing "+ str(price) + " and "+ str(transprice))

                            StockBalance = (CurrencyBalance) / transprice
                            CurrencyBalance = 0
                            # print(str(row) + ' Buy at: ' + str(transprice) + ' resulting stock balance: ' + str(StockBalance))
                            last_rec_price = transprice

                    elif  rec.ODR or rec.last_rec == 'Sell' :
                        # if (rec.ODR):

                            # print('ODR')
                        if StockBalance > 0.000001:
                            CurrencyBalance = (StockBalance * transprice) - cost
                            CurrencyBalance = CurrencyBalance - (CurrencyBalance * pcost)
                            StockBalance = 0
                            # print(str(row) + ' Sell at: ' + str(transprice) + ' resulting balance: ' + str(CurrencyBalance))

            prev_rec = rec.last_rec
            worth_now = CurrencyBalance
            if CurrencyBalance < 0.0001:
                worth_now = StockBalance * rec.Close
            worth.append(worth_now)



        # print('Stocks: ' + str(StockBalance))


        if StockBalance != 0.:
            stockValue = StockBalance * float(summarydf.Close.iloc[-1])
            # print('Value: ' + str(stockValue))
        else: stockValue = CurrencyBalance
        # print('Value: ' + str(stockValue))

        worth_otherwise = (StartCurrency/summarydf.Close.iloc[0])*summarydf.Close.iloc[-1]
        # print('Without action worth: ' + str(worth_otherwise))
        days = (datetime.strptime(summarydf.index[-1], '%Y-%m-%d') - datetime.strptime(summarydf.index[0], '%Y-%m-%d')).days
        # print (days)
        # gain ratio over just keep strategy
        print(str(stockValue/worth_otherwise)) #'Gain ratio: ' +
        # profit percentage
        print(str((stockValue / StartCurrency )))
        # withouot actiioon profit percentage
        print(str(worth_otherwise/StartCurrency))


        # print(str((summarydf.Close[-1]/summarydf.Close[0])*365/days))


    CurrencyBalance = StartCurrency
    StockBalance = 0
    started = False
    prev_rec = ''
    firstStarted = False

    if method2:
        print('Method 2')
        # buy/sell based on bands
        for row, rec in summarydf.iterrows():
            # first time rec happens on the day
            if started == False  and int(rec.Band) == 6: #todo add band changed
                started = True
                # print("started")
                firstStarted = False

            if started:
                transprice = rec.Close
                if (int(rec.Band) != int(prev_rec)) or firstStarted == False:
                    # print(str(rec.Band) + ' ' + str(prev_rec))
                    # transprice = float(rec[2])
                    firstStarted = True
                    if int(rec.Band) == 6:
                        if (CurrencyBalance >  0.00001):
                            started = True
                            CurrencyBalance = (CurrencyBalance - cost) - (CurrencyBalance * pcost)
                            # price =  dfbuy.at[mydatef]
                            # print("comparing "+ str(price) + " and "+ str(transprice))

                            StockBalance = (CurrencyBalance) / transprice
                            CurrencyBalance = 0
                            # print(str(row) + ' Buy at: ' + str(transprice) + ' resulting stock balance: ' + str(StockBalance))

                    elif (int(rec.Band) == 1) :
                        if (StockBalance >  0.000001):
                            CurrencyBalance = (StockBalance * transprice) - cost
                            CurrencyBalance = CurrencyBalance - (CurrencyBalance * pcost)
                            StockBalance = 0
                            # print(str(row) + ' Sell at: ' + str(transprice) + ' resulting balance: ' + str(CurrencyBalance))
            prev_rec = rec.Band


        # print('Stocks: ' + str(StockBalance))
        # print('Currency: ' + str(CurrencyBalance))

        if StockBalance != 0.:
            stockValue = StockBalance * float(summarydf.Close.iloc[-1])
            print('Equivalent to: ' + str(stockValue))
        else: stockValue = CurrencyBalance

        worth_otherwise = (StartCurrency/summarydf.Close.iloc[0])*summarydf.Close.iloc[-1]
        # print('Without action worth: ' + str(worth_otherwise))
        print('Gain ratio: ' + str(stockValue/worth_otherwise))
        # print('From ' + summarydf.index[0] + ' to ' + summarydf.index[-1])


    return worth







def getWorthML(summarydf):
    # date,last_rec,rec_ago,rec_price,Volume,Open,High,Low,Close,ATR,Top,Btm,Buy,Sell,ODR,Trend,WM,Band,last_pivot

    StartCurrency = summarydf.Close.iloc[0]
    CurrencyBalance = StartCurrency
    StockBalance = 0
    started = False
    pcost = 0.0026
    cost = 0.
    prev_rec = ''
    firstStarted = False

    worth = []


    method1 = True
    print('Running backtest')

    #TODO add switch
    if method1:
        # print('Method 1')
        # Based on Buy/Sell
        # old: 'Pair', 'Date', 'Price', 'Rec', 'Ago','ODR']
        for row, rec in summarydf.iterrows():

            threshold  = 0.95
            sell = rec.Sell_pred > threshold
            buy = rec.Buy_pred > threshold


            # first time rec happens on the day
            if started == False and buy: #rec.Buy_pred > 0.2:
                started = True
                print("started")

                last_rec_price = rec.Close

            if started:
                transprice = rec.Close
                # print((transprice / last_rec_price))
                # print (transprice)
                # print(last_rec_price)
                gain = transprice / last_rec_price

                # transprice = float(rec[2])

                if buy: #rec.Buy_pred > 0.2: # or gain < 0.95:
                    # print('Buy')
                    if CurrencyBalance > 0.00001:
                        CurrencyBalance = (CurrencyBalance - cost) - (CurrencyBalance * pcost)
                        # price =  dfbuy.at[mydatef]
                        # print("comparing "+ str(price) + " and "+ str(transprice))

                        StockBalance = (CurrencyBalance) / transprice
                        CurrencyBalance = 0
                        print(str(row) + ' Buy at: ' + str(transprice) + ' resulting stock balance: ' + str(StockBalance))
                        last_rec_price = transprice

                elif  rec.ODR or sell: #rec.Sell_pred > 0.2 :
                    # print('Sell')
                    # if (rec.ODR):

                        # print('ODR')
                    if StockBalance > 0.000001:
                        CurrencyBalance = (StockBalance * transprice) - cost
                        CurrencyBalance = CurrencyBalance - (CurrencyBalance * pcost)
                        StockBalance = 0
                        print(str(row) + ' Sell at: ' + str(transprice) + ' resulting balance: ' + str(CurrencyBalance))

            # prev_rec = rec.last_rec
            worth_now = CurrencyBalance
            if CurrencyBalance < 0.0001:
                worth_now = StockBalance * rec.Close
            worth.append(worth_now)



        # print('Stocks: ' + str(StockBalance))


        if StockBalance != 0.:
            stockValue = StockBalance * float(summarydf.Close.iloc[-1])
            # print('Value: ' + str(stockValue))
        else: stockValue = CurrencyBalance
        # print('Value: ' + str(stockValue))

        worth_otherwise = (StartCurrency/summarydf.Close.iloc[0])*summarydf.Close.iloc[-1]
        # print('Without action worth: ' + str(worth_otherwise))
        # days = (datetime.strptime(summarydf.index[-1], '%Y-%m-%d') - datetime.strptime(summarydf.index[0], '%Y-%m-%d')).days
        # print (days)
        # gain ratio over just keep strategy
        print(str(stockValue/worth_otherwise)) #'Gain ratio: ' +
        # profit percentage
        print(str((stockValue / StartCurrency )))
        # withouot actiioon profit percentage
        print(str(worth_otherwise/StartCurrency))


        # print(str((summarydf.Close[-1]/summarydf.Close[0])*365/days))



    return worth


class Backtest:
    def __init__(self, ticker, StartDay, EndDay):
        self.ticker = ticker
        self.StartDay = StartDay
        self.EndDay = EndDay
        self.balances = []
        self.balancesLMK = []
        self.LastEntryRecommendation = []
        self.gain = ""

    def runBacktest(self, w, pcost=0.0026, cost=0., TestDelay=False, fig=None):
        w.print("Test for " + str(self.ticker.symbol))

        ticker = self.ticker

        # todo pass LMK band for more informed entry/exit
        buy, sell, odr, h = ticker.getEntryExit()
        StartDay = h.index[0]
        # print("Buy lenght: " + str(len(buy)))

        dfbuy = buy['Close']
        dfsell = sell['Close']

        # entry point test
        StartCurrency = ticker.getPrice(StartDay)
        CurrencyBalance = StartCurrency
        StockBalance = 0

        # StartCurrencyLMK test
        StartCurrencyLMK = ticker.getPrice(StartDay)
        CurrencyBalanceLMK = StartCurrencyLMK
        StockBalanceLMK = 0
        LMKcount = 0
        buyLMK = False
        sellLMK = False
        prev_band = 0.


        OneDay = timedelta(days=1)
        mydate = StartDay
        started = False

        balances = [[]]
        balancesLMK = [[]]
        LastTransaction = []

        worth = pd.DataFrame(index=h.index.copy())
        worth_list = []

        ODR = ""
        trend = -1.
        EndDay = h.index[-1]
        prev_trend = -1.


        LMKtest = False

        for col, mydate in h.iterrows():
            # while mydate <= EndDay:

            # mydatef = mydate.strftime('%Y-%m-%d')

            if mydate['Btm']:
                btm = 'Btm'
            else: btm = ''
            if mydate['Top']:
                top = 'Top'
            else: top = ''

            # get ODR
            if (mydate.name in odr.index):
                ODR = "ODR"

            # get trendword
            trend = mydate['Trend']
            trendword = ''
            if trend == 1.:
                trendword = 'Up'
            elif trend == 2.:
                trendword = 'Down'



            buyLMK = False
            sellLMK = False
            ## naive: buy entering/exiting natural rally (5)
            ## todo reverse on reaction (band 2)
            band = int(mydate['Band'])
            if band == 5:
                if prev_band != band:
                    buyLMK = True
                    # print('buy  marker')
            else:
                if prev_band == 5:
                    sellLMK = True

            # if band == 2:
            #     if prev_band != band:
            #         sellLMK = True
            #         # print('buy  marker')
            # else:
            #     if prev_band == 2:
            #         sellLMK = True
                    # print('sell marker')

                # print("In natural rally")



            ## based on up or down column
            # if trend == prev_trend:
            #     LMKcount += 1
            #     sellLMK = False
            #     buyLMK = False
            #     if trend == 1.:
            #         trendword = 'Up'
            #     elif trend == 2.:
            #         trendword = 'Down'
            # else:
            #     LMKcount = 1
            #     if trend == 1.:
            #         buyLMK = True
            #         sellLMK = False
            #         trendword = 'Up'
            #     elif trend == 2.:
            #         sellLMK = True
            #         buyLMK = False
            #         trendword = 'Down'
            # prev_trend = trend








            prev_trend = trend
            prev_band = band

            # get band
            band = mydate['Trend']


            LastTransaction = ""
            # todo not allow split shares

            price = ''

            if (mydate.name in dfbuy.index):
                LastTransaction = 'Buy'
                price = dfbuy.at[mydate.name]
                self.LastEntryRecommendation = [self.ticker.getSymbol(), col, "{:.7f}".format(price),
                                                LastTransaction, str((h.index[-1] - col).days), ODR, btm, top] #str((h.iloc[-1].index - mydate).days)
            if (mydate.name in dfsell.index):
                LastTransaction = 'Sell'
                price = dfsell.at[mydate.name]
                self.LastEntryRecommendation = [self.ticker.getSymbol(), col, "{:.7f}".format(price),
                                                LastTransaction, str((h.index[-1] - col).days), ODR, btm, top] #str((h.iloc[-1].index - mydate).days)


            # if TestDelay and (EndDay != mydatef):
            #     tomorrow = mydate + timedelta(days=1)
            #     tomorrowf = tomorrow.strftime('%Y-%m-%d')
            #
            # else:
            #     tomorrowf = mydatef

            if TestDelay and (EndDay != mydate):
                transprice = ticker.getNextOpenPrice(mydate.name)  # getOpenPrice
            else:
                transprice = ticker.getPrice(mydate.name)  # getOpenPrice

            # print("comparing: ")
            # print(transprice)
            # print(ticker.getNextOpenPrice(mydatef))

            # print(StockBalance)
            ############################ entry exit test
            if StockBalance == 0:
                # BUY

                if mydate.name in dfbuy.index:

                    started = True

                    # print( str(mydatef) + 'BUY at' + str(dfbuy.at[mydatef]) + ', using ' + str(CurrencyBalance))
                    # print ()
                    CurrencyBalance = (CurrencyBalance - cost) - (CurrencyBalance * pcost)
                    # price =  dfbuy.at[mydatef]
                    # print("comparing "+ str(price) + " and "+ str(transprice))

                    StockBalance = (CurrencyBalance) / transprice
                    CurrencyBalance = 0

                    # todo last day did not show in summary

                    # print(StockBalance)
            else:
                # SELL
                if (mydate.name in dfsell.index) and started:
                    # print(mydatef + 'SELL at' + str(dfsell.at[mydatef]) + ', using stocks: ' + str(StockBalance))
                    #     print(ticker.getPrice(tomorrowf))
                    #     print(ticker.getPrice(mydatef))

                    CurrencyBalance = (StockBalance * transprice) - cost
                    CurrencyBalance = CurrencyBalance - (CurrencyBalance * pcost)
                    StockBalance = 0

                    # price = dfsell.at[mydatef]
                    # print("comparing "+ str(price) + " and "+ str(transprice))

                    # self.LastEntryRecommendation = [self.ticker.getSymbol(), mydatef, "{:.7f}".format(price), LastTransaction, str((EndDay - mydate).days), ODR]
                    # print(CurrencyBalance)
                    # print(mydate)


            ############################ up down trend test
            if LMKtest:
                if StockBalanceLMK == 0:
                    # BUY
                    if buyLMK == True:  #LMKcount == 1
                        started = True
                        # print('buy')

                        # print( str(mydatef) + 'BUY at' + str(dfbuy.at[mydatef]) + ', using ' + str(CurrencyBalance))
                        # print ()
                        CurrencyBalanceLMK = (CurrencyBalanceLMK - cost) - (CurrencyBalanceLMK * pcost)
                        # price =  dfbuy.at[mydatef]
                        # print("comparing "+ str(price) + " and "+ str(transprice))

                        StockBalanceLMK = (CurrencyBalanceLMK) / transprice
                        CurrencyBalanceLMK = 0


                        print(StockBalanceLMK)
                else:
                    # SELL
                    if sellLMK == True and started:  #LMKcount == 1 and
                        # print('sell')
                        # print(mydatef + 'SELL at' + str(dfsell.at[mydatef]) + ', using stocks: ' + str(StockBalance))
                        #     print(ticker.getPrice(tomorrowf))
                        #     print(ticker.getPrice(mydatef))

                        CurrencyBalanceLMK = (StockBalanceLMK * transprice) - cost
                        CurrencyBalanceLMK = CurrencyBalanceLMK - (CurrencyBalanceLMK * pcost)
                        StockBalanceLMK = 0

                        # price = dfsell.at[mydatef]
                        # print("comparing "+ str(price) + " and "+ str(transprice))

                        # self.LastEntryRecommendation = [self.ticker.getSymbol(), mydatef, "{:.7f}".format(price), LastTransaction, str((EndDay - mydate).days), ODR]
                        # print(CurrencyBalance)
                        # print(mydate)




            price = ticker.getPrice(mydate.name)
            balances.append([mydate.name, CurrencyBalance, StockBalance, price, LastTransaction, ODR, btm, top])

            # balancesLMK.append([mydatef, CurrencyBalance, StockBalance, price, trend, buyLMK, sellLMK])

            # keeping track of worth
            if CurrencyBalance == 0:
                worth_list.append(StockBalance * ticker.getPrice(mydate.name))
            else:
                worth_list.append(CurrencyBalance)

            ODR = ""

        for item in balances[-10:]:
            w.print(','.join(map(str, item)))
        # saveToDisk(balances)
        self.balances = balances


        # for item in balancesLMK[-40:]:
        #     w.print(','.join(map(str, item)))
        # # saveToDisk(balancesLMK)
        # self.balancesLMK = balancesLMK

        w.print("Last one day reversal: " + str(odr.tail(1).index[0].strftime('%Y-%m-%d')))  #

        w.print("Today's price: " + str(h['Close'].iloc[-1]))

        w.print('Stocks: ' + str(StockBalance))
        w.print('Currency: ' + str(CurrencyBalance))
        stockValue = CurrencyBalance

        if StockBalance != 0:
            w.print("Stocks equivalent to : " + str(StockBalance * h['Close'].iloc[-1]))
            stockValue = StockBalance * h['Close'].iloc[-1]

        w.print("Without action, current day worth: " + str(StartCurrency / h['Close'].iloc[0] * h['Close'].iloc[-1]))

        unchangedStockValue = StartCurrency / h['Close'].iloc[0] * h['Close'].iloc[-1]



        ################################################
        if LMKtest:
            w.print('StocksLMK: ' + str(StockBalanceLMK))
            w.print('CurrencyLMK: ' + str(CurrencyBalanceLMK))
            stockValueLMK = CurrencyBalanceLMK




        self.gain = "{:.0%}".format((stockValue / unchangedStockValue))

        # add worth to the graph
        worth['Worth'] = worth_list

        ticker.setWorth(worth_list)


        return w


    def saveToDisk(balances):
        # todo get ticker name
        # todo where is the file written?
        with open('ticker.csv', 'w') as f:
            # using csv.writer method from CSV package
            wr = csv.writer(f)
            wr.writerow(balances)

    def getBalance(self):
        return self.balances

    def getLastRecommendation(self):

        return self.LastEntryRecommendation

    def getGain(self):
        return self.gain


    def getBuySell(self):
        buy, sell, odr, h = self.ticker.getEntryExit()

        return h[h["Buy"] | h["Sell"]]
        # h[h["Top"] | h["Btm"]]


    def getLastRecInfo(self):
        recs = self.getBuySell()
        # compare index of last_rec wth last h index:
        rec_date = recs.index[-1]
        h_date = self.ticker.history.index[-1]
        rec_days_ago = (h_date - rec_date).days

        if recs['Buy'].iloc[-1]:
            last_rec = 'Buy'
        if recs['Sell'].iloc[-1]:
            last_rec = 'Sell'

        # returns last recommendation, days ago, closing price at that time
        return [h_date.strftime('%Y-%m-%d'), last_rec, rec_days_ago,recs['Close'].iloc[-1]]

    def getInterativeBacktestData(self):

        last_rec = self.getLastRecInfo()  #list
        last_hist = self.ticker.getLastHistory()  #

        for item in last_hist:
            last_rec.append(item)

        return last_rec





