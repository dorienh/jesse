import os
from lmk.ml import tools
import pandas as pd
from lmk.backtest.Finplot import Finplot


# script to preprocess data for the ML

if __name__ == '__main__':

    from scripts.main import analyses, testStock, testStockIterativelyFast, testML
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
    os.chdir(ROOT_DIR)

    error = False
    errorlist=[]


    #correct dataset: reverse csv files of folder and remove extra first line
    # gemini binance bittrex bitfinex poloniex
    directory = '../datasource/ml_candle/new/hourly/poloniex'
    save_directory = '../datasource/ml_final_dataset/hourly/'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # tools.reverseFile(f)
        print("Processing " + f)

        df = tools.preprocessFile(f)
        df = tools.addEntries(df)
        df, error = tools.addTechnical_indicators(df)


        # worth = tools.backtestEntries(df, Buyname='Buy_p15_a4', Sellname='Sell_p15_a4')
        # worth = tools.backtestEntries(df, Buyname='Buy_p40_a1', Sellname='Sell_p40_a1')
        # worth = tools.backtestEntries(df, Buyname='Buy_p300_a1', Sellname='Sell_p300_a1')


        # finplot = Finplot("", 40, history=df, showFig=True)
        # finplot.saveFig(filename='./analyses/charts/' + df['symbol'].iloc[1] + "_p40_a1.html")
        if not error:
            if (len(df.columns) != 206):
                error = True
                errorlist.append(f)
            else:
                print(len(df.columns))
                df.to_csv(save_directory+filename[:-2])
        else:
            print('error: ')
            print(len(df.columns))
            errorlist.append(f)
        del df

    print("ERRORLIST")
    print(errorlist)

    #Save to: datasource: ml_extended




    ### BACKTEST ML ###
    #todo: predict pTop and pBtm
    #df = tools.setSignals(df, Topname = 'pTop', Btmname = 'pBtm')
    # worth = tools.backtestEntries(df, Buyname='Buy', Sellname='Sell')


'''
BEST SETTINGS:

- HOURLY: p40_a1
- DAILY: p40_a1
- MIN: 


'''