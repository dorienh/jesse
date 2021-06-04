
import chart_studio.plotly as py
from plotly.offline import plot
# todo use renderer instead of offline
from lmk.ticker import ensure_columns_exist




class Finplot:


    def __init__(self, ticker, pivot_window_size, history = [], elements="C,CL,LMK,PV,PVL,ODR", ylimits=None, showFig=False):

        self.pivot_window_size = pivot_window_size

        if len(history) < 1:
            self.history = ticker.getHistory()
        else: self.history = history

        self.ticker = ticker
        """
        elements: elements that should be plotted, see below.
        ylimits: range of y axis. e.g. (0, 100)

        { # element and its dependent columns
          # Basic
          "C"       : ["Close",],   # Mark the tick['Close'] value with point.
          "CL"      : ["Close",],   # Mark the tick['Close'] value as a line.
          "HLC"     : ["High", "Low", "Close"], # Plot the HLC values in the tick.

          # Derived
          "ODR"     : ["Open", "High", "Low", "Close", "Volume"],   # One Day Reversal. Mark the ODR ticks.

          "PV"      : ["Close", "Top", "Btm"],  # Label the pivot point value.
          "PVL"     : ["Close", "Top", "Btm"],  # Plot a Line that connects the pivot points.

          "EE"      : ["Buy", "Sell"],  # Mark the idea Buy/Sell point.

          "BAND"    : ["Band", "WM"],   # Mark the point with its current LMK Band Level.
          "BANDL"   : ["Band", "Buy", "Sell"],  # Mark line segment according to its current band level.
          "WM"      : ["Trend", "WM"],  # Mark the current resistant or support line
        }
        """

        # https://plotly.com/python/candlestick-charts/

        h = self.history

        # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
        # https: // chart - studio.plotly.com / ~jackp / 17421 / plotly - candlestick - chart - in -python /  # /

        INCREASING_COLOR = '#87C38F'
        DECREASING_COLOR = '#DA2C38'

        data = [dict(
            type='candlestick',
            open=h.Open,
            high=h.High,
            low=h.Low,
            close=h.Close,
            x=h.index,
            yaxis='y2',
            name='GS',
            increasing=dict(line=dict(color=INCREASING_COLOR)),
            decreasing=dict(line=dict(color=DECREASING_COLOR)), visible='legendonly'
        )]

        layout = dict()

        fig = dict(data=data, layout=layout)

        # create layout object
        fig['layout'] = dict()
        fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
        fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True))
        fig['layout']['yaxis'] = dict(domain=[0, 0.2], showticklabels=False)
        fig['layout']['yaxis2'] = dict(domain=[0.2, 0.8], fixedrange=False)
        fig['layout']['legend'] = dict(orientation='h', y=0.84, x=.6, yanchor='bottom')  # , ,  )
        fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)
        fig['layout']['title'] = ticker.getSymbol()

        # add range buttong
        rangeselector = dict(
            visibe=True,
            x=0, y=0.9,
            bgcolor='rgba(150, 200, 250, 0.4)',
            font=dict(size=13),
            buttons=list([
                dict(count=1,
                     label='reset',
                     step='all'),
                dict(count=1,
                     label='1yr',
                     step='year',
                     stepmode='backward'),
                dict(count=3,
                     label='3 mo',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='1 mo',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ]))

        fig['layout']['xaxis']['rangeselector'] = rangeselector

        # def zoom(layout, xrange):
        #     in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
        #     fig.layout.yaxis.range = [in_view.High.min() - 10, in_view.High.max() + 10]
        #
        # fig.layout.on_change(zoom, 'xaxis.range')

        # set volume  bar chart colors
        colors = []

        for i in range(len(h.Close)):
            if i != 0:
                if h.Close.iloc[i] > h.Close.iloc[i - 1]:
                    colors.append(INCREASING_COLOR)
                else:
                    colors.append(DECREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)

        # add volume bar
        fig['data'].append(dict(x=h.index, y=h.Volume,
                                marker=dict(color=colors),
                                type='bar', yaxis='y', name='Volume'))

        # --------------------------------------------------------------

        h = ensure_columns_exist(h, ["CC", "ODR", "Top", "Btm", "Buy", "Sell"], self.pivot_window_size)

        buy = h[h["Buy"]]
        sell = h[h["Sell"]]
        odr = h[h["ODR"]]

        # https://plotly.com/python/line-and-scatter/
        # todo hide candlesticks by default

        fig['data'].append(dict(x=buy.index, y=buy.Close, type='scatter', mode='markers',
                                line=dict(width=1), legendgroup='rec',
                                marker=dict(color='#4B9555', size='10', symbol='cross'),
                                yaxis='y2', name='Buy'))

        fig['data'].append(dict(x=sell.index, y=sell.Close, type='scatter', mode='markers',
                                line=dict(width=1), legendgroup='rec',
                                marker=dict(color='#AD1F28', size='10', symbol='x'),
                                yaxis='y2', name='Sell'))


        if 'Buy_pred' in h.columns:
            buyp = h[h["Buy_pred"] > 0.95]
            sellp = h[h["Sell_pred"]> 0.95]
            fig['data'].append(dict(x=buyp.index, y=buyp.Close, type='scatter', mode='markers',
                                    line=dict(width=1), legendgroup='recp',
                                    marker=dict(color='#4B9555', size='10', symbol='cross'),
                                    yaxis='y2', name='Buy_pred'))

            fig['data'].append(dict(x=sellp.index, y=sellp.Close, type='scatter', mode='markers',
                                    line=dict(width=1), legendgroup='recp',
                                    marker=dict(color='#AD1F28', size='10', symbol='x'),
                                    yaxis='y2', name='Sell_pred'))

        fig['data'].append(dict(x=odr.index, y=odr.Close, text='ODR', type='scatter', mode='markers', legendgroup='rec',
                                line=dict(width=1), marker=dict(color='#F4F0BB', size='10', symbol='ODR', label='odr'),
                                yaxis='y2', name='ODR'))

        h = ensure_columns_exist(h, ["WM", "Trend", "Band"], self.pivot_window_size)

        top = h[h["Top"]]
        btm = h[h["Btm"]]

        fig['data'].append(dict(x=h.index, y=h.Close, text='Close', type='scatter', mode='line',
                                line=dict(width=1), marker=dict(color='#000000'),
                                yaxis='y2', name='Close'))

        # todo allow for atr ration not set to 1 (i.e. load it from ticker)
        atr_up = h['last_pivot'] + h['ATR'] / 2
        atr_down = h['last_pivot'] - h['ATR'] / 2

        fig['data'].append(dict(x=atr_down.index, y=atr_down, type='scatter', yaxis='y2',
                                line=dict(width=1),
                                marker=dict(color='grey'), hoverinfo='none',
                                legendgroup='ATR', name='Entry/Exit Band'))

        # visible': 'legendonly'

        fig['data'].append(dict(x=atr_up.index, y=atr_up, type='scatter', fill='tonexty', yaxis='y2',
                                line=dict(width=1),
                                marker=dict(color='grey'), hoverinfo='none',
                                legendgroup='ATR', name='Entry/Exit Band'))

        # top and bottoms
        fig['data'].append(dict(x=top.index, y=top.Close, text='Top', type='scatter', mode='markers', legendgroup='pivot',
                                visible='legendonly',
                                marker=dict(color='LightSkyBlue', size=10, symbol='triangle-up', opacity=0.8,
                                            line=dict(color='MediumPurple', width=1)),
                                yaxis='y2', name='Top'))

        fig['data'].append(dict(x=btm.index, y=btm.Close, text='Btm', type='scatter', mode='markers', legendgroup='pivot',
                                visible='legendonly',
                                marker=dict(color='LightSkyBlue', size=10, symbol='triangle-down', opacity=0.8,
                                            line=dict(color='MediumPurple', width=1)),
                                yaxis='y2', name='Btm'))

        # todo figure out other elements and bands

        # plot gains to fig

        if "Worth" in h:
               fig['data'].append(
                dict(x=h.index, y=h.Worth, text='Close', type='scatter', mode='line', visible='legendonly',
                     line=dict(width=1), marker=dict(color='#B27157'),
                     yaxis='y2', name='Worth'))

        # fig['data'].append(dict(x=h.index, y=h.last_pivot, type='scatter', yaxis='y2',
        #                      line=dict(width=1),
        #                      marker=dict(color='#ccc'), hoverinfo='none',
        #                      legendgroup='Last pivot', name='Last pivot'))

        # fig['data'].append(dict(x=h.index, y=h.ATR, type='scatter', yaxis='y2',
        #                         line=dict(width=1),
        #                         marker=dict(color='#eee'), hoverinfo='none',
        #                         legendgroup='ATR', name='ATR'))

        # Set x-axis title
        # fig.update_xaxes(title_text="xaxis title")
        #
        # # Set y-axes titles
        # fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
        # fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)


        # # -- Pivots --
        # @plot_elements("PV")
        # @columns("Top", "Btm", "Close", "Low", "High")
        # def plot_PV(ax, h):
        #     pivots = h[h["Top"] | h["Btm"]]
        #     for x, tick in pivots.iterrows():
        #         label = "%.2f" % tick["Close"]
        #         if tick["Top"]:  # crest
        #             y = tick["High"]
        #             # ax.text(x, y, label, color="g", alpha=.8)
        #         else:  # trough
        #             y = tick["Low"]
        #             # ax.text(x, y, label, color="r", alpha=.8)
        #
        # @plot_elements("PVL")
        # @columns("Top", "Btm", "Close")
        # def plot_PVL(ax, h):
        #     pivots = h[h["Top"] | h["Btm"]]
        #     # ax.plot(pivots.index, pivots["Close"], "-", color="blue", alpha=.3)
        #     r = h[h["Top"]]
        #     # ax.plot(r.index, r["Close"], "g^", alpha=1.0)
        #     r = h[h["Btm"]]
        #     # ax.plot(r.index, r["Close"], "rv", alpha=1.0)
        #
        # # -- LMK --
        # BAND_STYLE_MAP = {
        #     BAND_DNWARD: "rv",
        #     BAND_NAT_REACT: "m<",
        #     BAND_SEC_REACT: "m*",
        #     BAND_SEC_RALLY: "c*",
        #     BAND_NAT_RALLY: "c>",
        #     BAND_UPWARD: "g^",
        # }
        #
        # @plot_elements("BAND", "LMK")
        # @columns("Close", "WM", "Band")
        # def plot_BAND(ax, h):
        #     for band in range(BAND_DNWARD, BAND_UPWARD + 1):
        #         # if band in (BAND_SEC_REACT, BAND_SEC_RALLY): continue
        #         r = h[(h["WM"] == h["Close"]) & (h["Band"] == band)]
        #         # ax0.plot(r.index, r["Close"], BAND_STYLE_MAP[band], alpha=1.0)
        #
        # @plot_elements("BANDL")
        # @columns("Band", "Buy", "Sell", "Close")
        # def plot_BANDL(ax, h):
        #     chosen = ma.masked_where(~(h["Band"] >= BAND_NAT_RALLY), h["Close"])
        #     # if chosen.any():
        #         # ax.plot(h.index, chosen, "g-", linewidth=1, alpha=1)
        #
        #     chosen = ma.masked_where(~(h["Band"] <= BAND_NAT_REACT), h["Close"])
        #     # if chosen.any():
        #         # ax.plot(h.index, chosen, "r-", linewidth=1, alpha=1)
        #
        # @plot_elements("WM")
        # @columns("Trend", "WM", "ATR")
        # def plot_WM(ax, h):
        #     chosen = ma.masked_where(~(h['Trend'] == TREND_UP), h["WM"])
        #     # ax.plot(h.index, chosen, drawstyle="steps-post", color="g", linewidth=1.5)
        #     chosen = ma.masked_where(~(h['Trend'] == TREND_UP), h["WM"] - h["ATR"] * 2.0)
        #     # ax.plot(h.index, chosen, drawstyle="steps-post", color="r", alpha=.5)
        #     chosen = ma.masked_where(~(h['Trend'] == TREND_UP), h["WM"] - h["ATR"])
        #     # ax.plot(h.index, chosen, drawstyle="steps-post", color="b", alpha=.2)
        #
        #     chosen = ma.masked_where(~(h['Trend'] == TREND_DN), h["WM"])
        #     # ax.plot(h.index, chosen, drawstyle="steps-post", color="r", linewidth=1.5)
        #     chosen = ma.masked_where(~(h['Trend'] == TREND_DN), h["WM"] + h["ATR"] * 2.0)
        #     # ax.plot(h.index, chosen, drawstyle="steps-post", color="g", alpha=.5)
        #     chosen = ma.masked_where(~(h['Trend'] == TREND_DN), h["WM"] + h["ATR"])
        #     # ax.plot(h.index, chosen, drawstyle="steps-post", color="b", alpha=.2)
        #
        # @plot_elements("EE", "BS")
        # @columns("Buy", "Sell", "Close")
        # def plot_EE(ax, h):
        #     r = h[h["Buy"]]
        #     # ax0.plot(r.index, r["Close"], "g+", markersize=8, markeredgewidth=3, alpha=1)
        #     r = h[h["Sell"]]
        #     # ax0.plot(r.index, r["Close"], "r_", markersize=8, markeredgewidth=3, alpha=1)
        #
        # # Build the plotting function map ...
        # plot_functions = [f for f in locals().values() if callable(f) and hasattr(f, "elements")]
        # plot_dict = {}
        # for f in plot_functions:
        #     for element in f.elements:
        #         plot_dict[element] = f
        #
        # # do the real plotting ...
        # l = re.split("[-:,;.]", elements)
        # for c in l:
        #     _plot = plot_dict[c]
        # if c != "V":
        #     _plot(ax0, h)
        # else:
        #     _plot(ax1, h)

        ##end to port
        # moviing aaveage
        # def movingaverage(interval, window_size=10):
        #     window = np.ones(int(window_size)) / float(window_size)
        #     return np.convolve(interval, window, 'same')
        #
        # mv_y = movingaverage(df.Close)
        # mv_x = list(df.index)
        #
        # # Clip the ends
        # mv_x = mv_x[5:-5]
        # mv_y = mv_y[5:-5]
        #
        # fig['data'].append(dict(x=mv_x, y=mv_y, type='scatter', mode='lines',
        #                         line=dict(width=1),
        #                         marker=dict(color='#E377C2'),
        #                         yaxis='y2', name='Moving Average'))

        # bollinger bands
        # def bbands(price, window_size=10, num_of_std=5):
        #     rolling_mean = price.rolling(window=window_size).mean()
        #     rolling_std = price.rolling(window=window_size).std()
        #     upper_band = rolling_mean + (rolling_std * num_of_std)
        #     lower_band = rolling_mean - (rolling_std * num_of_std)
        #     return rolling_mean, upper_band, lower_band
        #
        # bb_avg, bb_upper, bb_lower = bbands(df.Close)
        #
        # fig['data'].append(dict(x=df.index, y=bb_upper, type='scatter', yaxis='y2',
        #                         line=dict(width=1),
        #                         marker=dict(color='#ccc'), hoverinfo='none',
        #                         legendgroup='Bollinger Bands', name='Bollinger Bands'))
        #
        # fig['data'].append(dict(x=df.index, y=bb_lower, type='scatter', yaxis='y2',
        #                         line=dict(width=1),
        #                         marker=dict(color='#ccc'), hoverinfo='none',
        #                         legendgroup='Bollinger Bands', showlegend=False))

        # py.iplot(fig)
        # plot

        # check https://plotly.com/python/creating-and-updating-figures/

        self.history = h
        self.fig = fig



    def saveFig(self, filename = ''):

        if filename == '':
            filename = './analyses/charts/' + self.ticker.symbol + '.html'

        plot(self.fig, filename=filename, validate=False, auto_open=False)
