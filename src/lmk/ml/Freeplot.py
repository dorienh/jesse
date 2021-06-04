
import chart_studio.plotly as py
from plotly.offline import plot
# todo use renderer instead of offline
from lmk.ticker import ensure_columns_exist




class Freeplot:


    def __init__(self, symbol, data, columns, ylimits=None, showFig=False):

        self.history = data
        self.symbol = symbol
        self.columns = columns
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
        fig['layout']['title'] = self.symbol

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

        for col in self.columns:

            thiscol = h[col]
            fig['data'].append(dict(x=thiscol.index, y=thiscol, type='scatter', mode='markers',
                                    line=dict(width=1), legendgroup='rec',
                                    marker=dict(color='#4B9555', size='10', symbol='cross'),
                                    yaxis='y2', name='Buy'))

        # https://plotly.com/python/line-and-scatter/
        # todo hide candlesticks by default


        #
        # fig['data'].append(dict(x=sell.index, y=sell.Close, type='scatter', mode='markers',
        #                         line=dict(width=1), legendgroup='rec',
        #                         marker=dict(color='#AD1F28', size='10', symbol='x'),
        #                         yaxis='y2', name='Sell'))
        #
        #
        # if 'Buy_pred' in h.columns:
        #     buyp = h[h["Buy_pred"] > 0.95]
        #     sellp = h[h["Sell_pred"]> 0.95]
        #     fig['data'].append(dict(x=buyp.index, y=buyp.Close, type='scatter', mode='markers',
        #                             line=dict(width=1), legendgroup='recp',
        #                             marker=dict(color='#4B9555', size='10', symbol='cross'),
        #                             yaxis='y2', name='Buy_pred'))
        #
        #     fig['data'].append(dict(x=sellp.index, y=sellp.Close, type='scatter', mode='markers',
        #                             line=dict(width=1), legendgroup='recp',
        #                             marker=dict(color='#AD1F28', size='10', symbol='x'),
        #                             yaxis='y2', name='Sell_pred'))
        #
        # fig['data'].append(dict(x=odr.index, y=odr.Close, text='ODR', type='scatter', mode='markers', legendgroup='rec',
        #                         line=dict(width=1), marker=dict(color='#F4F0BB', size='10', symbol='ODR', label='odr'),
        #                         yaxis='y2', name='ODR'))
        #
        # h = ensure_columns_exist(h, ["WM", "Trend", "Band"], self.pivot_window_size)
        #
        # top = h[h["Top"]]
        # btm = h[h["Btm"]]
        #
        # fig['data'].append(dict(x=h.index, y=h.Close, text='Close', type='scatter', mode='line',
        #                         line=dict(width=1), marker=dict(color='#000000'),
        #                         yaxis='y2', name='Close'))
        #
        # # todo allow for atr ration not set to 1 (i.e. load it from ticker)
        # atr_up = h['last_pivot'] + h['ATR'] / 2
        # atr_down = h['last_pivot'] - h['ATR'] / 2
        #
        # fig['data'].append(dict(x=atr_down.index, y=atr_down, type='scatter', yaxis='y2',
        #                         line=dict(width=1),
        #                         marker=dict(color='grey'), hoverinfo='none',
        #                         legendgroup='ATR', name='Entry/Exit Band'))
        #
        # # visible': 'legendonly'
        #
        # fig['data'].append(dict(x=atr_up.index, y=atr_up, type='scatter', fill='tonexty', yaxis='y2',
        #                         line=dict(width=1),
        #                         marker=dict(color='grey'), hoverinfo='none',
        #                         legendgroup='ATR', name='Entry/Exit Band'))
        #
        # # top and bottoms
        # fig['data'].append(dict(x=top.index, y=top.Close, text='Top', type='scatter', mode='markers', legendgroup='pivot',
        #                         visible='legendonly',
        #                         marker=dict(color='LightSkyBlue', size=10, symbol='triangle-up', opacity=0.8,
        #                                     line=dict(color='MediumPurple', width=1)),
        #                         yaxis='y2', name='Top'))
        #
        # fig['data'].append(dict(x=btm.index, y=btm.Close, text='Btm', type='scatter', mode='markers', legendgroup='pivot',
        #                         visible='legendonly',
        #                         marker=dict(color='LightSkyBlue', size=10, symbol='triangle-down', opacity=0.8,
        #                                     line=dict(color='MediumPurple', width=1)),
        #                         yaxis='y2', name='Btm'))
        #
        # # todo figure out other elements and bands
        #
        # # plot gains to fig
        #
        # if "Worth" in h:
        #        fig['data'].append(
        #         dict(x=h.index, y=h.Worth, text='Close', type='scatter', mode='line', visible='legendonly',
        #              line=dict(width=1), marker=dict(color='#B27157'),
        #              yaxis='y2', name='Worth'))


        self.history = h
        self.fig = fig
        # plot(self.fig, auto_open=True)
        self.saveFig()


    def saveFig(self, filename = ''):

        if filename == '':
            filename = './' + self.symbol + '.html'

        plot(self.fig, filename=filename, validate=False, auto_open=False)
