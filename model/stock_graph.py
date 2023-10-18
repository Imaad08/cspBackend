import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objs as go
from datetime import datetime


def get_stock_graph(stock_name):
    end_date = datetime.now()
    # Fetch data for the last year
    start_date = end_date - pd.Timedelta(days=365)

    # Fetch stock data using pandas_datareader
    df = web.DataReader(stock_name, 'yahoo', start_date, end_date)

    # Create a Plotly figure using fetched stock data
    graph = go.Figure(data=go.Candlestick(x=df.index,
                                          open=df['Open'],
                                          high=df['High'],
                                          low=df['Low'],
                                          close=df['Close'],
                                          name=stock_name))
    graph.update_layout(title=f'{stock_name} Stock Price',
                        xaxis_title='Date',
                        yaxis_title='Price')

    return graph
