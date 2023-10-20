from flask import Blueprint, jsonify
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
from flask_restful import Api, Resource
import numpy as np

stock_api = Blueprint('stock_api', __name__, url_prefix='/api/stocks')
api = Api(stock_api)


def get_stock_graph(stock_name):
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=4856)

    df = yf.download(stock_name, start=start_date, end=end_date)

    graph = go.Figure(data=go.Candlestick(x=df.index,
                                          open=df['Open'],
                                          high=df['High'],
                                          low=df['Low'],
                                          close=df['Close'],
                                          name=stock_name))
    graph.update_layout(title=f'{stock_name} Stock Price',
                        xaxis_title='Date',
                        yaxis_title='Price')

    graph_data = graph.to_dict()
    graph_data['data'][0]['x'] = graph_data['data'][0]['x'].astype(
        str).tolist()
    graph_data['data'][0]['open'] = graph_data['data'][0]['open'].tolist()
    graph_data['data'][0]['high'] = graph_data['data'][0]['high'].tolist()
    graph_data['data'][0]['low'] = graph_data['data'][0]['low'].tolist()
    graph_data['data'][0]['close'] = graph_data['data'][0]['close'].tolist()
    
    return graph_data


class _ReadStockGraph(Resource):
    def get(self, stock_name):
        graph = get_stock_graph(stock_name)
        return graph


api.add_resource(_ReadStockGraph, '/stock_graph/<string:stock_name>')


# Tanay added code

from flask import Blueprint, jsonify, request
import yfinance as yf
import pandas as pd
from flask_restful import Api, Resource
import plotly.graph_objs as go
from datetime import datetime

stock_api = Blueprint('stock_api', __name__, url_prefix='/api/stocks')
api = Api(stock_api)

# Function to get stock graph for a single ticker symbol
def get_stock_graph(stock_name):
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=4856)

    df = yf.download(stock_name, start=start_date, end=end_date)

    graph = go.Figure(data=go.Candlestick(x=df.index,
                                          open=df['Open'],
                                          high=df['High'],
                                          low=df['Low'],
                                          close=df['Close'],
                                          name=stock_name))
    graph.update_layout(title=f'{stock_name} Stock Price',
                        xaxis_title='Date',
                        yaxis_title='Price')

    graph_data = graph.to_dict()
    graph_data['data'][0]['x'] = graph_data['data'][0]['x'].astype(str).tolist()
    graph_data['data'][0]['open'] = graph_data['data'][0]['open'].tolist()
    graph_data['data'][0]['high'] = graph_data['data'][0]['high'].tolist()
    graph_data['data'][0]['low'] = graph_data['data'][0]['low'].tolist()
    graph_data['data'][0]['close'] = graph_data['data'][0]['close'].tolist()

    return graph_data

# Class for fetching stock data for multiple ticker symbols
class _ReadStockGraphs(Resource):
    def post(self):
        ticker_symbols = request.json.get('ticker_symbols', [])
        stock_data = {}
        
        for ticker_symbol in ticker_symbols:
            stock_data[ticker_symbol] = get_stock_graph(ticker_symbol)
        
        return stock_data

# Add the resource for fetching stock data for multiple ticker symbols
api.add_resource(_ReadStockGraphs, '/stock_graphs')

# Add a resource for fetching stock data for a single ticker symbol (unchanged)
api.add_resource(_ReadStockGraph, '/stock_graph/<string:stock_name>')
