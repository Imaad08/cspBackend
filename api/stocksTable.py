from flask import Blueprint
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
from flask_restful import Api, Resource
import json

stock_api = Blueprint('stock_api', __name__, url_prefix='/api/stocks')
api = Api(stock_api)


def get_stock_graph(stock_name):
    end_date = datetime.now()
    # Calculate the start date for the last 7 days
    start_date = end_date - timedelta(days=7)

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

    # Handle datetime objects in the graph_data
    for trace in graph_data['data']:
        for key, value in trace.items():
            if isinstance(value, datetime):
                trace[key] = value.isoformat()

    response_data = {
        "candlestick": graph_data
    }

    json_data = json.dumps(response_data)
    return json_data


class _ReadStockGraph(Resource):
    def get(self, stock_name):
        graph = get_stock_graph(stock_name)
        return graph


api.add_resource(_ReadStockGraph, '/stock_graph/<string:stock_name>')
