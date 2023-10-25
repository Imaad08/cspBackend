from flask import Blueprint, jsonify, request
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

class _GetStockInfo(Resource):
    def get(self, stock_name):
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=4856)
        
        df = yf.download(stock_name, start=start_date, end=end_date)

        stock_info = {
            "symbol": stock_name,
            "data": df.to_dict(),  
        }

        return jsonify(stock_info)

api.add_resource(_GetStockInfo, '/stock_info/<string:stock_name>')

class _UpdateStockData(Resource):
    def put(self, stock_name):
        data = request.get_json()  
        
        stock = yf.Ticker(stock_name)
        history = stock.history(period="1d")  
        updated_data = history.to_dict(orient='records')[0]

        response = {"message": "Stock data updated successfully", "updated_data": updated_data}
        return jsonify(response)

api.add_resource(_UpdateStockData, '/update_stock/<string:stock_name>')

