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

from flask import Flask, request, jsonify
import pandas_datareader.data as web
import pandas as pd
import yfinance as yf

app = Flask(__name)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    # Receive a list of ticker symbols from the front end
    ticker_symbols = request.json.get('ticker_symbols', [])

    all_data = {ticker: web.DataReader(ticker, 'stooq') for ticker in ticker_symbols}

    # Extract the 'Adjusted Closing Price' for each symbol
    price = pd.DataFrame({ticker: data['Close'] for ticker, data in all_data.items()})

    # Convert the DataFrame to a JSON response
    response = price.to_json(orient='split')
    
    return response

if __name__ == '__main__':
    app.run()

import requests

# Define the list of ticker symbols to fetch
ticker_symbols = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'AMZN', 'NFLX', 'QCOM', 'SBUX']

# Create a dictionary with the ticker symbols
data = {'ticker_symbols': ticker_symbols}

# Send a POST request to the backend
response = requests.post('http://your-backend-url/get_stock_data', json=data)

# Extract the JSON response
stock_data = response.json()

# Process or display the stock data as needed
print(stock_data)
