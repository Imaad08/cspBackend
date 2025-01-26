from flask import Blueprint, jsonify, request
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
from flask_restful import Api, Resource
import numpy as np
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import random
from flask import make_response


stock_api = Blueprint('stock_api', __name__, url_prefix='/api/stocks')
api = Api(stock_api)

CORS(stock_api)


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


class _GetLatestStockData(Resource):
    def get(self, stock_name):
        try:

            stock = yf.Ticker(stock_name)
            latest_data = stock.history(period="1d")

            if latest_data.empty:
                return jsonify({'error': 'No data found for the provided stock ticker.'}), 404

            latest_data = latest_data[['Open', 'High', 'Low', 'Close']]
            latest_data = latest_data.rename(
                columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})

            latest_data_dict = latest_data.to_dict(orient='records')[0]

            for key, value in latest_data_dict.items():
                if isinstance(value, (float, int)):
                    latest_data_dict[key] = round(value, 2)

            return jsonify(latest_data_dict)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


api.add_resource(_GetLatestStockData, '/latest_data/<string:stock_name>')

api.add_resource(_ReadStockGraph, '/stock_graph/<string:stock_name>')


def train_stock_prediction_model(stock_name):
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=4856)

    df = yf.download(stock_name, start=start_date, end=end_date)

    # preprocess the data
    df['Close'] = df['Close'].pct_change()  # calculate daily returns
    df = df.dropna()

    X = df[['Open', 'High', 'Low', 'Close']].values
    y = (df['Close'] > 0).astype(int)  # 1 if price increase 0 if decreased

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # create and train a DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model

# modify the get_stock_recommendation function


def get_stock_recommendation(stock_name, model):
    end_date = datetime.now()
    start_date = end_date - pd.Timedelta(days=1)

    df = yf.download(stock_name, start=start_date, end=end_date)

    if df.empty:
        return {'error': 'No data found for the provided stock ticker.'}, 404

    df['Close'] = df['Close'].pct_change().iloc[-1]

    X = df[['Open', 'High', 'Low', 'Close']].values
    # reshape the input for prediction
    prediction = model.predict(X.reshape(1, -1))

    if prediction == 1:
        recommendation = 'Buy'
        reason = f'The model predicts a price increase based on historical data. The value of prediction is {prediction[0]}. This means there will most likely be an increase in price overall'
    else:
        recommendation = 'Sell/Don\'t Buy'
        reason = f'The model predicts a price decrease based on historical data. The value of prediction is {prediction[0]}. This means there will most likely be a decrease in price overall'

    return {'recommendation': recommendation, 'reason': reason}


class _AnalyzeStock(Resource):
    def get(self, stock_name):
        try:
            model = train_stock_prediction_model(stock_name)
            recommendation = get_stock_recommendation(stock_name, model)
            return jsonify(recommendation)
        except Exception as e:
            return jsonify({'error': str(e)}), 500


api.add_resource(_AnalyzeStock, '/analyze/<string:stock_name>')


class _CalculateOptimalWeights(Resource):
    def post(self):
        data = request.get_json()
        stocks = data.get('stocks', [])

        if len(stocks) < 2:
            return jsonify({'error': 'At least 2 stocks are required to calculate optimal weighting.'}), 400

        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=4856)

        # log_ret datafframe for selected stocks
        log_returns = {}
        for stock_name in stocks:
            df = yf.download(stock_name, start=start_date, end=end_date)
            log_returns[stock_name] = np.log(
                df['Adj Close'] / df['Adj Close'].shift(1))

        log_ret = pd.DataFrame(log_returns)

        # simulate portfolio optimization
        num_ports = 6000
        num_stocks = len(stocks)

        all_weights = np.zeros((num_ports, num_stocks))
        ret_arr = np.zeros(num_ports)
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)

        for x in range(num_ports):
            # randomly generate and iterate weights
            weights = np.array([random.random() for _ in range(num_stocks)])
            weights /= np.sum(weights)
            all_weights[x, :] = weights

            # calculate expected return and volatility
            ret_arr[x] = np.sum(log_ret.mean() * weights) * 252
            vol_arr[x] = np.sqrt(
                np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

            # calculate Sharpe ratio
            sharpe_arr[x] = ret_arr[x] / vol_arr[x]

        # find the portfolio with the maximum sharpe ratio
        max_sharpe_idx = sharpe_arr.argmax()
        max_sharpe_ret = ret_arr[max_sharpe_idx]
        max_sharpe_vol = vol_arr[max_sharpe_idx]
        max_sharpe_weights = all_weights[max_sharpe_idx, :]
        max_sharpe = sharpe_arr[max_sharpe_idx]

        response_data = {
            'return': ret_arr.tolist(),
            'volatility': vol_arr.tolist(),
            'sharpe': sharpe_arr.tolist(),
            'max_sharpe_ret': max_sharpe_ret,
            'max_sharpe_vol': max_sharpe_vol,
            'max_sharpe_weights': max_sharpe_weights.tolist(),
            'max_sharpe': max_sharpe,
        }

        return jsonify(response_data)


api.add_resource(_CalculateOptimalWeights, '/optimal_weights')


class _GetStockPriceFiveYearsAgo(Resource):
    def get(self, stock_name):
        try:
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=5 * 365)

            # Fetch data over a wider range to avoid missing data
            df = yf.download(stock_name, start=start_date, end=start_date + pd.Timedelta(days=10))

            if df.empty:
                return make_response(jsonify({'error': 'No data found for the provided stock ticker.'}), 404)

            # Get the first available closing price
            price_five_years_ago = float(df['Close'].iloc[0])  # Convert to a JSON-serializable scalar

            return jsonify({'stock_name': stock_name, 'price_five_years_ago': round(price_five_years_ago, 2)})
        except Exception as e:
            return make_response(jsonify({'error': str(e)}), 500)



api.add_resource(_GetStockPriceFiveYearsAgo, '/price_five_years_ago/<string:stock_name>')