from flask_cors import CORS
import threading

# import "packages" from flask
# import render_template from "public" flask libraries
from flask import render_template

# import "packages" from "this" project
from __init__ import app, db  # Definitions initialization
from model.jokes import initJokes


# setup APIs
from api.covid import covid_api  # Blueprint import api definition
from api.joke import joke_api  # Blueprint import api definition
# from api.user import user_api  # Blueprint import api definition
# from api.player import player_api
from api.stocks import stock_api


# setup App pages
# Blueprint directory import projects definition
from projects.projects import app_projects


# Initialize the SQLAlchemy object to work with the Flask app instance
db.init_app(app)

# register URIs
app.register_blueprint(joke_api)  # register api routes
app.register_blueprint(covid_api)  # register api routes
# app.register_blueprint(user_api)  # register api routes
# app.register_blueprint(player_api)
app.register_blueprint(app_projects)  # register app pages
app.register_blueprint(stock_api)

CORS(app, resources={r"/api/stocks/*": {"origins": "http://127.0.0.1:4200"}})


@app.errorhandler(404)  # catch for URL not found
def page_not_found(e):
    # note that we set the 404 status explicitly
    return render_template('404.html'), 404


@app.route('/stock_graph/')
def stock_graph():
    return render_template('stock_graph.html')


@app.route('/')  # connects default URL to index() function
def index():
    return render_template("index.html")


@app.route('/table/')  # connects /stub/ URL to stub() function
def table():
    return render_template("table.html")


@app.before_request
def activate_job():  # activate these items
    initJokes()
    # initUsers()
    # initPlayers()


# this runs the application on the development server
if __name__ == "__main__":
    # change name for testing
    from flask_cors import CORS
    cors = CORS(app)
    app.run(debug=True, host="0.0.0.0", port="8765")
