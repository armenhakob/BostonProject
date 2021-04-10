
import flask
from flask import Flask, jsonify, request
from sklearn import linear_model, tree, neighbors, ensemble, svm
import json
import pickle
from model.best_model import BestModel
from model.random_models import RandomModels
from model.feature_metrics import FeatureMetrics
from sklearn.ensemble import RandomForestRegressor
from model import *
app = Flask(__name__)


def load_model():
    file_name = "/Users/armenhakobyan/PycharmProjects/BostonProject/pkl/BestModel.pkl"
    model = pickle.load(open(file_name, 'rb'))
    best_score = model.fit_predict_score()
    return str(best_score)


@app.route('/')
def hello():
    return "hello"


@app.route('/score', methods=['GET'])
def score():
    return load_model()


@app.route('/random')
def random_models():
    model = RandomModels()
    m = model.score()
    return str(m)


@app.route('/metrics')
def feature_importance():
    metrics = FeatureMetrics()
    data = metrics.load_data()
    X, y = metrics.split_data(data)
    scores = metrics.feature_score_counter(X, y)
    feature_importance = metrics.feature_importance_counter(X, y)
    return f' here is feature importance dict:\n{feature_importance},\nAnd here are feature scores:\n {scores}'


@app.route('/gago')
def ayo_gago():
    return "Hello Gago"


@app.errorhandler(500)
def internal_error(error):
    return 'something went wrong', 500


if __name__ == "__main__":
    app.run(host='localhost', port=5000)
    app.run(debug=True)
