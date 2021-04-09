
import flask
from flask import Flask, jsonify, request
import json
import pickle
from model.best_model import BestModel
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


@app.route('/gago')
def ayo_gago():
    return "Hello Gago"


@app.errorhandler(500)
def internal_error(error):
    return 'something went wrong', 500


if __name__ == "__main__":
    app.run(host='localhost', port=5000)
    app.run(debug=True)
