
import flask
from flask import Flask, jsonify, request
import json
import pickle
from model.best_model import BestModel
from sklearn.ensemble import RandomForestRegressor
from model import *
app = Flask(__name__)


def load_model():
    file_name = "model/BestModel.pkl"
    model = pickle.load(open(file_name, 'rb'))
    return str(model.fit_predict_score())

@app.route('/score')
def score():
    return load_model()


@app.route('/gago')
def ayo_gago():
    return "Hello Gago"


@app.route('/predict', methods=['GET'])
def predict():
    # load_model()
    # # parse the input features from the JSON Request
    # request_json = request.get_json()
    #
    # x = float(request_json['input'])
    #
    # # load the model
    # model = load_model()
    # prediction = model.predict([[x]])[0]
    #
    # response = json.dumps({'response': prediction})
    file_name = "/Users/armenhakobyan/PycharmProjects/BostonProject/model/model.pkl"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
    return jsonify(data)


# @app.errorhandler(500)
# def internal_error(error):
#     return '500', 500


if __name__ == "__main__":
    app.run(host='localhost', port=5000)
    app.run(debug=True)
