import pickle
from model.random_models import RandomModels
from model.best_model import BestModel


if __name__ == '__main__':
    model = pickle.load(open('../pkl/BestModel.pkl', 'rb'))
    model.fit_predict_score()
