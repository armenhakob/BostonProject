import pickle
from random_models import RandomModels
from best_model import BestModel

model = pickle.load(open('BestModel.pkl', 'rb'))

if __name__ == '__main__':
    model = pickle.load(open('BestModel.pkl', 'rb'))
    model.fit_predict_score()
