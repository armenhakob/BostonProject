import pickle
import numpy as np
from numpy.linalg.linalg import LinAlgError
import matplotlib.pyplot as plt
from best_model import BestModel
from random_models import RandomModels

model = pickle.load(open("model.pkl", "rb"))

if __name__ == '__main__':
    model = pickle.load(open("model.pkl", "rb"))
    # print(model.score())
    print(model.score())

