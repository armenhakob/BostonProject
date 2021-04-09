import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model, tree, neighbors, ensemble, svm
import pickle


class RandomModels:

    classifiers = [
        svm.SVR(),
        linear_model.SGDRegressor(),
        linear_model.BayesianRidge(),
        linear_model.LassoLars(),
        linear_model.ARDRegression(),
        linear_model.PassiveAggressiveRegressor(),
        linear_model.TheilSenRegressor(),
        linear_model.LinearRegression(),
        tree.DecisionTreeRegressor(),
        tree.ExtraTreeRegressor(),
        neighbors.KNeighborsRegressor(),
        ensemble.RandomForestRegressor()]

    def data_normalizer(self):
        address = 'Boston.txt'
        with open(address) as d_line:
            text = [line for line in d_line.readlines()]
            start_row = 0
        new_rows = []
        for i, l in enumerate(text[start_row:]):
            if not i % 2:
                new_line = l.strip('\n') + text[start_row + i + 1]
                new_rows.append(new_line)
        new_data = ''.join(new_rows)
        with open('boston_new.csv', 'w') as f:
            f.write(new_data)

    def load_the_data(self):
        self.data_normalizer()
        address = 'boston_new.csv'
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                        'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                        'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data = pd.read_csv(address, delim_whitespace=True, header=None, names=column_names)
        return data

    def detect_and_remove_outliers(self):
        data = self.load_the_data()
        data = data[data['RM'].apply(lambda x: 7.8 > x > 4.7)]
        data = data[data['LSTAT'].apply(lambda x: x < 31)]
        data = data[data['PTRATIO'].apply(lambda x: x > 13.5)]
        return data

    def split_data(self):
        # data = self.detect_and_remove_outliers()
        data = self.load_the_data()
        X = data.drop('MEDV', axis=1)
        y = data['MEDV']
        return X, y

    def scale(self):
        X, _ = self.split_data()
        std_scaler = preprocessing.StandardScaler()
        X = std_scaler.fit_transform(X)
        X = pd.DataFrame(X)
        return X

    def data_train_test_split(self):
        _, y = self.split_data()
        X = self.scale()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def model_chooser(self):
        models = {}
        for i in range(len(self.classifiers)):
            models[f'model_{i + 1}'] = self.classifiers[i]

        return models

    def fit(self):
        X_train, X_test, y_train, y_test = self.data_train_test_split()
        models = self.model_chooser()
        fitted = {}
        for key, value in models.items():
            fitted[key] = value.fit(X_train, y_train)
        return fitted, X_train, X_test, y_train, y_test

    def predict(self):
        pass

    def score(self):
        fitted, X_train, X_test, y_train, y_test = self.fit()
        scores = []
        for key, value in fitted.items():
            scores.append(value.score(X_test, y_test))
        # print(max(scores))
        return max(scores)


if __name__ == '__main__':
    model_1 = RandomModels()
    print(model_1.score())
    pickle.dump(model_1, open('../pkl/model.pkl', 'wb'))
