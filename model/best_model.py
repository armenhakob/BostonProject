import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import pickle


class BestModel:
    rand_forest = RandomForestRegressor()

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

    def split_data(self):
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.2)
        return X_train, X_test, y_train, y_test

    def fit_predict_score(self):
        X_train, X_test, y_train, y_test = self.data_train_test_split()
        self.rand_forest.fit(X_train, y_train)
        y_pred = self.rand_forest.predict(X_test)
        print(self.rand_forest.score(X_test, y_test))
        return self.rand_forest.score(X_test, y_test)


if __name__ == '__main__':
    model = BestModel()
    model.fit_predict_score()
    pickle.dump(model, open("BestModel.pkl", "wb"))

