import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor

# Global
pd.options.mode.chained_assignment = None


class Predictor(object):

    def __init__(self, dataframe, inputs=None, model=None, filename=None):

        self.frame = dataframe
        self.inputs = inputs
        self.model = model
        self.file = filename

    def train(self, size):

        # Data
        x, y, df = Predictor._get_xyframe(self)

        # Split into training and testing groups
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, shuffle=False)

        # Regression object
        m = Predictor._get_regressor(self)

        # Train model using training sets
        m.fit(x_train, y_train)

        # Save model
        joblib.dump(m, "Models/" + self.file + ".pkl")

        return df

    def predict(self):

        # Load the model to be used
        m = joblib.load("Models/" + self.file + ".pkl")

        # Perform prediction
        predictions = m.predict(self.frame[self.inputs].values)

        # Return column in dataframe
        self.frame['Predictions'] = np.array(predictions)

        return self.frame

    def check(self):

        # Data
        x, y, df = Predictor._get_xyframe(self)

        # Results
        fit_loss, pred_loss, num_points = [], [], []

        for sliver in np.linspace(500, len(y), 5):

            # New vectors
            x_new, y_new = x[x.index < sliver], y[y.index < sliver]

            # Split into training and testing groups
            x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=100, shuffle=False)

            # Fit model
            m = Predictor._get_regressor(self)
            m.fit(x_train, y_train)

            # Predict using testing sets
            prediction, validation = m.predict(x_test), m.predict(x_train)

            # Add results
            fit_loss.append(np.sum(np.abs(validation - y_train.as_matrix()))/len(x_new))
            pred_loss.append(np.sum(np.abs(prediction - y_test.as_matrix()))/len(x_test))
            num_points.append(len(x_new))

        return fit_loss, pred_loss, num_points

    def _keys(self):
        k = self.inputs.copy()
        k.append('Time')
        k.append('Exposure')

        return k

    def _get_regressor(self):

        if self.model == 'gbr':
            m = GradientBoostingRegressor(learning_rate=0.4, max_depth=5, n_estimators=500, loss='ls')

        elif self.model == 'nnr':
            m = KNeighborsRegressor(n_neighbors=5, weights='distance')

        elif self.model == 'svr':
            m = SVR(kernel='rbf', C=1e3, epsilon=1e-3, gamma=0.1)

        elif self.model == 'lmr':
            m = LinearRegression()

        elif self.model == 'krr':
            m = KernelRidge(kernel='rbf', gamma=0.1)

        elif self.model == 'brr':
            m = BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
                              fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
                              normalize=False, tol=0.001, verbose=False)

        elif self.model == 'gpr':
            m = GaussianProcessRegressor()

        elif self.model == 'mlp':
            m = MLPRegressor(hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001,
                             batch_size='auto', learning_rate='constant', learning_rate_init=0.01,
                             power_t=0.5, max_iter=1000, shuffle=True, random_state=9, tol=0.0001, verbose=False,
                             warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                             validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        else:
            m = AdaBoostRegressor(learning_rate=1.0, n_estimators=500, loss='linear')

        return m

    def _get_xyframe(self):
        df = self.frame[Predictor._keys(self)]
        x = df[self.inputs]
        y = df['Exposure']

        return x, y, df
