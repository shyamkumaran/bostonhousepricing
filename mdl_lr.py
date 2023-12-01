import pandas as pd
from sklearn import linear_model
import os
import numpy as np
import pickle

# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name, sep=";")
    x_parameter = []
    y_parameter = []
    # TODO: Replace the names of the fields 'square foot', 'price' for your own values
    for single_square_feet in data['square_feet']:
        x_parameter.append([float(single_square_feet)])

    for single_price_value in data['price']:
        y_parameter.append(float(single_price_value))
    return x_parameter, y_parameter

# Function for Fitting our data to Linear model
# noinspection PyPep8Naming
def linear_model_main(x_parameters, y_parameters, predict_value):
    # Create linear regression object
    #regr = linear_model.LinearRegression()
    regr.fit(list(x_parameters), list(y_parameters))
    # noinspection PyArgumentList
    #predict_outcome = regr.predict(predict_value)
    predict_outcome = regr.predict(np.array(predict_value).reshape(1, 1))
    predictions = {'intercept': regr.intercept_, 'coefficient': regr.coef_, 'predicted_value': predict_outcome}
    return predictions


X, Y = get_data('input_data.csv')
regr = linear_model.LinearRegression()
predicted_value = 700
result = linear_model_main(np.array(X), np.array(Y), predicted_value)
#pickle.dump(regr,"housemodel.mks")
outfile = "house_pricing.pkl"
with open(outfile, 'wb') as pickle_file:
    pickle.dump(regr, pickle_file)