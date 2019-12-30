import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import eikon as ek
import pandas as pd
import matplotlib.pyplot as plt
ek.set_app_key('b47a7929747c40149d93089db6b3e895a3f8bca5')
df = ek.get_timeseries(["CHRS.O"], start_date = "2001-10-10", end_date = "2019-02-05")
print(df)
df = df[['CLOSE']]
print(df)
forecast_out = 30
df['Prediction'] = df[['CLOSE']].shift(-forecast_out)
print(df)
X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)
y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence) 
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)
