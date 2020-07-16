import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


####Imports
df = pd.read_csv('cutmodel2.csv')
df= df.drop(columns=['Unnamed: 0'])
df = df.sample(20000)


####  split and report
X = df.drop(['Price'], axis=1)
df_list = list(X.columns)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('data split')

#### Create Model save and report
reg = LinearRegression().fit(X, y)
reg.fit(X_train, y_train)
filename = 'OLS_model.sav'
joblib.dump(reg, filename + '.gz', compress=('gzip', 3))
print('model finished and saved')


print('Training Labels Shape:', X_test.shape)
###### Predictions and Diagnostics
y_pred = reg.predict(X_test)
# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')



######## RF Equivlant ##########################
rfc = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfc.fit(X_train,y_train)
filename = 'RF_model.sav'
joblib.dump(rfc, filename + '.gz', compress=('gzip', 3))
#rfc = pickle.load(open(filename, 'rb'))
print('model finished and saved')

print('Training Labels Shape:', X_test.shape)
#### Create predictions save and report
predictions = rfc.predict(X_test)
filename2 = 'Predictions_model4.sav'
print('predictions finished and saved')

###### Model Diagnostics
errors = abs(predictions - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)

print('Accuracy:', round(accuracy, 2), '%.')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('Accuracy:', round(accuracy, 2), '%.')

### Print Estimator scores

importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(df_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
