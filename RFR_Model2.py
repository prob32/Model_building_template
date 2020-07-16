import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn import metrics
import pydotplus
import joblib
from sklearn import tree
import collections
import os


####Imports
df = pd.read_csv('mini_model_load.csv')
df= df.drop(columns=['Unnamed: 0'])
#df = df.sample(20000)


#### test train split and report
X = df.drop(['Price'], axis=1)
train_features = df.drop(['Price'], axis=1)
df_list = list(X.columns)
y = df['Price']
y = np.array(y)
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
print('data split')


#### Create Model save and report
rfc = RandomForestRegressor(n_estimators = 100, random_state = 42)
rfc.fit(X_train,y_train)
filename = 'mini_model_4.sav'
joblib.dump(rfc, filename + '.gz', compress=('gzip', 3))
#rfc = pickle.load(open(filename, 'rb'))
print('model finished and saved')


#### Create predictions save and report
predictions = rfc.predict(X_test)
filename2 = 'Predictions_model4.sav'
joblib.dump(predictions, filename2)
#predictions = pickle.load(open(filename2, 'rb'))
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

'''
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df3 = df3.head(25)
print(df3)
df3.to_csv('65_RF_model_PA.csv')

feature_results = pd.DataFrame({'feature': list(train_features.columns),
                                'importance': rfc.feature_importances_})
###Save to csv
feature_results.to_csv('Reduced_columns_mini.csv')

feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)
most_important_features = feature_results['feature'][:20]

indices = [list(train_features.columns).index(x) for x in most_important_features]



# Keep only the most important features
X_reduced = X[:, indices]
X_test_reduced = X_test[:, indices]

print('Most important training features shape: ', X_reduced.shape)
print('Most important testing  features shape: ', X_test_reduced.shape)

###Re-run
rfc.fit(X_reduced,y_train)
filename = 'mini_model_4_optimized.sav'
joblib.dump(rfc, filename)
#rfc = pickle.load(open(filename, 'rb'))
print('model finished and saved')

#### Create predictions save and report
predictions = rfc.predict(X_test_reduced)
filename2 = 'Predictions_model4_optimized.sav'
joblib.dump(predictions, filename2)
#predictions = pickle.load(open(filename2, 'rb'))
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
'''