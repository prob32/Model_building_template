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
df = pd.read_csv('mini_model_load.csv')
df= df.drop(columns=['Unnamed: 0'])



#### test train split and report
X = df.drop(['Price'], axis=1)
train_features = df.drop(['Price'], axis=1)

####Imports

feature_results=pd.read_csv('Reduced_columns_mini.csv')

feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)
most_important_features = feature_results['feature'][:40]


df2=pd.read_csv('mini_test2.csv')


list = df2['feature'].tolist()

df = df[list]

df.to_csv('cutmodel2.csv')


'''
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