import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

df = pd.read_csv('small_features.csv')




X = df.drop(['Price'], axis=1)

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

df_list = list(df.columns)
print(df_list)


price_mean = df["Price"].mean()
'''
features = np.array(df)

rfc.fit(X_train,y_train)

filename = 'finalized_model.sav'
pickle.dump(rfc, open(filename, 'wb'))

'''
#filename = 'finalized_model.sav'
#rfc = pickle.load(open(filename, 'rb'))




filename2 = 'prediction.sav'
predictions = pickle.load(open(filename2, 'rb'))

errors = abs(predictions - y_test)




mape = 100 * (errors / y_test)

accuracy = 100 - np.mean(mape)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Accuracy:', round(accuracy, 2), '%.')

baseline_errors = abs(price_mean - y_test)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))


'''
# Get numerical feature importances
importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(df_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
'''