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
os.environ["PATH"] += os.pathsep + r'C:\Users\Probinson\PycharmProjects\Carz\venv\Lib\site-packages\graphviz\graphviz-2.38\release\bin'

####Imports
df = pd.read_csv('wide_features.csv')
df= df.drop(columns=['Unnamed: 0'])



#### test train split and report
X = df.drop(['Price'], axis=1)
df_list = list(X.columns)
y = df['Price']
dt_target_names = [str(s) for s in y.unique()]

y = np.array(y)
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
print('data split')


rf = tree.DecisionTreeRegressor(max_depth = 11, random_state = 42)
rf.fit(X_train, y_train)
# Extract the small tree

# Save the tree as a png image

# Extract a single tree
dot_data = tree.export_graphviz(rf,
                                feature_names=df_list,
                                out_file=None,
                                filled=True,
                                rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)

colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png('tree.png')
