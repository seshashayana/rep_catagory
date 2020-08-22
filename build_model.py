# Import the relevant libraries

import pandas as pd                # Manage the dataframe
import numpy as np                 # Manage the numerical / array calculations

# Import the data from coma-seperated file to pandas dataframe

raw_data = pd.read_csv("repair_catagorization.csv")
raw_data.head()

raw_data.Interface = raw_data.Interface.apply(lambda x: 0 if x == "Yes" else 1)
    
dummies = pd.get_dummies(raw_data.Location)
dummies.head()

raw_data_1 = pd.concat([raw_data, dummies.drop("20-27", axis = 1)], axis = 1)
raw_data_1.head()

final_data = raw_data_1.drop(["Location", "Length", "Width"], axis = 1)
final_data.head()

X = final_data.drop("Complexity", axis = 1)
y = final_data.Complexity

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

log_regcv = LogisticRegressionCV(max_iter=25000)
log_regcv.fit(X_train, y_train)
log_regcv.score(X_test, y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
scores = cross_val_score(LogisticRegressionCV(max_iter = 15000), X, y, cv=cv)

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(log_regcv, f)

import json
columns = {
    'data_columns': [col for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))
