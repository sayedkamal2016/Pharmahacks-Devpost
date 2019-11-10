import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv("/root/Pharmahacks/MLstuff/(C)Merge_part3-n.csv")
le = preprocessing.LabelEncoder()
data['Receipt'] = le.fit_transform(data['Receipt'])
data['Expiry'] = le.fit_transform(data['Expiry'])
data['Name'] = le.fit_transform(data['Name'])
#data['On Hand Units'] = data['On Hand Units'].values.reshape(-1, 1)
#data['On Order Units'] = data['On Order Units'].values.reshape(-1, 1)
#data['Expiry'] = data['Expiry'].values.reshape(-1, 1)
#data['Receipt'] = data['Receipt'].values.reshape(-1, 1)
#data['Name'] = data['Name'].values.reshape(-1, 1)
data = data.drop(['Shelf'], axis = 1)
data = data.drop(['Receipt', 'Expiry', 'Name'], axis = 1)
diag_map = {'Wholesaler_location_1':1,
    'Wholesaler_location_2':2,
    'Wholesaler_location_3':3,
    'Wholesaler_location_4':4,
    'Wholesaler_location_5':5,
    'Wholesaler_location_6':6,
    'Wholesaler_location_7':7,
    'Wholesaler_location_8':8,
    'Wholesaler_location_9':9,
}
data['Location'] = data['Location'].map(diag_map)
print(data.shape)
print(data.describe())
print(data.dtypes)

# since ordered by ID number, shandomly shuffled it
#X, y = shuffle(data.data, data.target, random_state=13)
yVar = data['Sold/Returned']
xVar = data.loc[:, data.columns != 'Sold/Returned']
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size = 0.2)

gb = GradientBoostingClassifier()
model = gb.fit(X_train, y_train)
print(model.score(X_test, y_test))

params ={'n_estimators': 500, 'max_depth': 4, 'min_samples_split':2,
         'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)

# mse = mean_squared_error(y_test, clf.predict(X_test))
# print("MSE: %.4f" % mse)

test_score = np.zeros((params['n_estimators'],), dtype = np.float64)

for i, y_pred in enumerate(clf.staged_predict(X_test)):
     test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.title('Hi')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
          label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
feature_importance = clf.feature_importances_
#make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, data.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

