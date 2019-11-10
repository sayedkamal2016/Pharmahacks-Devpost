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

a = pd.read_csv("/root/Pharmahacks/MLstuff/[C]Merge_part3.csv")
b = pd.read_csv("/root/Pharmahacks/MLstuff/NewCompiled.csv")
sdate = a["Shipment date to Wholesaler"].str.split(".", n = 2, expand = True)
#a["SYear"] = sdate[2]
a["SMonth"] = sdate[1].astype(int, errors = "ignore")
a.drop(columns = ["Shipment date to Wholesaler"], inplace=True)
new = a.filter(["On Hand Units", "On Order Units", "Units Sold/Returned", "Name", "Location", "SMonth", "Quantity"], axis = 1)
noc = b.filter(["Name", "Year", "Month", "Manufact", "Type"], axis =1)
#print(new.dtypes)
#print(noc.dtypes)
#c = new.merge(noc, on="Name", how = "left")
#print(c.shape)
#c.loc[c.duplicated(subset=["Year", "Name", "Month"], keep='first'),:]
#print(c.shape)
#c.to_csv("/root/Pharmahacks/MLstuff/process.csv")
#new.fillna()
le = preprocessing.LabelEncoder()
new['Name'] =  new['Name'].astype(str)
new['Name'] = le.fit_transform(new['Name'])
new['Units Sold/Returned'] = pd.to_numeric(new['Units Sold/Returned'], errors='coerce')
#new.fillna(0)
#new['SMonth'] = new['SMonth'].astype(int)


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
new['Location'] = new['Location'].map(diag_map)
print(new.shape)
print(new.describe())
print(new.dtypes)

new.drop(columns=["SMonth"], axis = 1)
new = new.dropna()
yVar = new['Units Sold/Returned'] 
xVar = new.loc[:, new.columns != 'Units Sold/Returned'] 
X_train, X_test, y_train, y_test = train_test_split(xVar, yVar, test_size
 = 0.3) 
#gb = GradientBoostingClassifier()
#model = gb.fit(X_train, y_train)
#print(model.score(X_test, y_test))

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
plt.yticks(pos, new.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
#plt.show()
plt.savefig('Gradient Boosting.eps')