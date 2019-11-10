import pandas as pd
import numpy as np
from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline, make_union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv('/root/Pharmahacks/MLstuff/(C)Merge_part3-n.csv')
data = data.drop(['Receipt', 'Expiry', 'Shelf'], axis = 1)
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
le = LabelEncoder()
data['Name'] = le.fit_transform(data['Name'])

'''
# use multi-column label encoder from scikit tutorial
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
           
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
data = MultiColumnLabelEncoder(columns = ['Name']).fit_transform(data)
'''

x = data.loc[:, data.columns != 'Sold/Returned']
y = data['Sold/Returned']

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=1/3., random_state=42)

# function to select the columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("The DataFrame does not include the columns: %s" % cols_error)

pipeline = Pipeline(steps = [
        ("features", make_union(
                ColumnSelector(list(x)),
                )),
                ("model",RandomForestClassifier(random_state=42))
])

pipeline.fit(x_train, y_train)
pipeline.score(x_validation, y_validation)

print("RF Score before CV: %s" % pipeline.score(x_validation, y_validation))

# get list of hyperparameters
hyperparameters = { 'model__max_depth': [50, 70, 90],
                    'model__min_samples_leaf': [1, 2, 3]
                  }

# cv = k for k-fold
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(x_train, y_train)

print("RF Score after CV: %s" % clf.score(x_validation, y_validation))