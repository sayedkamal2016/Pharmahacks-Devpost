import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
from sklearn.preprocessing import Binarizer, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.cluster import KMeans   

# adapted from Clustering tutorial code
data = pd.read_csv('/root/Pharmahacks/MLstuff/Merge_part3-n.csv')
data = data.drop(['From Date', 'To Date', 'Location', 'Transaction Type'], axis=1)
data.fillna(value='0')
#print(data.head())
data['Location'] = data['Location'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
data.loc[:, data.columns != 'Units Sold/Returned'] = MinMaxScaler().fit_transform(data.loc[:, data.columns != 'Units Sold/Returned'])
data_elb = data.loc[:, data.columns != 'Units Sold/Returned']

sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data_elb)
    data_elb["clusters"] = kmeans.labels_
    #print(data["clusters"])
    # Inertia: Sum of distances of samples to their closest cluster center
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()