import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np

data = pd.read_csv("training.csv")

# now get the test data
X_test = pd.read_csv("real-data.csv")


data['from_date'] = "2000-01-01"

data['days_recorded']=(
    pd.to_datetime( data['date_recorded'],format='%Y-%m-%d')-
    pd.to_datetime( data['from_date'],format='%Y-%m-%d')).dt.days


X_test['from_date'] = "2000-01-01"

X_test['days_recorded']=(
    pd.to_datetime( X_test['date_recorded'],format='%Y-%m-%d')-
    pd.to_datetime( X_test['from_date'],format='%Y-%m-%d')).dt.days

y = data['status_group']
X = data.drop(['status_group','id','subvillage','wpt_name',
               'region_code',
               'extraction_type_group','extraction_type_class',
              'payment_type','quantity_group',
              'water_quality',
              'source_type',
              'source_class',
              'waterpoint_type_group',
              'management_group',
              'recorded_by','from_date','date_recorded'],axis=1)

X_test.drop(['id','subvillage','wpt_name',
               'region_code',
               'extraction_type_group','extraction_type_class',
              'payment_type','quantity_group',
              'water_quality',
              'source_type',
              'source_class',
              'waterpoint_type_group',
              'management_group',
              'recorded_by','from_date','date_recorded'],axis=1,inplace=True)


X_test['pos_x'] = np.around(np.multiply(np.cos(X_test[['latitude']]),np.cos(X_test[['longitude']])),decimals=2)
X_test['pos_y'] = np.around(np.multiply(np.cos(X_test[['latitude']]),np.sin(X_test[['longitude']])),decimals=2)
X_test['pos_z'] = np.around(np.sin(X_test[['latitude']]),decimals=2)
X_test.drop(['latitude','longitude'],axis=1,inplace=True)

X['pos_x'] = np.around(np.multiply(np.cos(X[['latitude']]),np.cos(X[['longitude']])),decimals=2)
X['pos_y'] = np.around(np.multiply(np.cos(X[['latitude']]),np.sin(X[['longitude']])),decimals=2)
X['pos_z'] = np.around(np.sin(X[['latitude']]),decimals=2)
X.drop(['latitude','longitude'],axis=1,inplace=True)

X['funder'].fillna("-",inplace=True)
X['installer'].fillna("-",inplace=True)
X['public_meeting'].fillna("-",inplace=True)
X['scheme_name'].fillna("-",inplace=True)
X['scheme_management'].fillna("-",inplace=True)
X['permit'].fillna("-",inplace=True)

X_test['funder'].fillna("-",inplace=True)
X_test['installer'].fillna("-",inplace=True)
X_test['public_meeting'].fillna("-",inplace=True)
X_test['scheme_name'].fillna("-",inplace=True)
X_test['scheme_management'].fillna("-",inplace=True)
X_test['permit'].fillna("-",inplace=True)

# Enocde the data
le = preprocessing.LabelEncoder()
for col in X:
   if X[col].dtype == 'object':
      con = X[col].append(X_test[col])
      le.fit(con.astype(str))
      X[col]= le.transform(X[col].astype(str))
      X_test[col] = le.transform(X_test[col].astype(str))

le1 = preprocessing.LabelEncoder()
le1.fit(y)
y= le1.transform(y)

from sklearn.ensemble import RandomForestClassifier

#Now run XGboost
model=RandomForestClassifier(n_estimators=500, max_depth=None, oob_score=True,warm_start=True)
model.fit(X,y)
y_test = model.predict(X_test)
y_test_trans = le1.inverse_transform(y_test)
np.savetxt("out.csv",y_test_trans,fmt="%s")
