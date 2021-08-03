# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 07:45:46 2021

@author: james
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 11:38:15 2021

@author: james
"""
import numpy as np
import pandas as pd
import tensorflow as tf

#Importing Dataset
df = pd.read_csv('bank-full.csv', sep=';')
df_ann = df[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'y']]

X = df_ann.iloc[:, :7].values
y = df_ann.iloc[:, -1].values


#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
le = LabelEncoder()


y[:] = le.fit_transform(y[:])


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here
   
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
       '''

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

X = MultiColumnLabelEncoder(columns = ['housing','loan', 'default']).fit_transform(df_ann)


#OneHotEncoder (3d+)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), ['job', 'marital', 'education'])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

X = np.asarray(X).astype('float32')
y= np.asarray(y).astype('float32')

#Split to training/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Building the ANN


#Initializing
ann = tf.keras.models.Sequential()

#Adding Input Layer 
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.2))

        
#Second Hidden Layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dropout(0.5))
        
#Output Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


#Training ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 20)



#Predicting Test Results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
pred_results = (np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
model_accuracy = accuracy_score(y_test, y_pred)
