# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:47:07 2018

@author: Андрей
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
#from keras.wrappers.scikit_learn import KerasRegressor
#from sklearn.cross_validation import train_test_split

train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')

train_csv = train_csv.replace(["male", "female"], [0,1])
test_csv = test_csv.replace(["male", "female"], [0,1])
train_csv = train_csv.replace(["S", "C", "Q"], [0,1,2])
test_csv = test_csv.replace(["S", "C", "Q"], [0,1,2])
train_csv = train_csv.fillna(0)
test_csv = test_csv.fillna(0)

y_train = train_csv[['Survived']]
#y_train = np.array(y_train.values)
#print (y_train.shape)

x_train = train_csv[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
x_test = test_csv[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
x_train=np.array(x_train.values, dtype=np.float32)
y_train=np.array(y_train.values, dtype=np.float32)
x_test=np.array(x_test.values, dtype=np.float32)

#print (x_train.shape)
        
#x_test = test_csv.loc[:,['Pclass','Sex','Age','SibSp','Parch']]
#x_test=np.array(x_test.values)
#print (x_test.shape)

# Среднее значение
#mean = x_train.mean(axis=0)
#print ('Среднее отклонение',mean)
# Стандартное отклонение
#std = x_train.std(axis=0)
#print (std)
#x_train -= mean
#x_train /= std
#x_test -= mean
#x_test /= std

seed = 42
np.random.seed(seed)

#X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.2)
  
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=30, verbose=2)

pred = np.round(model.predict(x_test))
#print (pred[0:10])
pred = pd.DataFrame(pred,columns=['Survived'])
#print (pred)
PasID = test_csv[["PassengerId"]]
#print(PasID)

result = pd.concat([PasID, pred], axis = 1)
#result = pd.DataFrame({'PassengerId':PasID,'Survived':pred[0]})
print (result)
result.to_csv('result.csv', index=False)
