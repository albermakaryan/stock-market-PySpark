from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import numpy as np
from pyspark.sql import SparkSession
from utils.data_preparation import split_data_lstm
from pyspark.sql.types import *
import pandas as pd
import json
import os
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pyspark.sql.functions as F
def r_squared(y_true,y_pred):

    r = 1 - np.sum((y_true-y_pred)**2)/(np.sum((y_true-y_true.mean())**2))
    return r

def train_lstm_regression(stock,train_size=0.9,spark=None,emotion=False,input_shape=10,batch_size=32,epoch=100):
    

    X_train, y_train, X_test, y_test = split_data_lstm(stock,train_size=train_size,spark=spark,emotion=emotion)

    train_data_generator = TimeseriesGenerator(
        data=X_train,
        targets=y_train,
        length=input_shape,  # Adjust the length as needed
        batch_size=batch_size,
        start_index=0,     # Start from the beginning of the time series
        end_index=None,   # End at the last available index
        shuffle=True
    )
    
    test_data_generator = TimeseriesGenerator(
        data=X_test,
        targets=y_test,
        length=input_shape,  # Adjust the length as needed
        batch_size=batch_size,
        start_index=0,     # Start from the beginning of the time series
        end_index=None,   # End at the last available index
        shuffle=True
    )

        
    # build model
    num_features = 4 if emotion else 2
    
    model = Sequential()
    
    model = Sequential()
    model.add(LSTM(150, activation='tanh', return_sequences=True,input_shape=(input_shape, num_features)))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(1))
    
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse')
    
    history = model.fit(train_data_generator,epochs=epoch,validation_data=test_data_generator,batch_size=batch_size,verbose=1)

    # X_test = X_test.reshape(-1,1,num_features)
   

    
    r_2_score = []
    for sample,target in test_data_generator:

        prediction = model.predict(sample)
    
        r_2 = r_squared(target,prediction)
        r_2_score.append(r_2)

    r_2_score = sum(r_2_score)/len(r_2_score)

    results = history.history
    results['test_r_2_score'] = r_2_score

    print(r_2_score)

    return model,results,r_2_score


def train_lstm_classification(stock,train_size=0.9,spark=None,emotion=False,input_shape=10,batch_size=32,epoch=100):
    

    X_train, y_train, X_test, y_test = split_data_lstm(stock,train_size=train_size,spark=spark,emotion=emotion)

    y_train = np.where(y_train<0,0,1)
    y_test = np.where(y_test<0,0,1)
    
    train_size = int(train_size*X_train.shape[0])

    X_valid = X_train[train_size:]
    y_valid = y_train[train_size:]    

    X_train = X_train[:train_size]
    y_train = y_train[:train_size]



    
    train_data_generator = TimeseriesGenerator(
        data=X_train,
        targets=y_train,
        length=input_shape,  # Adjust the length as needed
        batch_size=batch_size,
        start_index=0,     # Start from the beginning of the time series
        end_index=None,   # End at the last available index
        shuffle=True
    )
    
    valid_data_generator = TimeseriesGenerator(
        data=X_valid,
        targets=y_valid,
        length=input_shape,  # Adjust the length as needed
        batch_size=batch_size,
        start_index=0,     # Start from the beginning of the time series
        end_index=None,   # End at the last available index
        shuffle=True
    )
    
    test_data_generator = TimeseriesGenerator(
        data=X_test,
        targets=y_test,
        length=input_shape,  # Adjust the length as needed
        batch_size=batch_size,
        start_index=0,     # Start from the beginning of the time series
        end_index=None,   # End at the last available index
        shuffle=True
    )

        
    # build model
    num_features = 4 if emotion else 2
    
    model = Sequential()
    
    model = Sequential()
    model.add(LSTM(150, activation='tanh', return_sequences=True,input_shape=(input_shape, num_features)))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(1))
    
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    
    # model.compile(optimizer=Adam(learning_rate=learning_rate),loss='mse')
    
    history = model.fit(train_data_generator,epochs=epoch,validation_data=valid_data_generator,batch_size=batch_size,verbose=1)

    accuracy = []

    for sample,target in test_data_generator:

        acc = model.evaluate(sample,target)[1]
        print(acc)
        accuracy.append(acc)

    accuracy = sum(accuracy)/len(accuracy)

    results = history.history
    results['test_accuracy'] = accuracy
    
    return model,results
    