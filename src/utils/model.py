from elephas import ml,ml_model,mllib,enums,parameter,spark_model,utils,worker

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from pyspark.ml.feature import VectorAssembler

from pyspark.sql import Row
from elephas.utils.rdd_utils import to_simple_rdd


def lstm_model():
    
    num_features = 3
    input_shape = 30

    model = Sequential()

    model = Sequential()
    model.add(LSTM(150, activation='tanh', return_sequences=True, input_shape=(input_shape, num_features)))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(64))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    
    return model


