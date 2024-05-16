import numpy as np
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pyspark.sql.functions as F   
from pyspark.ml import Pipeline


def split_data_lstm(stock,train_size=0.8,spark=None,emotion=False):
    
    spark.sparkContext.setLogLevel("OFF")

    schema = StructType([

    StructField('date',StringType(),True),
    StructField('afinn_sentiment',FloatType(),True),
    StructField('pnn_sentiment',FloatType(),True),
    StructField('price_percent_change',FloatType(),True),
    StructField('volume_percent_change',FloatType(),True),
    StructField('next_day_price_percent_change_shifted',FloatType(),True)
    
    ])
    
    # 'date', 'afinn_sentiment', 'pnn_sentiment', 'price_percent_change', 'volume_percent_change', 'next_day_price_percent_change_shifted'
    df = spark.read.schema(schema).csv("../data/csv/"+stock+"/")


    # scale volumne 
    assembler = VectorAssembler(inputCols=['volume_percent_change'], outputCol="features")

    # Transform the data
    data = assembler.transform(df)
    
    # Initialize the StandardScaler
    scaler = StandardScaler(inputCol="features", outputCol="scaled_volume_percent_change", withMean=True, withStd=True)
    
    # Compute summary statistics by fitting the StandardScaler
    scaler_model = scaler.fit(data)
    
    # Scale features
    scaled_data = scaler_model.transform(data)
    
    firstelement = F.udf(lambda v:float(v[0]),FloatType())
    df = scaled_data.withColumn("volume_percent_change", firstelement("scaled_volume_percent_change"))
    
    df = df.select('date',
     'afinn_sentiment',
     'pnn_sentiment',
     'price_percent_change',
     'volume_percent_change',
     'next_day_price_percent_change_shifted')
    
    n = df.count()
    train_size = int(n*train_size)

    train_data = df.limit(train_size)
    test_data = df.subtract(train_data)

    if emotion:
        
        X_train = train_data.select('afinn_sentiment', 'pnn_sentiment', 'price_percent_change','volume_percent_change')
        y_train = train_data.select("price_percent_change")
    
    
        X_train = np.array(X_train.rdd.map(lambda x: [x.afinn_sentiment,x.pnn_sentiment,x.price_percent_change,x.volume_percent_change]).collect())
        y_train = np.array(y_train.rdd.map(lambda x: [x.price_percent_change]).collect())
    
    
        X_test = test_data.select('afinn_sentiment', 'pnn_sentiment', 'price_percent_change','volume_percent_change')
        y_test = test_data.select("price_percent_change")
    
        X_test = np.array(X_test.rdd.map(lambda x: [x.afinn_sentiment,x.pnn_sentiment,x.price_percent_change,x.volume_percent_change]).collect())
        y_test = np.array(y_test.rdd.map(lambda x: [x.price_percent_change]).collect())

    else:
        
        X_train = train_data.select('price_percent_change','volume_percent_change')
        y_train = train_data.select("price_percent_change")
    
    
        X_train = np.array(X_train.rdd.map(lambda x: [x.price_percent_change,x.volume_percent_change]).collect())
        y_train = np.array(y_train.rdd.map(lambda x: [x.price_percent_change]).collect())
    
    
        X_test = test_data.select('price_percent_change','volume_percent_change')
        y_test = test_data.select("price_percent_change")
    
        X_test = np.array(X_test.rdd.map(lambda x: [x.price_percent_change,x.volume_percent_change]).collect())
        y_test = np.array(y_test.rdd.map(lambda x: [x.price_percent_change]).collect())
    # sc = spark.sparkContext
    # simple_rdd = to_simple_rdd(sc, X_train,y_train)


    return X_train, y_train, X_test, y_test

def split_data_for_ml(stock=None,df=None,train_size=0.8,spark=None,emotion=False,classification=False,all=True,inference=True):


    df = df.sort(df["date"])
    spark.sparkContext.setLogLevel("OFF")
    
    if df is None:
        schema = StructType([
    
        StructField('date',StringType(),True),
        StructField('afinn_sentiment',FloatType(),True),
        StructField('pnn_sentiment',FloatType(),True),
        StructField('price_percent_change',FloatType(),True),
        StructField('volume_percent_change',FloatType(),True),
        StructField('next_day_price_percent_change_shifted',FloatType(),True)
        
        ])
        
        # 'date', 'afinn_sentiment', 'pnn_sentiment', 'price_percent_change', 'volume_percent_change', 'next_day_price_percent_change_shifted'
        df = spark.read.schema(schema).csv("../data/csv/"+stock+"/")
    
        # scale volume

        # scale volumne 
    assembler = VectorAssembler(inputCols=['volume_percent_change'], outputCol="features")

    # Transform the data
    data = assembler.transform(df)
    
    # Initialize the StandardScaler
    scaler = StandardScaler(inputCol="features", outputCol="scaled_volume_percent_change", withMean=True, withStd=True)
    
    # Compute summary statistics by fitting the StandardScaler
    scaler_model = scaler.fit(data)
    
    # Scale features
    scaled_data = scaler_model.transform(data)
    
    firstelement = F.udf(lambda v:float(v[0]),FloatType())
    df = scaled_data.withColumn("volume_percent_change", firstelement("scaled_volume_percent_change"))
    
    df = df.select('date',
     'afinn_sentiment',
     'pnn_sentiment',
     'price_percent_change',
     'volume_percent_change',
     'next_day_price_percent_change_shifted').withColumnRenamed("next_day_price_percent_change_shifted","label")


    # return df
    
    n = df.count()
    train_size = int(n*train_size)




 
    if emotion:

        
        assembler = VectorAssembler(inputCols=['afinn_sentiment', 'pnn_sentiment', 'price_percent_change', 'volume_percent_change'], \
                                    outputCol="features")

    else:
      
        assembler = VectorAssembler(inputCols=['price_percent_change', 'volume_percent_change'], \
                                    outputCol="features")

    if all:
        
        df = Pipeline(stages=[assembler]).fit(df).transform(df)
        
        df = df.select("features","label")
        
        if classification:
            df = df.withColumn("label", \
                                     F.when(F.col("label") >0, 1).otherwise(0))
            
        return df
            
    else:
        

        
        train_data = df.limit(train_size)
        test_data = df.subtract(train_data)
        train_data = Pipeline(stages=[assembler]).fit(train_data).transform(train_data)
        test_data = Pipeline(stages=[assembler]).fit(test_data).transform(test_data)

        train_data = train_data.select("features","label")
        test_data = test_data.select("features","label")

        if classification:
            train_data = train_data.withColumn("label", \
                                        F.when(F.col("label") >0, 1).otherwise(0))
            
            test_data = test_data.withColumn("label", \
                                        F.when(F.col("label") >0, 1).otherwise(0))
            
        return train_data,test_data

    