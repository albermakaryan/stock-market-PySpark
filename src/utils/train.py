from pyspark.sql import Row
from elephas.spark_model import SparkModel

from pyspark.sql import SparkSession


def spark_lstm(simple_rdd,model,epochs=1,batch_size=2,spark=None):
    
    stop = True if spark is None else False
    spark = SparkSession.builder.appName("train").getOrCreate() if spark is None else spark
    
    

    spark_model = SparkModel(model, frequency='epoch', mode='asynchronous',batch_size=batch_size)
    spark_model.fit(simple_rdd, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)

    if stop:
        spark.stop()
        
    return spark_model