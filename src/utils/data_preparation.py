

from pyspark.sql import SparkSession
import os


spark = SparkSession.builder.appName("data preparation").getOrCreate()



def make_dataframe(dir,path_to_save):
    
    
    
    df = spark.read.option("inferschema",'true').option("header",'true').json(dir)
    
    df.coalesce(1).write.csv(path_to_save, header=True)
    
    
    
    # df.write.csv(path_to_save,header=True)