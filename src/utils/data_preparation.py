
import sparknlp
import numpy as np
# Import the required modules and classes
from sparknlp.base import DocumentAssembler, Pipeline, Finisher

from sparknlp.annotator import (
    SentenceDetector,
    Tokenizer,
    Lemmatizer,
    SentimentDetector
)
import pyspark.sql.functions as F
# Start Spark Session

from sparknlp.pretrained import PretrainedPipeline
import afinn
from pyspark.sql import types
import yahooquery as yq
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pyspark.ml.feature import VectorAssembler
import seaborn as sns
from pyspark.streaming import dstream,StreamingContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StandardScaler
from pyspark.sql.window import Window
from elephas.utils.rdd_utils import to_simple_rdd



def get_scores(column='article',stock='Tesla',spark=None):
    
    

    data = spark.read.option("header","true").json("../data/json/"+stock)

    # Step 1: Transforms raw texts to `document` annotation
    document_assembler = (
        DocumentAssembler() \
        .setInputCol(column) \
        .setOutputCol("document")
    )

    schema = StructType([
    StructField("article",StringType(),True),
    StructField("description",StringType(),True),
    StructField("published date",StringType(),True),
    StructField("publisher_url",StringType(),True),
    StructField("title",StringType(),True),
    StructField("url",StringType(),True)
    ])

    
    
    # Step 2: Sentence Detection
    sentence_detector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
    
    # Step 3: Tokenization
    tokenizer = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
    
    # Step 4: Lemmatization
    lemmatizer= Lemmatizer().setInputCols("token").setOutputCol("lemma") \
                            .setDictionary("lemmas_small.txt", key_delimiter="->", value_delimiter="\t")
    
    # Step 5: Sentiment Detection
    sentiment_detector= (
        SentimentDetector() \
        .setInputCols(["lemma", "sentence"]) \
        .setOutputCol("sentiment_score") \
        .setDictionary("default-sentiment-dict.txt", ",")
    )

    
    # Step 6: Finisher
    finisher= (
        Finisher() \
        .setInputCols(["sentiment_score"]).setOutputCols("sentiment")
    )
    
    # Define the pipeline
    pipeline = Pipeline(
        stages=[
            document_assembler,
            sentence_detector, 
            tokenizer, 
            lemmatizer, 
            sentiment_detector, 
            finisher
        ]
    )
    
    def compute_with_afinn(text):
    
        return afinn.Afinn().score(text)
    
    compute_sentiment_score_udf = F.udf(compute_with_afinn, types.FloatType())

    # data = data.withColumnRe

    # scores
    result = pipeline.fit(data).transform(data)
    result = result.withColumn("afinn_sentiment",compute_sentiment_score_udf(F.col('article')))
    result = result.withColumn("pnn_sentiment",
                             F.when(F.array_contains(F.col("sentiment"), "positive"), 1)
                              .when(F.array_contains(F.col("sentiment"), "negative"), -1)
                              .otherwise(0))
    result = result.withColumnRenamed("published date","published_date")
    result = result.withColumn("published_date", F.regexp_extract(F.col("published_date"), r"^.*,\s*(.+)\s\d+:\d+:\d+", 1))

    func = F.udf(lambda x: dt.datetime.strptime(x, '%d %b %Y'), DateType())
    
    # Apply UDF to the column
    result = result.withColumn('published_date', func(result['published_date']))

    result = result.select('published_date','afinn_sentiment','pnn_sentiment')


    result = result.groupBy("published_date").agg(F.mean("afinn_sentiment").alias("afinn_sentiment_mean"),\
                                                  F.mean("pnn_sentiment").alias("pnn_sentiment"))

    
    feature_columns = ["afinn_sentiment_mean"]

    # scale
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="afinn_sentiment_scaled")

    # Scale the features using MinMaxScaler
    scaler = StandardScaler(inputCol="afinn_sentiment_scaled", outputCol="afinn_sentiment")
    
    # Create a pipeline to execute the assembler and scaler
    pipeline = Pipeline(stages=[assembler, scaler])
    
    # Fit and transform the pipeline
    result = pipeline.fit(result).transform(result)


    firstelement = F.udf(lambda v:float(v[0]),FloatType())
    result = result.withColumn("afinn_sentiment", firstelement("afinn_sentiment"))
    
    result = result.withColumnRenamed("published_date","date")
    
    result = result.select('date','afinn_sentiment','pnn_sentiment')


    

    return result


def prepare_price(stock,start_date=None,end_date=None,spark=None):
    


    price = yq.Ticker(stock).history(start=start_date,end=end_date).reset_index()
    # return price

    price_df = spark.createDataFrame(price)


    # only adj close and volume
    price_df = price_df.select("date","volume","adjclose")

    # scale
    
    def compute_percent_change(current_price, previous_price):
        if current_price is not None and previous_price is not None:
            return ((current_price - previous_price) / previous_price) * 100
        else:
            return None
    
    percent_change_udf = F.udf(compute_percent_change, FloatType())
    
    # Calculate percent change
    window_spec = Window.orderBy("date")
    
    price_df = price_df.withColumn("prev_price", F.lag("adjclose", 1).over(window_spec))
    price_df = price_df.withColumn("price_percent_change", percent_change_udf(F.col("adjclose"), F.col("prev_price")))
    
    
    price_df = price_df.withColumn("prev_volume", F.lag("volume", 1).over(window_spec))
    price_df = price_df.withColumn("volume_percent_change", percent_change_udf(F.col("volume"), F.col("prev_volume")))
    
    
    price_df = price_df.select('date','price_percent_change','volume_percent_change')   
    price_df = price_df.withColumn("next_day_price_percent_change_shifted", F.lead("price_percent_change", 1).over(window_spec))


    return price_df



def prepare_mix_data_lstm(scores,stock='NVIDIA',spark=None,train_size=0.8):
    
    
        
    tickers = {"NVIDIA":"NVDA",
              "Bitcoin":"BTC-USD",
              "Apple":"AAPL",
              "Tesla":"TSLA"}
    
    
    stock = tickers[stock]
    

    min_date = scores.agg(F.min("date")).collect()[0][0].strftime('%Y-%m-%d')
    max_date = scores.agg(F.max("date")).collect()[0][0].strftime('%Y-%m-%d')

    price_df = prepare_price(stock,min_date,max_date,spark=spark)

    df = scores.join(price_df, on="date", how="right")
    df = df.dropna()


    n = df.count()
    train_size = int(n*train_size)

    train_data = df.limit(train_size)
    test_data = df.subtract(train_data)

    X_train = train_data.select('afinn_sentiment', 'pnn_sentiment', 'price_percent_change','volume_percent_change')
    y_train = train_data.select("next_day_price_percent_change_shifted")

    X_train = np.array(X_train.rdd.map(lambda x: [x.afinn_sentiment,x.pnn_sentiment,x.price_percent_change,x.volume_percent_change]).collect())
    y_train = np.array(y_train.rdd.map(lambda x: [x.next_day_price_percent_change_shifted]).collect())


    X_test = test_data.select('afinn_sentiment', 'pnn_sentiment', 'price_percent_change','volume_percent_change')
    y_test = test_data.select("next_day_price_percent_change_shifted")

    X_test = np.array(X_test.rdd.map(lambda x: [x.afinn_sentiment,x.pnn_sentiment,x.price_percent_change,x.volume_percent_change]).collect())
    y_test = np.array(y_test.rdd.map(lambda x: [x.next_day_price_percent_change_shifted]).collect())
    
    sc = spark.sparkContext
    simple_rdd = to_simple_rdd(sc, X_train,y_train)

        
    return df, X_train, y_train, X_test, y_test, simple_rdd

    


def save_data(df,stock,spark):
    
    df.coalesce(1).write.csv("../data/model_data/csv/" + stock)
    
    
    
    
if __name__ == "__main__":
    
    spark = SparkSession.builder.appName("Stock Sentiment Analysis").getOrCreate()
    
    
    for stock in ['NVIDIA','Apple']:
        
        scores = get_scores(stock=stock,spark=spark)
        df,_,_,_,_,_ = prepare_mix_data_lstm(scores,stock=stock,spark=spark)
        save_data(df,stock,spark)
    
    spark.stop()