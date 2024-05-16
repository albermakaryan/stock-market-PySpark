from utils.get_news import get_stock_news
import datetime as dt
from utils.data_preparation import get_scores,prepare_price,prepare_mix_data_lstm

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sparknlp
from pyspark.ml.classification import LinearSVC,LinearSVCModel
from utils.split import split_data_for_ml
from pyspark.sql.types import DoubleType,IntegerType
import sparknlp
# Load the model
import warnings
from pyspark.sql.types import *
from pyspark.sql.functions import col,max
from pyspark.sql.functions import when
import datetime as dt


# Ignore all warnings
warnings.filterwarnings("ignore")

spark = sparknlp.start()


stock = "NVIDIA"
path_to_save = "../data/json/demo/"

get_stock_news(stock, 
                   dir_to_save=path_to_save,
                   end_date='today',
                   start_year=2024, start_month=5, start_day=3,
                   day_range=2)


scores = get_scores(stock='demo',spark=spark)
scores.show()

df = prepare_mix_data_lstm(scores,stock='NVIDIA',spark=spark)
df.show()


last_date = df.agg(max(col("date")).alias("date")).take(1)[0].date
    
    
predicted_day = last_date.day + 1
predicted_month = last_date.month
predicted_month = last_date.strftime("%B")

model_path_false = "../models/_binary_smvNVIDIA_emotion_False/"
model_path_true = "../models/_binary_smvNVIDIA_emotion_True/"


def bulish_bearish(model_path,emotion):

    
    data = split_data_for_ml(df=df,emotion=emotion,classification=True,spark=spark)

    loaded_model = LinearSVCModel.load(model_path)
    predictions = loaded_model.transform(data)

    pred = predictions.tail(1)[0].prediction


    if emotion:
        key = "without"
    else:
        key = "with"

    if pred == 1:

        trend = "bullish"
    else:
        trend = "bearish"
        
    print(f"\nSVM {key} sentiment scores predicts {trend} market for Nvidia stock for {predicted_day} of {predicted_month}!!\n")

    
    
for path,emotion in zip([model_path_false,model_path_true],[False,True]):

    
    bulish_bearish(path,emotion)



spark.stop()