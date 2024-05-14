from utils.data_preparation import get_scores,prepare_price,prepare_mix_data_lstm
import sparknlp

if __name__ == "__main__":
    
    spark = sparknlp.start() 
    
    scores = get_scores(spark=spark)

    # df, X_train, y_train, X_test, y_test, simple_rdd = prepare_mix_data_lstm(scores,)
    _,_,_,_,_,simple_rdd = prepare_mix_data_lstm(scores,spark=spark)
    
    print(simple_rdd)
    