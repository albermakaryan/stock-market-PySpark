from pyspark.sql import SparkSession
from gnews.gnews import GNews
import datetime as dt



def get_stock_news(stock:str, 
                   start_year=2018, start_month=1, start_day=1,
                   end_year=2024, end_month=12, end_day=31,
                   day_range=2):
    
    
    dates = [dt.datetime(year, month, day) \
                   for year in range(start_year, end_year+1) \
                       for month in range(start_month, end_month+1) \
                           for day in range(start_day, end_day+1, day_range)]
    
    n_dates = len(dates)
    
    start_date = dt.datetime(start_year, start_month, start_day)
    for i in range(n_dates):
        
        end_date = start_date + dt.timedelta(days=day_range)
        
        
        google_news = GNews()
        # google_news.period = '7d'  # News from last 7 days
        google_news.max_results = 10  # number of responses across a keyword
        google_news.country = 'United States'  # News from a specific country 
        google_news.language = 'english'  # News in a specific language
        # google_news.exclude_websites = ['yahoo.com', 'cnn.com']  # Exclude news from specific website i.e Yahoo.com and CNN.com
        google_news.start_date = start_date # Search from 1st Jan 2020
        google_news.end_date = end_date # Search until 1st March 2020
        
        result = google_news.get_news(stock)
        print(result)
        
        
        
        start_date = end_date
        
    
    
    # stocks = ['Apple','Google','Tesla','Meta & Facebook','NVIDIA']



if __name__ == "__main__":
    
    stocks = ['Apple','Google','Tesla','Meta & Facebook','NVIDIA']
    for stock in stocks:
        get_stock_news(stock)
        pass