from gnews.gnews import GNews
import datetime as dt
import numpy as np
import os
import json
import time
# import newspaper3k

def get_stock_news(stock:str, 
                   dir_to_save=None,
                   end_date='today',
                   start_year=2018, start_month=1, start_day=1,
                   end_year=2024, end_month=12, end_day=31,
                   day_range=2):
    
    
    # dates = [dt.datetime(year, month, day) \
    #                for year in range(start_year, end_year+1) \
    #                    for month in range(start_month, end_month+1) \
    #                        for day in range(start_day, end_day, day_range)]
    
    start_date = dt.datetime(start_year, start_month, start_day)
    
    if end_date == 'today':
        final_end_date = dt.datetime.today()
    else:
        final_end_date = dt.datetime(end_year, end_month, end_day)
        

    
    
    # start_date = dt.datetime(start_year, start_month, start_day)

    dir_to_save = os.path.join("../data/json/",stock)

    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)

    
    while True:
        
        end_date = start_date + dt.timedelta(days=day_range)
        
        print(f"Start date: {start_date}, End date: {end_date}")
        
        if start_date > final_end_date:
            break
        
        
        google_news = GNews()
        google_news.country = 'United States'  #  
        google_news.language = 'english'  # 
        google_news.start_date = start_date # 
        google_news.end_date = end_date # 
        
        results = google_news.get_news(stock)
        print("Number of articles:",len(results))
        row_i = 1
        for result in results:
            print("Required date range: ", start_date, end_date)
            print("Actual date: ",result['published date'])
            url = result['url']
            try:
                full_article = google_news.get_full_article(url).text
            except:
                full_article = np.nan
                
            result['publisher_url'] = result['publisher']['href']
            result['publisher'] = result['publisher']['title']
            result['article'] = full_article

            file_name = stock + "_"+start_date.strftime("%d-%B-%Y")+\
                        "-"+end_date.strftime("%d-%B-%Y")+"_"+str(row_i)+".json"

            file_path = os.path.join(dir_to_save,file_name)

            with open(file_path,'w') as f:
                json.dump(result,f)
            row_i += 1

        
        
        print()
        time.sleep(2)
        start_date = end_date + dt.timedelta(days=1)
        
    
    
    # stocks = ['Apple','Google','Tesla','Meta & Facebook','NVIDIA']


if __name__ == "__main__":
    
    stocks = ['Apple','Google','Tesla','Meta and Facebook','NVIDIA']
    for stock in stocks:
        result = get_stock_news(stock)
        break
