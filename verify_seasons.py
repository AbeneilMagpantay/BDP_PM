from datetime import datetime

def get_seasons_logic():
    seasons = None
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Logic from nba_data_fetcher.py
    if current_month >= 10:
        seasons = [f"{current_year}-{str(current_year+1)[-2:]}"]
    else:
        seasons = [f"{current_year-1}-{str(current_year)[-2:]}"]
        
    print(f"Current Date: {datetime.now()}")
    print(f"Calculated Seasons List: {seasons}")

get_seasons_logic()
