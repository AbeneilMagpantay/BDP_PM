import pandas as pd
from pathlib import Path

# Path to the cache file
parquet_file = Path('sports_betting_engine/data/nba_cache/nba_games_2025-26.parquet')
csv_output = Path('nba_sample_data.csv')

if not parquet_file.exists():
    print("Error: Parquet file not found.")
else:
    df = pd.read_parquet(parquet_file)
    
    # Sort to see the '0 points' games at the top
    df = df.sort_values(by='PTS', ascending=True)
    
    df.to_csv(csv_output, index=False)
    print(f"Successfully converted {len(df)} rows.")
    print(f"Saved to: {csv_output.absolute()}")
