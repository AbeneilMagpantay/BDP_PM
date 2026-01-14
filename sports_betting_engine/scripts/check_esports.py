"""Check available esports on The Odds API."""
import os
import requests
from dotenv import load_dotenv

load_dotenv('../.env')

api_key = os.getenv('ODDS_API_KEY')
r = requests.get('https://api.the-odds-api.com/v4/sports/', params={'apiKey': api_key})

sports = r.json()
print("=== All Available Sports ===")
for s in sports:
    if s.get('active'):
        print(f"{s['key']}: {s['title']} ({s['group']})")

print("\n=== Esports/Gaming ===")
for s in sports:
    key = s.get('key', '').lower()
    title = s.get('title', '').lower()
    group = s.get('group', '').lower()
    if any(x in key or x in title or x in group for x in ['dota', 'valorant', 'lol', 'csgo', 'cs2', 'esport', 'league of legends', 'counter-strike']):
        print(f"  {s['key']}: {s['title']} ({s['group']})")
