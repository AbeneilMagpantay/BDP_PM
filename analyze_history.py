"""Analyze betting history to evaluate model performance."""
import json
from pathlib import Path

history_path = Path(__file__).parent / "docs" / "data" / "history.json"

with open(history_path) as f:
    bets = json.load(f)

# Filter graded bets only
graded = [b for b in bets if b.get("status") == "GRADED"]

print(f"Total GRADED bets: {len(graded)}")
print()

# Overall stats
wins = [b for b in graded if b.get("result") == "WON"]
losses = [b for b in graded if b.get("result") == "LOST"]
pushes = [b for b in graded if b.get("result") == "PUSH"]

print(f"Record: {len(wins)}W - {len(losses)}L - {len(pushes)}P")
print(f"Win Rate: {len(wins)/len(graded)*100:.1f}%")

# Total profit (assuming $100 per bet)
total_profit = sum(b.get("profit", 0) for b in graded)
print(f"Total Profit: ${total_profit:.2f}")
print(f"ROI: {total_profit / (len(graded) * 100) * 100:.1f}%")
print()

# By sport
print("=" * 50)
print("BY SPORT")
print("=" * 50)
for sport in ["nfl", "nba", "soccer"]:
    sport_bets = [b for b in graded if b.get("sport") == sport]
    if not sport_bets:
        continue
    sport_wins = [b for b in sport_bets if b.get("result") == "WON"]
    sport_losses = [b for b in sport_bets if b.get("result") == "LOST"]
    sport_profit = sum(b.get("profit", 0) for b in sport_bets)
    print(f"{sport.upper()}: {len(sport_wins)}W-{len(sport_losses)}L ({len(sport_wins)/len(sport_bets)*100:.0f}%), Profit: ${sport_profit:.2f}")
print()

# High EV analysis (EV > 50%)
print("=" * 50)
print("HIGH EV BETS (EV > 50%)")
print("=" * 50)
high_ev = [b for b in graded if b.get("ev", 0) > 50]
high_ev_wins = [b for b in high_ev if b.get("result") == "WON"]
high_ev_profit = sum(b.get("profit", 0) for b in high_ev)
print(f"Count: {len(high_ev)}")
print(f"Record: {len(high_ev_wins)}W - {len(high_ev) - len(high_ev_wins)}L")
print(f"Win Rate: {len(high_ev_wins)/len(high_ev)*100:.1f}%")
print(f"Profit: ${high_ev_profit:.2f}")
print()

# Low EV analysis (EV < 50%)
print("=" * 50)
print("LOW EV BETS (EV < 50%)")
print("=" * 50)
low_ev = [b for b in graded if b.get("ev", 0) <= 50]
low_ev_wins = [b for b in low_ev if b.get("result") == "WON"]
low_ev_profit = sum(b.get("profit", 0) for b in low_ev)
print(f"Count: {len(low_ev)}")
print(f"Record: {len(low_ev_wins)}W - {len(low_ev) - len(low_ev_wins)}L")
print(f"Win Rate: {len(low_ev_wins)/len(low_ev)*100:.1f}%")
print(f"Profit: ${low_ev_profit:.2f}")
