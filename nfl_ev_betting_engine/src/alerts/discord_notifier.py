"""
Discord Notifier Module
=======================

Sends betting edge alerts to Discord via webhook.
Formats alerts with rich embeds for clear visualization.
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_webhook_url() -> str:
    """
    Get Discord webhook URL from environment variables.
    
    Returns:
        Webhook URL string
        
    Raises:
        ValueError: If webhook URL is not configured
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url or webhook_url == "your_webhook_url_here":
        raise ValueError(
            "DISCORD_WEBHOOK_URL not configured. "
            "Please set it in your .env file."
        )
    return webhook_url


def send_simple_message(
    message: str,
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a simple text message to Discord.
    
    Args:
        message: Text message to send
        webhook_url: Optional webhook URL (uses env var if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    url = webhook_url or get_webhook_url()
    
    payload = {
        "content": message
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending Discord message: {e}")
        return False


def send_edge_alert(
    edge: Dict[str, Any],
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a single edge alert with rich formatting.
    
    Args:
        edge: Edge dictionary with betting opportunity details
        webhook_url: Optional webhook URL
        
    Returns:
        True if successful, False otherwise
    """
    url = webhook_url or get_webhook_url()
    
    # Determine embed color based on EV
    ev = edge.get('ev', 0)
    if ev >= 10:
        color = 0x00FF00  # Green - Strong edge
    elif ev >= 5:
        color = 0xFFFF00  # Yellow - Good edge
    else:
        color = 0xFFA500  # Orange - Moderate edge
    
    # Build embed
    embed = {
        "title": f"🎯 Edge Alert: {edge.get('matchup', 'Unknown')}",
        "color": color,
        "fields": [
            {
                "name": "🏈 Bet",
                "value": f"**{edge.get('bet_team', 'Unknown')}** ({edge.get('bet_side', '').upper()})",
                "inline": True
            },
            {
                "name": "📊 Odds",
                "value": f"{edge.get('odds', 0):+d} @ {edge.get('bookmaker', 'Unknown')}",
                "inline": True
            },
            {
                "name": "📈 Expected Value",
                "value": f"**{edge.get('ev', 0):+.1f}%**",
                "inline": True
            },
            {
                "name": "🎲 Probabilities",
                "value": f"Model: {edge.get('model_prob', 0):.1%} | Implied: {edge.get('implied_prob', 0):.1%}",
                "inline": True
            },
            {
                "name": "📊 Edge",
                "value": f"{edge.get('edge', 0):+.1f}%",
                "inline": True
            },
            {
                "name": "💰 Kelly Bet",
                "value": f"{edge.get('kelly_bet', 0):.1f}% of bankroll",
                "inline": True
            }
        ],
        "footer": {
            "text": f"Game Time: {edge.get('commence_time', 'TBD')}"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    payload = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending Discord alert: {e}")
        return False


def send_daily_summary(
    edges: List[Dict[str, Any]],
    webhook_url: Optional[str] = None
) -> bool:
    """
    Send a daily summary of all detected edges.
    
    Args:
        edges: List of edge dictionaries
        webhook_url: Optional webhook URL
        
    Returns:
        True if successful, False otherwise
    """
    url = webhook_url or get_webhook_url()
    
    if not edges:
        # No edges found
        embed = {
            "title": "📊 Daily EV Report",
            "description": "No +EV opportunities found today.",
            "color": 0x808080,  # Gray
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        # Build summary
        total_ev = sum(e.get('ev', 0) for e in edges)
        best_edge = max(edges, key=lambda x: x.get('ev', 0))
        
        description = f"Found **{len(edges)}** betting opportunities\n"
        description += f"Best EV: **{best_edge.get('ev', 0):+.1f}%** on {best_edge.get('bet_team', 'Unknown')}"
        
        # Build fields for top opportunities
        fields = []
        for i, edge in enumerate(edges[:5], 1):  # Top 5
            fields.append({
                "name": f"#{i} {edge.get('bet_team', 'Unknown')}",
                "value": f"EV: {edge.get('ev', 0):+.1f}% | Odds: {edge.get('odds', 0):+d} @ {edge.get('bookmaker', 'Unknown')}",
                "inline": False
            })
        
        embed = {
            "title": "📊 Daily EV Report",
            "description": description,
            "color": 0x00FF00 if len(edges) > 0 else 0x808080,
            "fields": fields,
            "footer": {
                "text": f"NFL EV Betting Engine | {len(edges)} total opportunities"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    payload = {
        "embeds": [embed]
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending Discord summary: {e}")
        return False


def send_all_edge_alerts(
    edges: List[Dict[str, Any]],
    webhook_url: Optional[str] = None,
    include_summary: bool = True
) -> int:
    """
    Send alerts for all detected edges.
    
    Args:
        edges: List of edge dictionaries
        webhook_url: Optional webhook URL
        include_summary: Whether to send a summary first
        
    Returns:
        Number of successfully sent alerts
    """
    url = webhook_url or get_webhook_url()
    sent = 0
    
    if include_summary:
        send_daily_summary(edges, url)
    
    for edge in edges:
        if send_edge_alert(edge, url):
            sent += 1
    
    return sent


class DiscordNotifier:
    """
    Discord notification service for betting alerts.
    
    Provides a clean interface for sending various types of
    notifications to Discord.
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize the notifier.
        
        Args:
            webhook_url: Discord webhook URL (uses env var if not provided)
        """
        self.webhook_url = webhook_url or get_webhook_url()
    
    def test_connection(self) -> bool:
        """
        Test the webhook connection.
        
        Returns:
            True if connection successful
        """
        return send_simple_message(
            "🔔 NFL EV Betting Engine - Connection Test Successful!",
            self.webhook_url
        )
    
    def send_startup_message(self) -> bool:
        """Send a startup notification."""
        message = (
            "🚀 **NFL EV Betting Engine Started**\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Scanning for +EV opportunities..."
        )
        return send_simple_message(message, self.webhook_url)
    
    def send_edge_alerts(self, edges: List[Dict]) -> int:
        """Send all edge alerts."""
        return send_all_edge_alerts(edges, self.webhook_url)
    
    def send_error(self, error_message: str) -> bool:
        """Send an error notification."""
        message = f"⚠️ **Error**: {error_message}"
        return send_simple_message(message, self.webhook_url)


if __name__ == "__main__":
    print("Testing Discord Notifier...")
    
    # Check if webhook is configured
    try:
        webhook = get_webhook_url()
        print(f"Webhook configured: {webhook[:50]}...")
        
        # Test connection
        notifier = DiscordNotifier()
        if notifier.test_connection():
            print("✅ Test message sent successfully!")
        else:
            print("❌ Failed to send test message")
            
    except ValueError as e:
        print(f"\n⚠️ {e}")
        print("\nTo test, add your Discord webhook URL to .env:")
        print("DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...")
