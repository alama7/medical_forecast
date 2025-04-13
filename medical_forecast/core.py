import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

def calculate_device_scores(devices):
    """Calculate scores for individual devices."""
    # Add scoring logic here
    devices['Score'] = devices['Score'].fillna(0)
    return devices

def calculate_fleet_scores(devices):
    """
    Calculate fleet scores based on device data.
    
    Args:
        devices: DataFrame containing device information
    
    Returns:
        DataFrame: Fleet scores with columns DeviceType, FleetSize, TotalScore
    """
    # Check if Score column exists, if not create it
    if 'Score' not in devices.columns:
        # If Score doesn't exist, create a default score based on device age
        if 'PurchaseDate' in devices.columns:
            # Calculate age in years
            devices['Age'] = (pd.Timestamp.now() - pd.to_datetime(devices['PurchaseDate'])).dt.days / 365.25
            # Create a simple score based on age (inverse relationship)
            devices['Score'] = 100 - (devices['Age'] * 5).clip(0, 100)
        else:
            # If no PurchaseDate, use a default score
            devices['Score'] = 50
    
    # Group by device type and calculate fleet metrics
    fleet_scores = devices.groupby('DeviceType').agg({
        'DeviceID': 'count',
        'Score': 'mean'
    }).rename(columns={
        'DeviceID': 'FleetSize',
        'Score': 'TotalScore'
    }).reset_index()
    
    return fleet_scores

def get_utilization_rate(device_type):
    """Get utilization rate for a device type."""
    # Add utilization rate logic here
    return 0.75  # Placeholder

def save_utilization_cache(cache_data):
    """Save utilization cache to file."""
    with open('tests/data/test_utilization_cache.json', 'w') as f:
        json.dump(cache_data, f)

def load_lifecycle_cache():
    """Load lifecycle cache from file."""
    try:
        with open('tests/data/test_lifecycle_cache.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {} 