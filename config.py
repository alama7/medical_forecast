import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

CONFIG = {
    # Budget settings
    'ANNUAL_BUDGET': 200000,  # Annual budget for replacements
    
    # Risk scores
    'RISK_SCORES': {
        'III': 100,
        'II': 70,
        'I': 50
    },
    
    # Location scores
    'CRITICAL_LOCATIONS': ['ICU', 'ER', 'Radiology'],
    'LOCATION_SCORES': {
        'CRITICAL': 100,
        'NON_CRITICAL': 50
    },
    
    # Maintenance history weights
    'MAINTENANCE_WEIGHTS': {
        'REPAIR_MULTIPLIER': 2,
        'COSMETIC_MULTIPLIER': 1,
        'SCORE_MULTIPLIER': 10
    },
    
    # Score weightings
    'SCORE_WEIGHTS': {
        'AGE': 0.25,
        'MAINTENANCE_COST': 0.25,
        'RISK': 0.20,
        'MAINTENANCE_HISTORY': 0.20,
        'LOCATION': 0.05,
        'UTILIZATION': 0.05
    },
    
    # Analysis settings
    'ANALYSIS_YEAR': 2022,
    'DEFAULT_REPLACEMENT_COST': 50000,
    
    # File paths
    'DATA_DIR': DATA_DIR,
    'DEVICES_FILE': DATA_DIR / 'devices.csv',
    'WORK_ORDERS_FILE': DATA_DIR / 'work_orders.csv',
    'REPLACEMENT_COSTS_FILE': DATA_DIR / 'replacement_costs.csv',
    'UTILIZATION_CACHE_FILE': DATA_DIR / 'utilization_cache.csv',
    
    # OpenAI settings
    'OPENAI_MODEL': 'gpt-4',
    'OPENAI_PROMPT': """
    As a medical equipment expert, estimate the average daily utilization rate (in hours per day) 
    for a {device_type} in a typical hospital setting. Consider:
    - Standard operating hours
    - Emergency availability needs
    - Typical procedure durations
    - Patient volume
    - Required maintenance/downtime
    
    Provide only a number between 0 and 24 representing hours per day.
    """
}
