import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

# API key should be set in environment variables
if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

CONFIG = {
    # Budget settings
    'ANNUAL_BUDGET': 200000,  # Annual budget for replacements
    
    # Risk scores
    'RISK_SCORES': {
        'CRITICAL': 100,
        'HIGH': 75,
        'REG': 50,
        'LOW': 25
    },
    
    # Location scores
    'CRITICAL_LOCATIONS': ['ICU', 'ER', 'Radiology'],
    'LOCATION_SCORES': {
        'CRITICAL': 100,
        'NON_CRITICAL': 50
    },
    
    # Maintenance history weights
    'MAINTENANCE_WEIGHTS': {
        'COSMETIC_MULTIPLIER': 2,
        'USER_ERROR_MULTIPLIER': 1,
        'REPAIR_MULTIPLIER': 3,
        'SOFTWARE_MULTIPLIER': 2,
        'PM_MULTIPLIER': 0,  # Preventive Maintenance
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
    'DEFAULT_LIFECYCLE': 8,  # Default lifecycle in years if estimation fails
    'HIGH_PRIORITY_THRESHOLD': 75,  # Score threshold for high-priority fleets
    
    # File paths
    'DATA_DIR': DATA_DIR,
    'DEVICES_FILE': DATA_DIR / 'devices.csv',
    'WORK_ORDERS_FILE': DATA_DIR / 'work_orders.csv',
    'REPLACEMENT_COSTS_FILE': DATA_DIR / 'replacement_costs.csv',
    'UTILIZATION_CACHE_FILE': DATA_DIR / 'utilization_cache.csv',
    'LIFECYCLE_CACHE_FILE': DATA_DIR / 'lifecycle_cache.csv',
    
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
    """,
    
    # Maintenance categorization settings
    'MAINTENANCE_CATEGORIZATION': {
        'MODEL': 'gpt-3.5-turbo',  # Faster, cheaper model for categorization
        'BATCH_SIZE': 20,  # Number of work orders to process in a single API call
        'TEMPERATURE': 0.3,  # Lower temperature for more consistent results
        'MAX_TOKENS': 100  # Maximum tokens for batch responses
    }
} 