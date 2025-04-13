"""Test configuration file"""

CONFIG = {
    'DATA_DIR': 'tests/data',
    'DEVICES_FILE': 'tests/data/test_devices.csv',
    'WORK_ORDERS_FILE': 'tests/data/test_work_orders.csv',
    'REPLACEMENT_COSTS_FILE': 'tests/data/test_replacement_costs.csv',
    'UTILIZATION_CACHE_FILE': 'tests/data/test_utilization_cache.csv',
    'LIFECYCLE_CACHE_FILE': 'tests/data/test_lifecycle_cache.csv',
    'ANALYSIS_YEAR': 2024,
    'DEFAULT_REPLACEMENT_COST': 500000.0,
    'RISK_SCORES': {
        'High': 100,
        'Medium': 50,
        'Low': 25
    },
    'MAINTENANCE_WEIGHTS': {
        'COSMETIC_MULTIPLIER': 2,
        'USER_ERROR_MULTIPLIER': 1,
        'REPAIR_MULTIPLIER': 3,
        'SOFTWARE_MULTIPLIER': 2,
        'PM_MULTIPLIER': 0,
        'SCORE_MULTIPLIER': 10
    },
    'LOCATION_SCORES': {
        'CRITICAL': 100,
        'NON_CRITICAL': 50
    },
    'CRITICAL_LOCATIONS': ['Main', 'Emergency', 'ICU', 'OR'],
    'SCORE_WEIGHTS': {
        'AGE': 0.3,
        'MAINTENANCE_COST': 0.2,
        'RISK': 0.15,
        'MAINTENANCE_HISTORY': 0.15,
        'LOCATION': 0.1,
        'UTILIZATION': 0.1
    },
    'OPENAI_MODEL': 'gpt-3.5-turbo',
    'OPENAI_PROMPT': "What is the typical daily utilization rate (in hours) for a {device_type} in a hospital setting? Please respond with just a number between 0 and 24.",
    'HIGH_PRIORITY_THRESHOLD': 80,
    'MEDIUM_PRIORITY_THRESHOLD': 60,
    'MAINTENANCE_CATEGORIZATION': {
        'BATCH_SIZE': 10,
        'SYSTEM_PROMPT': "You are a medical equipment expert who understands maintenance work orders. For each work order, categorize it into one of these types: Preventive, Corrective, Software, Safety, User Error, Cosmetic, or PM. Respond with just the category name.",
        'USER_PROMPT': "Categorize this maintenance work order: {description}",
        'BATCH_PROMPT': "Categorize these maintenance work orders into types (Preventive, Corrective, Software, Safety, User Error, Cosmetic, or PM). For each work order, respond with just the category name, one per line:\n\n{descriptions}"
    },
    'EXPECTED_LIFECYCLES': {
        'MRI': 10,
        'CT': 8,
        'X-Ray': 5
    }
} 