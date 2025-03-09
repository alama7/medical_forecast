import pandas as pd
from datetime import datetime, timedelta
import openai
import time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from config import CONFIG
from data_processing import (
    get_device_category,
    calculate_price,
    estimate_missing_costs,
    MARKET_PRICES
)

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('equipment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# OpenAI API setup with error handling
try:
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OpenAI API key not found in environment variables")
except Exception as e:
    logger.error(f"Failed to setup OpenAI API: {e}")
    raise

# Create OpenAI client
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Add market prices constant
MARKET_PRICES = {
    ('Imaging', 'MRI'): {
        'base_price': 1000000,
        'premium_keywords': ['3T', 'TESLA', 'PREMIUM'],
        'budget_keywords': ['REFURBISHED', 'BASIC'],
        'premium_multiplier': 1.3,
        'budget_multiplier': 0.7
    },
    ('Imaging', 'CT'): {
        'base_price': 500000,
        'premium_keywords': ['PREMIUM', 'ADVANCED'],
        'budget_keywords': ['BASIC', 'STANDARD'],
        'premium_multiplier': 1.2,
        'budget_multiplier': 0.8
    },
    # Add more categories as needed
}

def load_or_create_utilization_cache() -> Dict[str, float]:
    """
    Load utilization cache from file or create new if not exists.
    
    Returns:
        Dict[str, float]: Dictionary mapping device types to utilization rates
    """
    try:
        cache_path = Path(CONFIG['UTILIZATION_CACHE_FILE'])
        if cache_path.exists():
            return pd.read_csv(cache_path).set_index('DeviceType')['UtilizationRate'].to_dict()
        return {}
    except Exception as e:
        logger.error(f"Error loading utilization cache: {e}")
        return {}

def get_utilization_rate(device_type: str, utilization_cache: Dict[str, float]) -> float:
    """
    Get utilization rate for a device type, either from cache or via API.
    
    Args:
        device_type: Type of medical device
        utilization_cache: Cache of known utilization rates
    
    Returns:
        float: Utilization rate in hours per day (0-24)
    """
    if device_type in utilization_cache:
        return utilization_cache[device_type]
    
    try:
        # Updated OpenAI API call for version 1.0.0+
        response = client.chat.completions.create(
            model=CONFIG['OPENAI_MODEL'],
            messages=[
                {"role": "system", "content": "You are a medical equipment expert who understands utilization rates of various medical equipment used in hospitals. Respond only with a number between 0 and 24."},
                {"role": "user", "content": CONFIG['OPENAI_PROMPT'].format(device_type=device_type)}
            ],
            temperature=0.3
        )
        
        rate = float(response.choices[0].message.content.strip())
        rate = min(max(rate, 0), 24)
        
        utilization_cache[device_type] = rate
        save_utilization_cache(utilization_cache)
        
        time.sleep(1)
        return rate
    
    except Exception as e:
        logger.error(f"Error getting utilization rate for {device_type}: {e}")
        return 12

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Load all required data files.
    
    Returns:
        tuple: (devices DataFrame, work orders DataFrame, replacement costs dict)
    """
    try:
        devices = pd.read_csv(Path(CONFIG['DEVICES_FILE']))
        # Ensure PurchaseDate is converted to datetime
        if 'PurchaseDate' in devices.columns:
            devices['PurchaseDate'] = pd.to_datetime(devices['PurchaseDate'], errors='coerce')
            
        # Load work orders and ensure Date is converted to datetime
        work_orders = pd.read_csv(Path(CONFIG['WORK_ORDERS_FILE']))
        if 'Date' in work_orders.columns:
            work_orders['Date'] = pd.to_datetime(work_orders['Date'], errors='coerce')
            
        replacement_costs_df = pd.read_csv(Path(CONFIG['REPLACEMENT_COSTS_FILE']))
        replacement_costs = dict(zip(
            replacement_costs_df['DeviceType'],
            replacement_costs_df['ReplacementCost']
        ))
        
        return devices, work_orders, replacement_costs
    
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def calculate_device_age(purchase_date: Union[str, pd.Timestamp]) -> float:
    """Calculate device age in years."""
    try:
        purchase_date = pd.to_datetime(purchase_date)
    except Exception as e:
        logger.error(f"Invalid purchase date format: {e}")
        return 0

    if pd.isna(purchase_date):
        logger.warning("Purchase date is missing")
        return 0

    return (datetime.now() - purchase_date).days / 365.25

def calculate_annual_maintenance_cost(device_id: str, 
                                   year: int, 
                                   work_orders: pd.DataFrame) -> float:
    """
    Calculate total maintenance cost for a device in a given year.
    
    Args:
        device_id: ID of the device
        year: Year to calculate maintenance cost for
        work_orders: DataFrame containing maintenance records
        
    Returns:
        float: Total maintenance cost for the year
        
    Raises:
        ValueError: If year is invalid
    """
    try:
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        # Ensure Date column is datetime type
        if 'Date' in work_orders.columns and not pd.api.types.is_datetime64_dtype(work_orders['Date']):
            logger.warning("Date column in work_orders is not datetime type. Converting...")
            work_orders['Date'] = pd.to_datetime(work_orders['Date'], errors='coerce')
        
        # Filter work orders for the device and date range
        wo = work_orders[(work_orders['DeviceID'] == device_id) &
                        (work_orders['Date'] >= start_date) &
                        (work_orders['Date'] <= end_date)]
        
        # Handle missing or NaN values in Cost column
        if 'Cost' not in wo.columns:
            logger.warning(f"Cost column missing in work orders for device {device_id}")
            return 0.0
            
        return wo['Cost'].fillna(0).sum()
    except ValueError as e:
        logger.error(f"Invalid year {year} for maintenance cost calculation: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calculating maintenance cost for device {device_id}: {e}")
        return 0.0

def calculate_device_scores(devices: pd.DataFrame, 
                          work_orders: pd.DataFrame, 
                          replacement_costs: Dict[str, float],
                          utilization_cache: Dict[str, float]) -> List[Dict]:
    """
    Calculate scores for each device based on various factors.
    
    Args:
        devices: DataFrame containing device information
        work_orders: DataFrame containing maintenance records
        replacement_costs: Dictionary of replacement costs by device type
        utilization_cache: Dictionary of device utilization rates
    
    Returns:
        List of dictionaries containing device scores
    """
    device_scores = []
    
    for index, row in devices.iterrows():
        try:
            device_id = row['DeviceID']
            device_type = row['DeviceType']
            
            if pd.isna(row['ExpectedLifecycle (years)']):
                logger.warning(f"Missing lifecycle data for device {device_id}")
                continue
                
            # Calculate device age score
            device_age = calculate_device_age(row['PurchaseDate'])
            expected_lifecycle = row['ExpectedLifecycle (years)']
            age_score = (device_age / expected_lifecycle) * 100

            # Calculate maintenance cost score
            annual_maintenance_cost = calculate_annual_maintenance_cost(device_id, CONFIG['ANALYSIS_YEAR'], work_orders)
            replacement_cost = replacement_costs.get(device_type, CONFIG['DEFAULT_REPLACEMENT_COST'])
            maintenance_cost_score = (annual_maintenance_cost / replacement_cost) * 100

            # Calculate risk score
            risk_score = CONFIG['RISK_SCORES'][row['RiskClass']]

            # Calculate maintenance history score
            maintenance_history_score = calculate_maintenance_history_score(device_id, work_orders)

            # Calculate location score
            location_score = calculate_location_score(row['Location'])

            # Calculate utilization score
            utilization_rate = get_utilization_rate(device_type, utilization_cache)
            utilization_score = (utilization_rate / 24) * 100

            total_score = calculate_total_score(
                age_score, maintenance_cost_score, risk_score,
                maintenance_history_score, location_score, utilization_score
            )

            device_scores.append({
                'DeviceID': device_id,
                'DeviceType': device_type,
                'TotalScore': total_score,
                'ReplacementCost': replacement_cost
            })
        except Exception as e:
            logger.error(f"Error processing device {device_id}: {e}")
            continue
    
    return device_scores

def calculate_maintenance_history_score(device_id: str, work_orders: pd.DataFrame) -> float:
    """Calculate maintenance history score for a device."""
    wo_device = work_orders[work_orders['DeviceID'] == device_id]
    functional_issues = wo_device[wo_device['MaintenanceType'] == 'Repair'].shape[0]
    cosmetic_issues = wo_device[wo_device['MaintenanceType'] == 'Cosmetic'].shape[0]
    
    return (
        (functional_issues * CONFIG['MAINTENANCE_WEIGHTS']['REPAIR_MULTIPLIER'] +
         cosmetic_issues * CONFIG['MAINTENANCE_WEIGHTS']['COSMETIC_MULTIPLIER']) *
        CONFIG['MAINTENANCE_WEIGHTS']['SCORE_MULTIPLIER']
    )

def calculate_location_score(location: str) -> float:
    """Calculate location score for a device."""
    return (CONFIG['LOCATION_SCORES']['CRITICAL'] 
            if location in CONFIG['CRITICAL_LOCATIONS'] 
            else CONFIG['LOCATION_SCORES']['NON_CRITICAL'])

def calculate_total_score(age_score: float, maintenance_cost_score: float, 
                         risk_score: float, maintenance_history_score: float,
                         location_score: float, utilization_score: float) -> float:
    """Calculate total score based on weighted components with safety checks."""
    try:
        return (
            (age_score * CONFIG['SCORE_WEIGHTS']['AGE']) +
            (maintenance_cost_score * CONFIG['SCORE_WEIGHTS']['MAINTENANCE_COST']) +
            (risk_score * CONFIG['SCORE_WEIGHTS']['RISK']) +
            (maintenance_history_score * CONFIG['SCORE_WEIGHTS']['MAINTENANCE_HISTORY']) +
            (location_score * CONFIG['SCORE_WEIGHTS']['LOCATION']) +
            (utilization_score * CONFIG['SCORE_WEIGHTS']['UTILIZATION'])
        )
    except ZeroDivisionError:
        logger.warning("Division by zero encountered in score calculation")
        return 0.0

def generate_forecast(device_scores: List[Dict], 
                     devices: pd.DataFrame,
                     replacement_costs: Dict[str, float]) -> List[Dict]:
    """Generate replacement forecast for the next 5 years."""
    forecast = []
    remaining_devices = device_scores.copy()
    
    for year in range(1, 6):
        yearly_replacements = []
        device_costs = {}  # New dictionary to track individual costs
        remaining_budget = CONFIG['ANNUAL_BUDGET']
        
        for device in remaining_devices[:]:  # Use slice to avoid modification during iteration
            try:
                device_type = devices[devices['DeviceID'] == device['DeviceID']]['DeviceType'].iloc[0]
                replacement_cost = replacement_costs.get(device_type, CONFIG['DEFAULT_REPLACEMENT_COST'])
                
                if remaining_budget >= replacement_cost:
                    yearly_replacements.append(device['DeviceID'])
                    device_costs[device['DeviceID']] = replacement_cost  # Store individual cost
                    remaining_budget -= replacement_cost
                    remaining_devices.remove(device)
            except IndexError:
                logger.warning(f"Device ID {device['DeviceID']} not found in devices DataFrame")
                continue
        
        total_cost = sum(device_costs.values())
        
        forecast.append({
            'Year': datetime.now().year + year,
            'DevicesToReplace': yearly_replacements,
            'DeviceCosts': device_costs,  # Add device costs to forecast
            'TotalCost': total_cost,
            'RemainingBudget': remaining_budget
        })
    
    return forecast

def output_forecast(forecast: List[Dict]):
    """Output the forecast results."""
    print("\nReplacement Forecast for Next 5 Years:")
    for entry in forecast:
        print(f"\nYear {entry['Year']}:")
        print("Devices to Replace:")
        if 'DeviceCosts' in entry:  # Add individual costs to the output
            for device_id, cost in entry['DeviceCosts'].items():
                print(f"- {device_id}: ${cost:,.2f}")
        print(f"Total Cost: ${entry['TotalCost']:,.2f}")
        print(f"Remaining Budget: ${entry['RemainingBudget']:,.2f}")

def save_utilization_cache(cache: Dict[str, float]):
    """Save utilization cache to file."""
    cache_df = pd.DataFrame(list(cache.items()), 
                          columns=['DeviceType', 'UtilizationRate'])
    cache_df.to_csv(CONFIG['UTILIZATION_CACHE_FILE'], index=False)



def main():
    try:
        # Ensure data directory exists
        data_dir = Path(CONFIG['DATA_DIR'])
        if not data_dir.exists():
            logger.info(f"Creating data directory at {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Starting equipment analysis")
        
        # Load all required data
        devices, work_orders, replacement_costs = load_data()
        utilization_cache = load_or_create_utilization_cache()
        
        # Process devices and calculate scores
        device_scores = calculate_device_scores(
            devices, 
            work_orders, 
            replacement_costs, 
            utilization_cache
        )
        
        # Generate and output forecast
        forecast = generate_forecast(device_scores, devices, replacement_costs)
        output_forecast(forecast)
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
