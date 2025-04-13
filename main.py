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
    MARKET_PRICES,
    PURCHASE_DATE_COLUMN,
    get_expected_lifecycle,
    load_or_create_lifecycle_cache,
    categorize_maintenance_type,
    batch_categorize_maintenance_types
)

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
        logger.error("OPENAI_API_KEY environment variable not found")
        raise ValueError("OPENAI_API_KEY environment variable not found")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found in environment variables")
except Exception as e:
    logger.error(f"Failed to setup OpenAI API: {e}")
    raise

# Create OpenAI client
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

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
        # Ensure data directory exists
        data_dir = Path(CONFIG['DATA_DIR'])
        if not data_dir.exists():
            logger.info(f"Creating data directory at {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
            
        devices = pd.read_csv(Path(CONFIG['DEVICES_FILE']))
        # Ensure PurchaseDate is converted to datetime
        if PURCHASE_DATE_COLUMN in devices.columns:
            devices[PURCHASE_DATE_COLUMN] = pd.to_datetime(devices[PURCHASE_DATE_COLUMN], errors='coerce')
            
        # Load work orders
        work_orders = pd.read_csv(Path(CONFIG['WORK_ORDERS_FILE']))
            
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

def preprocess_work_orders(work_orders: pd.DataFrame, batch_size: int = 10) -> pd.DataFrame:
    """
    Preprocess work orders data to ensure it has the MaintenanceType column.
    Uses ChatGPT to categorize work orders based on their description and type.
    Processes work orders in batches for efficiency.
    
    Args:
        work_orders: DataFrame containing work order records
        batch_size: Number of work orders to process in a single API call
        
    Returns:
        pd.DataFrame: Preprocessed work orders DataFrame
    """
    # Make a copy to avoid modifying the original DataFrame
    work_orders = work_orders.copy()
    
    # Ensure Date column is datetime type
    if 'Date' in work_orders.columns and not pd.api.types.is_datetime64_dtype(work_orders['Date']):
        logger.info("Converting Date column to datetime type")
        work_orders['Date'] = pd.to_datetime(work_orders['Date'], errors='coerce')
    
    # Check if we need to categorize work orders
    if 'MaintenanceType' not in work_orders.columns or work_orders['MaintenanceType'].isna().any():
        logger.info("Categorizing work orders using ChatGPT")
        
        # Create a temporary column for work order type if it exists
        work_order_type_col = None
        for possible_col in ['WorkOrderType', 'Type', 'Category', 'MaintenanceCategory']:
            if possible_col in work_orders.columns:
                work_order_type_col = possible_col
                break
        
        # Prepare data for batch processing
        work_orders_to_categorize = []
        indices_to_categorize = []
        
        # Identify which rows need categorization
        for idx, row in work_orders.iterrows():
            if 'MaintenanceType' not in work_orders.columns or pd.isna(row.get('MaintenanceType')):
                order_data = {}
                
                # Add description if available
                if 'Description' in work_orders.columns:
                    order_data['description'] = row['Description']
                
                # Add work order type if available
                if work_order_type_col:
                    order_data['work_order_type'] = row[work_order_type_col]
                
                # Only add if we have some data to categorize
                if order_data:
                    work_orders_to_categorize.append(order_data)
                    indices_to_categorize.append(idx)
        
        # If we have work orders to categorize
        if work_orders_to_categorize:
            logger.info(f"Batch processing {len(work_orders_to_categorize)} work orders")
            
            # Use batch categorization
            categories = batch_categorize_maintenance_types(work_orders_to_categorize, batch_size)
            
            # Create MaintenanceType column if it doesn't exist
            if 'MaintenanceType' not in work_orders.columns:
                work_orders['MaintenanceType'] = None
            
            # Update the DataFrame with the categorized values
            for i, idx in enumerate(indices_to_categorize):
                if i < len(categories):
                    work_orders.at[idx, 'MaintenanceType'] = categories[i]
                else:
                    # Default to Repair if we somehow didn't get a category
                    work_orders.at[idx, 'MaintenanceType'] = 'Repair'
        else:
            # No data to categorize, default to Repair
            logger.warning("No data available for categorization. Using default categorization.")
            work_orders['MaintenanceType'] = 'Repair'
    else:
        # MaintenanceType column already exists and has no NaN values
        # Still standardize it using our categorization function
        logger.info("Standardizing existing MaintenanceType column")
        
        # Prepare data for batch processing
        work_orders_to_standardize = []
        indices_to_standardize = []
        
        for idx, row in work_orders.iterrows():
            work_orders_to_standardize.append({
                'description': row['MaintenanceType']
            })
            indices_to_standardize.append(idx)
        
        # Use batch categorization for standardization
        standardized_categories = batch_categorize_maintenance_types(work_orders_to_standardize, batch_size)
        
        # Update the DataFrame with the standardized values
        for i, idx in enumerate(indices_to_standardize):
            if i < len(standardized_categories):
                work_orders.at[idx, 'MaintenanceType'] = standardized_categories[i]
    
    return work_orders

def calculate_device_scores(devices: pd.DataFrame, 
                          work_orders: pd.DataFrame, 
                          replacement_costs: Dict[str, float],
                          utilization_cache: Dict[str, float]) -> List[Dict]:
    """
    Calculate scores for each device based on various factors.
    
    Args:
        devices: DataFrame containing device information
        work_orders: DataFrame containing work order information
        replacement_costs: Dictionary of replacement costs
        utilization_cache: Dictionary of device utilization rates
    
    Returns:
        List[Dict]: List of device scores
    """
    logger.info("Calculating device scores...")
    
    # Load lifecycle cache
    lifecycle_cache = load_or_create_lifecycle_cache()
    
    device_scores = []
    
    for index, row in devices.iterrows():
        try:
            device_id = row['DeviceID']
            device_type = row['DeviceType']
            
            # Get expected lifecycle using our new function
            expected_lifecycle = get_expected_lifecycle(device_type, lifecycle_cache)
                
            # Calculate device age score
            device_age = calculate_device_age(row[PURCHASE_DATE_COLUMN])
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
            logger.info(f"Device {device_type} scored {total_score:.2f}")

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
    """
    Calculate maintenance history score for a device based on the types of maintenance performed.
    
    Each maintenance type has a different multiplier:
    - Cosmetic: multiplier of 2
    - User Error: multiplier of 1
    - Repair: multiplier of 3
    - Software: multiplier of 2
    - PM (Preventive Maintenance): multiplier of 0
    
    Args:
        device_id: ID of the device
        work_orders: DataFrame containing maintenance records
        
    Returns:
        float: Maintenance history score
    """
    wo_device = work_orders[work_orders['DeviceID'] == device_id]
    
    # Count occurrences of each maintenance type
    cosmetic_issues = wo_device[wo_device['MaintenanceType'] == 'Cosmetic'].shape[0]
    user_error_issues = wo_device[wo_device['MaintenanceType'] == 'User Error'].shape[0]
    repair_issues = wo_device[wo_device['MaintenanceType'] == 'Repair'].shape[0]
    software_issues = wo_device[wo_device['MaintenanceType'] == 'Software'].shape[0]
    pm_issues = wo_device[wo_device['MaintenanceType'] == 'PM'].shape[0]
    
    # Calculate weighted score
    return (
        (cosmetic_issues * CONFIG['MAINTENANCE_WEIGHTS']['COSMETIC_MULTIPLIER'] +
         user_error_issues * CONFIG['MAINTENANCE_WEIGHTS']['USER_ERROR_MULTIPLIER'] +
         repair_issues * CONFIG['MAINTENANCE_WEIGHTS']['REPAIR_MULTIPLIER'] +
         software_issues * CONFIG['MAINTENANCE_WEIGHTS']['SOFTWARE_MULTIPLIER'] +
         pm_issues * CONFIG['MAINTENANCE_WEIGHTS']['PM_MULTIPLIER']) *
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

def calculate_fleet_scores(devices: pd.DataFrame, 
                         work_orders: pd.DataFrame,
                         replacement_costs: Dict[str, float],
                         utilization_cache: Dict[str, float]) -> List[Dict]:
    """
    Calculate scores for each fleet of devices.
    
    Args:
        devices: DataFrame containing device information
        work_orders: DataFrame containing work order information
        replacement_costs: Dictionary of replacement costs
        utilization_cache: Dictionary of device utilization rates
    
    Returns:
        List[Dict]: List of fleet scores
    """
    logger.info("Calculating fleet scores...")
    
    # Load lifecycle cache
    lifecycle_cache = load_or_create_lifecycle_cache()
    
    # Group devices by type
    device_groups = devices.groupby('DeviceType')
    
    fleet_scores = []
    
    for device_type, fleet_devices in device_groups:
        try:
            # Calculate fleet-level metrics
            fleet_size = len(fleet_devices)
            
            # Calculate average age score for the fleet
            fleet_ages = [calculate_device_age(row[PURCHASE_DATE_COLUMN]) for _, row in fleet_devices.iterrows()]
            avg_fleet_age = sum(fleet_ages) / fleet_size if fleet_size > 0 else 0
            
            # Get expected lifecycle for this device type
            expected_lifecycle = get_expected_lifecycle(device_type, lifecycle_cache)
            age_score = (avg_fleet_age / expected_lifecycle) * 100
            
            # Calculate fleet maintenance cost score
            fleet_maintenance_costs = []
            for _, device in fleet_devices.iterrows():
                annual_cost = calculate_annual_maintenance_cost(device['DeviceID'], CONFIG['ANALYSIS_YEAR'], work_orders)
                fleet_maintenance_costs.append(annual_cost)
            
            avg_maintenance_cost = sum(fleet_maintenance_costs) / fleet_size if fleet_size > 0 else 0
            replacement_cost = replacement_costs.get(device_type, CONFIG['DEFAULT_REPLACEMENT_COST'])
            maintenance_cost_score = (avg_maintenance_cost / replacement_cost) * 100
            
            # Calculate fleet risk score (use highest risk in the fleet)
            risk_scores = [CONFIG['RISK_SCORES'][row['RiskClass']] for _, row in fleet_devices.iterrows()]
            risk_score = max(risk_scores) if risk_scores else 0
            
            # Calculate fleet maintenance history score
            fleet_maintenance_scores = []
            for _, device in fleet_devices.iterrows():
                history_score = calculate_maintenance_history_score(device['DeviceID'], work_orders)
                fleet_maintenance_scores.append(history_score)
            
            maintenance_history_score = sum(fleet_maintenance_scores) / fleet_size if fleet_size > 0 else 0
            
            # Calculate fleet location score (use highest location score in the fleet)
            location_scores = [calculate_location_score(row['Location']) for _, row in fleet_devices.iterrows()]
            location_score = max(location_scores) if location_scores else 0
            
            # Calculate fleet utilization score
            utilization_rate = get_utilization_rate(device_type, utilization_cache)
            utilization_score = (utilization_rate / 24) * 100
            
            # Calculate total fleet score
            total_score = calculate_total_score(
                age_score, maintenance_cost_score, risk_score,
                maintenance_history_score, location_score, utilization_score
            )
            
            # Calculate fleet replacement cost (cost to replace entire fleet)
            fleet_replacement_cost = replacement_cost * fleet_size
            
            logger.info(f"Fleet {device_type} scored {total_score:.2f}")
            
            fleet_scores.append({
                'DeviceType': device_type,
                'FleetSize': fleet_size,
                'TotalScore': total_score,
                'FleetReplacementCost': fleet_replacement_cost,
                'DeviceIDs': fleet_devices['DeviceID'].tolist(),
                'AvgAge': avg_fleet_age,
                'ExpectedLifecycle': expected_lifecycle
            })
            
        except Exception as e:
            logger.error(f"Error processing fleet {device_type}: {e}")
            continue
    
    return fleet_scores

def identify_very_old_devices(devices: pd.DataFrame, fleet_scores: List[Dict], work_orders: pd.DataFrame, replacement_costs: Dict[str, float]) -> List[Dict]:
    """
    Identify individual devices that are significantly older than their fleet average.
    
    Args:
        devices: DataFrame containing device information
        fleet_scores: List of fleet scores with average age information
        work_orders: DataFrame containing work order information
        replacement_costs: Dictionary of replacement costs by device type
        
    Returns:
        List[Dict]: List of very old devices that should be replaced individually
    """
    logger.info("Identifying very old devices for individual replacement...")
    
    very_old_devices = []
    
    # Create a dictionary of fleet information for easy lookup
    fleet_info = {fleet['DeviceType']: {
        'avg_age': fleet['AvgAge'],
        'expected_lifecycle': fleet['ExpectedLifecycle']
    } for fleet in fleet_scores}
    
    # Define threshold for "very old" - devices that are 50% older than their fleet average
    # and at least 80% of their expected lifecycle
    for _, device in devices.iterrows():
        device_type = device['DeviceType']
        
        # Skip if we don't have fleet information for this device type
        if device_type not in fleet_info:
            continue
        
        # Calculate device age
        device_age = calculate_device_age(device[PURCHASE_DATE_COLUMN])
        
        # Get fleet information
        fleet_avg_age = fleet_info[device_type]['avg_age']
        expected_lifecycle = fleet_info[device_type]['expected_lifecycle']
        
        # Check if device is significantly older than fleet average and approaching end of lifecycle
        age_ratio = device_age / fleet_avg_age if fleet_avg_age > 0 else 0
        lifecycle_ratio = device_age / expected_lifecycle if expected_lifecycle > 0 else 0
        
        if age_ratio > 1.5 and lifecycle_ratio > 0.8:
            # Calculate device-specific metrics
            maintenance_cost = calculate_annual_maintenance_cost(device['DeviceID'], CONFIG['ANALYSIS_YEAR'], work_orders)
            maintenance_history = calculate_maintenance_history_score(device['DeviceID'], work_orders)
            risk_score = CONFIG['RISK_SCORES'][device['RiskClass']]
            location_score = calculate_location_score(device['Location'])
            
            # Calculate device score (similar to fleet score but for individual device)
            device_score = calculate_total_score(
                (device_age / expected_lifecycle) * 100,  # Age score
                (maintenance_cost / replacement_costs.get(device_type, CONFIG['DEFAULT_REPLACEMENT_COST'])) * 100,  # Maintenance cost score
                risk_score,
                maintenance_history,
                location_score,
                100  # Assume high utilization for critical devices
            )
            
            very_old_devices.append({
                'DeviceID': device['DeviceID'],
                'DeviceType': device_type,
                'Age': device_age,
                'FleetAvgAge': fleet_avg_age,
                'ExpectedLifecycle': expected_lifecycle,
                'Score': device_score,
                'ReplacementCost': replacement_costs.get(device_type, CONFIG['DEFAULT_REPLACEMENT_COST']),
                'Location': device['Location'],
                'RiskClass': device['RiskClass']
            })
    
    # Sort very old devices by score (highest priority first)
    very_old_devices = sorted(very_old_devices, key=lambda x: x['Score'], reverse=True)
    
    logger.info(f"Identified {len(very_old_devices)} very old devices for individual replacement")
    return very_old_devices

def generate_fleet_forecast(fleet_scores: List[Dict], 
                          devices: pd.DataFrame,
                          replacement_costs: Dict[str, float],
                          work_orders: pd.DataFrame,
                          annual_budget: Optional[float] = None) -> List[Dict]:
    """
    Generate replacement forecast for the next 5 years at the fleet level.
    
    Args:
        fleet_scores: List of fleet scores
        devices: DataFrame containing device information
        replacement_costs: Dictionary of replacement costs
        work_orders: DataFrame containing work order information
        annual_budget: Optional annual budget. If None, forecast based on scores and age only.
    
    Returns:
        List[Dict]: Forecast for the next 5 years
    """
    forecast = []
    
    # Identify very old devices for individual replacement
    very_old_devices = identify_very_old_devices(devices, fleet_scores, work_orders, replacement_costs)
    
    # Sort fleet_scores by TotalScore in descending order (highest priority first)
    sorted_fleet_scores = sorted(fleet_scores, key=lambda x: x['TotalScore'], reverse=True)
    remaining_fleets = sorted_fleet_scores.copy()
    
    # If no budget provided, calculate replacement years based on scores and age
    if annual_budget is None:
        logger.info("No budget provided. Generating forecast based on scores and age.")
        return generate_unconstrained_forecast(fleet_scores, devices, replacement_costs, very_old_devices)
    
    # Budget-constrained forecast
    logger.info(f"Generating budget-constrained forecast with annual budget of ${annual_budget:,.2f}")
    for year in range(1, 6):
        yearly_replacements = []
        fleet_costs = {}  # Dictionary to track fleet costs
        remaining_budget = annual_budget
        
        # First, handle very old devices that need immediate replacement
        very_old_replacements = []
        for device in very_old_devices[:]:  # Use slice to avoid modification during iteration
            if device['Score'] > CONFIG['HIGH_PRIORITY_THRESHOLD'] and remaining_budget >= device['ReplacementCost']:
                very_old_replacements.append({
                    'DeviceID': device['DeviceID'],
                    'DeviceType': device['DeviceType'],
                    'Age': device['Age'],
                    'Score': device['Score'],
                    'IsVeryOldDevice': True
                })
                fleet_costs[f"Individual_{device['DeviceID']}"] = device['ReplacementCost']
                remaining_budget -= device['ReplacementCost']
                very_old_devices.remove(device)
        
        # Add very old device replacements to yearly replacements
        if very_old_replacements:
            yearly_replacements.extend(very_old_replacements)
        
        # Then handle fleet replacements
        for fleet in remaining_fleets[:]:  # Use slice to avoid modification during iteration
            try:
                fleet_replacement_cost = fleet['FleetReplacementCost']
                
                # Check if we can replace the entire fleet
                if remaining_budget >= fleet_replacement_cost:
                    yearly_replacements.append({
                        'DeviceType': fleet['DeviceType'],
                        'FleetSize': fleet['FleetSize'],
                        'DeviceIDs': fleet['DeviceIDs']
                    })
                    fleet_costs[fleet['DeviceType']] = fleet_replacement_cost
                    remaining_budget -= fleet_replacement_cost
                    remaining_fleets.remove(fleet)
                else:
                    # If we can't replace the entire fleet, check if we can do a partial replacement
                    # This is a strategic decision - you might want to replace a portion of the fleet
                    # to spread the cost over multiple years
                    if fleet['TotalScore'] > CONFIG['HIGH_PRIORITY_THRESHOLD']:
                        # For high-priority fleets, recommend replacing at least 20% of the fleet
                        min_replacement_size = max(1, int(fleet['FleetSize'] * 0.2))
                        partial_cost = replacement_costs.get(fleet['DeviceType'], CONFIG['DEFAULT_REPLACEMENT_COST']) * min_replacement_size
                        
                        if remaining_budget >= partial_cost:
                            # Select the oldest devices in the fleet for replacement
                            fleet_devices = devices[devices['DeviceType'] == fleet['DeviceType']]
                            fleet_devices = fleet_devices.sort_values(by=PURCHASE_DATE_COLUMN)
                            devices_to_replace = fleet_devices.head(min_replacement_size)['DeviceID'].tolist()
                            
                            yearly_replacements.append({
                                'DeviceType': fleet['DeviceType'],
                                'FleetSize': min_replacement_size,
                                'DeviceIDs': devices_to_replace,
                                'IsPartialReplacement': True
                            })
                            fleet_costs[fleet['DeviceType']] = partial_cost
                            remaining_budget -= partial_cost
            except Exception as e:
                logger.error(f"Error processing fleet {fleet['DeviceType']}: {e}")
                continue
        
        total_cost = sum(fleet_costs.values())
        
        forecast.append({
            'Year': datetime.now().year + year,
            'FleetsToReplace': yearly_replacements,
            'FleetCosts': fleet_costs,
            'TotalCost': total_cost,
            'RemainingBudget': remaining_budget
        })
    
    return forecast

def generate_unconstrained_forecast(fleet_scores: List[Dict], 
                                  devices: pd.DataFrame,
                                  replacement_costs: Dict[str, float],
                                  very_old_devices: List[Dict]) -> List[Dict]:
    """
    Generate replacement forecast based on scores and age without budget constraints.
    
    Args:
        fleet_scores: List of fleet scores
        devices: DataFrame containing device information
        replacement_costs: Dictionary of replacement costs
        very_old_devices: List of very old devices for individual replacement
    
    Returns:
        List[Dict]: Forecast for the next 5 years
    """
    logger.info("Generating unconstrained forecast based on scores and age")
    forecast = []
    
    # Sort fleet_scores by TotalScore in descending order (highest priority first)
    sorted_fleet_scores = sorted(fleet_scores, key=lambda x: x['TotalScore'], reverse=True)
    
    # Calculate replacement years for each fleet based on score and age
    fleet_replacements = []
    
    for fleet in sorted_fleet_scores:
        device_type = fleet['DeviceType']
        fleet_devices = devices[devices['DeviceType'] == device_type]
        
        # Sort devices by age (oldest first)
        fleet_devices = fleet_devices.sort_values(by=PURCHASE_DATE_COLUMN)
        
        # Calculate average age of the fleet
        fleet_ages = [calculate_device_age(row[PURCHASE_DATE_COLUMN]) for _, row in fleet_devices.iterrows()]
        avg_fleet_age = sum(fleet_ages) / len(fleet_ages) if fleet_ages else 0
        
        # Get expected lifecycle for this device type
        lifecycle_cache = load_or_create_lifecycle_cache()
        expected_lifecycle = get_expected_lifecycle(device_type, lifecycle_cache)
        
        # Calculate age factor (0-1 range, higher for older fleets)
        age_factor = min(avg_fleet_age / expected_lifecycle, 1.0)
        
        # Normalize score to 0-1 range (lower score is better)
        score_factor = 1 - (fleet['TotalScore'] / 100.0)
        
        # Determine replacement year based on age and score
        current_year = datetime.now().year
        
        # If device is past expected lifecycle
        if age_factor >= 1.0:
            # If score is low (good performance), extend replacement timeline
            if score_factor > 0.7:  # Score < 30
                # Extend replacement by 3-5 years based on score
                extension_years = int(3 + (score_factor - 0.7) * 10)  # 3-5 years extension
                replacement_year = current_year + extension_years
            elif score_factor > 0.5:  # Score < 50
                # Extend replacement by 1-2 years based on score
                extension_years = int(1 + (score_factor - 0.5) * 5)  # 1-2 years extension
                replacement_year = current_year + extension_years
            else:
                # High score (poor performance), recommend replacement soon
                replacement_year = current_year + 1
        # If device is approaching end-of-life (within 1-2 years)
        elif age_factor > 0.8:
            # If score is low (good performance), extend replacement timeline
            if score_factor > 0.7:  # Score < 30
                # Extend replacement by 2-3 years based on score
                extension_years = int(2 + (score_factor - 0.7) * 5)  # 2-3 years extension
                replacement_year = current_year + int((1 - age_factor) * expected_lifecycle) + extension_years
            elif score_factor > 0.5:  # Score < 50
                # Extend replacement by 1 year based on score
                replacement_year = current_year + int((1 - age_factor) * expected_lifecycle) + 1
            else:
                # High score (poor performance), recommend replacement at expected end-of-life
                replacement_year = current_year + int((1 - age_factor) * expected_lifecycle)
        else:
            # Device is newer, recommend replacement at expected end-of-life
            replacement_year = current_year + int((1 - age_factor) * expected_lifecycle)
        
        # Store the original replacement year before capping
        original_replacement_year = replacement_year
        
        # Limit replacement year to 10 years from now for display purposes
        max_replacement_year = current_year + 10
        capped_replacement_year = min(replacement_year, max_replacement_year)
        
        # Create fleet replacement entry
        fleet_entry = {
            'DeviceType': device_type,
            'FleetSize': fleet['FleetSize'],
            'DeviceIDs': fleet['DeviceIDs'],
            'ReplacementYear': capped_replacement_year,
            'OriginalReplacementYear': original_replacement_year,
            'FleetReplacementCost': fleet['FleetReplacementCost'],
            'Score': fleet['TotalScore'],
            'AvgAge': avg_fleet_age,
            'ExpectedLifecycle': expected_lifecycle,
            'YearsUntilReplacement': capped_replacement_year - current_year,
            'OriginalYearsUntilReplacement': original_replacement_year - current_year,
            'IsBeyondTenYears': original_replacement_year > max_replacement_year
        }
        
        fleet_replacements.append(fleet_entry)
    
    # Process very old devices for individual replacement
    very_old_replacements = []
    for device in very_old_devices:
        # Very old devices should be replaced soon, but consider their score
        score_factor = 1 - (device['Score'] / 100.0)
        
        # Determine replacement year based on score
        if score_factor > 0.7:  # Score < 30 (good performance)
            # Extend replacement by 1-2 years
            extension_years = int(1 + (score_factor - 0.7) * 5)
            replacement_year = current_year + extension_years
        elif score_factor > 0.5:  # Score < 50 (moderate performance)
            # Extend replacement by 1 year
            replacement_year = current_year + 1
        else:
            # High score (poor performance), recommend immediate replacement
            replacement_year = current_year
        
        # Store the original replacement year
        original_replacement_year = replacement_year
        
        # Limit replacement year to 10 years from now for display purposes
        max_replacement_year = current_year + 10
        capped_replacement_year = min(replacement_year, max_replacement_year)
        
        # Create very old device replacement entry
        device_entry = {
            'DeviceID': device['DeviceID'],
            'DeviceType': device['DeviceType'],
            'Age': device['Age'],
            'FleetAvgAge': device['FleetAvgAge'],
            'ExpectedLifecycle': device['ExpectedLifecycle'],
            'ReplacementYear': capped_replacement_year,
            'OriginalReplacementYear': original_replacement_year,
            'ReplacementCost': device['ReplacementCost'],
            'Score': device['Score'],
            'YearsUntilReplacement': capped_replacement_year - current_year,
            'OriginalYearsUntilReplacement': original_replacement_year - current_year,
            'IsBeyondTenYears': original_replacement_year > max_replacement_year,
            'IsVeryOldDevice': True,
            'Location': device['Location'],
            'RiskClass': device['RiskClass']
        }
        
        very_old_replacements.append(device_entry)
    
    # Group replacements by year
    # Find the maximum year in the forecast (limited to 10 years)
    max_year = min(max(
        max(f['ReplacementYear'] for f in fleet_replacements),
        max(d['ReplacementYear'] for d in very_old_replacements) if very_old_replacements else 0
    ), current_year + 10)
    
    # Create forecast entries for each year from current year to max year
    for year in range(current_year, max_year + 1):
        year_replacements = []
        
        # Add fleet replacements for this year
        year_fleet_replacements = [f for f in fleet_replacements if f['ReplacementYear'] == year]
        if year_fleet_replacements:
            year_replacements.extend([{
                'DeviceType': f['DeviceType'],
                'FleetSize': f['FleetSize'],
                'DeviceIDs': f['DeviceIDs'],
                'Score': f['Score'],
                'AvgAge': f['AvgAge'],
                'ExpectedLifecycle': f['ExpectedLifecycle'],
                'YearsUntilReplacement': f['YearsUntilReplacement'],
                'OriginalYearsUntilReplacement': f['OriginalYearsUntilReplacement'],
                'IsBeyondTenYears': f['IsBeyondTenYears']
            } for f in year_fleet_replacements])
        
        # Add very old device replacements for this year
        year_very_old_replacements = [d for d in very_old_replacements if d['ReplacementYear'] == year]
        if year_very_old_replacements:
            year_replacements.extend([{
                'DeviceID': d['DeviceID'],
                'DeviceType': d['DeviceType'],
                'Age': d['Age'],
                'FleetAvgAge': d['FleetAvgAge'],
                'ExpectedLifecycle': d['ExpectedLifecycle'],
                'Score': d['Score'],
                'YearsUntilReplacement': d['YearsUntilReplacement'],
                'OriginalYearsUntilReplacement': d['OriginalYearsUntilReplacement'],
                'IsBeyondTenYears': d['IsBeyondTenYears'],
                'IsVeryOldDevice': True,
                'Location': d['Location'],
                'RiskClass': d['RiskClass']
            } for d in year_very_old_replacements])
        
        if year_replacements:
            # Calculate costs
            fleet_costs = {f['DeviceType']: f['FleetReplacementCost'] for f in year_fleet_replacements}
            very_old_costs = {f"Individual_{d['DeviceID']}": d['ReplacementCost'] for d in year_very_old_replacements}
            
            # Combine costs
            all_costs = {**fleet_costs, **very_old_costs}
            total_cost = sum(all_costs.values())
            
            forecast.append({
                'Year': year,
                'FleetsToReplace': year_replacements,
                'FleetCosts': all_costs,
                'TotalCost': total_cost,
                'IsUnconstrained': True
            })
    
    # Add a special entry for fleets beyond 10 years
    beyond_ten_years = [f for f in fleet_replacements if f['IsBeyondTenYears']]
    if beyond_ten_years:
        forecast.append({
            'Year': 'Beyond 10 Years',
            'FleetsToReplace': [{
                'DeviceType': f['DeviceType'],
                'FleetSize': f['FleetSize'],
                'DeviceIDs': f['DeviceIDs'],
                'Score': f['Score'],
                'AvgAge': f['AvgAge'],
                'ExpectedLifecycle': f['ExpectedLifecycle'],
                'YearsUntilReplacement': f['OriginalYearsUntilReplacement'],
                'IsBeyondTenYears': True
            } for f in beyond_ten_years],
            'FleetCosts': {f['DeviceType']: f['FleetReplacementCost'] for f in beyond_ten_years},
            'TotalCost': sum(f['FleetReplacementCost'] for f in beyond_ten_years),
            'IsUnconstrained': True,
            'IsBeyondTenYears': True
        })
    
    # Add a special entry for very old devices beyond 10 years
    very_old_beyond_ten_years = [d for d in very_old_replacements if d['IsBeyondTenYears']]
    if very_old_beyond_ten_years:
        forecast.append({
            'Year': 'Beyond 10 Years',
            'FleetsToReplace': [{
                'DeviceID': d['DeviceID'],
                'DeviceType': d['DeviceType'],
                'Age': d['Age'],
                'FleetAvgAge': d['FleetAvgAge'],
                'ExpectedLifecycle': d['ExpectedLifecycle'],
                'Score': d['Score'],
                'YearsUntilReplacement': d['OriginalYearsUntilReplacement'],
                'IsBeyondTenYears': True,
                'IsVeryOldDevice': True,
                'Location': d['Location'],
                'RiskClass': d['RiskClass']
            } for d in very_old_beyond_ten_years],
            'FleetCosts': {f"Individual_{d['DeviceID']}": d['ReplacementCost'] for d in very_old_beyond_ten_years},
            'TotalCost': sum(d['ReplacementCost'] for d in very_old_beyond_ten_years),
            'IsUnconstrained': True,
            'IsBeyondTenYears': True,
            'IsVeryOldDevices': True
        })
    
    return forecast

def output_fleet_forecast(forecast: List[Dict], annual_budget: Optional[float] = None):
    """
    Output the fleet forecast results.
    
    Args:
        forecast: List of forecast entries
        annual_budget: Optional annual budget
    """
    if annual_budget is None:
        print("\nUnconstrained Fleet Replacement Forecast (Based on Scores and Age):")
    else:
        print(f"\nBudget-Constrained Fleet Replacement Forecast (Annual Budget: ${annual_budget:,.2f}):")
    
    for entry in forecast:
        # Check if this is the "Beyond 10 Years" entry
        if entry.get('IsBeyondTenYears', False) and entry['Year'] == 'Beyond 10 Years':
            if entry.get('IsVeryOldDevices', False):
                print("\nVery Old Devices Scheduled for Replacement Beyond 10 Years:")
                print("(These devices are significantly older than their fleet average but performing well)")
                
                for device in entry['FleetsToReplace']:
                    device_id = device['DeviceID']
                    device_type = device['DeviceType']
                    age = device['Age']
                    fleet_avg_age = device.get('FleetAvgAge', 'N/A')  # Use get() with default value
                    years_until_replacement = device['YearsUntilReplacement']
                    score = device['Score']
                    location = device.get('Location', 'N/A')  # Use get() with default value
                    risk_class = device.get('RiskClass', 'N/A')  # Use get() with default value
                    
                    print(f"- Device {device_id} ({device_type}) at {location} (Risk: {risk_class})")
                    print(f"  Age: {age:.1f} years (Fleet average: {fleet_avg_age if fleet_avg_age == 'N/A' else f'{fleet_avg_age:.1f} years'})")
                    print(f"  Replacement in {years_until_replacement} years (Year {datetime.now().year + years_until_replacement})")
                    print(f"  Score: {score}")
                
                print(f"Total Cost for Very Old Devices Beyond 10 Years: ${entry['TotalCost']:,.2f}")
            else:
                print("\nFleets Scheduled for Replacement Beyond 10 Years:")
                print("(These fleets are performing well and have longer expected lifecycles)")
                
                for fleet in entry['FleetsToReplace']:
                    device_type = fleet['DeviceType']
                    fleet_size = fleet['FleetSize']
                    years_until_replacement = fleet['YearsUntilReplacement']
                    score = fleet['Score']
                    avg_age = fleet['AvgAge']
                    expected_lifecycle = fleet['ExpectedLifecycle']
                    
                    print(f"- {device_type}: {fleet_size} devices")
                    print(f"  Replacement in {years_until_replacement} years (Year {datetime.now().year + years_until_replacement})")
                    print(f"  Score: {score}, Avg Age: {avg_age:.1f} years")
                    print(f"  Expected Lifecycle: {expected_lifecycle} years")
                
                print(f"Total Cost for Fleets Beyond 10 Years: ${entry['TotalCost']:,.2f}")
            continue
        
        print(f"\nYear {entry['Year']}:")
        print("Replacements:")
        
        # Group replacements by type (fleet vs. very old device)
        fleet_replacements = [r for r in entry['FleetsToReplace'] if not r.get('IsVeryOldDevice', False)]
        very_old_replacements = [r for r in entry['FleetsToReplace'] if r.get('IsVeryOldDevice', False)]
        
        # Output fleet replacements
        if fleet_replacements:
            print("Fleets to Replace:")
            for fleet in fleet_replacements:
                device_type = fleet['DeviceType']
                fleet_size = fleet['FleetSize']
                cost = entry['FleetCosts'].get(device_type, 0)
                
                replacement_type = "Partial" if fleet.get('IsPartialReplacement', False) else "Complete"
                print(f"- {device_type}: {replacement_type} replacement of {fleet_size} devices: ${cost:,.2f}")
                
                # For unconstrained forecast, show additional details
                if entry.get('IsUnconstrained', False):
                    print(f"  Score: {fleet.get('Score', 'N/A')}, Avg Age: {fleet.get('AvgAge', 'N/A'):.1f} years")
                    print(f"  Expected Lifecycle: {fleet.get('ExpectedLifecycle', 'N/A')} years")
                    print(f"  Years Until Replacement: {fleet.get('YearsUntilReplacement', 'N/A')}")
                    
                    # If this fleet is actually scheduled beyond 10 years but capped for display
                    if fleet.get('IsBeyondTenYears', False):
                        print(f"  Note: This fleet is actually scheduled for replacement in {fleet.get('OriginalYearsUntilReplacement', 'N/A')} years")
        
        # Output very old device replacements
        if very_old_replacements:
            print("Very Old Devices to Replace:")
            for device in very_old_replacements:
                device_id = device['DeviceID']
                device_type = device['DeviceType']
                age = device['Age']
                fleet_avg_age = device.get('FleetAvgAge', 'N/A')  # Use get() with default value
                cost = entry['FleetCosts'].get(f"Individual_{device_id}", 0)
                location = device.get('Location', 'N/A')  # Use get() with default value
                risk_class = device.get('RiskClass', 'N/A')  # Use get() with default value
                
                print(f"- Device {device_id} ({device_type}) at {location} (Risk: {risk_class})")
                print(f"  Age: {age:.1f} years (Fleet average: {fleet_avg_age if fleet_avg_age == 'N/A' else f'{fleet_avg_age:.1f} years'})")
                print(f"  Replacement Cost: ${cost:,.2f}")
                print(f"  Score: {device.get('Score', 'N/A')}")
                
                # If this device is actually scheduled beyond 10 years but capped for display
                if device.get('IsBeyondTenYears', False):
                    print(f"  Note: This device is actually scheduled for replacement in {device.get('OriginalYearsUntilReplacement', 'N/A')} years")
        
        print(f"Total Cost: ${entry['TotalCost']:,.2f}")
        if 'RemainingBudget' in entry:
            print(f"Remaining Budget: ${entry['RemainingBudget']:,.2f}")
        if entry.get('IsUnconstrained', False):
            print("(Based on scores and age, not budget-constrained)")

def export_forecast_to_csv(forecast: List[Dict], annual_budget: Optional[float] = None) -> str:
    """
    Export the fleet forecast results to a CSV file.
    
    Args:
        forecast: List of forecast entries
        annual_budget: Optional annual budget
        
    Returns:
        str: Path to the exported CSV file
    """
    # Create a list to store flattened forecast data
    csv_data = []
    
    # Determine forecast type
    forecast_type = "unconstrained" if annual_budget is None else "budget_constrained"
    
    # Flatten the forecast data for CSV format
    for entry in forecast:
        # Skip the "Beyond 10 Years" entry as we'll handle it separately
        if entry.get('IsBeyondTenYears', False) and entry['Year'] == 'Beyond 10 Years':
            # Handle fleet replacements beyond 10 years
            if not entry.get('IsVeryOldDevices', False):
                for fleet in entry['FleetsToReplace']:
                    device_type = fleet['DeviceType']
                    fleet_size = fleet['FleetSize']
                    replacement_cost = entry['FleetCosts'].get(device_type, 0)
                    
                    # Create a row for the CSV
                    row = {
                        'Year': 'Beyond 10 Years',
                        'DeviceType': device_type,
                        'FleetSize': fleet_size,
                        'ReplacementType': 'Complete',
                        'ReplacementCost': replacement_cost,
                        'TotalYearCost': entry['TotalCost'],
                        'RemainingBudget': None,
                        'ForecastType': forecast_type,
                        'Score': fleet.get('Score', 'N/A'),
                        'AvgAge': fleet.get('AvgAge', 'N/A'),
                        'ExpectedLifecycle': fleet.get('ExpectedLifecycle', 'N/A'),
                        'YearsUntilReplacement': fleet.get('YearsUntilReplacement', 'N/A'),
                        'IsBeyondTenYears': True,
                        'IsVeryOldDevice': False
                    }
                    
                    csv_data.append(row)
            # Handle very old devices beyond 10 years
            else:
                for device in entry['FleetsToReplace']:
                    device_id = device['DeviceID']
                    device_type = device['DeviceType']
                    age = device['Age']
                    fleet_avg_age = device.get('FleetAvgAge', 'N/A')  # Use get() with default value
                    replacement_cost = entry['FleetCosts'].get(f"Individual_{device_id}", 0)
                    location = device.get('Location', 'N/A')  # Use get() with default value
                    risk_class = device.get('RiskClass', 'N/A')  # Use get() with default value
                    
                    # Create a row for the CSV
                    row = {
                        'Year': 'Beyond 10 Years',
                        'DeviceID': device_id,
                        'DeviceType': device_type,
                        'Age': age,
                        'FleetAvgAge': fleet_avg_age,
                        'Location': location,
                        'RiskClass': risk_class,
                        'ReplacementCost': replacement_cost,
                        'TotalYearCost': entry['TotalCost'],
                        'RemainingBudget': None,
                        'ForecastType': forecast_type,
                        'Score': device.get('Score', 'N/A'),
                        'ExpectedLifecycle': device.get('ExpectedLifecycle', 'N/A'),
                        'YearsUntilReplacement': device.get('YearsUntilReplacement', 'N/A'),
                        'IsBeyondTenYears': True,
                        'IsVeryOldDevice': True
                    }
                    
                    csv_data.append(row)
            continue
        
        year = entry['Year']
        total_cost = entry['TotalCost']
        remaining_budget = entry.get('RemainingBudget', None)
        
        # Process each replacement in the year
        for replacement in entry['FleetsToReplace']:
            # Handle fleet replacements
            if not replacement.get('IsVeryOldDevice', False):
                device_type = replacement['DeviceType']
                fleet_size = replacement['FleetSize']
                replacement_cost = entry['FleetCosts'].get(device_type, 0)
                
                # Determine replacement type
                replacement_type = "Partial" if replacement.get('IsPartialReplacement', False) else "Complete"
                
                # Create a row for the CSV
                row = {
                    'Year': year,
                    'DeviceType': device_type,
                    'FleetSize': fleet_size,
                    'ReplacementType': replacement_type,
                    'ReplacementCost': replacement_cost,
                    'TotalYearCost': total_cost,
                    'RemainingBudget': remaining_budget,
                    'ForecastType': forecast_type,
                    'IsBeyondTenYears': replacement.get('IsBeyondTenYears', False),
                    'IsVeryOldDevice': False
                }
                
                # Add unconstrained forecast specific fields
                if entry.get('IsUnconstrained', False):
                    row.update({
                        'Score': replacement.get('Score', 'N/A'),
                        'AvgAge': replacement.get('AvgAge', 'N/A'),
                        'ExpectedLifecycle': replacement.get('ExpectedLifecycle', 'N/A'),
                        'YearsUntilReplacement': replacement.get('YearsUntilReplacement', 'N/A'),
                        'OriginalYearsUntilReplacement': replacement.get('OriginalYearsUntilReplacement', 'N/A')
                    })
                
                csv_data.append(row)
            # Handle very old device replacements
            else:
                device_id = replacement['DeviceID']
                device_type = replacement['DeviceType']
                age = replacement['Age']
                fleet_avg_age = replacement.get('FleetAvgAge', 'N/A')  # Use get() with default value
                replacement_cost = entry['FleetCosts'].get(f"Individual_{device_id}", 0)
                location = replacement.get('Location', 'N/A')  # Use get() with default value
                risk_class = replacement.get('RiskClass', 'N/A')  # Use get() with default value
                
                # Create a row for the CSV
                row = {
                    'Year': year,
                    'DeviceID': device_id,
                    'DeviceType': device_type,
                    'Age': age,
                    'FleetAvgAge': fleet_avg_age,
                    'Location': location,
                    'RiskClass': risk_class,
                    'ReplacementCost': replacement_cost,
                    'TotalYearCost': total_cost,
                    'RemainingBudget': remaining_budget,
                    'ForecastType': forecast_type,
                    'IsBeyondTenYears': replacement.get('IsBeyondTenYears', False),
                    'IsVeryOldDevice': True
                }
                
                # Add unconstrained forecast specific fields
                if entry.get('IsUnconstrained', False):
                    row.update({
                        'Score': replacement.get('Score', 'N/A'),
                        'ExpectedLifecycle': replacement.get('ExpectedLifecycle', 'N/A'),
                        'YearsUntilReplacement': replacement.get('YearsUntilReplacement', 'N/A'),
                        'OriginalYearsUntilReplacement': replacement.get('OriginalYearsUntilReplacement', 'N/A')
                    })
                
                csv_data.append(row)
    
    # Create DataFrame from the flattened data
    df = pd.DataFrame(csv_data)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fleet_forecast_{forecast_type}_{timestamp}.csv"
    
    # Export to CSV
    df.to_csv(filename, index=False)
    logger.info(f"Forecast exported to {filename}")
    
    return filename

def main(annual_budget: Optional[float] = None):
    """
    Main function to run the equipment analysis.
    
    Args:
        annual_budget: Optional annual budget for replacements
    """
    try:
        logger.info("Starting equipment analysis")
        
        # Load data
        devices, work_orders, replacement_costs = load_data()
        utilization_cache = load_or_create_utilization_cache()
        
        # Preprocess work orders to ensure MaintenanceType column exists
        # Use the batch size from config
        work_orders = preprocess_work_orders(work_orders, 
                                           batch_size=CONFIG['MAINTENANCE_CATEGORIZATION']['BATCH_SIZE'])
        
        # Process devices and calculate fleet scores
        fleet_scores = calculate_fleet_scores(devices, work_orders, replacement_costs, utilization_cache)
        
        # Identify very old devices for individual replacement
        very_old_devices = identify_very_old_devices(devices, fleet_scores, work_orders, replacement_costs)
        
        # Generate and output fleet forecast
        forecast = generate_fleet_forecast(fleet_scores, devices, replacement_costs, work_orders, annual_budget)
        output_fleet_forecast(forecast, annual_budget)
        
        # Export forecast to CSV
        csv_file = export_forecast_to_csv(forecast, annual_budget)
        print(f"\nForecast exported to CSV file: {csv_file}")
        
        logger.info("Equipment analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Medical Equipment Replacement Forecast')
    parser.add_argument('--budget', type=float, help='Annual budget for replacements (optional)')
    args = parser.parse_args()
    
    main(args.budget)
