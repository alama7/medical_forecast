from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
from config import CONFIG
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants for column names
PURCHASE_DATE_COLUMN = 'PurchaseDate'

# Move the MARKET_PRICES constant here
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

def get_device_category(device_type: str) -> Tuple[str, str]:
    """
    Categorize medical devices based on their description.
    
    Args:
        device_type: Description of the medical device
        
    Returns:
        Tuple containing (category, subcategory)
    """
    device_type = str(device_type).upper()
    
    # Imaging devices
    if any(term in device_type for term in ['MRI', 'MAGNETIC']):
        return ('Imaging', 'MRI')
    elif any(term in device_type for term in ['CT', 'CAT SCAN']):
        return ('Imaging', 'CT')
    elif any(term in device_type for term in ['X-RAY', 'XRAY']):
        return ('Imaging', 'X-Ray')
    
    # Monitoring devices
    elif any(term in device_type for term in ['MONITOR', 'ECG', 'EKG']):
        return ('Monitoring', 'Patient Monitor')
    
    # Default category
    return ('Other', 'General')

def calculate_price(device_info, market_price_info, reference_year: Optional[int] = None) -> float:
    """
    Calculates estimated price based on device characteristics and inflation
    
    Args:
        device_info: Device information dictionary
        market_price_info: Market price information dictionary
        reference_year: Optional reference year (defaults to current year)
        
    Returns:
        float: Calculated price
    """
    base_price = market_price_info['base_price']
    description = str(device_info['Asset Description']).upper()
    location = str(device_info['Location Description']).upper()
    
    # Use current year if reference_year not provided
    reference_year = reference_year or datetime.now().year
    
    # Calculate age-based adjustment with error handling
    try:
        date_accepted = pd.to_datetime(device_info['Date Accepted'])
        years_old = (pd.Timestamp(reference_year, 1, 1) - date_accepted).days / 365
        inflation_factor = (1 + CONFIG['COST_ESTIMATION']['INFLATION_RATE']) ** years_old
    except (ValueError, TypeError):
        logger.warning(f"Invalid date for device {device_info.get('Asset #', 'Unknown')}")
        inflation_factor = 1.0
    
    # Apply premium/budget adjustments
    if any(keyword in description or keyword in location 
           for keyword in market_price_info['premium_keywords']):
        adjusted_price = base_price * market_price_info['premium_multiplier']
    elif any(keyword in description or keyword in location 
            for keyword in market_price_info['budget_keywords']):
        adjusted_price = base_price * market_price_info['budget_multiplier']
    else:
        adjusted_price = base_price
    
    return adjusted_price * inflation_factor

def estimate_missing_costs(devices_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimates missing costs using enhanced categorization and market research.
    
    Args:
        devices_df: DataFrame containing device information
        
    Returns:
        Dict[str, float]: Dictionary mapping device types to estimated costs
        
    Raises:
        IOError: If unable to save estimation log
    """
    replacement_costs = {}
    cost_estimation_log = []
    
    for idx, device in devices_df.iterrows():
        try:
            device_type = str(device['Asset Description'])
            current_cost = device['Cost Basis']
            
            if pd.notna(current_cost) and current_cost > 0:
                replacement_costs[device_type] = current_cost
                continue
            
            category, subcategory = get_device_category(device_type)
            category_key = (category, subcategory)
            
            if category_key in MARKET_PRICES:
                price_info = MARKET_PRICES[category_key]
                estimated_cost = calculate_price(device, price_info)
            else:
                estimated_cost = estimate_cost_from_similar_devices(
                    device_type, devices_df
                )
            
            replacement_costs[device_type] = estimated_cost
            log_estimation(cost_estimation_log, device, category, 
                         subcategory, estimated_cost)
            
        except Exception as e:
            logger.error(f"Error estimating cost for device {device_type}: {e}")
            replacement_costs[device_type] = CONFIG['DEFAULT_REPLACEMENT_COST']
    
    try:
        save_estimation_log(cost_estimation_log)
    except IOError as e:
        logger.error(f"Failed to save estimation log: {e}")
    
    return replacement_costs

def estimate_cost_from_similar_devices(device_type: str, 
                                     devices_df: pd.DataFrame) -> float:
    """Helper function to estimate cost from similar devices."""
    similar_devices = devices_df[
        devices_df['Asset Description'].str.contains(
            device_type, case=False, na=False
        ) &
        (devices_df['Cost Basis'] > 0)
    ]['Cost Basis']
    
    if len(similar_devices) >= CONFIG['COST_ESTIMATION']['MIN_SAMPLE_SIZE']:
        return similar_devices.median()
    return CONFIG['DEFAULT_REPLACEMENT_COST']

def log_estimation(log: List[Dict], device: pd.Series, 
                  category: str, subcategory: str, 
                  estimated_cost: float) -> None:
    """Helper function to log cost estimation details."""
    log.append({
        'Asset_Number': device['Asset #'],
        'Device_Type': device['Asset Description'],
        'Category': category,
        'Subcategory': subcategory,
        'Estimated_Cost': estimated_cost,
        'Estimation_Method': 'Market Price' if (category, subcategory) in MARKET_PRICES
                           else 'Similar Devices Median'
    })

def save_estimation_log(log: List[Dict]) -> None:
    """Save estimation log to CSV file."""
    estimation_log_df = pd.DataFrame(log)
    estimation_log_df.to_csv('cost_estimation_log.csv', index=False)