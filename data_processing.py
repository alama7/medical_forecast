from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
from config import CONFIG
from pathlib import Path
import os
import openai

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

# Default lifecycle values for different device types
DEFAULT_LIFECYCLE = {
    'MRI Scanner': 10,
    'CT Scanner': 12,
    'X-Ray Machine': 8,
    'Ultrasound': 9,
    'Patient Monitor': 7,
    'Ventilator': 10,
    'Defibrillator': 8,
    'Infusion Pump': 7,
    'ECG Machine': 10,
    'Anesthesia Machine': 10,
    'Surgical Light': 15,
    'Surgical Table': 20,
    'Blood Gas Analyzer': 8,
    'Centrifuge': 10,
    'Microscope': 15,
    'Dialysis Machine': 8,
    'Mammography Unit': 10,
    'Portable X-Ray': 8,
    'EEG Machine': 10,
    'Blood Pressure Monitor': 5,
    'Pulse Oximeter': 5,
    'Infant Incubator': 8,
    'Phototherapy Unit': 6,
    'Cardiac Catheterization System': 15,
    'Endoscopy System': 8,
    'Chemotherapy Pump': 6,
    'Linear Accelerator': 15,
    'PET Scanner': 12,
    'Bone Densitometer': 10,
    'Respiratory Function Analyzer': 8,
    'Ophthalmoscope': 6,
    'Visual Field Analyzer': 8,
    'Surgical Microscope': 10,
    'Dental X-Ray': 8,
    'Dental Chair Unit': 10,
    'Sterilizer': 7,
    'Washer Disinfector': 7,
    'Hematology Analyzer': 8,
    'Chemistry Analyzer': 8
}

# Create OpenAI client
client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

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

def load_or_create_lifecycle_cache() -> Dict[str, float]:
    """
    Load lifecycle cache from file or create new if not exists.
    
    Returns:
        Dict[str, float]: Dictionary mapping device types to expected lifecycle in years
    """
    try:
        cache_path = Path(CONFIG.get('LIFECYCLE_CACHE_FILE', CONFIG['DATA_DIR'] / 'lifecycle_cache.csv'))
        if cache_path.exists():
            return pd.read_csv(cache_path).set_index('DeviceType')['ExpectedLifecycle'].to_dict()
        return {}
    except Exception as e:
        logger.error(f"Error loading lifecycle cache: {e}")
        return {}

def get_expected_lifecycle(device_type: str, lifecycle_cache: Dict[str, float] = None) -> float:
    """
    Get expected lifecycle for a device type, either from cache, default values, or via API.
    
    Args:
        device_type: Type of medical device
        lifecycle_cache: Cache of known lifecycle values
    
    Returns:
        float: Expected lifecycle in years
    """
    if lifecycle_cache is None:
        lifecycle_cache = load_or_create_lifecycle_cache()
        
    if device_type in lifecycle_cache:
        return lifecycle_cache[device_type]
    
    # Check if we have a default value for this device type
    if device_type in DEFAULT_LIFECYCLE:
        lifecycle_cache[device_type] = DEFAULT_LIFECYCLE[device_type]
        save_lifecycle_cache(lifecycle_cache)
        return DEFAULT_LIFECYCLE[device_type]
    
    try:
        # Use OpenAI API to estimate the expected lifecycle
        response = client.chat.completions.create(
            model=CONFIG.get('OPENAI_MODEL', 'gpt-4'),
            messages=[
                {"role": "system", "content": "You are a medical equipment expert who understands the expected lifecycle of various medical equipment used in hospitals. Respond only with a number representing years."},
                {"role": "user", "content": f"What is the expected lifecycle in years for a {device_type} in a typical hospital setting? Consider manufacturer recommendations, typical usage patterns, and technological obsolescence. Provide only a number representing years."}
            ],
            temperature=0.3
        )
        
        lifecycle = float(response.choices[0].message.content.strip())
        lifecycle = max(lifecycle, 1)  # Ensure minimum of 1 year
        
        lifecycle_cache[device_type] = lifecycle
        save_lifecycle_cache(lifecycle_cache)
        
        return lifecycle
    except Exception as e:
        logger.error(f"Error estimating lifecycle for {device_type}: {e}")
        # Return a default value if API call fails
        return CONFIG.get('DEFAULT_LIFECYCLE', 8)

def save_lifecycle_cache(cache: Dict[str, float]):
    """Save lifecycle cache to file."""
    try:
        cache_df = pd.DataFrame(list(cache.items()), 
                              columns=['DeviceType', 'ExpectedLifecycle'])
        cache_path = Path(CONFIG.get('LIFECYCLE_CACHE_FILE', CONFIG['DATA_DIR'] / 'lifecycle_cache.csv'))
        cache_df.to_csv(cache_path, index=False)
    except Exception as e:
        logger.error(f"Error saving lifecycle cache: {e}")

def categorize_maintenance_type(description: str, work_order_type: str = None) -> str:
    """
    Categorize maintenance work orders based on their description using ChatGPT.
    
    Categories:
    - Cosmetic: Appearance-related issues (multiplier of 2)
    - User Error: Issues caused by user mistakes (multiplier of 1)
    - Repair: Functional repairs of equipment (multiplier of 3)
    - Software: Software-related issues (multiplier of 2)
    - PM: Preventive Maintenance (multiplier of 0)
    
    Args:
        description: Description of the maintenance work
        work_order_type: Type of work order if available (optional)
        
    Returns:
        str: Maintenance type category
    """
    # If the description is already one of our categories, return it
    description_upper = str(description).upper()
    if description_upper in ['COSMETIC', 'USER ERROR', 'REPAIR', 'SOFTWARE', 'PM']:
        return description_upper.title()
    
    # If the work_order_type is already one of our categories, return it
    if work_order_type:
        work_order_type_upper = str(work_order_type).upper()
        if work_order_type_upper in ['COSMETIC', 'USER ERROR', 'REPAIR', 'SOFTWARE', 'PM', 'INSPECTION', 'CLEANING']:
            # Map inspection and cleaning to PM
            if work_order_type_upper in ['INSPECTION', 'CLEANING']:
                return 'PM'
            return work_order_type_upper.title()
    
    try:
        # Construct the prompt for ChatGPT
        prompt = f"""
        Categorize the following medical equipment maintenance work order into exactly one of these categories:
        - Cosmetic: Appearance-related issues (scratches, dents, paint, aesthetic issues)
        - User Error: Issues caused by user mistakes or improper use
        - Repair: Functional repairs of equipment components or systems
        - Software: Software-related issues, updates, configurations
        - PM: Preventive Maintenance, routine inspections, cleaning
        
        Work Order Description: {description}
        """
        
        # Add work order type to the prompt if available
        if work_order_type:
            prompt += f"\nWork Order Type: {work_order_type}"
            
        prompt += "\n\nCategory (respond with only one word):"
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=CONFIG['MAINTENANCE_CATEGORIZATION']['MODEL'],
            messages=[
                {"role": "system", "content": "You are a medical equipment maintenance expert. Categorize work orders into exactly one category from the list provided. Respond with only the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=CONFIG['MAINTENANCE_CATEGORIZATION']['TEMPERATURE'],
            max_tokens=10  # We only need a single word response
        )
        
        # Extract the category from the response
        category = response.choices[0].message.content.strip()
        
        # Normalize the category
        if "COSMETIC" in category.upper():
            return "Cosmetic"
        elif "USER ERROR" in category.upper() or "USER" in category.upper():
            return "User Error"
        elif "REPAIR" in category.upper():
            return "Repair"
        elif "SOFTWARE" in category.upper():
            return "Software"
        elif "PM" in category.upper() or "PREVENTIVE" in category.upper() or "MAINTENANCE" in category.upper() or "INSPECTION" in category.upper() or "CLEANING" in category.upper():
            return "PM"
        else:
            # Default to Repair if the response doesn't match any category
            logger.warning(f"Unrecognized category from API: {category}. Defaulting to Repair.")
            return "Repair"
            
    except Exception as e:
        # Log the error and fall back to the keyword-based approach
        logger.error(f"Error using ChatGPT for categorization: {e}. Falling back to keyword-based approach.")
        
        # Fall back to the keyword-based approach
        if any(term in description_upper for term in ['PM', 'PREVENTIVE', 'SCHEDULED', 'ROUTINE', 'INSPECTION', 'CLEANING']):
            return 'PM'
        elif any(term in description_upper for term in ['COSMETIC', 'APPEARANCE', 'SCRATCH', 'DENT', 'PAINT', 'AESTHETIC']):
            return 'Cosmetic'
        elif any(term in description_upper for term in ['SOFTWARE', 'FIRMWARE', 'UPDATE', 'PROGRAM', 'BOOT', 'REBOOT', 'SYSTEM', 'CONFIGURATION']):
            return 'Software'
        elif any(term in description_upper for term in ['USER ERROR', 'MISUSE', 'IMPROPER', 'TRAINING', 'OPERATOR', 'HUMAN ERROR']):
            return 'User Error'
        else:
            return 'Repair'

def batch_categorize_maintenance_types(work_orders_data: List[Dict[str, str]], batch_size: int = None) -> List[str]:
    """
    Categorize multiple work orders in batches to reduce API calls.
    
    Args:
        work_orders_data: List of dictionaries with 'description' and optional 'work_order_type' keys
        batch_size: Number of work orders to process in a single API call (defaults to config value)
        
    Returns:
        List[str]: List of maintenance type categories
    """
    # Use the batch size from config if not specified
    if batch_size is None:
        batch_size = CONFIG['MAINTENANCE_CATEGORIZATION']['BATCH_SIZE']
        
    results = []
    total_orders = len(work_orders_data)
    
    # Process in batches
    for i in range(0, total_orders, batch_size):
        batch = work_orders_data[i:i+batch_size]
        try:
            # Construct the prompt for ChatGPT with multiple work orders
            prompt = """
            Categorize each of the following medical equipment maintenance work orders into exactly one of these categories:
            - Cosmetic: Appearance-related issues (scratches, dents, paint, aesthetic issues)
            - User Error: Issues caused by user mistakes or improper use
            - Repair: Functional repairs of equipment components or systems
            - Software: Software-related issues, updates, configurations
            - PM: Preventive Maintenance, routine inspections, cleaning
            
            For each work order, respond with only the category name.
            """
            
            # Add each work order to the prompt
            for j, order in enumerate(batch):
                prompt += f"\n\nWork Order #{j+1}:"
                prompt += f"\nDescription: {order.get('description', 'N/A')}"
                if order.get('work_order_type'):
                    prompt += f"\nType: {order.get('work_order_type')}"
            
            prompt += "\n\nRespond with a list of categories, one per line, in the same order as the work orders:"
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model=CONFIG['MAINTENANCE_CATEGORIZATION']['MODEL'],
                messages=[
                    {"role": "system", "content": "You are a medical equipment maintenance expert. Categorize work orders into exactly one category from the list provided. Respond with only the category names, one per line, in the same order as the work orders."},
                    {"role": "user", "content": prompt}
                ],
                temperature=CONFIG['MAINTENANCE_CATEGORIZATION']['TEMPERATURE'],
                max_tokens=CONFIG['MAINTENANCE_CATEGORIZATION']['MAX_TOKENS']
            )
            
            # Extract the categories from the response
            categories = response.choices[0].message.content.strip().split('\n')
            
            # Normalize the categories and add to results
            for category in categories:
                category = category.strip()
                # Remove any numbering or prefixes
                category = ''.join([c for c in category if not (c.isdigit() or c in '.-:)')])
                category = category.strip()
                
                # Normalize the category
                if "COSMETIC" in category.upper():
                    results.append("Cosmetic")
                elif "USER ERROR" in category.upper() or "USER" in category.upper():
                    results.append("User Error")
                elif "REPAIR" in category.upper():
                    results.append("Repair")
                elif "SOFTWARE" in category.upper():
                    results.append("Software")
                elif "PM" in category.upper() or "PREVENTIVE" in category.upper() or "MAINTENANCE" in category.upper() or "INSPECTION" in category.upper() or "CLEANING" in category.upper():
                    results.append("PM")
                else:
                    # Default to Repair if the response doesn't match any category
                    logger.warning(f"Unrecognized category from API: {category}. Defaulting to Repair.")
                    results.append("Repair")
            
            # If we didn't get enough categories, fill in the rest with Repair
            while len(results) < i + len(batch):
                logger.warning("Not enough categories returned from API. Defaulting to Repair.")
                results.append("Repair")
                
        except Exception as e:
            # Log the error and fall back to the keyword-based approach for this batch
            logger.error(f"Error using ChatGPT for batch categorization: {e}. Falling back to individual categorization.")
            
            # Process each work order individually using the regular function
            for order in batch:
                results.append(categorize_maintenance_type(
                    order.get('description', ''), 
                    order.get('work_order_type')
                ))
    
    return results