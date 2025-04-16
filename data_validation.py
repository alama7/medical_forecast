import pandas as pd
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_devices_data(devices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the devices DataFrame to map columns to expected format.
    Attempts to identify and map columns based on their content and common naming patterns.
    
    Args:
        devices_df: Raw devices DataFrame from any source
        
    Returns:
        pd.DataFrame: Preprocessed devices DataFrame with standardized columns
    """
    # Create a copy to avoid modifying the original
    df = devices_df.copy()
    
    # List of possible column names for each required field
    device_id_patterns = ['Asset #', 'AssetID', 'DeviceID', 'Asset Number', 'ID', 'Asset_ID']
    device_type_patterns = ['Category', 'DeviceType', 'Type', 'Category Description', 'Asset Description', 'Sub-Category Description']
    purchase_date_patterns = ['Date Accepted', 'PurchaseDate', 'Purchase Date', 'Installation Date', 'Acquisition Date']
    location_patterns = ['Location', 'Site', 'Room', 'Department']
    cost_patterns = ['Cost', 'Cost Basis', 'Purchase Cost', 'Value']
    
    # Function to find matching column based on patterns
    def find_matching_column(columns, patterns):
        for pattern in patterns:
            matches = [col for col in columns if pattern.lower() in col.lower()]
            if matches:
                return matches[0]
        return None
    
    # Create column mapping based on detected columns
    column_mapping = {}
    
    # Find and map DeviceID column
    device_id_col = find_matching_column(df.columns, device_id_patterns)
    if device_id_col:
        column_mapping[device_id_col] = 'DeviceID'
    
    # Find and map DeviceType column
    device_type_col = find_matching_column(df.columns, device_type_patterns)
    if device_type_col:
        # If there's already a DeviceType column, drop it first
        if 'DeviceType' in df.columns:
            df = df.drop(columns=['DeviceType'])
        column_mapping[device_type_col] = 'DeviceType'
    
    # Find and map PurchaseDate column
    purchase_date_col = find_matching_column(df.columns, purchase_date_patterns)
    if purchase_date_col:
        column_mapping[purchase_date_col] = 'PurchaseDate'
    
    # Find and map Location column
    location_col = find_matching_column(df.columns, location_patterns)
    if location_col:
        column_mapping[location_col] = 'Location'
    
    # Find and map Cost column
    cost_col = find_matching_column(df.columns, cost_patterns)
    if cost_col:
        column_mapping[cost_col] = 'Cost'
    
    # Rename columns based on mapping
    df = df.rename(columns=column_mapping)
    
    # Add RiskClass if not present (default to 'REG')
    if 'RiskClass' not in df.columns:
        df['RiskClass'] = 'REG'
    
    # Print the column mapping for debugging
    print("\nDevice Data Column Mapping:")
    for orig, new in column_mapping.items():
        print(f"  {orig} -> {new}")
    
    return df

def preprocess_work_orders_data(work_orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the work orders DataFrame to map columns to expected format.
    Attempts to identify and map columns based on their content and common naming patterns.
    
    Args:
        work_orders_df: Raw work orders DataFrame from any source
        
    Returns:
        pd.DataFrame: Preprocessed work orders DataFrame with standardized columns
    """
    # Create a copy to avoid modifying the original
    df = work_orders_df.copy()
    
    # List of possible column names for each required field
    device_id_patterns = ['Asset #', 'AssetID', 'DeviceID', 'Asset Number', 'ID', 'Asset_ID']
    date_patterns = ['Date Created', 'Date', 'WorkOrderDate', 'Service Date', 'Maintenance Date']
    type_patterns = ['Type Code', 'MaintenanceType', 'Work Type', 'Service Type', 'Maintenance Category']
    description_patterns = ['Description', 'Work Description', 'Service Description', 'Notes']
    comments_patterns = ['Completion Comments', 'Comments', 'Notes', 'Resolution']
    cost_patterns = ['Cost', 'Service Cost', 'Maintenance Cost', 'Labor Cost']
    
    # Function to find matching column based on patterns
    def find_matching_column(columns, patterns):
        for pattern in patterns:
            matches = [col for col in columns if pattern.lower() in col.lower()]
            if matches:
                return matches[0]
        return None
    
    # Create column mapping based on detected columns
    column_mapping = {}
    
    # Find and map DeviceID column
    device_id_col = find_matching_column(df.columns, device_id_patterns)
    if device_id_col:
        column_mapping[device_id_col] = 'DeviceID'
    
    # Find and map Date column
    date_col = find_matching_column(df.columns, date_patterns)
    if date_col:
        column_mapping[date_col] = 'Date'
    
    # Find and map MaintenanceType column
    type_col = find_matching_column(df.columns, type_patterns)
    if type_col:
        column_mapping[type_col] = 'MaintenanceType'
    
    # Find and map Description column
    desc_col = find_matching_column(df.columns, description_patterns)
    if desc_col:
        column_mapping[desc_col] = 'Description'
    
    # Find and map Comments column
    comments_col = find_matching_column(df.columns, comments_patterns)
    if comments_col and comments_col != desc_col:  # Avoid mapping same column twice
        column_mapping[comments_col] = 'CompletionComments'
    
    # Find and map Cost column
    cost_col = find_matching_column(df.columns, cost_patterns)
    if cost_col:
        column_mapping[cost_col] = 'Cost'
    
    # Rename columns based on mapping
    df = df.rename(columns=column_mapping)
    
    # Print the column mapping for debugging
    print("\nWork Orders Column Mapping:")
    for orig, new in column_mapping.items():
        print(f"  {orig} -> {new}")
    
    return df

def validate_devices_data(devices_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the devices DataFrame for required columns and data quality.
    Works with standardized column names after preprocessing.
    
    Args:
        devices_df: DataFrame containing device information
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation issues)
    """
    issues = []
    
    # Required columns after preprocessing
    required_columns = {
        'DeviceID': 'Unique identifier for each device',
        'DeviceType': 'Type/category of the device',
        'PurchaseDate': 'Date when the device was purchased/accepted',
        'Location': 'Location of the device'
    }
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in devices_df.columns]
    if missing_columns:
        issues.append(f"Missing required columns after preprocessing: {', '.join(missing_columns)}")
        print("\nNote: The following columns are required and must be mapped from your input data:")
        for col, desc in required_columns.items():
            print(f"- {col}: {desc}")
    
    # Check for empty DataFrame
    if devices_df.empty:
        issues.append("Devices DataFrame is empty")
        return False, issues
    
    # Check for duplicate DeviceIDs
    if 'DeviceID' in devices_df.columns:
        duplicates = devices_df['DeviceID'].duplicated()
        if duplicates.any():
            duplicate_ids = devices_df[duplicates]['DeviceID'].tolist()
            issues.append(f"Found duplicate DeviceIDs: {duplicate_ids[:5]}...")
    
    # Check for missing values in required columns
    for col in required_columns:
        if col in devices_df.columns:
            missing_count = devices_df[col].isna().sum()
            if isinstance(missing_count, pd.Series):
                missing_count = missing_count.iloc[0]
            if missing_count > 0:
                issues.append(f"Found {missing_count} missing values in {col} column")
                if col == 'PurchaseDate':
                    print("\nNote: Missing PurchaseDate values will need to be handled. Options:")
                    print("1. Use a default date (e.g., earliest known date)")
                    print("2. Estimate based on other data")
                    print("3. Remove devices with missing dates")
    
    # Validate PurchaseDate format
    if 'PurchaseDate' in devices_df.columns:
        try:
            pd.to_datetime(devices_df['PurchaseDate'], errors='raise')
        except Exception as e:
            issues.append(f"Invalid date format in PurchaseDate column: {str(e)}")
    
    # Validate RiskClass values if present
    if 'RiskClass' in devices_df.columns:
        valid_risk_classes = ['CRITICAL', 'HIGH', 'REG', 'LOW']
        invalid_risks = devices_df[~devices_df['RiskClass'].isin(valid_risk_classes)]['RiskClass'].unique()
        if len(invalid_risks) > 0:
            issues.append(f"Found invalid RiskClass values: {invalid_risks}")
            print("\nNote: Valid RiskClass values are: CRITICAL, HIGH, REG, LOW")
            print("Default value 'REG' will be used for invalid or missing values")
    
    return len(issues) == 0, issues

def validate_work_orders_data(work_orders_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the work orders DataFrame for required columns and data quality.
    Works with standardized column names after preprocessing.
    
    Args:
        work_orders_df: DataFrame containing work order information
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation issues)
    """
    issues = []
    
    # Required columns after preprocessing
    required_columns = {
        'DeviceID': 'ID of the device the work order is for',
        'Date': 'Date of the work order'
    }
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in work_orders_df.columns]
    if missing_columns:
        issues.append(f"Missing required columns after preprocessing: {', '.join(missing_columns)}")
        print("\nNote: The following columns are required and must be mapped from your input data:")
        for col, desc in required_columns.items():
            print(f"- {col}: {desc}")
    
    # Check for empty DataFrame
    if work_orders_df.empty:
        issues.append("Work orders DataFrame is empty")
        return False, issues
    
    # Check for missing values in required columns
    for col in required_columns:
        if col in work_orders_df.columns:
            missing_count = work_orders_df[col].isna().sum()
            if isinstance(missing_count, pd.Series):
                missing_count = missing_count.iloc[0]
            if missing_count > 0:
                issues.append(f"Found {missing_count} missing values in {col} column")
    
    # Validate Date format
    if 'Date' in work_orders_df.columns:
        try:
            pd.to_datetime(work_orders_df['Date'], errors='raise')
        except Exception as e:
            issues.append(f"Invalid date format in Date column: {str(e)}")
    
    # Check for maintenance type information
    maintenance_info_columns = ['MaintenanceType', 'Description', 'CompletionComments']
    available_info_columns = [col for col in maintenance_info_columns if col in work_orders_df.columns]
    
    if not available_info_columns:
        issues.append("Need at least one of these columns for maintenance type information: " + 
                     ", ".join(maintenance_info_columns))
    elif 'MaintenanceType' in work_orders_df.columns:
        # If MaintenanceType is present, validate the values
        valid_types = ['Cosmetic', 'User Error', 'Repair', 'Software', 'PM', 'Inspection', 'Cleaning']
        invalid_types = [t for t in work_orders_df['MaintenanceType'].unique() if t not in valid_types]
        if len(invalid_types) > 0:
            issues.append(f"Found invalid maintenance types: {invalid_types}")
            print("\nNote: Valid maintenance types are:")
            print("- Cosmetic: Cosmetic repairs")
            print("- User Error: Issues caused by user error")
            print("- Repair: General repairs")
            print("- Software: Software-related issues")
            print("- PM: Preventive maintenance")
            print("- Inspection: Device inspections (will be mapped to PM)")
            print("- Cleaning: Device cleaning (will be mapped to PM)")
            print("\nOther types will be categorized based on description using the categorize_maintenance_type function")
    
    return len(issues) == 0, issues

def validate_replacement_costs(replacement_costs: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate the replacement costs dictionary.
    
    Args:
        replacement_costs: Dictionary mapping device types to replacement costs
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation issues)
    """
    issues = []
    
    # Check for empty dictionary
    if not replacement_costs:
        issues.append("Replacement costs dictionary is empty")
        return False, issues
    
    # Check for invalid costs
    invalid_costs = {device_type: cost for device_type, cost in replacement_costs.items() 
                    if not isinstance(cost, (int, float)) or cost <= 0}
    if invalid_costs:
        issues.append(f"Found invalid replacement costs for devices: {list(invalid_costs.keys())[:5]}...")
    
    return len(issues) == 0, issues

def validate_data_relationships(devices_df: pd.DataFrame, 
                              work_orders_df: pd.DataFrame,
                              replacement_costs: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate relationships between different data sources.
    
    Args:
        devices_df: DataFrame containing device information
        work_orders_df: DataFrame containing work order information
        replacement_costs: Dictionary mapping device types to replacement costs
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation issues)
    """
    issues = []
    
    # Check if work orders reference existing devices
    if 'DeviceID' in work_orders_df.columns and 'DeviceID' in devices_df.columns:
        invalid_device_refs = work_orders_df[~work_orders_df['DeviceID'].isin(devices_df['DeviceID'])]
        if not invalid_device_refs.empty:
            issues.append(f"Found {len(invalid_device_refs)} work orders referencing non-existent devices")
    
    # Check if replacement costs exist for all device types
    if 'DeviceType' in devices_df.columns:
        unique_device_types = set(devices_df['DeviceType'].dropna().unique())
        missing_costs = unique_device_types - set(replacement_costs.keys())
        if missing_costs:
            issues.append(f"Missing replacement costs for device types: {list(missing_costs)[:5]}...")
    
    return len(issues) == 0, issues

def validate_all_data(devices_df: pd.DataFrame,
                     work_orders_df: pd.DataFrame,
                     replacement_costs: Dict[str, float]) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Perform comprehensive validation of all data sources.
    
    Args:
        devices_df: DataFrame containing device information
        work_orders_df: DataFrame containing work order information
        replacement_costs: Dictionary mapping device types to replacement costs
        
    Returns:
        Tuple[bool, Dict[str, List[str]]]: (is_valid, dictionary of validation issues by data source)
    """
    all_issues = {}
    
    # Validate each data source
    devices_valid, devices_issues = validate_devices_data(devices_df)
    all_issues['devices'] = devices_issues
    
    work_orders_valid, work_orders_issues = validate_work_orders_data(work_orders_df)
    all_issues['work_orders'] = work_orders_issues
    
    costs_valid, costs_issues = validate_replacement_costs(replacement_costs)
    all_issues['replacement_costs'] = costs_issues
    
    # Validate relationships between data sources
    relationships_valid, relationship_issues = validate_data_relationships(
        devices_df, work_orders_df, replacement_costs
    )
    all_issues['relationships'] = relationship_issues
    
    # Overall validation result
    is_valid = all([
        devices_valid,
        work_orders_valid,
        costs_valid,
        relationships_valid
    ])
    
    return is_valid, all_issues

def format_validation_report(validation_results: Tuple[bool, Dict[str, List[str]]]) -> str:
    """
    Format validation results into a readable report.
    
    Args:
        validation_results: Tuple of (is_valid, issues dictionary)
        
    Returns:
        str: Formatted validation report
    """
    is_valid, issues = validation_results
    
    report = ["Data Validation Report", "=" * 50, ""]
    
    if is_valid:
        report.append("✅ All data validation checks passed successfully!")
    else:
        report.append("❌ Data validation found issues that need to be addressed:")
        report.append("")
        
        for data_source, data_issues in issues.items():
            if data_issues:
                report.append(f"\n{data_source.upper()}:")
                for issue in data_issues:
                    report.append(f"  • {issue}")
    
    return "\n".join(report)

def analyze_maintenance_types(work_orders_df):
    """Analyze maintenance types in the work orders data."""
    print("\nMaintenance Type Analysis")
    print("=" * 50)
    
    # Get unique maintenance types
    maintenance_types = work_orders_df['MaintenanceType'].unique()
    print("\nUnique maintenance types found:")
    for mtype in maintenance_types:
        count = len(work_orders_df[work_orders_df['MaintenanceType'] == mtype])
        print(f"- {mtype}: {count} occurrences")
    
    # Check for invalid maintenance types
    valid_types = ['Cosmetic', 'User Error', 'Repair', 'Software', 'PM', 'Inspection', 'Cleaning']
    invalid_types = [t for t in maintenance_types if t not in valid_types]
    
    if invalid_types:
        print("\nWARNING: Invalid maintenance types found:")
        for inv_type in invalid_types:
            print(f"- {inv_type}")
        print("\nValid maintenance types are:")
        print("- Cosmetic")
        print("- User Error")
        print("- Repair")
        print("- Software")
        print("- PM")
        print("- Inspection (will be mapped to PM)")
        print("- Cleaning (will be mapped to PM)")
    else:
        print("\nAll maintenance types are valid.")

if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    if len(sys.argv) != 3:
        print("Usage: python data_validation.py <devices_csv_file> <work_orders_csv_file>")
        sys.exit(1)
    
    # Get current timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nValidating input files...")
    print("=" * 50)
    
    # Load and preprocess the devices file
    devices_file = sys.argv[1]
    print(f"\nProcessing devices file: {devices_file}")
    print("-" * 30)
    print("Loading devices data...")
    devices_df = pd.read_csv(devices_file)
    print(f"Loaded {len(devices_df)} device records")
    print("\nOriginal device columns:")
    print(devices_df.columns.tolist())
    
    # Preprocess devices data
    print("\nPreprocessing devices data...")
    print("-" * 30)
    devices_df = preprocess_devices_data(devices_df)
    print("\nStandardized device columns:")
    print(devices_df.columns.tolist())
    
    # Load and preprocess the work orders file
    work_orders_file = sys.argv[2]
    print(f"\nProcessing work orders file: {work_orders_file}")
    print("-" * 30)
    print("Loading work orders data...")
    work_orders_df = pd.read_csv(work_orders_file)
    print(f"Loaded {len(work_orders_df)} work order records")
    print("\nOriginal work order columns:")
    print(work_orders_df.columns.tolist())
    
    # Preprocess work orders data
    print("\nPreprocessing work orders data...")
    print("-" * 30)
    work_orders_df = preprocess_work_orders_data(work_orders_df)
    print("\nStandardized work order columns:")
    print(work_orders_df.columns.tolist())
    
    print("\nValidating preprocessed data...")
    print("=" * 50)
    
    # Validate devices data
    print("\nValidating devices data:")
    print("-" * 30)
    devices_valid, devices_issues = validate_devices_data(devices_df)
    if devices_valid:
        print("✅ Devices data validation passed")
    else:
        print("❌ Devices data validation failed")
    
    # Validate work orders data
    print("\nValidating work orders data:")
    print("-" * 30)
    work_orders_valid, work_orders_issues = validate_work_orders_data(work_orders_df)
    if work_orders_valid:
        print("✅ Work orders data validation passed")
    else:
        print("❌ Work orders data validation failed")
    
    # Create sample replacement costs if DeviceType exists
    if 'DeviceType' in devices_df.columns:
        print("\nGenerating replacement costs...")
        print("-" * 30)
        # Get the first DeviceType column if there are duplicates
        device_type_col = [col for col in devices_df.columns if col == 'DeviceType'][0]
        device_types = np.unique(devices_df[device_type_col].dropna().values)
        replacement_costs = {str(device_type): 100000.0 for device_type in device_types}
        print(f"Generated replacement costs for {len(replacement_costs)} device types")
    else:
        print("\nWarning: DeviceType column not found in devices file. Using empty replacement costs.")
        replacement_costs = {}
    
    # Run validation
    is_valid, issues = validate_all_data(devices_df, work_orders_df, replacement_costs)
    print("\nValidation Results:")
    print("=" * 50)
    print(format_validation_report((is_valid, {
        'devices': devices_issues,
        'work_orders': work_orders_issues,
        'replacement_costs': [],
        'relationships': []
    })))
    
    # Save preprocessed data if validation passed
    if is_valid:
        print("\nSaving preprocessed data...")
        print("=" * 50)
        
        # Generate output filenames with timestamp
        devices_output = f"preprocessed_devices_{timestamp}.csv"
        work_orders_output = f"preprocessed_work_orders_{timestamp}.csv"
        
        # Save preprocessed data
        print("\nSaving devices data...")
        print("-" * 30)
        devices_df.to_csv(devices_output, index=False)
        print(f"Saved {len(devices_df)} device records to {devices_output}")
        
        print("\nSaving work orders data...")
        print("-" * 30)
        work_orders_df.to_csv(work_orders_output, index=False)
        print(f"Saved {len(work_orders_df)} work order records to {work_orders_output}")
        
        print("\nPreprocessing Summary:")
        print("=" * 50)
        print(f"Timestamp: {timestamp}")
        print(f"Input files:")
        print(f"- Devices: {devices_file}")
        print(f"- Work Orders: {work_orders_file}")
        print(f"\nOutput files:")
        print(f"- Devices: {devices_output}")
        print(f"- Work Orders: {work_orders_output}")
        print(f"\nData Summary:")
        print(f"- Total devices: {len(devices_df)}")
        print(f"- Total work orders: {len(work_orders_df)}")
        print(f"- Unique device types: {len(devices_df['DeviceType'].unique())}")
        print(f"- Date range: {devices_df['PurchaseDate'].min()} to {devices_df['PurchaseDate'].max()}")
        
        print("\nThese files are ready to be used with main.py")
    else:
        print("\nValidation failed. Please fix the issues before proceeding.") 