import pandas as pd
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(devices_file: str, work_orders_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return the devices and work orders data."""
    logger.info(f"Loading devices data from {devices_file}")
    devices_df = pd.read_csv(devices_file)
    
    logger.info(f"Loading work orders data from {work_orders_file}")
    work_orders_df = pd.read_csv(work_orders_file)
    
    # Map device columns
    device_column_map = {
        'Asset #': 'DeviceID',
        'Date Accepted': 'PurchaseDate',
        'Category Description': 'DeviceType',
        'Site': 'Location',
        'Cost Basis': 'Cost'
    }
    devices_df = devices_df.rename(columns=device_column_map)
    
    # Map work order columns
    work_order_column_map = {
        'Asset #': 'DeviceID',
        'Type Code': 'MaintenanceType',
        'Date Created': 'Date',
        'Description': 'Description',
        'Completion Comments': 'CompletionComments'
    }
    work_orders_df = work_orders_df.rename(columns=work_order_column_map)
    
    logger.info("Applied column mappings:")
    logger.info("Devices: " + ", ".join(f"{k} -> {v}" for k, v in device_column_map.items()))
    logger.info("Work Orders: " + ", ".join(f"{k} -> {v}" for k, v in work_order_column_map.items()))
    
    return devices_df, work_orders_df

def handle_missing_purchase_dates(devices_df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing purchase dates based on user choice."""
    if 'PurchaseDate' not in devices_df.columns:
        logger.warning("PurchaseDate column not found in devices data")
        return devices_df
    
    # Convert to datetime
    devices_df['PurchaseDate'] = pd.to_datetime(devices_df['PurchaseDate'], errors='coerce')
    missing_dates = devices_df['PurchaseDate'].isna().sum()
    
    if missing_dates == 0:
        return devices_df
    
    print(f"\nFound {missing_dates} missing PurchaseDate values")
    print("\nHow would you like to handle missing purchase dates?")
    print("1. Use earliest known date")
    print("2. Use a specific date")
    print("3. Remove devices with missing dates")
    print("4. Keep missing dates (not recommended for forecasting)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        earliest_date = devices_df['PurchaseDate'].min()
        if pd.isna(earliest_date):
            earliest_date = pd.Timestamp('2000-01-01')
        devices_df['PurchaseDate'] = devices_df['PurchaseDate'].fillna(earliest_date)
        logger.info(f"Filled missing dates with earliest date: {earliest_date}")
    
    elif choice == '2':
        date_str = input("Enter date (YYYY-MM-DD): ").strip()
        try:
            fill_date = pd.Timestamp(date_str)
            devices_df['PurchaseDate'] = devices_df['PurchaseDate'].fillna(fill_date)
            logger.info(f"Filled missing dates with {fill_date}")
        except:
            logger.error("Invalid date format. Keeping missing dates.")
    
    elif choice == '3':
        devices_df = devices_df.dropna(subset=['PurchaseDate'])
        logger.info(f"Removed {missing_dates} devices with missing dates")
    
    elif choice == '4':
        logger.warning("Keeping missing dates - this may affect forecasting accuracy")
    
    else:
        logger.warning("Invalid choice. Keeping missing dates.")
    
    return devices_df

def handle_missing_device_ids(work_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Handle work orders with missing device IDs based on user choice."""
    if 'DeviceID' not in work_orders_df.columns:
        logger.warning("DeviceID column not found in work orders data")
        return work_orders_df
    
    missing_ids = work_orders_df['DeviceID'].isna().sum()
    if missing_ids == 0:
        return work_orders_df
    
    print(f"\nFound {missing_ids} work orders with missing DeviceID values")
    print("\nHow would you like to handle work orders with missing DeviceIDs?")
    print("1. Remove work orders with missing DeviceIDs")
    print("2. Keep work orders with missing DeviceIDs (not recommended)")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == '1':
        work_orders_df = work_orders_df.dropna(subset=['DeviceID'])
        logger.info(f"Removed {missing_ids} work orders with missing DeviceIDs")
    
    elif choice == '2':
        logger.warning("Keeping work orders with missing DeviceIDs - these cannot be used for forecasting")
    
    else:
        logger.warning("Invalid choice. Keeping work orders with missing DeviceIDs.")
    
    return work_orders_df

def handle_maintenance_types(work_orders_df: pd.DataFrame) -> pd.DataFrame:
    """Handle maintenance type standardization based on user input."""
    if 'MaintenanceType' not in work_orders_df.columns:
        logger.warning("MaintenanceType column not found in work orders data")
        return work_orders_df
    
    # Get unique maintenance types
    maintenance_types = sorted(work_orders_df['MaintenanceType'].unique())
    print("\nCurrent maintenance types found:")
    for mtype in maintenance_types:
        count = len(work_orders_df[work_orders_df['MaintenanceType'] == mtype])
        print(f"- {mtype}: {count} occurrences")
    
    print("\nHow would you like to handle maintenance types?")
    print("1. Map types to standard categories")
    print("2. Delete specific maintenance types")
    print("3. Keep original types")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        print("\nEnter mappings for each maintenance type (press Enter to skip):")
        type_mapping = {}
        
        for mtype in maintenance_types:
            if pd.isna(mtype):
                continue
                
            print(f"\nCurrent type: {mtype}")
            print("Standard categories: PM, Repair, Cosmetic, User Error, Software")
            new_type = input("Map to (or press Enter to keep original): ").strip()
            
            if new_type:
                type_mapping[mtype] = new_type
                logger.info(f"Mapped {mtype} to {new_type}")
        
        # Apply mapping
        if type_mapping:
            work_orders_df['MaintenanceType'] = work_orders_df['MaintenanceType'].map(
                lambda x: type_mapping.get(x, x)
            )
            logger.info("Applied maintenance type mapping")
    
    elif choice == '2':
        print("\nSelect maintenance types to delete (enter numbers separated by commas):")
        for i, mtype in enumerate(maintenance_types, 1):
            count = len(work_orders_df[work_orders_df['MaintenanceType'] == mtype])
            print(f"{i}. {mtype}: {count} occurrences")
        
        types_to_delete = input("\nEnter numbers of types to delete (e.g., '1,3,4'): ").strip()
        try:
            # Convert input to list of indices
            delete_indices = [int(x.strip()) - 1 for x in types_to_delete.split(',')]
            types_to_remove = [maintenance_types[i] for i in delete_indices]
            
            # Count records to be deleted
            records_to_delete = work_orders_df['MaintenanceType'].isin(types_to_remove).sum()
            
            print(f"\nThis will delete {records_to_delete} work orders with types:")
            for mtype in types_to_remove:
                count = len(work_orders_df[work_orders_df['MaintenanceType'] == mtype])
                print(f"- {mtype}: {count} records")
            
            confirm = input("\nProceed with deletion? (y/n): ").strip().lower()
            if confirm == 'y':
                # Remove selected maintenance types
                work_orders_df = work_orders_df[~work_orders_df['MaintenanceType'].isin(types_to_remove)]
                logger.info(f"Removed {records_to_delete} work orders with maintenance types: {', '.join(types_to_remove)}")
            else:
                logger.info("Deletion cancelled")
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid input: {e}")
            logger.warning("Keeping all maintenance types")
    
    elif choice == '3':
        logger.info("Keeping original maintenance types")
    
    else:
        logger.warning("Invalid choice. Keeping original maintenance types.")
    
    # Print final maintenance type statistics
    print("\nFinal maintenance type distribution:")
    for mtype in sorted(work_orders_df['MaintenanceType'].unique()):
        count = len(work_orders_df[work_orders_df['MaintenanceType'] == mtype])
        print(f"- {mtype}: {count} occurrences")
    
    return work_orders_df

def save_cleaned_data(devices_df: pd.DataFrame, work_orders_df: pd.DataFrame, 
                     devices_file: str, work_orders_file: str):
    """Save cleaned data with timestamps."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate output filenames
    devices_output = f"cleaned_devices_{timestamp}.csv"
    work_orders_output = f"cleaned_work_orders_{timestamp}.csv"
    
    # Save files
    logger.info(f"Saving cleaned devices data to {devices_output}")
    devices_df.to_csv(devices_output, index=False)
    
    logger.info(f"Saving cleaned work orders data to {work_orders_output}")
    work_orders_df.to_csv(work_orders_output, index=False)
    
    print("\nCleaned data saved to:")
    print(f"- Devices: {devices_output}")
    print(f"- Work Orders: {work_orders_output}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_cleaner.py <devices_csv_file> <work_orders_csv_file>")
        sys.exit(1)
    
    # Load data
    devices_file = sys.argv[1]
    work_orders_file = sys.argv[2]
    devices_df, work_orders_df = load_data(devices_file, work_orders_file)
    
    print("\nData Cleaning Tool")
    print("=" * 50)
    
    # Handle each type of issue
    devices_df = handle_missing_purchase_dates(devices_df)
    work_orders_df = handle_missing_device_ids(work_orders_df)
    work_orders_df = handle_maintenance_types(work_orders_df)
    
    # Save cleaned data
    save_cleaned_data(devices_df, work_orders_df, devices_file, work_orders_file)

if __name__ == "__main__":
    main() 