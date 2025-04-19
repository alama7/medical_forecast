#!/usr/bin/env python
"""
Medical Device Category Manager

A command-line tool for managing medical device categories, including:
- Viewing all categories and subcategories
- Adding new categories and subcategories
- Removing categories and subcategories
- Processing device files for categorization
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .device_categorization import DeviceCategorizer

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def view_categories(categorizer: DeviceCategorizer):
    """Display all categories and subcategories."""
    categories = categorizer.get_all_categories()
    
    print("\nDevice Categories:")
    print("=" * 50)
    
    for category, subcategories in categories.items():
        print(f"\n{category}:")
        
        # Check if subcategories is a list or a dictionary
        if isinstance(subcategories, list):
            # Simple list of subcategories
            for subcategory in subcategories:
                print(f"  - {subcategory}")
        elif isinstance(subcategories, dict) and 'subcategories' in subcategories:
            # Nested structure with subcategories as a dictionary
            for subcategory_name in subcategories['subcategories'].keys():
                print(f"  - {subcategory_name}")
        elif isinstance(subcategories, dict):
            # Direct dictionary of subcategories
            for subcategory_name in subcategories.keys():
                print(f"  - {subcategory_name}")

def add_category(categorizer: DeviceCategorizer, category: str, subcategory: str):
    """Add a new category and subcategory."""
    if category in categorizer.categories_data['categories']:
        if subcategory in categorizer.categories_data['categories'][category]:
            logger.warning(f"Subcategory '{subcategory}' already exists in category '{category}'")
            return
        
        categorizer.categories_data['categories'][category].append(subcategory)
    else:
        categorizer.categories_data['categories'][category] = [subcategory]
    
    categorizer._save_categories()
    logger.info(f"Added subcategory '{subcategory}' to category '{category}'")

def remove_category(categorizer: DeviceCategorizer, category: str, subcategory: Optional[str] = None):
    """Remove a category or subcategory."""
    if category not in categorizer.categories_data['categories']:
        logger.warning(f"Category '{category}' does not exist")
        return
    
    if subcategory:
        if subcategory not in categorizer.categories_data['categories'][category]:
            logger.warning(f"Subcategory '{subcategory}' does not exist in category '{category}'")
            return
        
        categorizer.categories_data['categories'][category].remove(subcategory)
        logger.info(f"Removed subcategory '{subcategory}' from category '{category}'")
        
        # If no subcategories left, remove the category
        if not categorizer.categories_data['categories'][category]:
            del categorizer.categories_data['categories'][category]
            logger.info(f"Removed empty category '{category}'")
    else:
        del categorizer.categories_data['categories'][category]
        logger.info(f"Removed category '{category}' and all its subcategories")
    
    categorizer._save_categories()

def process_device_file(categorizer: DeviceCategorizer, file_path: str, strict_mode: bool = True):
    """Process a device file and categorize all devices."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Processing {len(df)} devices from {file_path}")
        
        # Check for required columns
        required_cols = ['Asset Description', 'Manufacturer Name', 'Model']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return
        
        # Set strict mode
        categorizer.strict_mode = strict_mode
        
        # Process each device
        results = []
        for idx, row in df.iterrows():
            description = row['Asset Description']
            manufacturer = row['Manufacturer Name'] if pd.notna(row['Manufacturer Name']) else None
            model = row['Model'] if pd.notna(row['Model']) else None
            
            category, subcategory = categorizer.categorize_device(description, manufacturer, model)
            results.append({
                'Asset Description': description,
                'Manufacturer': manufacturer,
                'Model': model,
                'Category': category,
                'Subcategory': subcategory
            })
            
            if idx % 100 == 0:
                logger.info(f"Processed {idx} devices...")
        
        # Save results
        output_file = f"categorized_devices_{Path(file_path).stem}.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        logger.info(f"Saved categorization results to {output_file}")
        
        # If in strict mode, report uncategorized devices
        if strict_mode:
            uncategorized = categorizer.get_uncategorized_devices_report()
            if uncategorized:
                logger.warning(f"Found {len(uncategorized)} devices that couldn't be categorized")
                uncategorized_file = f"uncategorized_devices_{Path(file_path).stem}.csv"
                pd.DataFrame(uncategorized).to_csv(uncategorized_file, index=False)
                logger.info(f"Saved uncategorized devices to {uncategorized_file}")
        
    except Exception as e:
        logger.error(f"Error processing device file: {e}")

def main():
    """Main entry point for the category manager."""
    parser = argparse.ArgumentParser(description="Medical Device Category Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # View categories command
    view_parser = subparsers.add_parser("view", help="View all categories and subcategories")
    
    # Add category command
    add_parser = subparsers.add_parser("add", help="Add a new category or subcategory")
    add_parser.add_argument("category", help="Category name")
    add_parser.add_argument("subcategory", help="Subcategory name")
    
    # Remove category command
    remove_parser = subparsers.add_parser("remove", help="Remove a category or subcategory")
    remove_parser.add_argument("category", help="Category name")
    remove_parser.add_argument("--subcategory", help="Subcategory name (optional)")
    
    # Process file command
    process_parser = subparsers.add_parser("process", help="Process a device file")
    process_parser.add_argument("file", help="Path to the device CSV file")
    process_parser.add_argument("--flexible", action="store_true", 
                               help="Allow creation of new categories (default: strict mode)")
    
    args = parser.parse_args()
    
    # Initialize the categorizer
    categorizer = DeviceCategorizer()
    
    if args.command == "view":
        view_categories(categorizer)
        return 0
    
    elif args.command == "add":
        add_category(categorizer, args.category, args.subcategory)
        return 0
    
    elif args.command == "remove":
        remove_category(categorizer, args.category, args.subcategory)
        return 0
    
    elif args.command == "process":
        process_device_file(categorizer, args.file, not args.flexible)
        return 0
    
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 