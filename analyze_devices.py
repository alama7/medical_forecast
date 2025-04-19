import pandas as pd
import logging
import yaml
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_device_types():
    try:
        # Read the CSV file
        df = pd.read_csv('SHassets.csv')
        
        # Create a hierarchical structure
        categories = defaultdict(lambda: defaultdict(int))
        
        # Group by Category and Sub-Category
        for _, row in df.iterrows():
            category = str(row['Category Description']).strip()
            subcategory = str(row['Sub-Category Description']).strip()
            if pd.notna(category) and pd.notna(subcategory):
                categories[category][subcategory] += 1
        
        # Convert to YAML structure
        yaml_dict = {
            'categories': {},
            'manufacturers': {},
            'category_keywords': {}
        }
        
        # Add categories and subcategories
        for category, subcats in categories.items():
            if category and category != 'nan':
                yaml_dict['categories'][category] = []
                for subcat, count in subcats.items():
                    if subcat and subcat != 'nan':
                        yaml_dict['categories'][category].append({
                            'name': subcat,
                            'count': count
                        })
        
        # Write to YAML file
        with open('config/device_categories.yaml', 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
        
        # Print summary
        print("\nCategory Hierarchy:")
        print("=" * 50)
        for category, subcats in categories.items():
            if category and category != 'nan':
                total = sum(subcats.values())
                print(f"\n{category} (Total: {total} devices):")
                for subcat, count in sorted(subcats.items(), key=lambda x: x[1], reverse=True):
                    if subcat and subcat != 'nan':
                        print(f"  - {subcat}: {count} devices")
        
        # Print manufacturer analysis
        print("\nManufacturer Analysis:")
        print("=" * 50)
        manufacturer_counts = df['Manufacturer Name'].value_counts()
        print("\nTop Manufacturers:")
        for mfg, count in manufacturer_counts.head(10).items():
            if pd.notna(mfg):
                print(f"- {mfg}: {count} devices")
        
        # Save manufacturer info
        yaml_dict['manufacturers'] = {
            mfg: {
                'device_count': count,
                'specialties': [],  # To be filled based on their devices
                'known_models': []  # To be filled based on actual models
            }
            for mfg, count in manufacturer_counts.items()
            if pd.notna(mfg) and count > 10  # Only include major manufacturers
        }
        
        # Update YAML file with manufacturer info
        with open('config/device_categories.yaml', 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
        
    except Exception as e:
        logger.error(f"Error analyzing device types: {e}")

if __name__ == "__main__":
    analyze_device_types() 