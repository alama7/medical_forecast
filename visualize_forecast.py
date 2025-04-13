import pandas as pd
import os
import matplotlib.pyplot as plt
from medical_forecast import (
    calculate_fleet_scores,
    generate_visualization
)
from main import generate_fleet_forecast

# Create sample data
sample_devices = pd.DataFrame({
    'DeviceType': ['CT', 'MRI', 'X-Ray', 'CT', 'MRI', 'X-Ray'],
    'PurchaseDate': ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01'],
    'Score': [80, 90, 70, 85, 75, 95],
    'FleetSize': [2, 1, 1, 2, 1, 1]
})

# Calculate fleet scores
fleet_scores = calculate_fleet_scores(sample_devices)

# Generate forecast
forecast_data = generate_fleet_forecast(fleet_scores, sample_devices)

# Convert forecast data to DataFrame if it's a list
if isinstance(forecast_data, list):
    # Process each year's forecast data
    forecast_df = pd.DataFrame()
    
    for year_entry in forecast_data:
        year = year_entry['Year']
        
        # Process fleet replacements for this year
        for fleet in year_entry.get('FleetsToReplace', []):
            row = {
                'Year': year,
                'DeviceType': fleet['DeviceType'],
                'FleetSize': fleet.get('FleetSize', 1),
                'TotalScore': fleet.get('Score', 0),
                'ReplacementCost': year_entry['FleetCosts'].get(fleet['DeviceType'], 0),
                'TotalYearCost': year_entry['TotalCost'],
                'IsVeryOldDevice': fleet.get('IsVeryOldDevice', False)
            }
            
            # Add additional fields if available
            if 'AvgAge' in fleet:
                row['AvgAge'] = fleet['AvgAge']
            if 'ExpectedLifecycle' in fleet:
                row['ExpectedLifecycle'] = fleet['ExpectedLifecycle']
            if 'YearsUntilReplacement' in fleet:
                row['YearsUntilReplacement'] = fleet['YearsUntilReplacement']
            if 'Location' in fleet:
                row['Location'] = fleet['Location']
            
            forecast_df = pd.concat([forecast_df, pd.DataFrame([row])], ignore_index=True)
    
    # If forecast_df is empty, create a default entry
    if forecast_df.empty:
        forecast_df = pd.DataFrame({
            'Year': [pd.Timestamp.now().year],
            'DeviceType': ['No Data'],
            'FleetSize': [0],
            'TotalScore': [0],
            'ReplacementCost': [0],
            'TotalYearCost': [0],
            'IsVeryOldDevice': [False]
        })
    
    forecast_data = forecast_df

# Add fleet scores to forecast data
forecast_data = forecast_data.merge(fleet_scores[['DeviceType', 'TotalScore']], on='DeviceType', how='left')

# Calculate device ages
forecast_data['Age'] = (pd.Timestamp.now() - pd.to_datetime(sample_devices['PurchaseDate'])).dt.days / 365.25

# Generate all visualization types
visualization_types = ['replacement_timeline', 'fleet_scores', 'device_ages']
for viz_type in visualization_types:
    filepath = generate_visualization(forecast_data, viz_type)
    print(f"Generated {viz_type} visualization: {filepath}")

# Display the visualizations
print("\nOpening visualizations...")
for viz_type in visualization_types:
    filepath = os.path.join('output', f'{viz_type}_*.png')
    # Find the most recent file matching the pattern
    files = sorted([f for f in os.listdir('output') if f.startswith(viz_type) and f.endswith('.png')])
    if files:
        latest_file = os.path.join('output', files[-1])
        print(f"Opening {latest_file}")
        # Open the image with the default image viewer
        os.system(f"open {latest_file}")  # For macOS
        # For Windows, use: os.system(f"start {latest_file}")
        # For Linux, use: os.system(f"xdg-open {latest_file}")

print("\nVisualizations have been generated and opened.")
print("You can find them in the 'output' directory.") 