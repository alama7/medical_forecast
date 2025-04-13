import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import logging
from main import generate_fleet_forecast, load_data, calculate_fleet_scores, preprocess_work_orders, load_or_create_utilization_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for caching
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'last_budget' not in st.session_state:
    st.session_state.last_budget = None
if 'devices' not in st.session_state:
    st.session_state.devices = None
if 'work_orders' not in st.session_state:
    st.session_state.work_orders = None
if 'replacement_costs' not in st.session_state:
    st.session_state.replacement_costs = None
if 'fleet_scores' not in st.session_state:
    st.session_state.fleet_scores = None
if 'utilization_cache' not in st.session_state:
    st.session_state.utilization_cache = None
if 'all_devices_df' not in st.session_state:
    st.session_state.all_devices_df = None
if 'device_counts' not in st.session_state:
    st.session_state.device_counts = None

st.set_page_config(page_title="Medical Device Forecast Dashboard", layout="wide")

# Title and description
st.title("Medical Device Forecast Dashboard")
st.markdown("""
This dashboard provides an interactive view of medical device forecasts, including:
- Device replacement timelines
- Fleet scores and performance metrics
- Age distribution analysis
- Location-based analysis
""")

# Sidebar - Budget Configuration
st.sidebar.header('Forecast Configuration')

# Budget input
budget_option = st.sidebar.radio(
    "Forecast Type",
    ["Unconstrained", "Budget Constrained"]
)

budget = None
if budget_option == "Budget Constrained":
    budget = st.sidebar.number_input(
        "Annual Budget ($)",
        min_value=0,
        value=1000000,
        step=100000,
        format="%d"
    )
    st.sidebar.markdown(f"**Annual Budget:** ${budget:,}")
    logger.info(f"Budget set to: ${budget:,}")

# Load data if not already loaded
if st.session_state.devices is None:
    st.session_state.devices, st.session_state.work_orders, st.session_state.replacement_costs = load_data()
    logger.info(f"Loaded {len(st.session_state.devices)} devices, {len(st.session_state.work_orders)} work orders")
    
    # Load utilization cache
    st.session_state.utilization_cache = load_or_create_utilization_cache()
    logger.info(f"Loaded utilization cache with {len(st.session_state.utilization_cache)} entries")
    
    # Preprocess work orders
    st.session_state.work_orders = preprocess_work_orders(st.session_state.work_orders)
    logger.info(f"Preprocessed {len(st.session_state.work_orders)} work orders")
    
    # Calculate fleet scores
    st.session_state.fleet_scores = calculate_fleet_scores(
        st.session_state.devices, 
        st.session_state.work_orders, 
        st.session_state.replacement_costs, 
        st.session_state.utilization_cache
    )
    logger.info(f"Calculated scores for {len(st.session_state.fleet_scores)} fleets")
    
    # Create a DataFrame with all devices for reference
    st.session_state.all_devices_df = st.session_state.devices.copy()
    logger.info(f"Created reference DataFrame with {len(st.session_state.all_devices_df)} devices")
    
    # Calculate device counts by type
    st.session_state.device_counts = st.session_state.all_devices_df['DeviceType'].value_counts().to_dict()
    logger.info(f"Calculated device counts for {len(st.session_state.device_counts)} device types")

# Check if we need to regenerate the forecast
regenerate_forecast = False
if budget != st.session_state.last_budget:
    regenerate_forecast = True
    st.session_state.last_budget = budget

# Generate forecast data if needed
if st.session_state.forecast_data is None or regenerate_forecast:
    if budget is not None:
        st.sidebar.success(f"Generating budget-constrained forecast with ${budget:,} annual budget")
        logger.info(f"Generating budget-constrained forecast with budget: ${budget:,}")
        st.session_state.forecast_data = generate_fleet_forecast(
            st.session_state.fleet_scores, 
            st.session_state.devices, 
            st.session_state.replacement_costs, 
            st.session_state.work_orders, 
            budget
        )
    else:
        st.sidebar.info("Generating unconstrained forecast")
        logger.info("Generating unconstrained forecast")
        st.session_state.forecast_data = generate_fleet_forecast(
            st.session_state.fleet_scores, 
            st.session_state.devices, 
            st.session_state.replacement_costs, 
            st.session_state.work_orders
        )
    
    # Log the forecast data structure
    if isinstance(st.session_state.forecast_data, list):
        logger.info(f"Forecast data is a list with {len(st.session_state.forecast_data)} entries")
        for i, entry in enumerate(st.session_state.forecast_data):
            logger.info(f"Entry {i}: Year={entry.get('Year')}, TotalCost=${entry.get('TotalCost', 0):,.2f}")
    else:
        logger.info(f"Forecast data is a DataFrame with shape {st.session_state.forecast_data.shape}")

# Convert forecast data to DataFrame if it's a list
forecast_data = st.session_state.forecast_data
if isinstance(forecast_data, list):
    # Process each year's forecast data
    forecast_df = pd.DataFrame()
    
    for year_entry in forecast_data:
        year = year_entry['Year']
        
        # Process fleet replacements for this year
        for fleet in year_entry.get('FleetsToReplace', []):
            device_type = fleet['DeviceType']
            fleet_size = fleet.get('FleetSize', 1)
            
            # Get the total number of devices of this type
            total_devices = st.session_state.device_counts.get(device_type, 0)
            
            row = {
                'Year': year,
                'DeviceType': device_type,
                'FleetSize': fleet_size,
                'TotalDevices': total_devices,  # Add total devices of this type
                'ReplacementCost': year_entry['FleetCosts'].get(device_type, 0),
                'TotalYearCost': year_entry['TotalCost'],
                'IsVeryOldDevice': fleet.get('IsVeryOldDevice', False)
            }
            
            # Add score if available
            if 'Score' in fleet:
                row['TotalScore'] = fleet['Score']
            
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
            'Year': [datetime.now().year],
            'DeviceType': ['No Data'],
            'FleetSize': [0],
            'TotalDevices': [0],
            'TotalScore': [0],
            'ReplacementCost': [0],
            'TotalYearCost': [0],
            'IsVeryOldDevice': [False]
        })
    
    forecast_data = forecast_df
    logger.info(f"Converted forecast data to DataFrame with shape {forecast_data.shape}")

# Convert fleet scores to DataFrame
fleet_scores_df = pd.DataFrame(st.session_state.fleet_scores)
logger.info(f"Fleet scores columns: {fleet_scores_df.columns.tolist()}")

# Merge forecast data with fleet scores
if 'TotalScore' not in forecast_data.columns and 'DeviceType' in fleet_scores_df.columns:
    # Try to merge with fleet scores to get TotalScore
    if 'Score' in fleet_scores_df.columns:
        fleet_scores_df = fleet_scores_df.rename(columns={'Score': 'TotalScore'})
    
    merged_data = forecast_data.merge(
        fleet_scores_df[['DeviceType', 'TotalScore', 'FleetSize']],
        on='DeviceType',
        how='left'
    )
else:
    merged_data = forecast_data

# Ensure TotalScore column exists
if 'TotalScore' not in merged_data.columns:
    merged_data['TotalScore'] = 0

# Calculate device ages
merged_data['Age'] = (datetime.now() - pd.to_datetime(st.session_state.devices['PurchaseDate'])).dt.days / 365.25

# Add location information if available
if 'Location' in st.session_state.devices.columns:
    # Create a mapping of DeviceType to Location
    device_location_map = st.session_state.devices.groupby('DeviceType')['Location'].first().to_dict()
    
    # Add Location column to merged_data if it doesn't exist
    if 'Location' not in merged_data.columns:
        merged_data['Location'] = merged_data['DeviceType'].map(device_location_map)
    
    # Fill any missing locations with 'Unknown'
    merged_data['Location'] = merged_data['Location'].fillna('Unknown')
    
    logger.info(f"Added location information. Unique locations: {merged_data['Location'].nunique()}")

# Convert Year column to string to handle 'Beyond 10 Years' values
merged_data['Year'] = merged_data['Year'].astype(str)

# Debug section to show data counts
with st.expander("Debug Information"):
    st.write(f"Total devices in database: {len(st.session_state.all_devices_df)}")
    st.write(f"Devices in forecast: {len(merged_data)}")
    
    # Count devices by year
    year_counts = merged_data.groupby('Year').agg({
        'DeviceType': 'count',
        'FleetSize': 'sum'
    }).reset_index()
    year_counts.columns = ['Year', 'Device Types', 'Total Devices']
    st.write("Devices by year:")
    st.dataframe(year_counts)
    
    # Count devices by device type
    device_type_counts = merged_data.groupby('DeviceType').agg({
        'FleetSize': 'sum',
        'TotalDevices': 'first'
    }).reset_index()
    device_type_counts.columns = ['Device Type', 'Devices in Forecast', 'Total Devices of Type']
    st.write("Devices by type:")
    st.dataframe(device_type_counts)
    
    # Show all device types in the database
    all_device_types = st.session_state.all_devices_df['DeviceType'].unique()
    st.write(f"All device types in database: {len(all_device_types)}")
    st.write(all_device_types)
    
    # Show device types in forecast
    forecast_device_types = merged_data['DeviceType'].unique()
    st.write(f"Device types in forecast: {len(forecast_device_types)}")
    st.write(forecast_device_types)
    
    # Check for missing device types
    missing_types = set(all_device_types) - set(forecast_device_types)
    if missing_types:
        st.warning(f"Missing device types in forecast: {missing_types}")
    
    # For unconstrained forecast, show all devices
    if budget is None:
        st.subheader("All Devices (Unconstrained Forecast)")
        st.dataframe(st.session_state.all_devices_df)

# Sidebar filters
st.sidebar.header('Filters')

# Device type filter
device_types = ['All'] + list(merged_data['DeviceType'].unique())
selected_device_type = st.sidebar.selectbox('Device Type', device_types)

# Location filter
if 'Location' in merged_data.columns:
    locations = ['All'] + list(merged_data['Location'].unique())
    selected_location = st.sidebar.selectbox('Location', locations)
else:
    selected_location = 'All'

# Age range filter
min_age = merged_data['Age'].min()
max_age = merged_data['Age'].max()
age_range = st.sidebar.slider('Age Range (years)', min_age, max_age, (min_age, max_age))

# Score range filter
min_score = merged_data['TotalScore'].min()
max_score = merged_data['TotalScore'].max()

# Handle the case where min_score and max_score are identical
if min_score == max_score:
    # If they're identical, create a small range around the value
    score_range = (min_score - 0.1, max_score + 0.1)
    st.sidebar.info(f"All devices have the same score: {min_score:.2f}")
else:
    score_range = st.sidebar.slider('Score Range', min_score, max_score, (min_score, max_score))

# Apply filters
filtered_data = merged_data.copy()

if selected_device_type != 'All':
    filtered_data = filtered_data[filtered_data['DeviceType'] == selected_device_type]

if selected_location != 'All' and 'Location' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['Location'] == selected_location]

filtered_data = filtered_data[
    (filtered_data['Age'] >= age_range[0]) &
    (filtered_data['Age'] <= age_range[1]) &
    (filtered_data['TotalScore'] >= score_range[0]) &
    (filtered_data['TotalScore'] <= score_range[1])
]

# For unconstrained forecast, include all devices that aren't in the forecast
if budget is None and st.session_state.all_devices_df is not None:
    # Get device types in the forecast
    forecast_device_types = set(merged_data['DeviceType'].unique())
    
    # Get devices not in the forecast
    missing_devices = st.session_state.all_devices_df[~st.session_state.all_devices_df['DeviceType'].isin(forecast_device_types)]
    
    if not missing_devices.empty:
        # Add missing devices to filtered_data with a special year
        missing_data = []
        for _, device in missing_devices.iterrows():
            device_type = device['DeviceType']
            total_devices = st.session_state.device_counts.get(device_type, 0)
            
            missing_data.append({
                'Year': 'Not Scheduled',
                'DeviceType': device_type,
                'FleetSize': 1,
                'TotalDevices': total_devices,
                'TotalScore': 0,  # Default score
                'ReplacementCost': st.session_state.replacement_costs.get(device_type, 0),
                'TotalYearCost': 0,
                'IsVeryOldDevice': False,
                'Age': (datetime.now() - pd.to_datetime(device['PurchaseDate'])).dt.days / 365.25,
                'Location': device.get('Location', 'Unknown')
            })
        
        # Add missing devices to filtered_data
        if missing_data:
            missing_df = pd.DataFrame(missing_data)
            filtered_data = pd.concat([filtered_data, missing_df], ignore_index=True)
            logger.info(f"Added {len(missing_data)} missing devices to the filtered data")

# Display key metrics
st.header('Key Metrics')
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_devices_in_forecast = filtered_data['FleetSize'].sum()
    st.metric('Total Devices in Forecast', total_devices_in_forecast)
with col2:
    st.metric('Average Fleet Score', f"{filtered_data['TotalScore'].mean():.1f}")
with col3:
    st.metric('Average Device Age', f"{filtered_data['Age'].mean():.1f} years")
with col4:
    urgent_replacements = len(filtered_data[filtered_data['IsVeryOldDevice']])
    st.metric('Urgent Replacements', urgent_replacements)

# Replacement Timeline
st.header('Replacement Timeline')

# Convert Year back to numeric for plotting, handling 'Beyond 10 Years' as a special case
timeline_data = filtered_data.copy()
timeline_data['YearNum'] = timeline_data['Year'].apply(lambda x: 11 if x == 'Beyond 10 Years' else (12 if x == 'Not Scheduled' else float(x)))

# Sort by numeric year
timeline_data = timeline_data.sort_values('YearNum')

# Group by year for the chart
timeline_grouped = timeline_data.groupby('Year').agg({
    'DeviceType': 'count',
    'FleetSize': 'sum',
    'TotalYearCost': 'sum'
}).reset_index()

# Create the chart
fig = px.bar(timeline_grouped, x='Year', y='FleetSize', 
             title='Device Replacements by Year',
             labels={'FleetSize': 'Number of Devices', 'Year': 'Year'})
fig.add_scatter(x=timeline_grouped['Year'], y=timeline_grouped['TotalYearCost'], 
                name='Total Cost', yaxis='y2', line=dict(color='red'))

# Add budget line if budget is specified
if budget is not None:
    fig.add_hline(y=budget, line_dash="dash", line_color="green", 
                  annotation_text=f"Budget: ${budget:,}", 
                  annotation_position="bottom right")

fig.update_layout(yaxis2=dict(title='Total Cost ($)', overlaying='y', side='right'))
st.plotly_chart(fig, use_container_width=True)

# Fleet Analysis
st.header('Fleet Analysis')
col1, col2 = st.columns(2)

with col1:
    # Fleet scores by device type
    fig = px.box(filtered_data, x='DeviceType', y='TotalScore', 
                 title='Fleet Scores by Device Type')
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Device age distribution
    fig = px.histogram(filtered_data, x='Age', nbins=20, 
                      title='Device Age Distribution',
                      labels={'Age': 'Age (years)', 'count': 'Number of Devices'})
    st.plotly_chart(fig, use_container_width=True)

# Location Analysis - Always show this section if Location data is available
if 'Location' in filtered_data.columns and filtered_data['Location'].nunique() > 1:
    st.header('Location Analysis')
    col1, col2 = st.columns(2)

    with col1:
        # Device distribution by location
        location_counts = filtered_data.groupby('Location')['FleetSize'].sum().reset_index()
        fig = px.pie(values=location_counts['FleetSize'], names=location_counts['Location'], 
                     title='Device Distribution by Location')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fleet scores by location
        fig = px.box(filtered_data, x='Location', y='TotalScore', 
                     title='Fleet Scores by Location')
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Add a map of replacements by location over time
    st.subheader('Replacements by Location Over Time')
    
    # Group data by Year and Location
    location_timeline = filtered_data.groupby(['Year', 'Location'])['FleetSize'].sum().reset_index()
    
    # Create a stacked bar chart
    fig = px.bar(location_timeline, x='Year', y='FleetSize', color='Location',
                 title='Replacements by Location Over Time',
                 labels={'FleetSize': 'Number of Devices', 'Year': 'Year'})
    st.plotly_chart(fig, use_container_width=True)
elif 'Location' in filtered_data.columns:
    st.info("Location analysis is not available because there is only one location in the data.")

# Detailed Data View
st.header('Detailed Data View')

# Get available columns
available_columns = filtered_data.columns.tolist()
logger.info(f"Available columns: {available_columns}")

# Define all possible columns
all_columns = ['DeviceType', 'Location', 'Age', 'TotalScore', 'FleetSize', 'TotalDevices', 'Year', 'ReplacementCost', 'TotalYearCost', 'IsVeryOldDevice']

# Filter to only include columns that exist in the DataFrame
existing_columns = [col for col in all_columns if col in available_columns]

# Set default columns to show
default_columns = [col for col in ['DeviceType', 'Year', 'TotalScore', 'Age', 'FleetSize', 'TotalDevices'] if col in existing_columns]

columns_to_show = st.multiselect(
    'Select columns to display',
    existing_columns,
    default=default_columns
)

if columns_to_show:
    st.dataframe(filtered_data[columns_to_show])

# Export options
st.header('Export Data')
export_format = st.radio('Export Format', ['CSV', 'Excel'])

if st.button('Export'):
    if export_format == 'CSV':
        filtered_data.to_csv('filtered_data.csv', index=False)
        st.success('Data exported to filtered_data.csv')
    else:
        filtered_data.to_excel('filtered_data.xlsx', index=False)
        st.success('Data exported to filtered_data.xlsx')

else:
    st.error("No data available. Please check the data files in the 'data' directory.") 