import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def export_to_csv(data, filename=None):
    """
    Export data to CSV file.
    
    Args:
        data (pd.DataFrame): Data to export
        filename (str, optional): Output filename. If None, generates timestamped name.
    
    Returns:
        str: Path to the exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'forecast_{timestamp}.csv'
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)
    
    data.to_csv(filepath, index=False)
    return filepath

def export_to_excel(data, filename=None, sheet_name='Forecast'):
    """
    Export data to Excel file.
    
    Args:
        data (pd.DataFrame): Data to export
        filename (str, optional): Output filename. If None, generates timestamped name.
        sheet_name (str): Name of the Excel sheet
    
    Returns:
        str: Path to the exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'forecast_{timestamp}.xlsx'
    
    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)
    
    data.to_excel(filepath, sheet_name=sheet_name, index=False)
    return filepath

def export_to_json(data, filename=None):
    """
    Export data to JSON file.
    
    Args:
        data (pd.DataFrame): Data to export
        filename (str, optional): Output filename. If None, generates timestamped name.
    
    Returns:
        str: Path to the exported file
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'forecast_{timestamp}.json'
    
    os.makedirs('output', exist_ok=True)
    filepath = os.path.join('output', filename)
    
    data.to_json(filepath, orient='records', date_format='iso')
    return filepath

def generate_visualization(data, output_type='replacement_timeline'):
    """
    Generate visualizations from the forecast data.
    
    Args:
        data (pd.DataFrame): Forecast data
        output_type (str): Type of visualization to generate
            - 'replacement_timeline': Timeline of device replacements
            - 'fleet_scores': Distribution of fleet scores
            - 'device_ages': Age distribution of devices
    
    Returns:
        str: Path to the saved visualization
    """
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    plt.figure(figsize=(12, 6))
    
    if output_type == 'replacement_timeline':
        # Group by replacement year and count devices
        yearly_replacements = data.groupby('ReplacementYear').size()
        yearly_replacements.plot(kind='bar')
        plt.title('Device Replacements by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Devices')
        plt.xticks(rotation=45)
        
    elif output_type == 'fleet_scores':
        # Create box plot of fleet scores
        sns.boxplot(data=data, x='DeviceType', y='FleetScore')
        plt.title('Fleet Scores by Device Type')
        plt.xlabel('Device Type')
        plt.ylabel('Fleet Score')
        plt.xticks(rotation=45)
        
    elif output_type == 'device_ages':
        # Create histogram of device ages
        sns.histplot(data=data, x='Age', bins=20)
        plt.title('Distribution of Device Ages')
        plt.xlabel('Age (years)')
        plt.ylabel('Count')
    
    plt.tight_layout()
    filepath = os.path.join('output', f'{output_type}_{timestamp}.png')
    plt.savefig(filepath)
    plt.close()
    
    return filepath

def generate_report(data, output_format='html'):
    """
    Generate a comprehensive report of the forecast.
    
    Args:
        data (pd.DataFrame): Forecast data
        output_format (str): Format of the report ('html' or 'pdf')
    
    Returns:
        str: Path to the generated report
    """
    os.makedirs('output', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_format == 'html':
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Medical Device Forecast Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Medical Device Forecast Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <table>
                <tr>
                    <th>Total Devices</th>
                    <td>{len(data)}</td>
                </tr>
                <tr>
                    <th>Device Types</th>
                    <td>{', '.join(data['DeviceType'].unique())}</td>
                </tr>
                <tr>
                    <th>Average Fleet Score</th>
                    <td>{data['FleetScore'].mean():.2f}</td>
                </tr>
            </table>
            
            <h2>Forecast Details</h2>
            {data.to_html(index=False)}
        </body>
        </html>
        """
        
        filepath = os.path.join('output', f'forecast_report_{timestamp}.html')
        with open(filepath, 'w') as f:
            f.write(html_content)
            
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    return filepath 