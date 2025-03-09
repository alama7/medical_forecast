import pytest
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add parent directory to path to import main module
sys.path.append(str(Path(__file__).parent.parent))

from main import (
    calculate_device_age,
    calculate_annual_maintenance_cost,
    get_utilization_rate,
    calculate_maintenance_history_score,
    calculate_location_score,
    calculate_total_score,
    get_device_category,
    calculate_price,
    generate_forecast,
    save_utilization_cache
)

@pytest.fixture
def sample_devices():
    return pd.DataFrame({
        'DeviceID': ['D1', 'D2'],
        'DeviceType': ['MRI', 'CT'],
        'PurchaseDate': [
            datetime.now() - timedelta(days=365*2),
            datetime.now() - timedelta(days=365*5)
        ],
        'ExpectedLifecycle (years)': [10, 8],
        'RiskClass': ['III', 'II'],
        'Location': ['ICU', 'Radiology']
    })

@pytest.fixture
def sample_work_orders():
    return pd.DataFrame({
        'DeviceID': ['D1', 'D1', 'D2'],
        'Date': [
            datetime(2022, 1, 15),
            datetime(2022, 6, 15),
            datetime(2022, 3, 15)
        ],
        'Cost': [1000, 2000, 1500],
        'MaintenanceType': ['Repair', 'Cosmetic', 'Repair']
    })

@pytest.fixture
def sample_market_price_info():
    return {
        'base_price': 1000000,
        'premium_keywords': ['PREMIUM'],
        'budget_keywords': ['BASIC'],
        'premium_multiplier': 1.3,
        'budget_multiplier': 0.7
    }

def test_calculate_device_age():
    purchase_date = datetime.now() - timedelta(days=365*2)
    age = calculate_device_age(purchase_date)
    assert 1.9 <= age <= 2.1

def test_calculate_device_age_with_none():
    age = calculate_device_age(None)
    assert age == 0

def test_calculate_annual_maintenance_cost(sample_work_orders):
    cost = calculate_annual_maintenance_cost('D1', 2022, sample_work_orders)
    assert cost == 3000

def test_calculate_annual_maintenance_cost_no_records(sample_work_orders):
    cost = calculate_annual_maintenance_cost('D3', 2022, sample_work_orders)
    assert cost == 0

def test_calculate_annual_maintenance_cost_invalid_year(sample_work_orders):
    with pytest.raises(ValueError):
        calculate_annual_maintenance_cost('D1', -1, sample_work_orders)

def test_get_utilization_rate():
    cache = {'MRI': 12.5}
    rate = get_utilization_rate('MRI', cache)
    assert rate == 12.5

@patch('openai.ChatCompletion.create')
def test_get_utilization_rate_api_call(mock_create):
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="15.5"))]
    )
    cache = {}
    rate = get_utilization_rate('CT', cache)
    assert rate == 15.5
    mock_create.assert_called_once()

@patch('openai.ChatCompletion.create')
def test_get_utilization_rate_api_error(mock_create):
    mock_create.side_effect = Exception("API Error")
    cache = {}
    rate = get_utilization_rate('CT', cache)
    assert rate == 12  # Default value

def test_calculate_maintenance_history_score(sample_work_orders):
    score = calculate_maintenance_history_score('D1', sample_work_orders)
    expected_score = (1 * 2 + 1 * 1) * 10  # 1 repair * 2 + 1 cosmetic * 1 * multiplier
    assert score == expected_score

def test_calculate_location_score():
    assert calculate_location_score('ICU') == 100
    assert calculate_location_score('General Ward') == 50

def test_calculate_total_score():
    score = calculate_total_score(
        age_score=50,
        maintenance_cost_score=30,
        risk_score=100,
        maintenance_history_score=40,
        location_score=100,
        utilization_score=80
    )
    expected_score = (
        50 * 0.25 +  # age
        30 * 0.25 +  # maintenance cost
        100 * 0.20 + # risk
        40 * 0.20 +  # maintenance history
        100 * 0.05 + # location
        80 * 0.05    # utilization
    )
    assert score == expected_score

def test_get_device_category():
    assert get_device_category('MRI SCANNER') == ('Imaging', 'MRI')
    assert get_device_category('CT MACHINE') == ('Imaging', 'CT')
    assert get_device_category('PATIENT MONITOR') == ('Monitoring', 'Patient Monitor')
    assert get_device_category('UNKNOWN DEVICE') == ('Other', 'General')

def test_calculate_price(sample_market_price_info):
    device_info = {
        'Asset Description': 'PREMIUM MRI',
        'Location Description': 'ICU',
        'Date Accepted': '2020-01-01'
    }
    price = calculate_price(device_info, sample_market_price_info, 2022)
    assert price > sample_market_price_info['base_price']

def test_calculate_price_invalid_date(sample_market_price_info):
    device_info = {
        'Asset Description': 'PREMIUM MRI',
        'Location Description': 'ICU',
        'Date Accepted': 'invalid_date'
    }
    price = calculate_price(device_info, sample_market_price_info)
    assert price > 0

def test_save_utilization_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = Path(tmpdir) / 'test_cache.csv'
        cache = {'MRI': 12.5, 'CT': 15.0}
        
        with patch('main.CONFIG', {'UTILIZATION_CACHE_FILE': cache_file}):
            save_utilization_cache(cache)
            
            # Verify file contents
            df = pd.read_csv(cache_file)
            assert set(df['DeviceType']) == {'MRI', 'CT'}
            assert df[df['DeviceType'] == 'MRI']['UtilizationRate'].iloc[0] == 12.5

def test_generate_forecast(sample_devices):
    device_scores = [
        {'DeviceID': 'D1', 'DeviceType': 'MRI', 'TotalScore': 90, 'ReplacementCost': 100000},
        {'DeviceID': 'D2', 'DeviceType': 'CT', 'TotalScore': 80, 'ReplacementCost': 50000}
    ]
    replacement_costs = {'MRI': 100000, 'CT': 50000}
    
    forecast = generate_forecast(device_scores, sample_devices, replacement_costs)
    
    assert len(forecast) == 5  # 5-year forecast
    assert isinstance(forecast[0]['Year'], int)
    assert isinstance(forecast[0]['TotalCost'], (int, float))
    assert isinstance(forecast[0]['DevicesToReplace'], list) 