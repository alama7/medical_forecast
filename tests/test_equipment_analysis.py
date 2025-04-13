import pytest
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import tempfile
import json
import os

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
from data_processing import (
    PURCHASE_DATE_COLUMN,
    get_expected_lifecycle,
    load_or_create_lifecycle_cache,
    save_lifecycle_cache
)

# Mock the get_expected_lifecycle function for testing
@pytest.fixture
def mock_get_expected_lifecycle():
    with patch('data_processing.get_expected_lifecycle') as mock:
        mock.side_effect = lambda device_type, lifecycle_cache=None: 10 if device_type == 'MRI' else 8
        yield mock

@pytest.fixture
def sample_devices():
    return pd.DataFrame({
        'DeviceID': ['D1', 'D2'],
        'DeviceType': ['MRI', 'CT'],
        PURCHASE_DATE_COLUMN: [
            datetime.now() - timedelta(days=365*2),
            datetime.now() - timedelta(days=365*5)
        ],
        'RiskClass': ['CRITICAL', 'HIGH'],
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

def test_calculate_device_scores(sample_devices, sample_work_orders, mock_get_expected_lifecycle):
    from main import calculate_device_scores
    
    # Mock replacement costs and utilization cache
    replacement_costs = {'MRI': 1000000, 'CT': 500000}
    utilization_cache = {'MRI': 12, 'CT': 16}
    
    scores = calculate_device_scores(sample_devices, sample_work_orders, 
                                   replacement_costs, utilization_cache)
    
    assert len(scores) == 2
    assert 'DeviceID' in scores[0]
    assert 'TotalScore' in scores[0]
    
    # Verify get_expected_lifecycle was called
    assert mock_get_expected_lifecycle.call_count == 2 

def test_load_or_create_lifecycle_cache():
    from data_processing import load_or_create_lifecycle_cache
    
    # Create a temporary cache file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
        temp_path = temp.name
        
        # Write test data
        pd.DataFrame({
            'DeviceType': ['MRI Scanner', 'CT Scanner'],
            'ExpectedLifecycle': [10, 12]
        }).to_csv(temp_path, index=False)
        
        # Mock CONFIG
        with patch('data_processing.CONFIG', {'LIFECYCLE_CACHE_FILE': temp_path}):
            cache = load_or_create_lifecycle_cache()
            
            assert len(cache) == 2
            assert cache['MRI Scanner'] == 10
            assert cache['CT Scanner'] == 12
    
    # Clean up
    os.unlink(temp_path)

def test_get_expected_lifecycle_from_cache():
    # Test getting lifecycle from cache
    lifecycle_cache = {'MRI Scanner': 10, 'CT Scanner': 12}
    
    result = get_expected_lifecycle('MRI Scanner', lifecycle_cache)
    assert result == 10

def test_get_expected_lifecycle_from_defaults():
    # Test getting lifecycle from defaults
    with patch('data_processing.save_lifecycle_cache') as mock_save:
        lifecycle_cache = {}
        
        # This should use the DEFAULT_LIFECYCLE dictionary
        result = get_expected_lifecycle('MRI Scanner', lifecycle_cache)
        
        assert result == 10  # Value from DEFAULT_LIFECYCLE
        assert lifecycle_cache['MRI Scanner'] == 10
        mock_save.assert_called_once()

@patch('openai.OpenAI')
def test_get_expected_lifecycle_from_api(mock_openai):
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "9.5"
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch('data_processing.save_lifecycle_cache'):
        lifecycle_cache = {}
        
        # Use a device type not in DEFAULT_LIFECYCLE
        result = get_expected_lifecycle('New Device Type', lifecycle_cache)
        
        assert result == 9.5
        assert lifecycle_cache['New Device Type'] == 9.5
        mock_client.chat.completions.create.assert_called_once()

def test_save_lifecycle_cache():
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp:
        cache_file = temp.name
    
    try:
        # Create test data
        cache = {'MRI Scanner': 10, 'CT Scanner': 12}
        
        # Mock CONFIG
        with patch('data_processing.CONFIG', {'LIFECYCLE_CACHE_FILE': cache_file}):
            save_lifecycle_cache(cache)
            
            # Verify file was created with correct data
            df = pd.read_csv(cache_file)
            assert len(df) == 2
            assert set(df['DeviceType']) == {'MRI Scanner', 'CT Scanner'}
            assert set(df['ExpectedLifecycle']) == {10, 12}
    finally:
        # Clean up
        os.unlink(cache_file) 