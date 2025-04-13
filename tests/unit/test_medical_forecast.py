import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import os
import shutil
from datetime import datetime
from medical_forecast import (
    calculate_device_scores,
    calculate_fleet_scores,
    get_utilization_rate,
    save_utilization_cache,
    load_lifecycle_cache
)
from main import generate_fleet_forecast

class TestMedicalForecast(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before all tests."""
        # Sample test data
        cls.sample_devices = pd.DataFrame({
            'DeviceType': ['CT', 'MRI'],
            'PurchaseDate': ['2015-01-01', '2016-01-01'],
            'Score': [80, 90],
            'FleetSize': [2, 1]
        })
        
        cls.sample_utilization_cache = {
            'CT': {'utilization_rate': 0.75},
            'MRI': {'utilization_rate': 0.85}
        }
        
        # Create test cache files
        os.makedirs('tests/data', exist_ok=True)
        with open('tests/data/test_utilization_cache.json', 'w') as f:
            json.dump(cls.sample_utilization_cache, f)
            
        with open('tests/data/test_lifecycle_cache.json', 'w') as f:
            json.dump({}, f)

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.devices = self.sample_devices.copy()

    @patch('medical_forecast.get_utilization_rate')
    def test_calculate_device_scores(self, mock_get_utilization):
        """Test calculate_device_scores function."""
        mock_get_utilization.return_value = 0.75
        
        result = calculate_device_scores(self.devices)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('Score' in result.columns)
        
    @patch('medical_forecast.get_utilization_rate')
    def test_calculate_fleet_scores(self, mock_get_utilization):
        """Test calculate_fleet_scores function."""
        mock_get_utilization.return_value = 0.75
        
        result = calculate_fleet_scores(self.devices)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('FleetScore' in result.columns)
        
    @patch('medical_forecast.get_utilization_rate')
    @patch('medical_forecast.save_utilization_cache')
    @patch('medical_forecast.load_lifecycle_cache')
    def test_generate_fleet_forecast(self, mock_lifecycle, mock_save_util, mock_get_util):
        """Test generate_fleet_forecast function."""
        mock_get_util.return_value = 0.75
        mock_lifecycle.return_value = {'CT': 10, 'MRI': 12}
        
        result = generate_fleet_forecast(self.devices)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue('ReplacementYear' in result.columns)
        
    def test_get_utilization_rate(self):
        """Test get_utilization_rate function."""
        device_type = 'CT'
        result = get_utilization_rate(device_type)
        self.assertIsInstance(result, float)
        self.assertTrue(0 <= result <= 1)
        
    def test_save_utilization_cache(self):
        """Test save_utilization_cache function."""
        cache_data = {'CT': {'utilization_rate': 0.75}}
        save_utilization_cache(cache_data)
        
        with open('tests/data/test_utilization_cache.json', 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data, cache_data)
        
    def test_load_lifecycle_cache(self):
        """Test load_lifecycle_cache function."""
        result = load_lifecycle_cache()
        self.assertIsInstance(result, dict)

    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists('tests/data/test_utilization_cache.json'):
            os.remove('tests/data/test_utilization_cache.json')
        if os.path.exists('tests/data/test_lifecycle_cache.json'):
            os.remove('tests/data/test_lifecycle_cache.json')
            
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        if os.path.exists('tests/data'):
            shutil.rmtree('tests/data')

if __name__ == '__main__':
    unittest.main() 