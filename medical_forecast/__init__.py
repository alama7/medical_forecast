"""
Medical Forecast Package

A package for analyzing and categorizing medical devices, managing maintenance schedules,
and forecasting equipment needs.
"""

__version__ = "1.0.0"

from .device_categorization import DeviceCategorizer
from .category_manager import (
    view_categories,
    add_category,
    remove_category,
    process_device_file
)

__all__ = [
    'DeviceCategorizer',
    'view_categories',
    'add_category',
    'remove_category',
    'process_device_file'
] 