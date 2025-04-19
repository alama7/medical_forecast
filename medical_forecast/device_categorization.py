import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import openai
import logging
import os
from config import CONFIG

logger = logging.getLogger(__name__)

class DeviceCategorizer:
    def __init__(self, categories_file: str = "config/device_categories.yaml", strict_mode: bool = True):
        """Initialize the device categorizer with the master categories file."""
        self.categories_file = Path(categories_file)
        self.categories_data = self._load_categories()
        self.client = openai.OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.strict_mode = strict_mode
        self.uncategorized_devices = []  # Track devices that couldn't be categorized

    def _load_categories(self) -> Dict:
        """Load the categories from the YAML file."""
        try:
            with open(self.categories_file, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading categories file: {e}")
            raise

    def _save_categories(self):
        """Save the updated categories to the YAML file."""
        try:
            with open(self.categories_file, 'w') as f:
                yaml.dump(self.categories_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Updated categories saved to {self.categories_file}")
        except Exception as e:
            logger.error(f"Error saving categories file: {e}")

    def get_manufacturer_info(self, manufacturer: str, model: str) -> Optional[Dict]:
        """Get manufacturer and model information if available."""
        manufacturer = manufacturer.upper()
        model = model.upper()
        
        if manufacturer in self.categories_data['manufacturers']:
            mfg_data = self.categories_data['manufacturers'][manufacturer]
            
            # Check if the model is known
            for known_model, categories in mfg_data['known_models'].items():
                if known_model.upper() in model:
                    return {
                        'manufacturer': manufacturer,
                        'model': known_model,
                        'categories': categories,
                        'specialties': mfg_data['specialties']
                    }
            
            # If model not found, return just manufacturer specialties
            return {
                'manufacturer': manufacturer,
                'specialties': mfg_data['specialties']
            }
        return None

    def categorize_device(self, 
                         device_description: str,
                         manufacturer: Optional[str] = None,
                         model: Optional[str] = None) -> Tuple[str, str]:
        """
        Categorize a device using AI and the master categories.
        
        Args:
            device_description: Description of the device
            manufacturer: Optional manufacturer name
            model: Optional model number/name
            
        Returns:
            Tuple of (category, subcategory)
        """
        # First try to get manufacturer/model info
        mfg_info = None
        if manufacturer and model:
            mfg_info = self.get_manufacturer_info(manufacturer, model)

        # Prepare the prompt for the AI
        prompt = f"""
        Categorize this medical device into the most appropriate category and subcategory from the following list:

        Categories and Subcategories:
        {yaml.dump(self.categories_data['categories'], default_flow_style=False)}

        Device Description: {device_description}
        """

        if mfg_info:
            prompt += f"""
            Manufacturer Information:
            - Manufacturer: {mfg_info['manufacturer']}
            - Known Specialties: {', '.join(mfg_info['specialties'])}
            """
            if 'categories' in mfg_info:
                prompt += f"- Known Categories for this Model: {', '.join(mfg_info['categories'])}\n"

        if self.strict_mode:
            prompt += """
            IMPORTANT: You must ONLY use categories and subcategories from the list above.
            Do not create new categories or subcategories.
            If you cannot find a good match, use the most general category that applies.
            """
        else:
            prompt += """
            If you cannot find a good match in the existing categories, suggest a new category and subcategory.
            """

        prompt += """
        Respond with ONLY the category and subcategory in this exact format:
        Category: [category]
        Subcategory: [subcategory]
        """

        try:
            response = self.client.chat.completions.create(
                model=CONFIG['OPENAI_MODEL'],
                messages=[
                    {"role": "system", "content": "You are a medical equipment expert who understands device categorization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            category = None
            subcategory = None
            
            for line in response_text.split('\n'):
                if line.startswith('Category:'):
                    category = line.replace('Category:', '').strip()
                elif line.startswith('Subcategory:'):
                    subcategory = line.replace('Subcategory:', '').strip()
            
            # Validate the response against our master categories
            if category and subcategory:
                if (category in self.categories_data['categories'] and 
                    subcategory in self.categories_data['categories'][category]):
                    return category, subcategory
                
                # If in strict mode and category doesn't exist, use fallback
                if self.strict_mode:
                    # Track this device for reporting
                    self.uncategorized_devices.append({
                        'description': device_description,
                        'manufacturer': manufacturer,
                        'model': model,
                        'suggested_category': category,
                        'suggested_subcategory': subcategory
                    })
                    return self._fallback_keyword_matching(device_description)
                
                # If not in strict mode, add the new category
                if category not in self.categories_data['categories']:
                    self.categories_data['categories'][category] = []
                if subcategory not in self.categories_data['categories'][category]:
                    self.categories_data['categories'][category].append(subcategory)
                    # Save the updated categories
                    self._save_categories()
                return category, subcategory
            
            # If AI response doesn't match our categories, fall back to keyword matching
            return self._fallback_keyword_matching(device_description)
            
        except Exception as e:
            logger.error(f"Error in AI categorization: {e}")
            return self._fallback_keyword_matching(device_description)

    def _fallback_keyword_matching(self, device_description: str) -> Tuple[str, str]:
        """Fallback method using keyword matching if AI categorization fails."""
        device_description = device_description.upper()
        
        # Check each category's keywords
        for category, keywords in self.categories_data['category_keywords'].items():
            if any(keyword.upper() in device_description for keyword in keywords):
                # Find the most specific subcategory that matches
                for subcategory in self.categories_data['categories'][category]:
                    if subcategory.upper() in device_description:
                        return category, subcategory
                # If no specific subcategory found, return the first one
                return category, self.categories_data['categories'][category][0]
        
        # If no match found, return Other/General
        return "Other", "General"

    def validate_category(self, category: str, subcategory: str) -> bool:
        """Validate if a category/subcategory pair exists in our master list."""
        return (category in self.categories_data['categories'] and 
                subcategory in self.categories_data['categories'][category])

    def get_all_categories(self) -> Dict:
        """Get all available categories and subcategories."""
        return self.categories_data['categories']

    def get_category_keywords(self) -> Dict[str, List[str]]:
        """Get all category keywords for matching."""
        return self.categories_data['category_keywords']
        
    def get_uncategorized_devices_report(self) -> List[Dict]:
        """Get a report of devices that couldn't be categorized in strict mode."""
        return self.uncategorized_devices 