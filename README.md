# Medical Equipment Replacement Forecast

A system for analyzing medical equipment fleets and generating replacement forecasts based on various factors including age, maintenance history, risk level, and budget constraints.

## Setup

1. Clone the repository
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up your configuration:
   - Copy `config.template.py` to `config.py`
   - Set your OpenAI API key as an environment variable:
     ```bash
     export OPENAI_API_KEY='your-api-key-here'  # On Windows: set OPENAI_API_KEY=your-api-key-here
     ```
   - Adjust other settings in `config.py` as needed

## Usage

1. Place your data files in the `data` directory:
   - `devices.csv`: List of medical devices
   - `work_orders.csv`: Maintenance work orders
   - `replacement_costs.csv`: Replacement costs for devices

2. Run the main script:
   ```bash
   python main.py
   ```

3. View the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Configuration

The `config.py` file contains various settings that can be adjusted:

- Budget settings
- Risk scores
- Location scores
- Maintenance history weights
- Score weightings
- Analysis settings
- File paths
- OpenAI settings

See `config.template.py` for a complete list of configurable options.

## Output

The system generates:
- Replacement forecasts for the next 5 years
- Fleet analysis reports
- Interactive visualizations via the dashboard
- CSV exports of forecast data

## License

MIT License
