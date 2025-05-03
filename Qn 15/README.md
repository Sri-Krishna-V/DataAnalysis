# Financial Fraud Cybercrime Analysis in India (2020-2022)

## Project Objective

This project analyzes the evolution of financial fraud-related cybercrimes across Indian states from 2020 to 2022. Financial fraud constitutes a significant portion (approximately 64.8%) of total cybercrime cases in India, making it a crucial area for analysis and policy intervention. The analysis uses time series forecasting (ARIMA) to understand trends and predict future patterns.

## Datasets Used

The analysis primarily uses the dataset:
- **"cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv"**

This dataset was selected because it provides:
- Comprehensive state-wise data covering all Indian states and union territories
- Year-by-year data for the exact period of interest (2020-2022)
- Specific categorization of cybercrimes by motive, including "Fraud or Illegal Gain" which directly corresponds to financial fraud
- Official data from the National Crime Record Bureau (NCRB), ensuring reliability
- Consistent data structure that facilitates time series analysis

## Analysis Summary

The analysis follows a structured approach to understand the evolution of financial fraud-related cybercrimes:

1. **Data Preparation and Cleaning**:
   - Focused on the "Fraud or Illegal Gain" category as the primary indicator of financial fraud
   - Filtered data for the years 2020-2022
   - Handled missing values and ensured consistency across state names

2. **Exploratory Data Analysis**:
   - Examined the percentage of fraud-related cases in total cybercrimes
   - Identified top states with the highest financial fraud cases
   - Analyzed year-to-year growth rates in fraud cases

3. **Trend Analysis**:
   - Tracked the national-level evolution of financial fraud cases
   - Compared state-wise growth patterns
   - Identified states with significant increases or decreases

4. **Time Series Forecasting**:
   - Applied ARIMA (AutoRegressive Integrated Moving Average) models to forecast fraud cases for top states
   - Projected trends for 2023-2024 based on historical patterns
   - Provided confidence intervals for forecasts

5. **Key Findings**:
   - Financial fraud cybercrimes showed a 32.1% increase from 2020 to 2022 at the national level
   - States like Telangana, Karnataka, and Maharashtra consistently ranked among the top states for fraud cases
   - Several states showed significant year-over-year growth in fraud cases, particularly from 2021 to 2022
   - Forecasts suggest continued growth in fraud cases for most high-incidence states

## Visualizations Chosen

The analysis includes five carefully selected visualizations that best illustrate the evolution of financial fraud-related cybercrimes:

1. **Time Series Trend Plot** (`financial_fraud_trend.png`): 
   - Shows the overall trend in financial fraud cybercrimes for all of India from 2020-2022
   - Provides clear view of year-to-year changes and growth percentages
   - Helps establish the national context before diving into state-specific analysis

2. **Comparative Bar Chart** (`top_states_comparison.png`):
   - Displays the top 10 states with highest fraud cases, comparing their values across 2020-2022
   - Allows direct visual comparison of how states rank and how their cases have changed over time
   - Highlights which states have the largest absolute number of cases

3. **Stacked Area Plot** (`state_fraud_distribution.png`):
   - Visualizes how the total fraud cases are distributed among the top 5 states over time
   - Shows the relative contribution of each state to the total and how these proportions change
   - Provides insight into which states are driving the overall increase in fraud cases

4. **Choropleth Map** (`fraud_choropleth_2022.png`):
   - Geographic visualization of fraud cases across all states for 2022
   - Allows for immediate identification of hotspots and regional patterns
   - Provides spatial context that other visualizations cannot

5. **Bubble Plot Comparison** (`fraud_bubble_plot_2021_vs_2022.png`):
   - Compares fraud cases between 2021 and 2022 for top states
   - Bubble size represents the magnitude of change, while position shows increase/decrease
   - Color indicates growth rate (positive/negative)
   - Effectively shows both absolute numbers and growth rates simultaneously

6. **ARIMA Forecasting Plot** (`arima_forecast_top_5_states.png`):
   - Projects future trends for the top 5 states through 2024 using time series forecasting
   - Includes uncertainty ranges to reflect confidence in predictions
   - Provides actionable insights for policy planning and resource allocation

These visualizations were chosen because they provide complementary perspectives on the data, from national trends to state comparisons to geographic distribution to future projections, creating a comprehensive picture of financial fraud cybercrimes in India.

## Execution Instructions

To run the analysis script:

1. Ensure you have Python 3.7+ installed with the required dependencies (see below)
2. Navigate to the directory containing the script
3. Run the script using:
   ```
   python financial_fraud_analysis.py
   ```
4. The script will generate all visualizations and save them in the current directory
5. The analysis results and progress will be displayed in the console

## Dependencies

The following Python libraries are required:

```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.12.0
plotly>=5.3.0
geopandas>=0.10.0
```

You can install these dependencies using:

```
pip install pandas numpy matplotlib seaborn statsmodels plotly geopandas
```

Note: Some visualizations use Plotly for interactive features. If Plotly fails to render the choropleth map, the script will automatically fall back to a static visualization using Matplotlib.