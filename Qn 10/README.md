# Geographical Distribution of Fraud-Related Cybercrimes in India (2019-2022)

This project analyzes and visualizes the geographic distribution of fraud-related cybercrimes across Indian states from 2019 to 2022 using data from the National Crime Records Bureau (NCRB).

## Project Objective

The primary objective is to answer the following question:

**What are the geographic distributions of fraud-related cybercrimes across Indian states from 2018 to 2022?**

The analysis aims to identify patterns, trends, and hotspots of fraud-related cybercrime activities across different states in India, while also examining year-over-year changes and growth rates.

## Datasets Used

After reviewing all available datasets in the collection, this analysis uses:

**Cybercrime by Motives Dataset** (`cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv`)

This dataset was selected because it:
- Contains detailed breakdown of cybercrimes by different motives across all Indian states
- Includes a specific category for "Fraud or Illegal Gain" which directly addresses the analysis question
- Provides comprehensive state-wise data for the years 2019-2022
- Offers consistent measurement and categorization across all states and years

While the dataset doesn't include 2018 data (only covering 2019-2022), it provides the most comprehensive and directly relevant information for analyzing fraud-related cybercrime distribution across Indian states.

## Analysis Summary

The analysis follows these key steps:

1. **Data Preprocessing**:
   - Filter for "Fraud or Illegal Gain" cybercrimes
   - Remove aggregate "All India" rows
   - Clean and standardize data formats

2. **Exploratory Data Analysis**:
   - Calculate yearly summary statistics
   - Identify top states with highest fraud cybercrime incidence
   - Calculate year-over-year growth rates
   - Identify states with highest growth rates

3. **Visualization and Analysis**:
   - Create geographic distribution visualizations (bar chart showing state-wise distribution)
   - Analyze time trends for top states
   - Compare growth patterns across states
   - Examine per capita rates using population estimates
   - Compare 2021-2022 changes through bubble plot analysis
   - Generate forecasts for future trends

## Key Findings

1. **Geographic Distribution**: 
   - Southern and western states generally show higher incidences of fraud-related cybercrimes
   - Karnataka, Telangana, Maharashtra, and Uttar Pradesh consistently rank among the highest in raw numbers

2. **Temporal Trends**:
   - Overall increasing trend from 2019-2022 across most states
   - Some states show dramatic year-over-year growth, particularly tech hubs

3. **Per Capita Analysis**:
   - Tech-hub states show higher per capita fraud cybercrime rates
   - Smaller states with tech centers show disproportionately high per capita rates

4. **Growth Patterns**:
   - Growth is not uniform across states, with some showing much faster increase than others
   - Specific growth hotspots identified in regions with emerging digital economies

## Visualizations Chosen

The analysis includes five core visualizations:

1. **State-wise Distribution Bar Chart**: Shows the geographic distribution of fraud cybercrimes across all states for the most recent year (2022). This visualization was chosen to provide a clear comparison of absolute fraud cybercrime numbers across different states, highlighting which regions have the highest incidence.

2. **Time Series Plot of Top 5 States**: Tracks fraud cybercrime trends in the five most affected states from 2019-2022. This visualization was selected to reveal how fraud cybercrime patterns have evolved over time in the most impacted regions.

3. **Year-over-Year Growth Heatmap**: Visualizes growth rates across top 15 states. This heatmap was chosen to identify which states are experiencing the fastest increase in fraud cybercrimes, highlighting emerging hotspots.

4. **Per Capita Fraud Cybercrime Chart**: Adjusts for population differences to show which states have the highest rate of fraud cybercrimes per 100,000 residents. This visualization was selected because raw numbers can be misleading when states have vastly different populations.

5. **2021 vs 2022 Bubble Plot Comparison**: Compares fraud cybercrime rates between 2021 and 2022, with bubbles sized by 2022 values and colored by growth rate. This visualization was chosen to provide a multidimensional view of recent changes, showing both absolute values and relative growth in a single chart.

Additionally, an ARIMA forecast visualization was included as a bonus to project potential future trends for the top 5 states.

## Execution Instructions

To run the analysis:

1. Ensure you have Python 3.6+ installed
2. Install the required dependencies (see requirements section)
3. Run the script:
   ```
   python fraud_cybercrime_analysis.py
   ```
4. Generated visualizations will be saved in the output directory

## Dependencies

The script requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels (optional, for ARIMA forecasting)

These can be installed via pip:
```
pip install pandas numpy matplotlib seaborn statsmodels
```

## Output Files

The script generates the following visualization files:
- `fraud_choropleth_2022.png`: State-wise distribution of fraud cybercrimes in 2022
- `fraud_trends_top_5_states.png`: Time series trends for top 5 states
- `fraud_growth_heatmap.png`: Heatmap of year-over-year growth rates
- `fraud_per_capita_2022.png`: Per capita fraud cybercrime rates by state
- `fraud_bubble_plot_2021_vs_2022.png`: Comparison of 2021 vs 2022 fraud cybercrimes
- `arima_forecast_top_5_states.png`: Forecast of future trends for top states