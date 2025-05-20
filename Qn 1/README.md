# Cybercrime Trend Analysis in India's Tech Hubs vs Financial Centers (2011-2022)

## Project Objective

This project analyzes how cybercrime trends have evolved in India's major technology hubs (Bengaluru, Hyderabad) compared to financial centers (Mumbai) from 2011 to 2022. We employ time series analysis techniques, particularly ARIMA (Autoregressive Integrated Moving Average) modeling, to understand historical patterns and predict future trends.

## Dataset Used

For this analysis, we selected:
- **cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv**

This dataset was chosen for the following reasons:
1. It provides comprehensive data on cybercrime cases for all three cities of interest (Bengaluru, Hyderabad, and Mumbai)
2. It covers the entire time period from 2011 to 2022
3. It offers absolute values of total cybercrime cases, which is ideal for time series analysis
4. The data is structured consistently, making it easier to perform comparative analysis
5. The dataset contains fewer missing values and is more reliable for the specific cities being studied

## Analysis Summary

Our analysis follows these key steps:

1. **Data Preparation**: Loading and cleaning the dataset, filtering for the three cities of interest over the period 2011-2022
2. **Exploratory Data Analysis**: Creating visualizations to understand trends, patterns, and variations in cybercrime cases
3. **Time Series Analysis**: Using ARIMA modeling to analyze historical patterns and forecast future trends
4. **Comparative Analysis**: Comparing technology hubs with financial centers to identify differences in cybercrime evolution
5. **Period Analysis**: Examining changes in cybercrime patterns across different time periods

### Key Findings

1. **Growth Patterns**: Technology hubs, particularly Bengaluru, have experienced more rapid growth in cybercrime cases compared to Mumbai
2. **Volatility**: Bengaluru shows higher volatility in year-on-year growth rates than both Hyderabad and Mumbai
3. **Period Shift**: All three cities show a significant increase in cybercrime cases in the 2017-2022 period compared to 2011-2016
4. **Tech vs Financial**: On average, tech hubs consistently report more cybercrime cases than the financial center (Mumbai)
5. **Future Projections**: ARIMA forecasts suggest continued growth in cybercrime cases in all three cities

## Visualizations Chosen

1. **Line Chart (Yearly Trends)**: Shows the yearly progression of cybercrime cases for each city, helping identify overall trends and pattern changes. This visualization directly addresses the question of how cybercrime has evolved over time.

2. **Box Plots (Period Comparison)**: Compares the distribution and variability of cybercrime cases between early period (2011-2016) and recent period (2017-2022). This helps understand how the magnitude and spread of cybercrime cases have changed over time.

3. **ARIMA Time Series Analysis & Forecast**: Provides time series modeling of cybercrime trends for each city with forecasts for the next three years. This visualization combines statistical modeling with visual representation to understand underlying patterns.

4. **Year-on-Year Growth Rate Chart**: Visualizes the percentage change in cybercrime cases each year, highlighting periods of acceleration or deceleration in cybercrime growth.

5. **Tech Hubs vs Financial Center Comparison**: Directly compares the average of tech hubs (Bengaluru and Hyderabad) with the financial center (Mumbai) to highlight the differences in cybercrime evolution between these types of urban centers.

## Execution Instructions

To run the analysis:

1. Ensure Python 3.6+ is installed on your system
2. Install the required dependencies (see below)
3. Run the script:
   ```
   python cybercrime_analysis.py
   ```
4. The visualizations will be saved to the "Qn 1" folder

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- pmdarima

Install all dependencies using:
```
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn pmdarima
```

## Future Work

- Include demographic data to normalize cybercrime cases by population
- Analyze the types of cybercrimes common in tech hubs vs financial centers
- Incorporate socioeconomic factors to identify potential correlations with cybercrime trends
