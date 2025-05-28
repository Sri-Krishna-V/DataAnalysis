# Kolkata Cybercrime Volatility Analysis

## Project Objective
This project investigates why Kolkata experiences significant volatility in cybercrime trends compared to other major metropolitan cities in India. The analysis aims to quantify and visualize this volatility, identify potential patterns or causes, and implement predictive models to better understand the observed fluctuations.

## Datasets Used

The primary dataset used for this analysis is:
- **"cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv"**

This dataset was chosen because:
1. It contains **year-wise data spanning from 2014 to 2022**, providing sufficient temporal coverage to analyze volatility trends
2. It includes **data for all major metropolitan cities** in India, enabling direct comparison between Kolkata and other metros
3. The data is consistently reported in **absolute numbers** across all years and cities, ensuring reliable comparisons
4. It contains **complete records without significant missing values** for the metros being analyzed

As a supplementary dataset, we also utilized:
- **"cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv"**

This secondary dataset provides breakdowns of cybercrime types, which helps analyze whether Kolkata's volatility is associated with specific categories of cybercrimes.

## Analysis Summary

The analysis follows these key steps:

1. **Data Preprocessing**:
   - Filtering data to focus on major metro cities (Kolkata, Delhi, Mumbai, Chennai, Bengaluru, Hyderabad)
   - Converting data types and ensuring consistency
   - Handling any missing values or anomalies

2. **Volatility Quantification**:
   - Calculating year-over-year percentage changes
   - Computing standard deviation of changes as a volatility metric
   - Determining coefficient of variation to normalize volatility across cities with different baseline crime rates
   - Calculating range between maximum and minimum changes

3. **Exploratory Data Analysis**:
   - Analyzing time series trends across cities
   - Identifying years with abnormal fluctuations
   - Comparing Kolkata's patterns with other major metros
   - Investigating potential correlations with cybercrime types

4. **Key Findings**:
   - Kolkata shows the highest coefficient of variation among major metros, indicating greater relative volatility
   - The city experiences more dramatic swings between increases and decreases in cybercrime rates
   - Years 2016-2018 and 2020-2021 show particularly significant fluctuations for Kolkata
   - The composition of cybercrime types in Kolkata differs from other metros, potentially explaining some of the volatility

## Visualizations Chosen

1. **Time Series Plot** (`cybercrime_trend_analysis.png`):
   - **Purpose**: Visualizes the raw cybercrime numbers across years for all major metros
   - **Justification**: Shows the actual magnitude of fluctuations and allows for direct visual comparison of volatility patterns
   - **Insights**: Clearly illustrates how Kolkata's trend line shows more abrupt changes compared to smoother progressions in other cities

2. **Year-over-Year Change Heatmap** (`yoy_change_heatmap.png`):
   - **Purpose**: Displays percentage changes between consecutive years for each city
   - **Justification**: Color coding helps instantly identify extreme fluctuations and patterns across cities and years
   - **Insights**: Kolkata's cells show more extreme color variations, quantifying the observed volatility

3. **Volatility Comparison Bar Chart** (`volatility_comparison.png`):
   - **Purpose**: Directly compares volatility metrics (coefficient of variation and range) across cities
   - **Justification**: Provides a clear ranking of cities by volatility, removing the time dimension to focus on overall patterns
   - **Insights**: Confirms if Kolkata objectively demonstrates higher volatility metrics than other metros

4. **Volatility Distribution Box Plot** (`volatility_distribution.png`):
   - **Purpose**: Shows the statistical distribution of year-over-year changes for each city
   - **Justification**: Box plots reveal the median, quartiles, and outliers in the volatility data
   - **Insights**: Visualizes whether Kolkata has wider interquartile range or more outliers than other metros

5. **Crime Type Composition** (`crime_type_comparison.png`):
   - **Purpose**: Displays the breakdown of cybercrime types for each city
   - **Justification**: Helps identify if Kolkata's volatility correlates with a different composition of crime types
   - **Insights**: May reveal if Kolkata has higher proportions of more volatile crime categories

## Modeling Approach

### Baseline Model
A **Linear Regression** model was chosen as the baseline because:
- It assumes a simple linear trend in cybercrime numbers
- It represents the most straightforward forecasting approach
- It serves as a good benchmark for more complex models
- It's commonly used in time series forecasting as a naive baseline

### Main Model
An **ARIMA (AutoRegressive Integrated Moving Average)** model was selected as the main model because:
- It's specifically designed for time series data with potential volatility
- It can account for trends, seasonality, and autocorrelation
- It's well-suited for data with irregular patterns and fluctuations
- The model automatically selects optimal parameters (p, d, q) for the Kolkata data

### Evaluation Strategy
Models are compared using:
- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors
- **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values
- **Visual comparison of forecasts**: The model comparison plot shows actual data alongside predictions from both models

This dual-model approach allows us to determine whether Kolkata's cybercrime patterns require a more sophisticated model (ARIMA) that can account for the observed volatility, or if a simple linear model is sufficient.
