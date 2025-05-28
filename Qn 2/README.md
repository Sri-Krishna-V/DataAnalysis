# Cybercrime Prevalence Analysis Across Metropolitan Areas in India (2012-2022)

## Project Objective

This project analyzes the trend of cybercrime prevalence across metropolitan areas in India from 2012 to 2022. The analysis employs time series modeling techniques, specifically ARIMA (AutoRegressive Integrated Moving Average), to identify patterns, trends, and make predictions about cybercrime rates in major Indian cities.

## Dataset Used

For this analysis, we selected the following dataset:

**"cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv"**

This dataset was chosen for the following reasons:

1. **Relevance**: The dataset directly addresses our research question as it contains year-wise, state-wise, and city-wise data on the number of cybercrimes committed in India.

2. **Completeness**: It includes data from 2003 to 2022, covering our target period of 2012-2022.

3. **Granularity**: The data is broken down by metropolitan areas, allowing us to analyze trends specifically in urban centers.

4. **Data Quality**: The dataset is sourced from the National Crime Records Bureau (NCRB) of India, which is the official organization responsible for collecting and analyzing crime data in the country, ensuring reliability and authenticity.

5. **Quantitative Measures**: The absolute number of cybercrimes provides a clear metric to track changes over time and compare between cities.

## Analysis Summary

The analysis follows these key steps:

1. **Data Preprocessing**:
   - Filtering data for the years 2012-2022
   - Cleaning missing values
   - Focusing on metropolitan cities
   - Removing aggregate entries

2. **Exploratory Data Analysis (EDA)**:
   - Analyzing basic statistics of cybercrime numbers
   - Identifying cities with the highest cybercrime rates
   - Examining year-over-year changes in cybercrime prevalence
   - Comparing trends across different metropolitan areas

3. **Time Series Analysis**:
   - Implementing ARIMA modeling to understand temporal patterns
   - Comparing ARIMA forecasts with baseline predictions
   - Evaluating model performance using mean squared error

4. **Growth Rate Analysis**:
   - Calculating year-over-year growth rates for top metropolitan cities
   - Identifying cities with accelerating or decelerating cybercrime rates

## Visualizations Chosen

The following five visualization types were selected for their effectiveness in addressing different aspects of the research question:

1. **Time Series Plot (total_cybercrime_trend.png)**:
   - Shows the overall trend of cybercrimes across all metropolitan cities from 2012 to 2022
   - Helps identify whether cybercrime is increasing, decreasing, or remaining stable over time
   - Reveals any significant year-to-year variations or anomalies

2. **Bar Chart (top_cities_2022.png)**:
   - Compares the top 10 cities with the highest cybercrime cases in 2022
   - Provides a clear visual ranking of metropolitan areas by cybercrime prevalence
   - Helps identify which cities are most affected by cybercrimes in the most recent data year

3. **Heatmap (cybercrime_heatmap.png)**:
   - Displays the year-by-year cybercrime data for the top 5 cities in a matrix format
   - Enables easy tracking of individual city trends while allowing for cross-city comparisons
   - The color intensity provides an intuitive understanding of cybercrime severity

4. **Growth Rate Line Chart (cybercrime_growth_rates.png)**:
   - Visualizes the year-over-year percentage change in cybercrime for top cities
   - Reveals which cities are experiencing accelerating or decelerating rates of cybercrime
   - Helps identify cities with similar cybercrime growth patterns

5. **ARIMA vs. Baseline Forecasting Plot (arima_vs_baseline_forecasting.png)**:
   - Compares the performance of the ARIMA time series model against a naive baseline approach
   - Demonstrates the predictive capability of the models using the last two years as test data
   - Provides quantitative metrics (MSE) to evaluate model performance
   - Illustrates the value of sophisticated time series analysis over simple forecasting methods

These visualizations collectively provide a comprehensive view of cybercrime trends across metropolitan India, addressing temporal patterns, geographic variations, growth rates, and predictive modeling.
