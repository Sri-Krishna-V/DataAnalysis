# Cybercrime Prevalence Analysis in Metropolitan Areas of India (2012-2022)

## Project Objective

This project analyzes the trend of cybercrime prevalence across metropolitan areas in India from 2012 to 2022. The analysis employs time series techniques, specifically ARIMA (Autoregressive Integrated Moving Average) modeling, to understand historical trends and forecast future cybercrime patterns in major Indian cities.

## Dataset Selection

For this analysis, we selected the following dataset:
- **"Year, State, and City-wise Total Number of Cyber Crimes Committed in India"** from the National Crime Records Bureau (NCRB)

This dataset was chosen because:
1. It contains comprehensive data about cybercrime incidents across major metropolitan areas in India
2. It covers our entire period of interest (2012-2022)
3. It provides consistent recording of cybercrime cases across metropolitan cities
4. The data is from an authoritative source (NCRB), ensuring reliability and accuracy
5. It includes absolute values of cybercrime incidents, allowing for direct comparison between cities and across years

## Analysis Approach

The analysis follows these key steps:

1. **Data Preparation**: Filtering data for the years 2012-2022 and cleaning any inconsistencies
2. **Identification of Top Metropolitan Areas**: Determining the cities with highest cybercrime prevalence
3. **Trend Analysis**: Examining year-by-year trends in cybercrime cases for top cities
4. **Growth Analysis**: Calculating year-on-year growth rates and compound annual growth rates (CAGR)
5. **Time Series Analysis**: Using ARIMA models to forecast future cybercrime trends for major cities
6. **Visualization**: Creating informative visualizations including line charts for trends and heatmaps for crime intensity

## Key Findings

1. There has been a significant overall increase in cybercrime cases across metropolitan areas in India from 2012 to 2022
2. Certain cities (e.g., Bengaluru, Mumbai, Hyderabad) have experienced particularly high growth in cybercrime incidents
3. The ARIMA forecasts suggest continuing growth in cybercrime cases for major metropolitan areas
4. The distribution of cybercrime cases is heavily skewed, with a few metropolitan areas accounting for a large proportion of all cases
5. Year-on-year growth rates show periods of acceleration and deceleration in cybercrime reporting

## Running the Analysis

To run the analysis script:

1. Ensure you have Python 3.6+ installed with the following libraries:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - statsmodels

2. Navigate to the project directory and run:
   ```
   python cybercrime_metropolitan_analysis.py
   ```

3. The script will generate multiple visualizations saved in the `visualizations` directory:
   - Yearly cybercrime trends for top metropolitan cities
   - Cybercrime intensity heatmap
   - Growth rate analysis
   - ARIMA forecasts
   - Distribution of cybercrime cases in the most recent year
   - CAGR analysis

## Conclusion

The analysis reveals important patterns in cybercrime prevalence across Indian metropolitan areas. The significant growth in cybercrime cases, particularly in major tech hubs like Bengaluru and Hyderabad, highlights the need for enhanced cybersecurity measures in these regions. The forecasting component provides valuable insights for law enforcement agencies to allocate resources effectively in anticipation of future cybercrime trends.