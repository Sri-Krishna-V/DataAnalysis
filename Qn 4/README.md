# Cybercrime Rate Changes in Indian Cities During COVID-19 (2019-2021)

## Project Objective

This project analyzes which Indian cities showed the most significant changes in cybercrime rates due to the COVID-19 pandemic (2019-2021) using Support Vector Machine (SVM) classification techniques. The analysis aims to identify patterns in cybercrime rate changes and classify cities based on the magnitude and direction of these changes during the pandemic period.

## Datasets Used

For this analysis, the following dataset was selected:

- **cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv**

### Justification for Dataset Selection

This dataset was chosen for the following reasons:

1. **City-level Data**: The dataset provides granular city-wise cybercrime statistics, which is essential for comparing changes across different urban centers in India.

2. **Temporal Coverage**: The dataset covers the years 2019-2021, perfectly capturing the pre-pandemic, initial outbreak, and ongoing pandemic periods necessary for a comprehensive COVID-19 impact analysis.

3. **Consistent Metrics**: The dataset uses consistent measurement units ("value in absolute number") across all cities and years, making comparisons statistically valid.

4. **Data Completeness**: The dataset has minimal missing values for major cities, ensuring robust analysis without significant imputation requirements.

5. **Direct Relevance**: The dataset directly reports cybercrime incidents by city, allowing for straightforward measurement of change rates without complex data transformations.

## Analysis Summary

The analysis follows a structured approach to understand cybercrime patterns during the COVID-19 pandemic:

1. **Data Preprocessing**: The dataset was filtered to the relevant years (2019-2021) and cleaned by removing aggregated rows and handling missing values.

2. **Exploratory Data Analysis (EDA)**: Initial exploration revealed significant variations in cybercrime rates across cities during the pandemic, with some cities experiencing substantial increases while others showed decreases.

3. **Change Pattern Classification**: Cities were classified based on their cybercrime rate change patterns:
   - Continuous Increase: Cities that saw rising rates throughout the pandemic
   - Continuous Decrease: Cities with falling rates throughout
   - Increase then Decrease: Cities that initially saw a rise followed by a decline
   - Decrease then Increase: Cities that initially saw a drop followed by a rise

4. **Significance Classification**: Cities were further classified as having either "Significant" (>50% change) or "Moderate" changes in cybercrime rates over the pandemic period.

5. **Predictive Modeling**: SVM classification was used to predict whether a city experienced significant changes based on various features, with a Logistic Regression model serving as a baseline for comparison.

## Key Findings

- Several major cities, including Bengaluru, Hyderabad, and Delhi, experienced substantial changes in cybercrime rates during the pandemic.
- The most common pattern observed was "Increase then Decrease," suggesting an initial surge in cybercrimes during early lockdowns followed by a reduction as adaptations occurred.
- The SVM model demonstrated superior performance in classifying cities by significance of change compared to the baseline Logistic Regression model.
- The percentage change between 2019-2021 and the absolute cybercrime values in 2020 emerged as the most important features for classification.

## Visualizations Chosen

The analysis employs five distinct visualization types, each selected for specific analytical purposes:

1. **Bar Chart of Percent Changes**: Shows the magnitude and direction of change in cybercrime rates for top cities between 2019-2021, providing a clear comparison of which cities were most affected during the pandemic.

2. **Line Plot of Yearly Trends**: Illustrates the progression of cybercrime rates across the three-year period for major cities, revealing temporal patterns and allowing for the identification of potential COVID-19 impact points.

3. **Heatmap of Cybercrime Rates**: Presents a comprehensive view of cybercrime rates by city and year, using color intensity to highlight hotspots and facilitate multi-city comparisons within a single visualization.

4. **Change Pattern Classification Visualizations**: Combines pie and scatter plots to categorize cities by their change patterns, offering both aggregate distribution information and individual city positioning in the change space.

5. **Model Comparison Visualization**: Compares the performance of SVM and Logistic Regression models through accuracy metrics and confusion matrices, while also highlighting feature importance to understand key predictors of significant cybercrime changes.

Each visualization type was selected to address a specific aspect of the analysis question, collectively providing a comprehensive understanding of how COVID-19 affected cybercrime rates across Indian cities.