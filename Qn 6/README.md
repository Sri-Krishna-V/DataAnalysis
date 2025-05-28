# Cybercrime Pattern Analysis: Bengaluru, Hyderabad, and Pune (2022)

## Project Objective

This project aims to analyze and compare cybercrime patterns across three major Indian tech hubs: Bengaluru, Hyderabad, and Pune for the year 2022. Using K-means clustering and various data visualization techniques, we investigate how these cities differ in terms of cybercrime types, frequencies, and overall patterns.

## Dataset Used

For this analysis, we selected the following dataset:

**"cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv"**

This dataset was chosen because:
1. **Relevance**: It contains detailed cybercrime data for all three cities of interest.
2. **Granularity**: It breaks down cybercrimes into categories and subcategories, allowing for in-depth analysis.
3. **Timeframe**: It includes data for 2022, which matches our target year.
4. **Completeness**: The dataset has minimal missing values and a consistent structure.
5. **Quality**: It provides absolute numbers of cases, allowing for quantitative comparison.

## Analysis Summary

Our analysis pipeline included:

1. **Data Loading and Preprocessing**:
   - Filtering data for Bengaluru, Hyderabad, and Pune in 2022
   - Handling missing values in subcategory fields
   - Creating a comprehensive category system that combines main and subcategories

2. **Exploratory Data Analysis**:
   - Comparing total cybercrime cases across cities
   - Identifying most prevalent cybercrime categories in each city
   - Analyzing the distribution of different crime types

3. **Clustering Analysis**:
   - Applying K-means clustering to identify similarities between cities
   - Using Principal Component Analysis (PCA) for dimensionality reduction
   - Evaluating cluster quality with silhouette scores and Calinski-Harabasz index

4. **Model Comparison**:
   - Using a baseline model (mean predictor) for comparison
   - Evaluating K-means based prediction approach
   - Comparing model performance using RMSE

## Visualizations Chosen

1. **Bar Chart of Top Crimes by City**
   - **Purpose**: Directly compares the most prevalent cybercrime categories across the three cities
   - **Insight**: Shows which specific crime types are disproportionately high in each city
   - **Value**: Provides an immediate visual indicator of the key crime challenges each city faces

2. **Crime Intensity Heatmap**
   - **Purpose**: Displays the intensity of different crime types across cities using color gradients
   - **Insight**: Shows patterns and outliers through color intensity
   - **Value**: Provides a comprehensive overview of all crime categories in a compact visual format

3. **K-means Clustering Visualization**
   - **Purpose**: Shows how cities group together based on their cybercrime patterns
   - **Insight**: Reveals which cities have similar cybercrime profiles
   - **Value**: Helps identify underlying patterns that might not be apparent in direct comparisons

4. **Radar Chart of Cluster Feature Importance**
   - **Purpose**: Displays the key features (crime types) that define each cluster
   - **Insight**: Shows what crime characteristics make cities similar or different
   - **Value**: Helps understand the defining features of cybercrime patterns in each city group

5. **Crime Distribution Comparison**
   - **Purpose**: Shows both absolute and percentage distribution of crime categories by city
   - **Insight**: Reveals how the composition of cybercrimes differs across cities
   - **Value**: Allows for proportional comparison regardless of the total number of cases

6. **Model Comparison Visualization** (additional)
   - **Purpose**: Compares the performance of baseline and K-means models
   - **Insight**: Shows how much better clustering-based prediction works compared to simple averages
   - **Value**: Quantifies the benefit of using clustering for understanding cybercrime patterns

## Modeling Approach

### Baseline Model
- **Type**: DummyRegressor with 'mean' strategy
- **Purpose**: Provides a simple benchmark by predicting the mean value of total cybercrime cases
- **Evaluation**: Uses Root Mean Square Error (RMSE) to measure prediction accuracy

### Main Model: K-means Clustering
- **Approach**: Groups cities based on their cybercrime profiles
- **Implementation**:
  - Standardized data to ensure fair comparison across crime categories
  - Used PCA for dimensionality reduction
  - Determined optimal number of clusters using silhouette score
  - For prediction, used cluster means as the predicted value for cities in each cluster
- **Evaluation**: Compared RMSE with the baseline model to measure improvement

### Evaluation Strategy
- The comparison assesses how well the clustering approach captures the patterns in the data compared to the simple baseline
- Lower RMSE indicates better performance
- We calculated the percentage improvement to quantify how much better the K-means approach is

## Key Findings

The analysis reveals distinct cybercrime patterns across the three cities:

1. **Bengaluru** shows significantly higher volumes of cybercrime, particularly in identity theft and cheating by personation categories, which might be related to its status as a major IT hub.

2. **Hyderabad** has a more balanced distribution across crime categories with notable numbers in ransomware and identity theft crimes.

3. **Pune** shows lower overall cybercrime numbers but has proportionally higher cases in specific categories like privacy violations and identity theft.

The clustering analysis successfully grouped cities with similar cybercrime profiles, providing valuable insights into the underlying patterns that might inform targeted cybersecurity strategies for each city.
