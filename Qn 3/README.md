# COVID-19 Impact on Cybercrime Reporting in Major Indian Cities

## Project Objective

This project aims to analyze the extent to which the COVID-19 pandemic (2020-2021) impacted the reporting of cybercrimes in major Indian cities. Using classification techniques (primarily Random Forest) and visualization methods, we examine patterns and trends in cybercrime reporting before, during, and after the pandemic.

## Datasets Used

For this analysis, we primarily used two datasets from the National Crime Records Bureau (NCRB) of India:

1. **Primary Dataset**: `cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv`
   - Contains total cybercrime figures for major Indian cities from 2003 to 2022
   - City-level granularity allows for detailed geographical analysis
   - Annual data allows for clear temporal comparison of pre-pandemic, pandemic, and post-pandemic periods

2. **Supporting Dataset**: `cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv`
   - Provides breakdown of specific cybercrime types at the city level
   - Allows for analysis of which crime categories were most affected by the pandemic

These datasets were chosen because they:
- Offer appropriate temporal coverage (2018-2022) for our analysis period
- Provide city-level granularity, allowing for more nuanced analysis than state-level data
- Include categorized crime data, enabling analysis of which crime types increased during the pandemic
- Come from an authoritative source (NCRB), ensuring data reliability

## Analysis Summary

Our analysis examines cybercrime patterns across three distinct periods:
- **Pre-COVID** (2018-2019)
- **COVID** (2020-2021)
- **Post-COVID** (2022)

The analytical approach included:

1. **Data Preprocessing**:
   - Filtering relevant years (2018-2022)
   - Adding period classifications (Pre-COVID, COVID, Post-COVID)
   - Calculating population-adjusted crime rates
   - Merging datasets for comprehensive analysis

2. **Exploratory Data Analysis**:
   - Comparing total cybercrime figures across different periods
   - Analyzing city-specific trends before, during, and after the pandemic
   - Examining growth rates of different cybercrime categories

3. **Machine Learning Analysis**:
   - Training a Random Forest classifier to identify factors distinguishing COVID-period cybercrimes
   - Using a Decision Tree as a baseline model for comparison
   - Extracting feature importance to understand key influences on cybercrime patterns

## Key Findings

1. Most major Indian cities experienced significant changes in cybercrime reporting during the COVID-19 pandemic
2. Tech hubs like Bengaluru and Hyderabad showed different patterns compared to financial centers like Mumbai
3. Certain categories of cybercrimes increased substantially during the pandemic, particularly:
   - Identity theft
   - Online fraud
   - Computer-related offenses

## Visualizations Chosen

1. **Total Crimes by Period Bar Chart**
   - Shows aggregate cybercrime counts across Pre-COVID, COVID, and Post-COVID periods
   - Provides a clear macro view of the pandemic's overall impact on cybercrime reporting
   - Reveals the direction and magnitude of change across periods

2. **City Comparison Bar Chart**
   - Compares cybercrime cases across top 10 cities for all three periods
   - Includes growth rate calculations to quantify pandemic impact
   - Highlights geographical variations in pandemic effects on cybercrime

3. **Crime Types Heatmap**
   - Visualizes how different categories of cybercrimes changed during the pandemic
   - Uses color intensity to show volume and text annotations to show percentage changes
   - Identifies which crime types were most affected by pandemic conditions

4. **Random Forest Feature Importance Chart**
   - Ranks factors that best predict whether a crime occurred during the COVID period
   - Provides insight into which variables were most strongly associated with pandemic-era cybercrimes
   - Supports data-driven understanding of pandemic cybercrime dynamics

5. **Decision Tree Visualization**
   - Illustrates the classification logic used to distinguish COVID-period crimes
   - Provides an interpretable view of how different factors interact
   - Highlights key decision points and thresholds in the classification process

These visualization types were selected because they effectively communicate different aspects of the pandemic's impact on cybercrime patterns, from macro-level trends to detailed breakdowns of crime categories and predictive factors.

## Machine Learning Approach

Our analysis employs a carefully designed machine learning methodology to avoid overfitting while achieving the required 75%+ accuracy:

1. **Feature Engineering and Selection**:
   - Created informative features related to cybercrime, digital literacy, and internet penetration
   - Used correlation analysis to eliminate highly correlated features that could cause overfitting
   - Applied an aggressive feature selection approach to keep only the most predictive features
   - Implemented proper feature scaling for consistent model performance

2. **Temporal Data Splitting**:
   - Used time-based train/test splitting instead of random splitting to ensure realistic evaluation
   - Earlier years (training data) are used to predict later years (test data)
   - This approach better simulates real-world prediction scenarios

3. **Robust Model Evaluation**:
   - Implemented 5-fold cross-validation to get reliable performance estimates
   - Measured the gap between training and testing performance to detect overfitting
   - Added learning curve visualization to diagnose bias-variance tradeoff
   - Evaluated models on multiple metrics: accuracy, precision, recall, and F1 score

4. **Model Regularization**:
   - Applied appropriate regularization to all models to prevent overfitting
   - Used shallower decision trees with pruning to avoid capturing noise
   - Implemented L1/L2 regularization for linear and tree-based models
   - Reduced model complexity across the board (fewer estimators, smaller networks, etc.)

5. **Model Selection Strategy**:
   - Selected the final model based on the smallest gap between cross-validation and test accuracy
   - This approach emphasizes models that generalize well rather than just having high test accuracy
   - Prioritized interpretable models with clear feature importance for better insights
