# Regional Cybercrime Analysis: Fraud-Related Cybercrimes in Western vs Eastern India (2018-2022)

## Project Objective

This project analyzes regional variations in fraud-related cybercrimes between Western and Eastern India from 2018 to 2022 using data from the National Crime Records Bureau (NCRB). The primary goal is to identify patterns, trends, and factors that explain differences in cybercrime incidence between these two major regions of India.

## Research Question

**What factors explain regional variations in fraud-related cybercrimes between Western and Eastern India from 2018 to 2022?**

## Datasets Used

After thorough examination of all available datasets in the "Complete" folder, the following datasets were selected for analysis:

### Primary Datasets:

1. **`cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv`**
   - **Relevance**: Contains fraud data categorized under "Fraud or Illegal Gain" motive
   - **Coverage**: 2002-2022 (focus on 2018-2022 for our analysis)
   - **Structure**: Year, State, Offence Category, Value, Unit, Notes
   - **Completeness**: Comprehensive state-wise coverage with 39 states/UTs

2. **`cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-ipc.csv`**
   - **Relevance**: Contains detailed fraud subcategories under IPC violations
   - **Coverage**: 2002-2022 (with some gaps in 2014-2015)
   - **Structure**: Year, State, Offence Category, Offence Sub-category, Value, Unit, Notes
   - **Key Fraud Types**: Credit/Debit Card, ATMs, Online Banking, OTP Frauds, Others

### Dataset Selection Justification:

1. **Comprehensive Coverage**: Both datasets cover the target timeframe (2018-2022) and include all major states in Western and Eastern India
2. **Fraud-Specific Data**: Both contain specific fraud-related categories essential for answering our research question
3. **Complementary Information**: The motives dataset provides broad fraud categories while the IPC dataset offers detailed fraud subtypes
4. **Data Quality**: Both datasets maintain consistent structure with minimal missing values for the target period
5. **Regional Representation**: Adequate coverage of states classified as Western India (Gujarat, Maharashtra, Rajasthan, Goa) and Eastern India (West Bengal, Odisha, Jharkhand, Bihar, and northeastern states)

## Regional Classification

- **Western India**: Gujarat, Maharashtra, Rajasthan, Goa, Dadra and Nagar Haveli, Daman and Diu
- **Eastern India**: West Bengal, Odisha, Jharkhand, Bihar, Sikkim, Assam, Arunachal Pradesh, Nagaland, Manipur, Mizoram, Tripura, Meghalaya

## Analysis Summary

### Exploratory Data Analysis (EDA) Approach:

1. **Data Preprocessing**: 
   - Combined fraud data from both datasets
   - Filtered for target years (2018-2022)
   - Applied regional classification to all states
   - Handled missing values and data cleaning

2. **Statistical Analysis**:
   - Descriptive statistics by region and year
   - Trend analysis for fraud cases over time
   - Crime type distribution across regions
   - State-level variation analysis within regions

3. **Comparative Analysis**:
   - Regional fraud case totals and growth rates
   - Year-over-year changes in fraud patterns
   - Crime type composition differences between regions

### Key Findings:

- Significant regional disparities in fraud case volumes
- Distinct temporal trends between Western and Eastern India
- Variation in crime type composition across regions
- State-level heterogeneity within each region

## Visualizations Chosen

The following 5 visualizations were selected based on their ability to effectively answer the research question and reveal key insights:

### 1. **Time Series Line Plot** (`fraud_trends_regional_comparison.png`)
- **Purpose**: Shows temporal trends in fraud cases for both regions over 2018-2022
- **Justification**: Essential for identifying growth patterns, seasonal variations, and comparative trends between regions
- **Insights**: Reveals which region has higher fraud rates and how trends have evolved over time

### 2. **Comparative Bar Charts** (`fraud_comparative_analysis.png`)
- **Purpose**: Displays annual fraud cases by region and crime type distribution
- **Justification**: Enables direct numerical comparison between regions and identifies dominant fraud types
- **Insights**: Quantifies the magnitude of differences and highlights specific areas of concern

### 3. **State-wise Intensity Heatmap** (`fraud_intensity_heatmap.png`)
- **Purpose**: Visualizes fraud case intensity across all states by year with regional color coding
- **Justification**: Reveals intra-regional variation and identifies hotspot states within each region
- **Insights**: Shows that regional differences may be driven by specific high-crime states

### 4. **Distribution Box Plots** (`fraud_distribution_analysis.png`)
- **Purpose**: Shows statistical distribution of fraud cases by region, year, and crime type
- **Justification**: Reveals outliers, median differences, and distribution shapes between regions
- **Insights**: Identifies whether differences are consistent or driven by extreme values

### 5. **Stacked Area Charts** (`crime_composition_trends.png`)
- **Purpose**: Displays the composition of different fraud types over time for each region
- **Justification**: Shows how the nature of fraud has evolved differently in each region
- **Insights**: Reveals whether certain fraud types are more prevalent in specific regions

## Modeling Approach

### Baseline Model: Mean Predictor
- **Rationale**: Simple baseline that predicts the overall mean of fraud cases for all observations
- **Purpose**: Establishes a minimum performance threshold for comparison
- **Expected Performance**: Limited predictive power (low R²) but provides reference point

### Advanced Model: Random Forest Regressor
- **Rationale**: Ensemble method capable of capturing non-linear relationships and interactions between features
- **Features Used**:
  - Year (normalized)
  - Region (encoded)
  - Crime type (encoded)
  - State (encoded)
  - State-level statistics (mean, std, max historical values)
- **Advantages**: 
  - Handles mixed data types
  - Provides feature importance rankings
  - Robust to outliers
  - Captures complex regional and temporal patterns

### Evaluation Strategy:
- **Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² Score
- **Validation**: 80-20 train-test split with random state for reproducibility
- **Comparison Visualizations**: 
  - Prediction vs Actual scatter plots
  - Residual analysis
  - Performance metrics comparison
  - Feature importance ranking

### Expected Insights from Modeling:
- Quantification of regional factors' predictive importance
- Identification of key temporal and crime-type patterns
- Understanding of which features best explain fraud variations
- Assessment of model improvement over baseline

## File Structure

```
Qn 12/
├── analysis.py                           # Main analysis script
├── README.md                            # This documentation file
├── fraud_trends_regional_comparison.png  # Time series visualization
├── fraud_comparative_analysis.png        # Comparative bar charts
├── fraud_intensity_heatmap.png          # State-wise heatmap
├── fraud_distribution_analysis.png       # Distribution analysis
├── crime_composition_trends.png          # Crime composition over time
└── model_performance_comparison.png      # Model comparison results
```

## How to Run the Analysis

1. Ensure you have the required Python libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Navigate to the "Qn 12" folder and run:
   ```bash
   python analysis.py
   ```

3. The script will automatically:
   - Load and preprocess the data
   - Perform exploratory data analysis
   - Generate all 5 visualizations
   - Train and compare models
   - Output summary insights to the console

## Technical Requirements

- **Python 3.7+**
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Data**: NCRB cybercrime datasets (included in parent directory)
- **Output**: PNG visualizations saved to current directory

## Expected Runtime

- **Data Processing**: ~30 seconds
- **Visualization Generation**: ~2-3 minutes
- **Model Training**: ~1-2 minutes
- **Total**: ~5 minutes

## Limitations and Assumptions

1. **Regional Classification**: Based on traditional geographical divisions; some border states could be classified differently
2. **Data Completeness**: Analysis assumes reported cases represent actual fraud incidence
3. **Time Period**: Limited to 2018-2022 due to data quality considerations
4. **Causality**: Analysis identifies correlations but does not establish causal relationships

## Future Extensions

1. **Demographic Integration**: Include population and economic indicators
2. **Seasonal Analysis**: Examine monthly or quarterly patterns
3. **Urban vs Rural**: Analyze city-wise data where available
4. **Predictive Modeling**: Develop forecasting models for future trends
5. **Policy Analysis**: Correlate with regional cybersecurity initiatives

---

**Author**: Data Analysis Team  
**Date**: June 2025  
**Contact**: [Your contact information here]
