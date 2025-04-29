# Relationship Between IPC 'Fraud' and IT Act 'Cheating by Personation' in Cyber Crimes

## Project Objective

This project analyzes the relationship between two major categories of cyber crimes in India:

1. **IPC Fraud Cases**: Cyber crimes registered under the Indian Penal Code (IPC) as 'Fraud', which includes subcategories like credit/debit card fraud, ATM fraud, online banking fraud, OTP frauds, etc.

2. **IT Act Cheating by Personation Cases**: Crimes registered under the Information Technology Act specifically for "Cheating by personation by using computer resource"

The central research question is: **Do states with high IPC 'Fraud' cases show proportionally higher IT Act 'Cheating by Personation' cases?**

## Datasets Used

The analysis utilizes two primary datasets from the National Crime Records Bureau (NCRB) of India:

1. **cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-ipc.csv**:
   - Contains state-wise and year-wise data for cyber crimes registered under the IPC
   - Includes detailed breakdown of various types of frauds
   - Provides data for years 2020-2022

2. **cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv**:
   - Contains state-wise and year-wise data for cyber crimes registered under the IT Act
   - Includes "Cheating by personation by using computer resource" as a subcategory of Computer Related Offences
   - Covers the same time period (2020-2022)

These datasets were chosen because they directly contain the variables needed for our analysis and provide detailed state-wise and temporal information allowing for comprehensive comparative analysis.

## Analysis Overview

The analysis follows these steps:

1. **Data Preprocessing**:
   - Extracting relevant IPC fraud cases and IT Act cheating by personation cases
   - Merging the datasets by state and year
   - Calculating total cyber crime figures and ratios

2. **Exploratory Data Analysis**:
   - Summary statistics of both types of crimes across states
   - Identification of states with highest incidence of each crime type
   - Visualization of the relationship through scatter plots

3. **Proportional Odds Model Analysis**:
   - Categorizing states into quartiles based on crime incidence
   - Applying a Proportional Odds Model to test the relationship
   - Visualizing the results and calculating correlation coefficients

4. **Standard Visualizations**:
   - Diverging bar charts comparing both crime types for top states
   - Heatmaps showing the ratios between the two crime types by state
   - Additional visualizations of the statistical model results

5. **Advanced Visualization Techniques**:
   - **Trend Analysis Visualizations**: 
     - Stacked area charts showing temporal trends of both crime types
     - Year-over-year growth rate comparisons
     - Time series analysis with LOWESS smoothing for top states
   
   - **Regional Distribution Analysis**:
     - Boxplots showing distribution of crimes across different regions of India
     - Split violin plots for comparing regional distributions between crime types
   
   - **Quadrant Analysis**:
     - Four-quadrant categorization of states based on crime rates
     - Visual mapping of states into high/low fraud and high/low personation categories
   
   - **Cluster Analysis**:
     - K-means clustering to identify groups of states with similar crime patterns
     - Principal Component Analysis (PCA) for dimensionality reduction and visualization
     - 3D visualizations of state clusters based on crime metrics
     - Radar charts profiling each cluster's characteristics
   
   - **Advanced Correlation Visualizations**:
     - Correlation heatmaps across multiple crime metrics
     - Joint distribution plots with regression analysis
     - Pearson and Spearman correlation coefficients with statistical significance testing

## Key Findings

1. There is a positive correlation between states with high IPC Fraud cases and high IT Act Cheating by Personation cases, suggesting that these two types of cyber crimes tend to co-occur geographically.

2. Major urban states/UTs like Karnataka, Telangana, Uttar Pradesh, Maharashtra, and Tamil Nadu consistently show high numbers in both categories, indicating that urbanization and digital penetration may be common risk factors.

3. The proportional odds model indicates that as states move from lower to higher quartiles of IPC Fraud cases, the odds of them being in higher quartiles of IT Act Cheating by Personation cases increase significantly.

4. There are some notable exceptions to the general trend, with certain states showing disproportionately high rates of one crime type compared to the other, suggesting possible differences in reporting, law enforcement focus, or specific criminal activities in those regions.

5. Year-over-year analysis shows increasing trends in both categories of cyber crimes across most states, pointing to the growing challenge of digital fraud in India.

6. Cluster analysis reveals distinct groupings of states with similar cybercrime patterns, allowing for targeted policy interventions based on similar profiles.

7. Regional distribution analysis shows significant variance in cybercrime patterns across different regions of India, with South and North regions generally experiencing higher rates of both crime types.

## Instructions

To run the analysis:

1. Ensure you have Python 3.7+ installed with the following libraries:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - statsmodels
   - sklearn (scikit-learn)
   - scipy

2. Run the script with the following command:
   ```
   python fraud_personation_analysis.py
   ```

3. The script will generate various visualizations in the "Qn 14" folder.

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- scipy

## Output Files

The script generates the following visualizations:

### Basic Visualizations:
1. `fraud_vs_personation_scatter.png`: Scatter plot showing relationship between the two crime types
2. `diverging_bar_chart_2021.png` and `diverging_bar_chart_2022.png`: Diverging bar charts comparing the two crime types for top states
3. `fraud_personation_ratio_heatmap_2021.png` and `fraud_personation_ratio_heatmap_2022.png`: Heatmaps showing the ratio between the two crime types
4. `proportional_odds_analysis.png`: Visualization of the proportional odds model results

### Advanced Visualizations:
5. `trend_stacked_area_chart.png`: Stacked area chart showing trends over time
6. `growth_rate_comparison.png`: Bar chart comparing year-over-year growth rates
7. `top_states_trend_analysis.png`: Multi-panel plot showing trends for top states
8. `regional_distribution_boxplots.png`: Boxplots showing regional distribution of crimes
9. `regional_violin_plots.png`: Split violin plots showing distribution by region and crime type
10. `quadrant_analysis.png`: Scatter plot with quadrant analysis of states
11. `cluster_analysis_pca.png`: PCA-based visualization of state clusters
12. `cluster_analysis_3d.png`: 3D visualization of state clusters
13. `cluster_radar_charts.png`: Radar charts profiling each cluster
14. `correlation_heatmap.png`: Heatmap of correlation between crime metrics
15. `joint_distribution_plot.png`: Joint distribution plot with regression analysis

## Future Improvements

- Incorporate population data to calculate per capita crime rates
- Add geographical visualizations with state boundaries (choropleth maps)
- Extend the analysis to include more years and explore temporal trends
- Investigate the role of other factors like internet penetration, digital literacy, etc.
- Apply more advanced machine learning techniques for prediction and anomaly detection
- Create an interactive dashboard for exploring the relationships dynamically