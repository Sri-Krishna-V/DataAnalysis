#!/usr/bin/env python
# coding: utf-8

"""
Analysis of Fraud-Related Cybercrimes Across Indian States (2021-2022)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import warnings
from matplotlib.ticker import FuncFormatter
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

# File path
file_path = r'c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv'

# Load the data
print("Loading data...")
df = pd.read_csv(file_path)
print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")

# Display dataset structure
print("\nDataset overview:")
print(df.head())

# Clean and prepare the data
print("\nPreparing data for analysis...")

# Filter for Fraud or Illegal Gain crimes
fraud_df = df[df['offence_category'] == 'Fraud or Illegal Gain'].copy()

# Check the years available in the dataset
print("\nYears available in the dataset:", sorted(fraud_df['year'].unique()))

# Filter for 2021 and 2022
fraud_df = fraud_df[fraud_df['year'].isin([2021, 2022])].copy()
print(f"Filtered data for 2021-2022: {fraud_df.shape[0]} rows")

# Convert value column to numeric
fraud_df['value'] = pd.to_numeric(fraud_df['value'], errors='coerce')

# Remove 'All India' to focus on states only
fraud_df = fraud_df[fraud_df['state'] != 'All India'].copy()

# Create a pivot table for easier analysis (state vs. year)
fraud_pivot = fraud_df.pivot_table(index='state', columns='year', values='value', aggfunc='sum')
fraud_pivot = fraud_pivot.fillna(0)
print("\nPivot table of fraud crimes by state and year:")
print(fraud_pivot.head())

# Calculate the change in fraud cases from 2021 to 2022
fraud_pivot['change'] = fraud_pivot[2022] - fraud_pivot[2021]
fraud_pivot['percent_change'] = (fraud_pivot['change'] / fraud_pivot[2021] * 100).fillna(0)

# Sort by absolute number of fraud cases in 2022 (descending)
fraud_pivot = fraud_pivot.sort_values(by=2022, ascending=False)

print("\nData preparation complete.")
print("\nBeginning analysis...")

# Create output directory if it doesn't exist
output_dir = r'c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\Qn 11'
os.makedirs(output_dir, exist_ok=True)

# Analysis 1: Top 10 states with highest fraud cases in 2022
print("\nAnalysis 1: Top 10 states with highest fraud cases in 2022")
top_10_states = fraud_pivot.head(10).copy()

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=top_10_states.index, y=top_10_states[2022], palette='viridis')
plt.title('Top 10 States with Highest Fraud-Related Cybercrimes in 2022', fontsize=16)
plt.xlabel('State', fontsize=14)
plt.ylabel('Number of Fraud Cases', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'bottom', fontsize=12)

# Save figure
plt.savefig(os.path.join(output_dir, 'top_10_states_fraud_2022.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analysis 2: States with highest percentage increase in fraud cases
print("\nAnalysis 2: States with highest percentage increase in fraud cases")

# Filter for states with at least 10 fraud cases in 2021 to avoid misleading percentage changes
significant_states = fraud_pivot[fraud_pivot[2021] >= 10].copy()
significant_states = significant_states.sort_values(by='percent_change', ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=significant_states.head(10).index, y=significant_states.head(10)['percent_change'], palette='plasma')
plt.title('Top 10 States with Highest % Increase in Fraud Cybercrimes (2021-2022)', fontsize=16)
plt.xlabel('State', fontsize=14)
plt.ylabel('Percentage Increase (%)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Add value labels on bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'bottom', fontsize=12)

# Save figure
plt.savefig(os.path.join(output_dir, 'top_10_states_percent_increase.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analysis 3: Line graph comparing 2021 vs 2022 fraud cases across top states
print("\nAnalysis 3: Line graph comparing 2021 vs 2022 fraud cases across top states")

# Get top 15 states by 2022 fraud cases
top_15_states = fraud_pivot.head(15).copy()

# Reshape the data for line plotting
top_15_melted = pd.melt(
    top_15_states.reset_index(), 
    id_vars='state', 
    value_vars=[2021, 2022],
    var_name='year', 
    value_name='fraud_cases'
)

plt.figure(figsize=(14, 10))
sns.lineplot(
    data=top_15_melted, 
    x='year', 
    y='fraud_cases', 
    hue='state', 
    marker='o', 
    linewidth=2.5,
    markersize=10
)

plt.title('Fraud-Related Cybercrimes Trends in Top 15 States (2021-2022)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Fraud Cases', fontsize=14)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(output_dir, 'fraud_trend_top_15_states.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analysis 4: ARIMA Time Series Forecasting
print("\nAnalysis 4: ARIMA Time Series Forecasting for top states")

# Function to forecast using ARIMA
def arima_forecast(data, state_name):
    """
    Performs ARIMA forecasting for a given state's fraud data
    """
    # Convert to a time series 
    ts_data = pd.Series(data.values, index=pd.date_range(start='2021', periods=len(data), freq='Y'))
    
    # ARIMA forecasting (using simple parameters since we have limited data points)
    # For real-world use with more data points, parameter tuning would be recommended
    try:
        model = ARIMA(ts_data, order=(1,0,0))  # Simple AR(1) model due to limited data points
        model_fit = model.fit()
        
        # Forecast for 2023 and 2024
        forecast = model_fit.forecast(steps=2)
        
        # Create forecast table
        forecast_data = pd.DataFrame({
            'Year': [2021, 2022, 2023, 2024],
            'Value': list(ts_data.values) + list(forecast)
        })
        
        return forecast_data
    except:
        print(f"Could not perform ARIMA forecast for {state_name} due to insufficient data")
        return None

# Select top 5 states for ARIMA forecasting
top_5_states = fraud_pivot.head(5).copy()
plt.figure(figsize=(14, 10))

# Store forecasts for combined plot
all_forecasts = {}

for state in top_5_states.index:
    state_data = fraud_pivot.loc[state, [2021, 2022]]
    forecast_data = arima_forecast(state_data, state)
    
    if forecast_data is not None:
        all_forecasts[state] = forecast_data
        
        # Plot line
        plt.plot(forecast_data['Year'], forecast_data['Value'], marker='o', linewidth=2, markersize=8, label=state)
        
        # Add annotations for forecasted values
        for i in range(2, 4):
            plt.annotate(f"{forecast_data['Value'][i]:.0f}", 
                        (forecast_data['Year'][i], forecast_data['Value'][i]),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

# Set plot attributes
plt.axvline(x=2022, color='gray', linestyle='--', alpha=0.7, label='Forecast begins')
plt.title('ARIMA Forecast of Fraud-Related Cybercrimes for Top 5 States (2021-2024)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Fraud Cases', fontsize=14)
plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(output_dir, 'arima_forecast_top_5_states.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analysis 5: Heatmap showing percentage change across all states
print("\nAnalysis 5: Heatmap showing percentage change across all states")

# Prepare data for heatmap
heatmap_data = fraud_pivot[['percent_change']].copy()
heatmap_data = heatmap_data.sort_values(by='percent_change', ascending=False)

plt.figure(figsize=(14, 12))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='coolwarm', linewidths=.5, 
                cbar_kws={'label': 'Percentage Change (%)'})

plt.title('Percentage Change in Fraud Cybercrimes by State (2021-2022)', fontsize=16)
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(output_dir, 'fraud_percentage_change_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# Analysis 6: Choropleth Map of India (if geopandas is available)
try:
    import geopandas as gpd
    
    print("\nAnalysis 6: Choropleth Map of India showing fraud cases in 2022")
    
    # Try to load India shapefile (this is a placeholder - you'll need to provide actual shapefile)
    india_shapefile_path = os.path.join(output_dir, 'india_states.shp')
    
    if os.path.exists(india_shapefile_path):
        india_map = gpd.read_file(india_shapefile_path)
        
        # Merge with our fraud data
        merged_data = india_map.merge(fraud_pivot.reset_index(), left_on='STATE_NAME', right_on='state', how='left')
        
        # Plot choropleth
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        merged_data.plot(column=2022, 
                        cmap='YlOrRd', 
                        linewidth=0.8, 
                        ax=ax, 
                        edgecolor='0.8',
                        legend=True,
                        legend_kwds={'label': "Fraud Cases in 2022"})
        
        ax.set_title('Fraud-Related Cybercrimes Across Indian States in 2022', fontsize=16)
        ax.axis('off')
        
        # Save figure
        plt.savefig(os.path.join(output_dir, 'fraud_choropleth_map.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("India shapefile not found. Skipping choropleth map.")
except ImportError:
    print("Geopandas not available. Skipping choropleth map.")

print("\nAnalysis complete. All visualizations saved to the Qn 11 folder.")