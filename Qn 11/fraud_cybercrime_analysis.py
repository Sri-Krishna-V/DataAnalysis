#!/usr/bin/env python
# coding: utf-8

"""
Enhanced Analysis of Fraud-Related Cybercrimes Across Indian States (2021-2022)
===============================================================================

This script provides a comprehensive analysis of fraud-related cybercrimes in India
with integrated baseline models for comparative evaluation. It includes:

1. Multiple baseline model types for contextual analysis
2. Enhanced visualizations with baseline integration
3. Modular code structure with dispatcher pattern
4. Comprehensive reporting of analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from scipy import stats
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set styling for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("tab10")

class CybercrimeBaselines:
    """
    Comprehensive baseline models for cybercrime analysis
    """
    
    @staticmethod
    def national_average_baseline(df, metric_col, year=None):
        """Calculate national average as baseline"""
        if year:
            year_data = df[df['year'] == year]
            return year_data[metric_col].mean()
        return df[metric_col].mean()
    
    @staticmethod
    def zero_growth_baseline():
        """Zero growth baseline for percentage change analysis"""
        return 0.0
    
    @staticmethod
    def population_weighted_baseline(df, value_col, pop_col):
        """Population-weighted baseline (when population data available)"""
        total_cases = df[value_col].sum()
        total_population = df[pop_col].sum()
        return (total_cases / total_population) * 100000  # Per lakh population
    
    @staticmethod
    def naive_forecast_baseline(series, horizon):
        """Naive forecasting baseline - last value repeated"""
        last_value = series.iloc[-1] if len(series) > 0 else 0
        return [last_value] * horizon
    
    @staticmethod
    def simple_moving_average_baseline(series, window=2, horizon=2):
        """Simple moving average baseline"""
        if len(series) < window:
            return CybercrimeBaselines.naive_forecast_baseline(series, horizon)
        ma_value = series.tail(window).mean()
        return [ma_value] * horizon
    
    @staticmethod
    def regional_average_baseline(df, state_col, value_col, regional_mapping):
        """Calculate regional average baselines"""
        regional_avg = {}
        for region, states_list in regional_mapping.items():
            region_data = df[df[state_col].isin(states_list)]
            regional_avg[region] = region_data[value_col].mean()
        return regional_avg
    
    @staticmethod
    def percentile_baseline(df, metric_col, percentile=50):
        """Percentile-based baseline (median by default)"""
        return np.percentile(df[metric_col], percentile)
    
    @staticmethod
    def linear_trend_baseline(series, horizon):
        """Linear trend extrapolation baseline"""
        if len(series) < 2:
            return CybercrimeBaselines.naive_forecast_baseline(series, horizon)
        
        x = np.arange(len(series))
        slope, intercept, _, _, _ = stats.linregress(x, series)
        
        future_x = np.arange(len(series), len(series) + horizon)
        return [slope * i + intercept for i in future_x]

# File path - Use a path that works across different environments
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 
                        'cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv')
# Fallback to the original path if the relative path doesn't exist
if not os.path.exists(file_path):
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
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Enhanced Visualization Functions
def create_enhanced_percentage_increase_chart(pivot_df, output_dir='.'):
    """Enhanced percentage increase chart with multiple baselines"""
    # Filter states with significant base cases to avoid misleading percentages
    significant_states = pivot_df[pivot_df[2021] >= 10].copy()
    significant_states = significant_states.sort_values(by='percent_change', ascending=False)
    top_10 = significant_states.head(10)
    
    # Calculate baselines
    zero_growth = CybercrimeBaselines.zero_growth_baseline()
    national_avg_growth = CybercrimeBaselines.national_average_baseline(
        significant_states.reset_index(), 'percent_change'
    )
    median_growth = CybercrimeBaselines.percentile_baseline(
        significant_states.reset_index(), 'percent_change', 50
    )
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create bar plot with baseline-based color coding
    bars = ax.bar(range(len(top_10)), top_10['percent_change'], 
                  alpha=0.8, edgecolor='navy', linewidth=1.2)
    
    # Color coding based on performance vs baselines
    for i, (bar, value) in enumerate(zip(bars, top_10['percent_change'])):
        if value > national_avg_growth:
            bar.set_color('darkred')
            bar.set_alpha(0.9)
        elif value > median_growth:
            bar.set_color('orange')
            bar.set_alpha(0.8)
        else:
            bar.set_color('lightblue')
            bar.set_alpha(0.7)
            
        # Add value labels
        ax.annotate(f'{value:.1f}%', 
                   (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add baseline lines with clear legends
    ax.axhline(zero_growth, color='red', linestyle='--', linewidth=2.5, 
               label=f'Zero Growth Baseline: {zero_growth:.1f}%', alpha=0.8)
    ax.axhline(national_avg_growth, color='orange', linestyle='-.', linewidth=2.5,
               label=f'National Average: {national_avg_growth:.1f}%', alpha=0.8)
    ax.axhline(median_growth, color='green', linestyle=':', linewidth=2.5,
               label=f'National Median: {median_growth:.1f}%', alpha=0.8)
    
    # Enhanced styling
    ax.set_xlabel('States', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage Increase (%)', fontsize=14, fontweight='bold')
    ax.set_title('Top 10 States: Fraud Cybercrime Growth vs Multiple Baselines (2021-2022)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(range(len(top_10)))
    ax.set_xticklabels(top_10.index, rotation=45, ha='right', fontsize=12)
    
    # Enhanced legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_percentage_increase_with_baselines.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return analysis summary
    above_national = len(top_10[top_10['percent_change'] > national_avg_growth])
    return {
        'chart_type': 'percentage_increase',
        'baseline_types': ['zero_growth', 'national_average', 'national_median'],
        'states_above_national': above_national,
        'total_states': len(top_10),
        'performance_ratio': above_national / len(top_10)
    }

def create_enhanced_arima_forecast(pivot_df, output_dir='.'):
    """Enhanced ARIMA forecast with multiple baseline models"""
    top_5_states = pivot_df.head(5).copy()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Colors for different models
    state_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    baseline_colors = ['gray', 'brown', 'purple']
    
    years = [2021, 2022, 2023, 2024]
    forecast_years = [2023, 2024]
    
    all_forecasts = {}
    
    for i, state in enumerate(top_5_states.index[:5]):
        state_data = top_5_states.loc[state, [2021, 2022]]
        
        # ARIMA Forecast
        try:
            ts_data = pd.Series(state_data.values, index=pd.date_range(start='2021', periods=2, freq='Y'))
            model = ARIMA(ts_data, order=(1,0,0))
            model_fit = model.fit()
            arima_forecast = model_fit.forecast(steps=2)
            arima_values = list(state_data.values) + list(arima_forecast)
        except:
            slope = state_data.iloc[1] - state_data.iloc[0]
            arima_forecast = [state_data.iloc[1] + slope, state_data.iloc[1] + 2*slope]
            arima_values = list(state_data.values) + arima_forecast
        
        # Baseline Models
        naive_forecast = CybercrimeBaselines.naive_forecast_baseline(state_data, 2)
        ma_forecast = CybercrimeBaselines.simple_moving_average_baseline(state_data, 2, 2)
        trend_forecast = CybercrimeBaselines.linear_trend_baseline(state_data, 2)
        
        # Store all forecasts
        all_forecasts[state] = {
            'arima': arima_values,
            'naive': list(state_data.values) + naive_forecast,
            'ma': list(state_data.values) + ma_forecast,
            'trend': list(state_data.values) + trend_forecast
        }
        
        # Plot ARIMA forecast (main line)
        ax.plot(years, arima_values, marker='o', linewidth=3, markersize=8, 
                color=state_colors[i], label=f'{state} (ARIMA)', alpha=0.9)
        
        # Plot baselines for first state only (to avoid cluttering)
        if i == 0:
            ax.plot(years, all_forecasts[state]['naive'], marker='s', linestyle='--', 
                   linewidth=2, color=baseline_colors[0], alpha=0.7,
                   label=f'{state} Naive Baseline')
            ax.plot(years, all_forecasts[state]['ma'], marker='^', linestyle='-.', 
                   linewidth=2, color=baseline_colors[1], alpha=0.7,
                   label=f'{state} Moving Avg Baseline')
            ax.plot(years, all_forecasts[state]['trend'], marker='d', linestyle=':', 
                   linewidth=2, color=baseline_colors[2], alpha=0.7,
                   label=f'{state} Linear Trend Baseline')
    
    # Add vertical line to separate historical and forecast data
    ax.axvline(x=2022.5, color='black', linestyle=':', linewidth=2, alpha=0.5, 
               label='Forecast Boundary')
    
    # Enhanced styling
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Fraud Cases', fontsize=14, fontweight='bold')
    ax.set_title('Enhanced ARIMA Forecast vs Baseline Models\nTop 5 States Fraud Cybercrime (2021-2024)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Format y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'enhanced_arima_forecast_with_baselines.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'chart_type': 'arima_forecast',
        'baseline_types': ['naive', 'moving_average', 'linear_trend'],
        'states_forecasted': len(top_5_states),
        'forecast_horizon': 2,
        'forecast_data': all_forecasts
    }

def create_enhanced_visualization_with_baselines(data, chart_type, output_dir='.', **kwargs):
    """
    Master function for creating enhanced visualizations with appropriate baselines

    Parameters:
    -----------
    data : pandas.DataFrame
        Input cybercrime data pivot table
    chart_type : str
        Type of chart ('percentage', 'forecast')
    output_dir : str
        Directory to save output visualizations
    **kwargs : additional parameters for specific chart types

    Returns:
    --------
    dict
        Analysis results summary
    """
    if chart_type == 'percentage':
        return create_enhanced_percentage_increase_chart(data, output_dir, **kwargs)
    elif chart_type == 'forecast':
        return create_enhanced_arima_forecast(data, output_dir, **kwargs)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")

def main_enhanced_analysis(file_path, output_dir=None):
    """
    Main function to run complete enhanced analysis with baselines
    
    Parameters:
    -----------
    file_path : str
        Path to the input CSV file
    output_dir : str, optional
        Directory to save output visualizations, defaults to script location if None
    
    Returns:
    --------
    dict
        Collection of analysis results
    """
    # Use default output directory if none provided
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nEnhanced Cybercrime Analysis with Integrated Baseline Models")
    print("=" * 70)
    
    # Load and prepare data
    print("Loading and preparing data...")
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
    print("\nBeginning enhanced analysis...")
    
    # Run analyses
    results = {}
    
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
    
    # Analysis 2: States with highest percentage increase in fraud cases with baseline integration
    print("\nAnalysis 2: States with highest percentage increase in fraud cases (with baselines)")
    results['percentage'] = create_enhanced_visualization_with_baselines(fraud_pivot, 'percentage', output_dir)
    
    # Print summary of baseline analysis
    print(f"  - States above national average growth: {results['percentage']['states_above_national']} of {results['percentage']['total_states']}")
    print(f"  - Performance ratio: {results['percentage']['performance_ratio']:.2f}")
    print(f"  - Baseline types used: {', '.join(results['percentage']['baseline_types'])}")
    print("  - Enhanced visualization saved as 'enhanced_percentage_increase_with_baselines.png'")
    
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
    
    # Analysis 4: Enhanced ARIMA Time Series Forecasting with multiple baselines
    print("\nAnalysis 4: Enhanced ARIMA Time Series Forecasting with multiple baselines")
    results['forecast'] = create_enhanced_visualization_with_baselines(fraud_pivot, 'forecast', output_dir)
    
    # Print summary of forecast analysis
    print(f"  - Number of states forecasted: {results['forecast']['states_forecasted']}")
    print(f"  - Forecast horizon: {results['forecast']['forecast_horizon']} years")
    print(f"  - Baseline types used: {', '.join(results['forecast']['baseline_types'])}")
    print("  - Enhanced visualization saved as 'enhanced_arima_forecast_with_baselines.png'")
    
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
    
    print("\nAnalysis complete. All visualizations saved to the output folder.")
    print("\nSUMMARY OF BASELINE INTEGRATIONS:")
    print("-" * 50)
    for analysis_type, result in results.items():
        print(f"{analysis_type.upper()}:")
        print(f"  Baseline types: {', '.join(result['baseline_types'])}")
        if 'performance_ratio' in result:
            print(f"  Performance ratio: {result['performance_ratio']:.2f}")
    
    return results

# If this script is run directly, execute the main enhanced analysis
if __name__ == "__main__":
    # Define a more appropriate file path for different environments
    try:
        # Try to use the file_path defined above
        if not os.path.exists(file_path):
            print(f"Warning: File not found at {file_path}")
            # Try to find the file in the current directory or its parent
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alternative_paths = [
                os.path.join(current_dir, 'cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv'),
                os.path.join(current_dir, '..', 'cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv')
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    print(f"Using alternative file path: {file_path}")
                    break
        
        # Run the enhanced analysis
        results = main_enhanced_analysis(file_path, output_dir)
        print("\nEnhanced analysis with baseline models integration complete!")
    except Exception as e:
        print(f"Error running enhanced analysis: {str(e)}")