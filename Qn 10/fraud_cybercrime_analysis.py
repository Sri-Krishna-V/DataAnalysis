#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of Geographic Distribution of Fraud-Related Cybercrimes in India (2019-2022)
---------------------
This script analyzes and visualizes the geographic distribution of fraud-related cybercrimes 
across Indian states from 2019 to 2022 using data from the National Crime Records Bureau (NCRB).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Blues_r")

# Baseline Model Definitions


class BaselineModels:
    @staticmethod
    def historical_average(df, group_col, value_col, years):
        """Calculate historical average value for a given grouping."""
        return df[df['year'].isin(years)].groupby(group_col)[value_col].mean().mean()

    @staticmethod
    def national_trend(df, value_col):
        """Calculate national trend as yearly sum."""
        return df.groupby('year')[value_col].sum().reset_index()

    @staticmethod
    def naive_forecast(series, horizon):
        """Generate naive forecast by repeating last value."""
        last_value = series.iloc[-1]
        return [last_value] * horizon


# Define file paths
BASE_DIR = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete"
INPUT_FILE = os.path.join(
    BASE_DIR, "cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Qn 10")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load and preprocess the data


def load_and_preprocess_data(file_path):
    """
    Load and preprocess cybercrime data
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Filter for fraud-related crimes
    fraud_df = df[df['offence_category'] == 'Fraud or Illegal Gain'].copy()

    # Check if we have data for the required years
    available_years = sorted(fraud_df['year'].unique())
    print(f"Available years in dataset: {available_years}")

    # Filter out 'All India' total rows for state-level analysis
    fraud_df = fraud_df[fraud_df['state'] != 'All India']

    # Clean and transform data
    fraud_df['value'] = pd.to_numeric(fraud_df['value'], errors='coerce')
    fraud_df = fraud_df.dropna(subset=['value'])

    return fraud_df

# Function for data exploration and summary


def explore_data(df):
    """
    Perform exploratory data analysis
    """
    # Basic summary statistics
    summary = df.groupby('year')['value'].agg(
        ['sum', 'mean', 'median', 'min', 'max']).reset_index()
    print("\nYearly Summary Statistics for Fraud-Related Cybercrimes:")
    print(summary)

    # Top states with highest fraud cybercrimes (across all years)
    top_states = df.groupby('state')['value'].sum(
    ).sort_values(ascending=False).head(10)
    print("\nTop 10 States with Highest Fraud-Related Cybercrimes (2019-2022):")
    print(top_states)

    # State with highest yearly growth
    pivot_df = df.pivot_table(
        index='state', columns='year', values='value', aggfunc='sum').fillna(0)

    # Calculate year-over-year growth rates (2020-2022)
    growth_states = []
    for year in range(2020, 2023):
        pivot_df[f'growth_{year}'] = (
            (pivot_df[year] - pivot_df[year-1]) / pivot_df[year-1]) * 100

    avg_growth = pivot_df[['growth_2020',
                           'growth_2021', 'growth_2022']].mean(axis=1)
    top_growth_states = avg_growth.sort_values(ascending=False).head(5)
    print("\nTop 5 States with Highest Average Growth Rate (2020-2022):")
    print(top_growth_states)

    return top_states, top_growth_states, pivot_df

# VISUALIZATION 1: Choropleth Map of Fraud Cybercrimes by State (Latest Year)


def create_choropleth_map(df):
    """
    Create a choropleth map showing geographic distribution of fraud cybercrimes
    with national average baseline
    """
    print("\nCreating choropleth map with baseline...")
    # Get the most recent year's data
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].copy()

    # Calculate baseline (national average)
    national_avg = BaselineModels.historical_average(
        df, 'state', 'value', [latest_year])

    # Try to load India shapefile - this is a placeholder as proper geospatial data would be needed
    # In a real implementation, you would need to have the India shapefile
    try:
        # Create a simple dataframe with state names for demonstration
        states_data = latest_data[['state', 'value']].set_index('state')

        # For this example, we'll use a bar chart instead of an actual map
        # as loading shapefiles is environment-dependent
        plt.figure(figsize=(12, 10))
        states_sorted = states_data.sort_values('value', ascending=False)
        ax = sns.barplot(x=states_sorted.index, y=states_sorted['value'])

        # Add baseline comparison
        ax.axhline(national_avg, color='r', linestyle='--',
                   linewidth=2, label=f'National Average: {national_avg:,.0f}')

        plt.title(
            f'Fraud-Related Cybercrimes by State in India ({latest_year}) with National Baseline', fontsize=14)
        plt.xlabel('State', fontsize=12)
        plt.ylabel('Number of Fraud Cybercrimes', fontsize=12)
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(
            OUTPUT_DIR, f'fraud_choropleth_{latest_year}_with_baseline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved choropleth map with baseline to {output_path}")

    except Exception as e:
        print(f"Error creating choropleth map: {e}")

# VISUALIZATION 2: Time Series Plot of Top 5 States with Highest Fraud Cybercrimes


def create_time_series_plot(df):
    """
    Create a time series plot showing fraud cybercrime trends in top 5 states
    with national trend baseline
    """
    print("\nCreating time series plot with national baseline...")
    # Identify top 5 states based on the most recent year
    latest_year = df['year'].max()
    top_states = df[df['year'] == latest_year].nlargest(5, 'value')[
        'state'].unique()

    # Filter data for top 5 states
    top_states_data = df[df['state'].isin(top_states)]

    # Calculate national trend baseline
    national_trend = BaselineModels.national_trend(df, 'value')

    # Create the time series plot
    plt.figure(figsize=(12, 8))
    for state in top_states:
        state_data = top_states_data[top_states_data['state'] == state]
        plt.plot(state_data['year'], state_data['value'],
                 marker='o', linewidth=2, label=state)

    # Plot national trend baseline
    plt.plot(national_trend['year'], national_trend['value'],
             color='black', linestyle=':', linewidth=3,
             label='National Trend Baseline')

    plt.title(
        'Fraud-Related Cybercrime Trends in Top 5 States vs National Baseline (2019-2022)', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Fraud Cybercrimes', fontsize=12)
    plt.xticks(top_states_data['year'].unique())
    plt.legend(title='State', title_fontsize=12, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(OUTPUT_DIR, 'fraud_trends_with_baseline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved time series plot with baseline to {output_path}")

# VISUALIZATION 3: Heatmap of Year-over-Year Growth Rate


def create_growth_heatmap(pivot_df):
    """
    Create a heatmap showing year-over-year growth rates for fraud cybercrimes
    with zero-growth baseline
    """
    print("\nCreating growth rate heatmap with baseline...")
    # Select top 15 states based on 2022 values
    top_states = pivot_df[2022].sort_values(ascending=False).head(15).index
    growth_df = pivot_df.loc[top_states, [
        'growth_2020', 'growth_2021', 'growth_2022']]

    # Replace infinite values with NaN and then with a large number
    growth_df = growth_df.replace([np.inf, -np.inf], np.nan)
    growth_df = growth_df.fillna(0)

    # Cap growth rates for better visualization
    growth_df = growth_df.clip(-100, 200)

    # Create baseline comparison - zero growth is the baseline
    baseline_growth = 0  # Zero-growth baseline

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(growth_df)
    mask[np.isnan(growth_df)] = True

    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    ax = sns.heatmap(growth_df, annot=True, fmt='.1f', cmap=cmap,
                     center=baseline_growth, linewidths=.5, cbar_kws={"shrink": .8},
                     mask=mask)

    plt.title(
        'Year-over-Year Growth Rate (%) vs Zero-Growth Baseline in Top 15 States', fontsize=14)
    plt.xlabel('Year-over-Year Growth', fontsize=12)
    plt.ylabel('State', fontsize=12)

    # Rename the column labels for clarity
    ax.set_xticklabels(['2019-2020', '2020-2021', '2021-2022'])
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        OUTPUT_DIR, 'fraud_growth_heatmap_with_baseline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved growth heatmap with baseline to {output_path}")

# VISUALIZATION 4: Bar chart showing per capita fraud cybercrimes


def create_per_capita_chart(df):
    """
    Create a bar chart showing per capita fraud cybercrimes in 2022
    with national per capita baseline
    """
    print("\nCreating per capita fraud cybercrime chart with baseline...")
    # This requires population data by state which we don't have in the dataset
    # We'll create a placeholder visualization with mock population data for demonstration

    # Get 2022 data
    data_2022 = df[df['year'] == 2022].copy()

    # Mock population data (in millions) - in a real scenario, actual census data should be used
    population_data = {
        'Karnataka': 61.0, 'Telangana': 35.0, 'Maharashtra': 112.0, 'Uttar Pradesh': 200.0,
        'Tamil Nadu': 72.0, 'Gujarat': 60.0, 'Rajasthan': 68.0, 'Delhi': 16.8, 'Kerala': 33.0,
        'Madhya Pradesh': 72.0, 'Bihar': 104.0, 'Odisha': 41.0, 'Haryana': 25.0,
        'Jharkhand': 33.0, 'Punjab': 27.0, 'Andhra Pradesh': 49.0, 'West Bengal': 91.0,
        # Add rough estimates for other states/UTs
        'Assam': 31.0, 'Chhattisgarh': 25.0, 'Uttarakhand': 10.0, 'Jammu and Kashmir': 12.0,
        'Himachal Pradesh': 7.0, 'Goa': 1.5, 'Tripura': 4.0, 'Meghalaya': 3.0, 'Manipur': 2.7,
        'Nagaland': 2.0, 'Arunachal Pradesh': 1.4, 'Mizoram': 1.1, 'Sikkim': 0.6,
        'Chandigarh': 1.0, 'Puducherry': 1.2, 'Andaman and Nicobar Islands': 0.4,
        'Dadra and Nagar Haveli and Daman and Diu': 0.6, 'Lakshadweep': 0.1, 'Ladakh': 0.3
    }

    # Add population data to the dataframe
    data_2022['population'] = data_2022['state'].map(population_data)

    # Calculate per capita cybercrimes (per 100,000 people)
    data_2022['per_capita'] = (
        data_2022['value'] / (data_2022['population'] * 1e6)) * 100000

    # Calculate national per capita baseline
    national_per_capita = (data_2022['value'].sum() /
                           data_2022['population'].sum()) * 100000

    # Sort by per capita value and get top 15
    data_2022 = data_2022.sort_values('per_capita', ascending=False).head(15)

    # Create the bar chart
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='per_capita', y='state',
                     data=data_2022, palette='viridis')

    # Add baseline reference
    plt.axvline(national_per_capita, color='green', linestyle='--',
                label=f'National Per Capita: {national_per_capita:.1f}')

    plt.title(
        'Fraud-Related Cybercrimes per 100,000 People (2022) with National Baseline', fontsize=14)
    plt.xlabel('Fraud Cybercrimes per 100,000 People', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        OUTPUT_DIR, 'fraud_per_capita_2022_with_baseline.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per capita chart with baseline to {output_path}")

# VISUALIZATION 5: Bubble plot comparing 2021 vs 2022 fraud cybercrimes


def create_comparison_bubble_plot(df):
    """
    Create a bubble plot comparing fraud cybercrimes between 2021 and 2022
    """
    print("\nCreating comparison bubble plot...")
    # Get data for 2021 and 2022
    pivot_df = df.pivot_table(
        index='state', columns='year', values='value', aggfunc='sum').fillna(0)

    # Calculate growth from 2021 to 2022
    comparison_df = pd.DataFrame({
        'state': pivot_df.index,
        'value_2021': pivot_df[2021],
        'value_2022': pivot_df[2022]
    })

    # Calculate growth percentage
    comparison_df['growth'] = ((comparison_df['value_2022'] - comparison_df['value_2021']) /
                               comparison_df['value_2021'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Get top 20 states by 2022 values for better visualization
    top_states = comparison_df.nlargest(20, 'value_2022')

    # Create the bubble plot
    plt.figure(figsize=(12, 10))

    # Define color mapping based on growth
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(
        top_states['growth'].min(), top_states['growth'].max())

    # Create scatter plot with bubble size representing 2022 values
    scatter = plt.scatter(
        x=top_states['value_2021'],
        y=top_states['value_2022'],
        s=top_states['value_2022'] / 50,  # Scale bubble size
        c=top_states['growth'],
        cmap=cmap,
        alpha=0.7,
        edgecolors='black',
        linewidths=1
    )

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Growth Rate (%)', rotation=270, labelpad=20)

    # Add state labels
    for i, row in top_states.iterrows():
        plt.annotate(row['state'],
                     (row['value_2021'], row['value_2022']),
                     fontsize=9,
                     ha='center',
                     va='bottom')

    # Add reference line (y=x)
    max_val = max(top_states['value_2021'].max(),
                  top_states['value_2022'].max())
    plt.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3)

    # Add labels and title
    plt.xlabel('Fraud Cybercrimes in 2021', fontsize=12)
    plt.ylabel('Fraud Cybercrimes in 2022', fontsize=12)
    plt.title('Comparison of Fraud-Related Cybercrimes: 2021 vs 2022', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        OUTPUT_DIR, 'fraud_bubble_plot_2021_vs_2022.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison bubble plot to {output_path}")

# VISUALIZATION BONUS: ARIMA Forecast for Top 5 States (since we have limited time series data)


def create_arima_forecast(df):
    """
    Create a simple ARIMA forecast for the top 5 states with naive baseline
    """
    print("\nCreating ARIMA forecast plot with naive baseline...")

    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Identify top 5 states based on the most recent year
        latest_year = df['year'].max()
        top_states = df[df['year'] == latest_year].nlargest(5, 'value')[
            'state'].unique()

        # Filter data for top 5 states
        top_states_data = df[df['state'].isin(top_states)]

        # Create the forecast plot
        plt.figure(figsize=(14, 10))

        for state in top_states:
            state_data = top_states_data[top_states_data['state'] == state]
            state_data = state_data.sort_values('year')

            # Fit ARIMA model (very simple due to limited data points)
            # In reality, you'd want at least 30+ data points for a good ARIMA model
            try:
                model = ARIMA(state_data['value'].values, order=(1, 1, 0))
                model_fit = model.fit()

                # Forecast next 3 years
                forecast = model_fit.forecast(steps=3)
                future_years = [latest_year + i for i in range(1, 4)]

                # Plot historical data and forecast
                plt.plot(state_data['year'], state_data['value'],
                         marker='o', linewidth=2, label=f'{state} (Actual)')
                plt.plot(future_years, forecast, marker='x', linestyle='--',
                         linewidth=1.5, label=f'{state} (Forecast)')

                # Add naive forecast baseline
                naive_forecast = BaselineModels.naive_forecast(
                    state_data['value'], 3)
                plt.plot(future_years, naive_forecast, marker='x', linestyle=':',
                         linewidth=1, label=f'{state} (Naive Baseline)')
            except:
                # If ARIMA fails due to limited data, just plot the historical data
                plt.plot(state_data['year'], state_data['value'],
                         marker='o', linewidth=2, label=f'{state}')

        plt.title(
            'Fraud-Related Cybercrime Forecast with Naive Baseline for Top 5 States', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Fraud Cybercrimes', fontsize=12)
        plt.xticks(list(df['year'].unique()) + future_years)
        plt.legend(title='State', title_fontsize=12, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(
            OUTPUT_DIR, 'arima_forecast_with_baseline.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ARIMA forecast plot with baseline to {output_path}")

    except ImportError:
        print("Statsmodels not available, skipping ARIMA forecast...")
    except Exception as e:
        print(f"Error creating ARIMA forecast: {e}")

# Main execution function


def main():
    """
    Main execution function
    """
    # Load and preprocess the data
    fraud_df = load_and_preprocess_data(INPUT_FILE)

    # Basic data exploration
    top_states, top_growth_states, pivot_df = explore_data(fraud_df)

    # Create visualizations
    create_choropleth_map(fraud_df)
    create_time_series_plot(fraud_df)
    create_growth_heatmap(pivot_df)
    create_per_capita_chart(fraud_df)
    create_comparison_bubble_plot(fraud_df)
    create_arima_forecast(fraud_df)

    print("\nAnalysis complete! All visualizations have been saved to the output directory.")


if __name__ == "__main__":
    main()
