#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cybercrime Prevalence Analysis in Metropolitan Areas of India (2012-2022)
=========================================================================
This script analyzes the trend of cybercrime prevalence across metropolitan 
areas in India from 2012 to 2022 using time series analysis (ARIMA).

Data source: National Crime Records Bureau (NCRB) - Crime in India reports
"""

import os
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'

# Define the path to the data file
data_path = r'C:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv'

# Set output directory to current directory (Qn 2 folder)
output_dir = '.'


def load_and_prepare_data(file_path):
    """
    Load and prepare the dataset for analysis
    """
    print("Loading and preparing data...")
    df = pd.read_csv(file_path)

    # Filter data for years 2012-2022
    df = df[(df['year'] >= 2012) & (df['year'] <= 2022)]

    # Drop rows with missing values
    df = df.dropna(subset=['value'])

    # Filter out non-metropolitan areas and aggregate entries
    # We consider only rows where 'city' is not 'Total Cities'
    df = df[df['city'] != 'Total Cities']

    print(
        f"Data loaded with {len(df)} records spanning from {df['year'].min()} to {df['year'].max()}")
    return df


def identify_top_metro_cities(df, n=10):
    """
    Identify the top n metropolitan cities by total cybercrime cases
    """
    city_totals = df.groupby('city')['value'].sum().reset_index()
    top_cities = city_totals.nlargest(n, 'value')['city'].tolist()
    print(
        f"Top {n} metropolitan cities by cybercrime cases: {', '.join(top_cities)}")
    return top_cities


def analyze_yearly_trend(df, top_cities):
    """
    Analyze yearly trends of cybercrime in top metropolitan cities
    """
    print("Analyzing yearly trends...")

    # Filter data for top cities and pivot to get yearly counts
    top_cities_df = df[df['city'].isin(top_cities)]
    yearly_pivot = top_cities_df.pivot_table(
        index='year', columns='city', values='value', aggfunc='sum'
    ).fillna(0)

    # Plot the trends
    plt.figure(figsize=(14, 10))
    for city in yearly_pivot.columns:
        plt.plot(yearly_pivot.index,
                 yearly_pivot[city], marker='o', linewidth=2, label=city)

    plt.title(
        'Yearly Cybercrime Trends in Top Metropolitan Cities (2012-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.xticks(yearly_pivot.index)

    # Save the plot
    plt.savefig(f'{output_dir}/yearly_cybercrime_trends_top_metros.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    return yearly_pivot


def create_crime_intensity_heatmap(df, top_cities):
    """
    Create a heatmap showing cybercrime intensity across top cities over years
    """
    print("Creating crime intensity heatmap...")

    # Filter data for top cities and pivot to get yearly counts
    top_cities_df = df[df['city'].isin(top_cities)]
    yearly_pivot = top_cities_df.pivot_table(
        index='year', columns='city', values='value', aggfunc='sum'
    ).fillna(0)

    # Create the heatmap
    plt.figure(figsize=(16, 10))
    ax = sns.heatmap(
        yearly_pivot,
        annot=True,
        fmt='g',
        cmap='viridis',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Cybercrime Cases'}
    )

    plt.title(
        'Cybercrime Intensity Across Top Metropolitan Cities (2012-2022)', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Year', fontsize=14)
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    plt.savefig(f'{output_dir}/cybercrime_intensity_heatmap.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def calculate_growth_rates(yearly_data):
    """
    Calculate year-on-year growth rates for cybercrime cases
    """
    print("Calculating growth rates...")

    # Calculate percentage growth for each city
    growth_rates = yearly_data.pct_change() * 100

    # Plot growth rates
    plt.figure(figsize=(14, 10))
    for city in growth_rates.columns:
        plt.plot(growth_rates.index[1:], growth_rates[city]
                 [1:], marker='o', linewidth=2, label=city)

    plt.title(
        'Year-on-Year Growth Rate of Cybercrime Cases (2013-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Growth Rate (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.xticks(growth_rates.index[1:])

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)

    # Save the plot
    plt.savefig(f'{output_dir}/cybercrime_growth_rates.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    return growth_rates


def perform_arima_forecasting(yearly_data, top_cities, forecast_periods=3):
    """
    Perform ARIMA forecasting for top 5 cities with highest cybercrime cases
    """
    print("Performing ARIMA forecasting...")

    # Select top 5 cities from our top cities list based on total cases
    city_totals = yearly_data.sum().sort_values(ascending=False)
    forecast_cities = city_totals.index[:5].tolist()

    plt.figure(figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, city in enumerate(forecast_cities):
        # Prepare data for ARIMA
        city_data = yearly_data[city].astype(float)

        # Fit ARIMA model (p,d,q) = (1,1,1) as a simple yet effective model
        # This can be optimized using auto_arima if statsmodels.tsa.statespace.sarimax is available
        model = ARIMA(city_data, order=(1, 1, 1))
        model_fit = model.fit()

        # Forecast next 3 years
        forecast_result = model_fit.forecast(steps=forecast_periods)

        # Plot actual vs forecast
        plt.plot(yearly_data.index, city_data, marker='o',
                 color=colors[i], label=f"{city} (Actual)")

        # Create forecast years (next 3 years after the last year in data)
        forecast_years = range(
            yearly_data.index[-1] + 1, yearly_data.index[-1] + forecast_periods + 1)

        plt.plot(forecast_years, forecast_result, marker='x', linestyle='--',
                 color=colors[i], label=f"{city} (Forecast)")

    plt.title(
        'ARIMA Forecast for Top 5 Metropolitan Cities (2023-2025)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)

    # Set x-ticks to be integers only
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save the plot
    plt.savefig(f'{output_dir}/arima_forecast_top_metros.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def analyze_metropolitan_distribution(df, year=2022):
    """
    Analyze the distribution of cybercrime cases across metropolitan areas for the most recent year
    """
    print(f"Analyzing metropolitan distribution for {year}...")

    # Filter data for the specified year
    year_data = df[df['year'] == year]

    # Get top 15 cities by number of cases
    top_15_cities = year_data.nlargest(15, 'value')

    # Create a bar plot
    plt.figure(figsize=(14, 10))
    bars = plt.bar(top_15_cities['city'], top_15_cities['value'],
                   color=sns.color_palette("viridis", 15))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{int(height)}',
                 ha='center', va='bottom', rotation=0, fontsize=10)

    plt.title(
        f'Top 15 Metropolitan Cities by Cybercrime Cases ({year})', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(f'{output_dir}/top_metros_distribution_{year}.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def calculate_cagr(df, top_cities):
    """
    Calculate Compound Annual Growth Rate (CAGR) for top metropolitan cities
    """
    print("Calculating CAGR for top cities...")

    # Filter data for top cities
    top_cities_df = df[df['city'].isin(top_cities)]

    # Get the first and last years in the dataset
    first_year = df['year'].min()
    last_year = df['year'].max()
    time_period = last_year - first_year

    # Calculate CAGR for each city
    cagr_data = []

    for city in top_cities:
        city_data = top_cities_df[top_cities_df['city'] == city]

        # Get the first and last year values
        first_year_value = city_data[city_data['year']
                                     == first_year]['value'].values
        last_year_value = city_data[city_data['year']
                                    == last_year]['value'].values

        # Check if we have valid data points
        if len(first_year_value) > 0 and len(last_year_value) > 0:
            first_year_value = first_year_value[0]
            last_year_value = last_year_value[0]

            # Calculate CAGR
            if first_year_value > 0:  # Avoid division by zero
                cagr = (((last_year_value / first_year_value)
                        ** (1 / time_period)) - 1) * 100
            else:
                cagr = float('inf')  # Handle cases where initial value is 0

            cagr_data.append({
                'City': city,
                'Initial Value (2012)': first_year_value,
                'Final Value (2022)': last_year_value,
                'CAGR (%)': cagr if cagr != float('inf') else 'N/A'
            })

    # Create a DataFrame and sort by CAGR
    cagr_df = pd.DataFrame(cagr_data)
    cagr_df = cagr_df.sort_values('CAGR (%)', ascending=False)

    # Plot CAGR values
    plt.figure(figsize=(14, 10))

    # Filter out 'N/A' values for plotting
    plot_df = cagr_df[cagr_df['CAGR (%)'] != 'N/A'].copy()

    # Create the bar chart
    bars = plt.bar(plot_df['City'], plot_df['CAGR (%)'],
                   color=sns.color_palette("viridis", len(plot_df)))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%',
                 ha='center', va='bottom', rotation=0, fontsize=10)

    plt.title(
        f'Compound Annual Growth Rate of Cybercrime Cases ({first_year}-{last_year})', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('CAGR (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(f'{output_dir}/cybercrime_cagr.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    print("CAGR Results:")
    print(cagr_df)

    return cagr_df


def main():
    """
    Main function to execute the analysis workflow
    """
    print("\n=== Cybercrime Prevalence Analysis in Metropolitan Areas of India (2012-2022) ===\n")

    # Load and prepare the data
    df = load_and_prepare_data(data_path)

    # Identify top metropolitan cities
    top_metros = identify_top_metro_cities(df, n=10)

    # Analyze yearly trends
    yearly_data = analyze_yearly_trend(df, top_metros)

    # Create crime intensity heatmap
    create_crime_intensity_heatmap(df, top_metros)

    # Calculate growth rates
    calculate_growth_rates(yearly_data)

    # Perform ARIMA forecasting
    perform_arima_forecasting(yearly_data, top_metros)

    # Analyze metropolitan distribution for the most recent year
    analyze_metropolitan_distribution(df)

    # Calculate CAGR for top cities
    calculate_cagr(df, top_metros)

    print("\n=== Analysis completed successfully! ===")
    print(
        f"All visualizations have been saved to the '{output_dir}' directory.")


if __name__ == "__main__":
    main()
