#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cybercrime Trends Analysis in India's Tech Hubs vs Financial Centers (2011-2022)

This script analyzes cybercrime trends in India's major technology hubs (Bengaluru, Hyderabad)
compared to financial centers (Mumbai) from 2011 to 2022 using time series analysis (ARIMA).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import matplotlib.dates as mdates
import warnings

# Ignore warnings to keep output clean
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
OUTPUT_DIR = 'c:/Users/srikr/Desktop/Studies/Self/Papers/Data Analysis/Complete/Qn 1'


def load_data():
    """Load and preprocess the cybercrime dataset using provided data."""

    # Using the provided data for 2011-2022
    bengaluru = [121, 349, 417, 675, 1042, 762,
                 2743, 5253, 10555, 8892, 6423, 9940]
    hyderabad = [67, 42, 160, 386, 369, 291, 328, 428, 1379, 2553, 3303, 4436]
    mumbai = [33, 105, 132, 608, 979, 980, 1362, 1482, 2527, 2433, 2883, 4724]

    # Years from 2011 to 2022
    years = list(range(2011, 2023))

    # Create a list of dictionaries for easy conversion to DataFrame
    data = []
    for year_idx, year in enumerate(years):
        data.append({'year': year, 'city': 'Bengaluru',
                    'value': bengaluru[year_idx]})
        data.append({'year': year, 'city': 'Hyderabad',
                    'value': hyderabad[year_idx]})
        data.append({'year': year, 'city': 'Mumbai',
                    'value': mumbai[year_idx]})

    # Convert to DataFrame
    df = pd.DataFrame(data)

    print(f"Data loaded successfully with {len(df)} records.")
    print(f"Years range: {min(df['year'])} - {max(df['year'])}")
    print(f"Cities included: {', '.join(df['city'].unique())}")

    return df


def create_time_series_data(df):
    """Create time series data for the cities."""

    # Pivot data for time series analysis
    pivot_df = df.pivot(index='year', columns='city', values='value')

    # Fill any missing values (if any)
    pivot_df = pivot_df.fillna(0)

    return pivot_df


def visualize_trends(pivot_df):
    """Create a line chart showing yearly crime trends per city."""

    plt.figure(figsize=(12, 8))

    for city in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[city],
                 marker='o', linewidth=2, label=city)

    plt.title(
        'Cybercrime Trends in Tech Hubs vs Financial Center (2011-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.xticks(pivot_df.index, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add annotations for major trend changes
    for city in pivot_df.columns:
        max_idx = pivot_df[city].idxmax()
        max_val = pivot_df[city].max()
        plt.annotate(f'{max_val}',
                     xy=(max_idx, max_val),
                     xytext=(0, 10),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'cybercrime_yearly_trends.png'), dpi=300)
    plt.close()


def visualize_boxplots(pivot_df):
    """Create box plots to show crime variability in different time periods."""

    # Create two time periods for comparison
    early_period = pivot_df.loc[2011:2016].reset_index().melt(
        id_vars='year', value_name='cases')
    early_period['period'] = '2011-2016'

    late_period = pivot_df.loc[2017:2022].reset_index().melt(
        id_vars='year', value_name='cases')
    late_period['period'] = '2017-2022'

    # Combine the data
    combined_df = pd.concat([early_period, late_period])

    # Create boxplots
    plt.figure(figsize=(14, 8))

    sns.boxplot(x='city', y='cases', hue='period',
                data=combined_df, palette='Set2')

    plt.title(
        'Cybercrime Variability: Early Period (2011-2016) vs Recent Period (2017-2022)', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.legend(title='Time Period', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'cybercrime_period_comparison_boxplots.png'), dpi=300)
    plt.close()


def run_arima_analysis(pivot_df):
    """Perform ARIMA time series analysis and forecasting, compared with a base model."""

    # First, get results from the base model (Moving Average)
    ma_results, ma_mse, window_size = implement_base_model(pivot_df)

    # ARIMA model evaluation results
    arima_mse = {}

    # Increased figure height for additional plots
    plt.figure(figsize=(15, 15))

    # Define colors for the plot
    colors = {
        'Bengaluru': 'tab:blue',
        'Hyderabad': 'tab:orange',
        'Mumbai': 'tab:green'
    }

    for i, city in enumerate(pivot_df.columns):
        # Prepare data
        data = pivot_df[city]

        # Find best ARIMA model using auto_arima
        stepwise_model = auto_arima(data,
                                    start_p=0, start_q=0,
                                    max_p=3, max_q=3, max_d=1,
                                    seasonal=False,
                                    trace=False,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True)

        best_order = stepwise_model.order
        print(f"Best ARIMA order for {city}: {best_order}")

        # Fit ARIMA model
        model = ARIMA(data, order=best_order)
        model_fit = model.fit()

        # Calculate ARIMA model MSE (in-sample)
        arima_pred = model_fit.predict(start=window_size-1, end=len(data)-1)
        arima_mse[city] = mean_squared_error(
            data.iloc[window_size-1:], arima_pred)
        print(f"ARIMA MSE for {city}: {arima_mse[city]:.2f}")

        # Forecast next 3 years
        forecast_steps = 3
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_index = range(
            data.index[-1] + 1, data.index[-1] + forecast_steps + 1)

        # Plot actual data and forecast
        ax = plt.subplot(3, 1, i+1)
        plt.plot(data.index, data, marker='o',
                 label=f'Actual {city} Data', color=colors[city])
        plt.plot(forecast_index, forecast, marker='*', linestyle='--',
                 label=f'ARIMA Forecast', color=colors[city], alpha=0.7)

        # Plot Moving Average predictions
        plt.plot(ma_results.index[window_size-1:], ma_results[f"{city}_MA"].iloc[window_size-1:],
                 marker='x', linestyle='-.', color='red',
                 label=f'Moving Average (Base Model)')

        # Add confidence intervals for forecast
        pred_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()

        ax.fill_between(forecast_index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                        color=colors[city], alpha=0.1, label='95% Confidence Interval')

        plt.title(
            f'Time Series Analysis for {city}: ARIMA vs Moving Average\nARIMA MSE: {arima_mse[city]:.2f}, MA MSE: {ma_mse[city]:.2f}',
            fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Cybercrime Cases', fontsize=12)
        plt.xticks(list(data.index) + list(forecast_index))
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'arima_vs_baseline_forecasting.png'), dpi=300)
    plt.close()

    # Create a model comparison visualization
    compare_models(pivot_df, arima_mse, ma_mse)

    return pivot_df

    # Plot actual data and forecast
    ax = plt.subplot(3, 1, i+1)
    plt.plot(data.index, data, marker='o',
             label=f'Actual {city} Data', color=colors[city])
    plt.plot(forecast_index, forecast, marker='*', linestyle='--',
             label=f'ARIMA Forecast', color=colors[city], alpha=0.7)

    # Plot Moving Average predictions
    plt.plot(ma_results.index[window_size-1:], ma_results[f"{city}_MA"].iloc[window_size-1:],
             marker='x', linestyle='-.', color='red',
             label=f'Moving Average (Base Model)')

    # Add confidence intervals for forecast
    pred_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()

    ax.fill_between(forecast_index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                    color=colors[city], alpha=0.1, label='95% Confidence Interval')

    plt.title(
        f'Time Series Analysis for {city}: ARIMA vs Moving Average\nARIMA MSE: {arima_mse[city]:.2f}, MA MSE: {ma_mse[city]:.2f}',
        fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Cybercrime Cases', fontsize=12)
    plt.xticks(list(data.index) + list(forecast_index))
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'arima_vs_baseline_forecasting.png'), dpi=300)
    plt.close()

    # Create a model comparison visualization
    compare_models(pivot_df, arima_mse, ma_mse)

    return pivot_df

    # Plot actual data and forecast
    ax = plt.subplot(3, 1, i+1)
    plt.plot(data.index, data, marker='o',
             label=f'Actual {city} Data', color=colors[city])
    plt.plot(forecast_index, forecast, marker='*', linestyle='--',
             label=f'ARIMA Forecast', color=colors[city], alpha=0.7)

    # Add confidence intervals for forecast
    pred_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()

    ax.fill_between(forecast_index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                    color=colors[city], alpha=0.1, label='95% Confidence Interval')

    plt.title(
        f'ARIMA Time Series Analysis and Forecast for {city}', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Cybercrime Cases', fontsize=12)
    plt.xticks(list(data.index) + list(forecast_index))
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'arima_forecasting.png'), dpi=300)
    plt.close()

    return pivot_df


def implement_base_model(pivot_df):
    """Implement a simple Moving Average as the base model for comparison with ARIMA."""

    # Create a DataFrame to store the results
    ma_results = pd.DataFrame(index=pivot_df.index)

    # Window size for moving average (3 years)
    window_size = 3

    # Calculate moving average for each city
    for city in pivot_df.columns:
        # Calculate moving average
        ma_results[f"{city}_MA"] = pivot_df[city].rolling(
            window=window_size).mean()

    # Calculate MSE for each city
    mse_results = {}
    for city in pivot_df.columns:
        # Skip the first window_size-1 values that don't have MA predictions
        actual = pivot_df[city].iloc[window_size-1:]
        predicted = ma_results[f"{city}_MA"].iloc[window_size-1:]
        mse = mean_squared_error(actual, predicted)
        mse_results[city] = mse
        print(f"Moving Average MSE for {city}: {mse:.2f}")

    return ma_results, mse_results, window_size


def compare_models(pivot_df, arima_mse, ma_mse):
    """Create a comparison visualization between ARIMA and Moving Average models."""

    # Prepare data for the bar chart
    cities = list(pivot_df.columns)
    arima_errors = [arima_mse[city] for city in cities]
    ma_errors = [ma_mse[city] for city in cities]

    # Create figure
    plt.figure(figsize=(12, 8))

    # Set width of bars
    barWidth = 0.35

    # Set positions for the bars
    r1 = np.arange(len(cities))
    r2 = [x + barWidth for x in r1]

    # Create bars
    plt.bar(r1, arima_errors, width=barWidth,
            color='blue', alpha=0.7, label='ARIMA Model')
    plt.bar(r2, ma_errors, width=barWidth, color='red',
            alpha=0.7, label='Moving Average (Base Model)')

    # Add labels and legend
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Mean Squared Error (Lower is Better)', fontsize=14)
    plt.title('Model Comparison: ARIMA vs Moving Average Base Model', fontsize=16)
    plt.xticks([r + barWidth/2 for r in range(len(cities))], cities)
    plt.legend()

    # Add text labels on bars
    for i in range(len(cities)):
        plt.text(r1[i], arima_errors[i] + 50,
                 f'{arima_errors[i]:.0f}', ha='center', va='bottom')
        plt.text(r2[i], ma_errors[i] + 50,
                 f'{ma_errors[i]:.0f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=300)
    plt.close()


def visualize_growth_rates(pivot_df):
    """Visualize year-on-year growth rates."""

    # Calculate year-on-year percentage growth
    growth_df = pivot_df.pct_change() * 100
    growth_df = growth_df.round(1)

    # Plot growth rates
    plt.figure(figsize=(12, 8))

    for city in growth_df.columns:
        plt.plot(growth_df.index[1:], growth_df[city]
                 [1:], marker='o', linewidth=2, label=city)

    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(
        'Year-on-Year Growth Rate of Cybercrime Cases (2012-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Growth Rate (%)', fontsize=14)
    plt.xticks(growth_df.index[1:], rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'cybercrime_growth_rates.png'), dpi=300)
    plt.close()


def compare_tech_vs_financial(pivot_df):
    """Compare tech hubs (avg of Bengaluru and Hyderabad) with financial center (Mumbai)."""

    # Create a new dataframe with tech hubs average
    compare_df = pd.DataFrame(index=pivot_df.index)
    compare_df['Tech Hubs Avg'] = pivot_df[[
        'Bengaluru', 'Hyderabad']].mean(axis=1)
    compare_df['Financial Center'] = pivot_df['Mumbai']

    # Plot comparison
    plt.figure(figsize=(12, 8))

    plt.plot(compare_df.index, compare_df['Tech Hubs Avg'], marker='o',
             linewidth=2, label='Tech Hubs (Bengaluru, Hyderabad) Avg')
    plt.plot(compare_df.index, compare_df['Financial Center'], marker='s',
             linewidth=2, label='Financial Center (Mumbai)')

    plt.title(
        'Comparison: Tech Hubs vs Financial Center Cybercrime Trends', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.xticks(compare_df.index, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Calculate and display the ratio of tech hubs to financial center
    for year in compare_df.index:
        ratio = compare_df.loc[year, 'Tech Hubs Avg'] / \
            compare_df.loc[year, 'Financial Center']
        plt.annotate(f'{ratio:.1f}x',
                     xy=(year, (compare_df.loc[year, 'Tech Hubs Avg'] +
                                compare_df.loc[year, 'Financial Center'])/2),
                     xytext=(0, 0),
                     textcoords='offset points',
                     ha='center',
                     fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'tech_vs_financial_comparison.png'), dpi=300)
    plt.close()


def main():
    """Main function to orchestrate the analysis."""

    print("Starting Cybercrime Trends Analysis...")

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load and preprocess data
    df = load_data()
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Create time series data
    pivot_df = create_time_series_data(df)
    print("Time series data created.")

    # Visualize data
    print("Generating visualizations...")
    visualize_trends(pivot_df)
    visualize_boxplots(pivot_df)
    visualize_growth_rates(pivot_df)
    compare_tech_vs_financial(pivot_df)

    # Perform ARIMA analysis with baseline model comparison
    print("Performing ARIMA time series analysis with baseline model comparison...")
    run_arima_analysis(pivot_df)

    print("Analysis completed. All visualizations saved to the 'Qn 1' folder.")

    # Print summary statistics
    print("\nSummary Statistics for Cybercrime Cases (2011-2022):")
    print(pivot_df.describe().round(0))


if __name__ == "__main__":
    main()
