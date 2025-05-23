#!/usr/bin/env python
# coding: utf-8

"""
Kolkata Cybercrime Volatility Analysis

This script analyzes the volatility in cybercrime trends in Kolkata compared to other major metros
using data from the National Crime Records Bureau (NCRB) of India.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot style and figure size
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Set a consistent color palette for the metro cities
METRO_COLORS = {
    'Kolkata': '#FF5733',    # Vibrant orange-red
    'Delhi': '#33A8FF',      # Bright blue
    'Mumbai': '#33FF57',     # Bright green
    'Chennai': '#A833FF',    # Purple
    'Bengaluru': '#FFBD33',  # Gold
    'Hyderabad': '#FF33A8'   # Pink
}


def load_and_preprocess_data():
    """
    Load and preprocess the city-wise cybercrime data.

    Returns:
        DataFrame: Preprocessed cybercrime data
    """
    # Define file path
    file_path = '../cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv'

    # Load data
    df = pd.read_csv(file_path)

    # Basic preprocessing
    df = df[df['city'] != 'Total Cities']  # Remove summary rows
    df['year'] = df['year'].astype(int)    # Ensure year is integer

    # Handle missing values before converting to integer
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])  # Remove rows with NaN values
    df['value'] = df['value'].astype(int)  # Ensure count is integer

    # Filter only major metro cities
    metro_cities = ['Kolkata', 'Delhi', 'Mumbai',
                    'Chennai', 'Bengaluru', 'Hyderabad']
    df_metros = df[df['city'].isin(metro_cities)]

    return df_metros


def calculate_volatility_metrics(df):
    """
    Calculate volatility metrics for each city.

    Args:
        df (DataFrame): Preprocessed data

    Returns:
        DataFrame: Dataframe with volatility metrics by city
    """
    # Pivot the data to get a time series for each city
    city_time_series = df.pivot(index='year', columns='city', values='value')

    # Calculate year-over-year percent change
    pct_change = city_time_series.pct_change().dropna()

    # Calculate volatility metrics
    volatility_metrics = pd.DataFrame({
        'std_dev': pct_change.std(),
        'max_change': pct_change.max(),
        'min_change': pct_change.min(),
        'range': pct_change.max() - pct_change.min(),
        'mean_abs_change': pct_change.abs().mean()
    })

    # Add coefficient of variation (CV) - standard deviation divided by mean
    volatility_metrics['cv'] = city_time_series.std() / city_time_series.mean()

    # Calculate average crime count for context
    volatility_metrics['avg_crimes'] = city_time_series.mean()

    return volatility_metrics, city_time_series, pct_change


def visualize_time_series(city_time_series, output_path):
    """
    Create a time series line plot of cybercrime trends across major metros.

    Args:
        city_time_series (DataFrame): Time series data of cybercrimes by city
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(14, 8))

    for city in city_time_series.columns:
        if city in METRO_COLORS:
            plt.plot(city_time_series.index, city_time_series[city],
                     marker='o', linewidth=2.5, label=city, color=METRO_COLORS[city])

    plt.title(
        'Cybercrime Trends in Major Indian Metro Cities (2014-2022)', fontsize=18)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cyber Crimes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(city_time_series.index)

    # Add annotations for Kolkata's significant changes
    kolkata_data = city_time_series['Kolkata']

    # Find years with significant changes for Kolkata
    for i in range(1, len(kolkata_data)):
        # 40% change
        if abs(kolkata_data.iloc[i] - kolkata_data.iloc[i-1]) / kolkata_data.iloc[i-1] > 0.4:
            year = kolkata_data.index[i]
            value = kolkata_data.iloc[i]
            pct_change = (
                kolkata_data.iloc[i] - kolkata_data.iloc[i-1]) / kolkata_data.iloc[i-1] * 100
            plt.annotate(f'{pct_change:.1f}%',
                         xy=(year, value),
                         xytext=(0, 10),
                         textcoords='offset points',
                         ha='center',
                         fontweight='bold',
                         color=METRO_COLORS['Kolkata'])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_year_over_year_change(pct_change, output_path):
    """
    Create a heatmap of year-over-year percent changes in cybercrime rates.

    Args:
        pct_change (DataFrame): Year-over-year percent changes
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(14, 8))

    # Convert to percentage for better readability
    heatmap_data = pct_change * 100

    # Create the heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn_r', fmt='.1f',
                linewidths=0.5, cbar_kws={'label': 'Percent Change (%)'})

    plt.title('Year-over-Year Percent Change in Cybercrime Rates', fontsize=18)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Year', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_volatility_comparison(volatility_metrics, output_path):
    """
    Create a bar chart comparing volatility metrics across cities.

    Args:
        volatility_metrics (DataFrame): Volatility metrics by city
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(14, 10))

    # Create subplot for coefficient of variation (main volatility metric)
    plt.subplot(2, 1, 1)
    city_colors = [METRO_COLORS.get(city, '#CCCCCC')
                   for city in volatility_metrics.index]

    # Sort by coefficient of variation in descending order
    cv_sorted = volatility_metrics.sort_values('cv', ascending=False)

    bars = plt.bar(cv_sorted.index, cv_sorted['cv'], color=[
                   METRO_COLORS.get(city, '#CCCCCC') for city in cv_sorted.index])

    # Highlight Kolkata
    for i, city in enumerate(cv_sorted.index):
        if city == 'Kolkata':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)

    plt.title(
        'Coefficient of Variation in Cybercrime Rates (Higher = More Volatile)', fontsize=16)
    plt.ylabel('Coefficient of Variation', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)

    # Create subplot for range of percent changes
    plt.subplot(2, 1, 2)

    # Calculate range between max and min percentage changes
    range_sorted = volatility_metrics.sort_values('range', ascending=False)

    # Plot the range as a bar
    bars = plt.bar(range_sorted.index, range_sorted['range'],
                   color=[METRO_COLORS.get(city, '#CCCCCC') for city in range_sorted.index])

    # Highlight Kolkata
    for i, city in enumerate(range_sorted.index):
        if city == 'Kolkata':
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(2)

    plt.title(
        'Range Between Maximum and Minimum Year-over-Year Changes', fontsize=16)
    plt.ylabel('Range (as proportion)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_volatility_distribution(pct_change, output_path):
    """
    Create box plots showing the distribution of year-over-year changes by city.

    Args:
        pct_change (DataFrame): Year-over-year percent changes
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(14, 8))

    # Melt the dataframe for easier plotting
    melted_df = pct_change.reset_index().melt(
        id_vars='year', var_name='city', value_name='pct_change')

    # Convert to percentage for better readability
    melted_df['pct_change'] = melted_df['pct_change'] * 100

    # Create boxplot
    sns.boxplot(x='city', y='pct_change', data=melted_df,
                palette=METRO_COLORS, whis=[5, 95],
                boxprops=dict(alpha=0.7))

    # Add individual points
    sns.stripplot(x='city', y='pct_change', data=melted_df,
                  size=8, color='black', alpha=0.6)

    plt.title(
        'Distribution of Year-over-Year Percent Changes in Cybercrime Rates', fontsize=18)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Percent Change (%)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_crime_type_comparison(output_path):
    """
    Create a stacked bar chart showing the composition of cybercrime types for Kolkata vs. other metros.

    Args:
        output_path (str): Path to save the visualization
    """
    # Load the detailed crime type data
    try:
        file_path = '../cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv'
        crime_types_df = pd.read_csv(file_path)

        # Filter for the most recent year and relevant cities
        recent_year = crime_types_df['year'].max()
        metro_cities = ['Kolkata', 'Delhi', 'Mumbai',
                        'Chennai', 'Bengaluru', 'Hyderabad']
        recent_data = crime_types_df[(crime_types_df['year'] == recent_year) &
                                     (crime_types_df['city'].isin(metro_cities))]

        # Filter out the 'Total' rows and group data by offense category
        filtered_data = recent_data[recent_data['offence_category'] != 'Total']

        # Focus on main offense categories
        main_categories = [
            'Computer Related Offences',
            'Publication or transmission of Obscene or Sexually Explicit Act in Electronic Form',
            'Cyber Terrorism',
            'Tampering Computer Source documents',
            'Unauthorized access or attempt to access to protected computer system'
        ]

        # Filter for main categories and aggregate data
        filtered_data = filtered_data[filtered_data['offence_category'].isin(
            main_categories)]

        # Sum up values by city and offense category
        category_data = filtered_data.groupby(['city', 'offence_category'])[
            'value'].sum().reset_index()

        # Create a pivot table for plotting
        pivot_data = category_data.pivot(
            index='city', columns='offence_category', values='value').fillna(0)

        # Normalize to show percentage composition
        normalized_data = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

        # Plotting
        plt.figure(figsize=(14, 10))
        normalized_data.plot(kind='bar', stacked=True,
                             colormap='tab10', figsize=(14, 8))

        plt.title(
            f'Cybercrime Type Composition by City ({recent_year})', fontsize=18)
        plt.xlabel('City', fontsize=14)
        plt.ylabel('Percentage of Total Cybercrimes (%)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(title='Cybercrime Type',
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in crime type comparison: {e}")
        # If there's an error, create a note about it
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Unable to generate crime type comparison due to: {e}",
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def train_and_compare_models(city_time_series, output_path):
    """
    Train baseline and ARIMA models for Kolkata's cybercrime data and compare their performance.

    Args:
        city_time_series (DataFrame): Time series data of cybercrimes by city
        output_path (str): Path to save the visualization
    """
    # Focus on Kolkata's data
    kolkata_data = city_time_series['Kolkata'].dropna()

    if len(kolkata_data) < 5:  # Need minimum data points for modeling
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Insufficient data points for modeling",
                 horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Prepare data for modeling
    train_size = int(len(kolkata_data) * 0.7)
    train_data = kolkata_data.iloc[:train_size]
    test_data = kolkata_data.iloc[train_size:]

    # Create X variables (years) for linear regression
    X_train = np.array(range(len(train_data))).reshape(-1, 1)
    X_test = np.array(range(len(train_data), len(kolkata_data))).reshape(-1, 1)
    y_train = train_data.values
    y_test = test_data.values

    # Train baseline model (linear regression)
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    # Train ARIMA model
    try:
        # Find best order for ARIMA
        best_aic = float('inf')
        best_order = None

        # Try different combinations of p, d, q values
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                    except:
                        continue

        # Train with best order
        if best_order:
            arima_model = ARIMA(train_data, order=best_order)
            arima_results = arima_model.fit()

            # Make predictions
            arima_pred = arima_results.forecast(steps=len(test_data))
            arima_rmse = np.sqrt(mean_squared_error(y_test, arima_pred))
            arima_mae = mean_absolute_error(y_test, arima_pred)
        else:
            # Fallback to simple order if automatic selection fails
            arima_model = ARIMA(train_data, order=(1, 1, 0))
            arima_results = arima_model.fit()
            arima_pred = arima_results.forecast(steps=len(test_data))
            arima_rmse = np.sqrt(mean_squared_error(y_test, arima_pred))
            arima_mae = mean_absolute_error(y_test, arima_pred)

    except Exception as e:
        print(f"Error in ARIMA modeling: {e}")
        arima_pred = np.array([np.nan] * len(test_data))
        arima_rmse = np.nan
        arima_mae = np.nan

    # Create comparison visualization
    plt.figure(figsize=(14, 10))

    # Plot original data
    plt.plot(kolkata_data.index, kolkata_data.values, 'o-', color=METRO_COLORS['Kolkata'],
             linewidth=2, label='Actual Data', markersize=8)

    # Plot train/test split
    plt.axvline(x=train_data.index[-1], color='gray', linestyle='--', alpha=0.7,
                label='Train/Test Split')

    # Plot baseline predictions
    if not np.isnan(baseline_pred).all():
        plt.plot(test_data.index, baseline_pred, 's-', color='blue', linewidth=2,
                 label=f'Baseline Model (RMSE: {baseline_rmse:.2f})', markersize=8)

    # Plot ARIMA predictions
    if not np.isnan(arima_pred).all():
        plt.plot(test_data.index, arima_pred, '^-', color='green', linewidth=2,
                 label=f'ARIMA Model (RMSE: {arima_rmse:.2f})', markersize=8)

    plt.title(
        'Kolkata Cybercrime Forecasting: Baseline vs. ARIMA Model', fontsize=18)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cyber Crimes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Create a table with model performance metrics
    plt.figtext(0.5, 0.01,
                f"Model Performance Comparison:\n"
                f"Baseline Model - RMSE: {baseline_rmse:.2f}, MAE: {baseline_mae:.2f}\n"
                f"ARIMA Model - RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}",
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the metrics table
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to execute the analysis and create visualizations."""
    print("Loading and preprocessing data...")
    df_metros = load_and_preprocess_data()

    print("Calculating volatility metrics...")
    volatility_metrics, city_time_series, pct_change = calculate_volatility_metrics(
        df_metros)
    print("\nVolatility Analysis Results:")
    print(volatility_metrics.sort_values('cv', ascending=False))

    print("\nCreating visualizations...")

    # Visualization 1: Time series plot
    visualize_time_series(city_time_series, os.path.join(
        output_dir, 'cybercrime_trend_analysis.png'))

    # Visualization 2: Year-over-year change heatmap
    visualize_year_over_year_change(
        pct_change, os.path.join(output_dir, 'yoy_change_heatmap.png'))

    # Visualization 3: Volatility comparison bar chart
    visualize_volatility_comparison(volatility_metrics, os.path.join(
        output_dir, 'volatility_comparison.png'))

    # Visualization 4: Volatility distribution box plot
    visualize_volatility_distribution(pct_change, os.path.join(
        output_dir, 'volatility_distribution.png'))

    # Visualization 5: Crime type comparison
    visualize_crime_type_comparison(os.path.join(
        output_dir, 'crime_type_comparison.png'))

    # Model comparison
    train_and_compare_models(city_time_series, os.path.join(
        output_dir, 'model_comparison.png'))

    print("Analysis complete. All visualizations have been saved to the output directory.")


if __name__ == "__main__":
    main()
