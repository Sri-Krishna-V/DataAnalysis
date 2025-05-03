#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import warnings
from matplotlib.ticker import FuncFormatter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import json
import matplotlib.colors as mcolors

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib plots - using default style instead of seaborn
plt.style.use('default')
colors = sns.color_palette('viridis', 10)

# Function to load data


def load_data():
    """
    Load and prepare the dataset for analysis
    """
    # Path to the data file from the current directory - using absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(
        parent_dir, "cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv")

    print(f"Loading data from: {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Clean and prepare data
    df = df.rename(columns={'offence_category': 'crime_type'})

    # Print basic info about the data
    print("Dataset shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    print("\nNull Values:", df.isnull().sum().sum())

    return df

# Function to filter data for fraud-related crimes


def filter_fraud_data(df):
    """
    Extract financial fraud-related cybercrimes from the dataset
    """
    # Filter for fraud or illegal gain
    fraud_df = df[df['crime_type'] == 'Fraud or Illegal Gain'].copy()

    # Filter for years 2020-2022
    fraud_df = fraud_df[(fraud_df['year'] >= 2020) &
                        (fraud_df['year'] <= 2022)]

    # Print information about the filtered data
    print("\nFinancial Fraud Data Shape:", fraud_df.shape)
    print("\nYear distribution:")
    print(fraud_df['year'].value_counts().sort_index())

    return fraud_df

# Function for exploratory data analysis


def exploratory_analysis(df, fraud_df):
    """
    Perform exploratory data analysis on the crime data
    """
    print("\n--- Exploratory Data Analysis ---")

    # Calculate total crimes and fraud percentage
    total_crimes_by_year = df.groupby('year')['value'].sum()
    fraud_by_year = fraud_df.groupby('year')['value'].sum()

    fraud_percentage = (fraud_by_year / total_crimes_by_year) * 100

    print("\nPercentage of Financial Fraud in Total Cybercrimes:")
    for year, percentage in fraud_percentage.items():
        print(f"Year {year}: {percentage:.2f}%")

    # Top 10 states with highest fraud cases in 2022
    top_states_2022 = fraud_df[fraud_df['year'] == 2022].groupby('state')[
        'value'].sum().nlargest(10)
    print("\nTop 10 States with Highest Fraud Cases in 2022:")
    print(top_states_2022)

    return total_crimes_by_year, fraud_by_year

# Function to create time series plot of fraud cases by year


def plot_fraud_trend(fraud_df):
    """
    Create a time series plot showing the trend of financial fraud cybercrimes from 2020-2022
    """
    # Aggregate fraud data by year
    yearly_fraud = fraud_df.groupby('year')['value'].sum()

    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_fraud.index, yearly_fraud.values,
             marker='o', linewidth=2, color=colors[0])
    for year, value in zip(yearly_fraud.index, yearly_fraud.values):
        plt.text(year, value + 500, f'{value:,.0f}',
                 ha='center', fontweight='bold')

    # Add styling and labels
    plt.title('Financial Fraud Cybercrimes in India (2020-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(yearly_fraud.index)

    # Format y-axis with thousand separators
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add annotations
    plt.annotate('32.1% increase\nfrom 2020 to 2022', xy=(2021, yearly_fraud[2021]),
                 xytext=(2021, yearly_fraud[2021] - 4000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('financial_fraud_trend.png', dpi=300)
    plt.close()

    return 'financial_fraud_trend.png'

# Function to create bar chart of top states with fraud cases


def plot_top_states_comparison(fraud_df):
    """
    Create a bar chart comparing top states with fraud cases across years
    """
    # Get top 10 states based on 2022 data
    top_states_2022 = fraud_df[fraud_df['year'] == 2022].groupby(
        'state')['value'].sum().nlargest(10).index.tolist()

    # Filter data for top states
    top_states_df = fraud_df[fraud_df['state'].isin(top_states_2022)]

    # Pivot data for plotting
    pivot_df = top_states_df.pivot_table(
        index='state', columns='year', values='value', aggfunc='sum')

    # Sort by 2022 values
    pivot_df = pivot_df.reindex(top_states_df[top_states_df['year'] == 2022].groupby(
        'state')['value'].sum().nlargest(10).index)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set width for bars
    bar_width = 0.25
    r1 = np.arange(len(pivot_df))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Create bars
    ax.bar(r1, pivot_df[2020], width=bar_width, label='2020',
           color=colors[0], edgecolor='black', linewidth=0.5)
    ax.bar(r2, pivot_df[2021], width=bar_width, label='2021',
           color=colors[3], edgecolor='black', linewidth=0.5)
    ax.bar(r3, pivot_df[2022], width=bar_width, label='2022',
           color=colors[6], edgecolor='black', linewidth=0.5)

    # Add labels and styling
    ax.set_title(
        'Financial Fraud Cybercrimes: Top 10 States (2020-2022)', fontsize=16)
    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    ax.set_xticks([r + bar_width for r in range(len(pivot_df))])
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Format y-axis with thousand separators
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add value labels on top of bars
    for i, year in enumerate([2020, 2021, 2022]):
        for j, value in enumerate(pivot_df[year]):
            ax.text(j + i*bar_width, value + 50, f'{int(value):,}',
                    ha='center', va='bottom', fontsize=8, rotation=90, fontweight='bold')

    plt.tight_layout()
    plt.savefig('top_states_comparison.png', dpi=300)
    plt.close()

    return 'top_states_comparison.png'

# Function to create stacked area plot showing year-wise distribution


def plot_state_fraud_distribution(fraud_df):
    """
    Create a stacked area plot showing distribution of fraud cases across states by year
    """
    # Exclude 'All India' from data first
    filtered_fraud_df = fraud_df[fraud_df['state'] != 'All India']

    # Get top 5 states based on total cases over 3 years (excluding 'All India')
    top_states = filtered_fraud_df.groupby(
        'state')['value'].sum().nlargest(5).index.tolist()

    other_states = [state for state in filtered_fraud_df['state'].unique()
                    if state not in top_states]

    # Create a pivot table for better data handling
    pivot_df = filtered_fraud_df.pivot_table(
        index='year', columns='state', values='value', aggfunc='sum'
    ).fillna(0)

    # Select top states columns
    top_states_data = pivot_df[top_states]

    # Create 'Other States' column
    other_states_data = pivot_df[other_states].sum(axis=1)

    # Combine data for plotting
    plot_data = pd.concat(
        [top_states_data, other_states_data.rename('Other States')], axis=1)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create stacked area plot with more robust approach
    years = plot_data.index
    y_data = [plot_data[col].values for col in plot_data.columns]
    labels = plot_data.columns

    plt.stackplot(years, y_data, labels=labels,
                  alpha=0.8, colors=colors[:len(labels)])

    # Add styling
    plt.title(
        'Distribution of Financial Fraud Cybercrimes Across Major States (2020-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(years)

    # Format y-axis
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f'{int(x):,}'))

    plt.tight_layout()
    plt.savefig('state_fraud_distribution.png', dpi=300)
    plt.close()

    return 'state_fraud_distribution.png'

# Function to create choropleth map of fraud cases


def plot_fraud_choropleth(fraud_df):
    """
    Create a choropleth map showing the distribution of fraud cases across India
    """
    # Filter data for 2022
    fraud_2022 = fraud_df[fraud_df['year'] == 2022].copy()

    # Drop 'All India' as it's not a geographical entity
    fraud_2022 = fraud_2022[fraud_2022['state'] != 'All India']

    # Aggregate by state
    state_fraud = fraud_2022.groupby('state')['value'].sum().reset_index()

    # Create state name mapping for consistent naming with geojson
    state_name_mapping = {
        'Andaman & Nicobar Islands': 'Andaman & Nicobar',
        'Dadra and Nagar Haveli and Daman and Diu': 'Dadra & Nagar Haveli',
        'Jammu & Kashmir': 'Jammu & Kashmir',
        'NCT of Delhi': 'Delhi',
        'Telengana': 'Telangana'
    }

    # Apply mapping for consistency
    state_fraud['state'] = state_fraud['state'].replace(state_name_mapping)

    try:
        # Use a more reliable geojson URL
        geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"

        # Create choropleth figure
        fig = px.choropleth(
            state_fraud,
            geojson=geojson_url,
            featureidkey='properties.ST_NM',
            locations='state',
            color='value',
            color_continuous_scale='Viridis',
            title='Financial Fraud Cybercrimes Across India (2022)',
            labels={'value': 'Number of Cases'}
        )

        # Update layout for better visualization
        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title="Number of Cases",
                tickformat=",d"
            ),
            title={
                'text': 'Financial Fraud Cybercrimes Across India (2022)',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )

        # Save as HTML file since plotly figures are interactive
        fig.write_html('fraud_choropleth_2022.html')

        # Also save as PNG for static viewing
        fig.write_image('fraud_choropleth_2022.png', width=1200, height=800)

        return 'fraud_choropleth_2022.html'

    except Exception as e:
        print(f"Could not create choropleth with Plotly: {e}")

        # Create an alternative visualization using matplotlib
        plt.figure(figsize=(12, 10))

        # Sort for better visualization
        state_fraud = state_fraud.sort_values('value', ascending=False)

        # Create horizontal bar chart as alternative
        ax = sns.barplot(x='value', y='state',
                         data=state_fraud.head(20), palette='viridis')

        # Add value labels
        for i, v in enumerate(state_fraud.head(20)['value']):
            ax.text(v + 10, i, f'{int(v):,}', va='center')

        plt.title(
            'Top 20 States by Financial Fraud Cybercrimes (2022)', fontsize=16)
        plt.xlabel('Number of Cases', fontsize=12)
        plt.ylabel('State', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')

        # Format x-axis
        plt.gca().xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{int(x):,}'))

        plt.tight_layout()
        plt.savefig('fraud_by_state_2022.png', dpi=300)
        plt.close()

        return 'fraud_by_state_2022.png'

# Function to create bubble plot of fraud cases relative to population


def plot_fraud_bubble(fraud_df):
    """
    Create a bubble plot comparing 2021 and 2022 fraud cases for top states
    """
    # Get data for 2021 and 2022
    fraud_2021 = fraud_df[fraud_df['year'] == 2021].copy()
    fraud_2022 = fraud_df[fraud_df['year'] == 2022].copy()

    # Drop 'All India'
    fraud_2021 = fraud_2021[fraud_2021['state'] != 'All India']
    fraud_2022 = fraud_2022[fraud_2022['state'] != 'All India']

    # Aggregate by state
    state_fraud_2021 = fraud_2021.groupby('state')['value'].sum().reset_index()
    state_fraud_2022 = fraud_2022.groupby('state')['value'].sum().reset_index()

    # Merge data for 2021 and 2022
    merged_data = pd.merge(state_fraud_2021, state_fraud_2022,
                           on='state', suffixes=('_2021', '_2022'))

    # Calculate growth percentage with error handling for zero values
    merged_data['growth'] = merged_data.apply(
        lambda row: 0 if row['value_2021'] == 0 else
        ((row['value_2022'] - row['value_2021']) /
         max(1, row['value_2021'])) * 100,
        axis=1
    )

    # Create bubble plot
    plt.figure(figsize=(14, 10))

    # Filter for top 15 states by 2022 cases for better visualization
    top_states = merged_data.nlargest(15, 'value_2022')

    # Normalize bubble sizes for better visualization
    size_factor = 20
    min_size = 100

    # Create scatter plot with normalized bubble sizes
    scatter = plt.scatter(
        top_states['value_2021'],
        top_states['value_2022'],
        s=np.abs(top_states['growth']) * size_factor +
        min_size,  # Adjusted size calculation
        c=top_states['growth'],
        cmap='coolwarm',  # Red for negative, Blue for positive
        alpha=0.7,
        edgecolors='black',
        linewidths=1
    )

    # Add diagonal line for reference (y=x)
    max_val = max(top_states['value_2021'].max(),
                  top_states['value_2022'].max())
    plt.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.5)

    # Label points with state names and growth percentage
    for i, row in top_states.iterrows():
        # Position label based on whether point is above or below diagonal
        if row['value_2022'] > row['value_2021']:
            # Point above diagonal
            xytext_offset = (5, 10)
        else:
            # Point below diagonal
            xytext_offset = (5, -15)

        plt.annotate(
            f"{row['state']} ({row['growth']:.1f}%)",
            (row['value_2021'], row['value_2022']),
            xytext=xytext_offset,
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3",
                      fc="white", ec="gray", alpha=0.8)
        )

    # Add styling
    plt.title(
        'Financial Fraud Cybercrimes: 2021 vs. 2022 (Top 15 States)', fontsize=16)
    plt.xlabel('Number of Cases in 2021', fontsize=12)
    plt.ylabel('Number of Cases in 2022', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add colorbar for growth percentage
    cbar = plt.colorbar(scatter)
    cbar.set_label('Growth Rate (%)', rotation=270, labelpad=20)

    # Format axes with thousand separators
    plt.gca().xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add text annotation explaining the plot
    plt.figtext(0.5, 0.01, "Bubble size represents the magnitude of change from 2021 to 2022.\n"
                "Points above diagonal line indicate increase in cases, below indicate decrease.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('fraud_bubble_plot_2021_vs_2022.png', dpi=300)
    plt.close()

    return 'fraud_bubble_plot_2021_vs_2022.png'

# Function to perform ARIMA time series forecasting


def perform_arima_forecast(fraud_df):
    """
    Perform ARIMA time series forecasting for top 5 states
    """
    # Get top 5 states by total fraud cases
    top_5_states = fraud_df.groupby(
        'state')['value'].sum().nlargest(5).index.tolist()

    # Filter data for top 5 states
    top_states_data = fraud_df[fraud_df['state'].isin(top_5_states)]

    # Set up the figure
    plt.figure(figsize=(14, 10))

    # Colors for different states
    state_colors = dict(zip(top_5_states, colors[:5]))

    # Plot actual data and forecasts for each state
    for i, state in enumerate(top_5_states):
        state_data = top_states_data[top_states_data['state'] == state]
        state_yearly = state_data.groupby('year')['value'].sum()

        # Plot actual data points
        plt.plot(state_yearly.index, state_yearly.values, 'o-', label=f"{state} (Actual)",
                 color=state_colors[state], linewidth=2, markersize=8)

        # Fit ARIMA model
        try:
            # Simple forecasting based on limited data points
            # Using a simple model due to limited historical data
            model = ARIMA(state_yearly.values, order=(1, 1, 0))
            model_fit = model.fit()

            # Generate forecast for next 2 years
            forecast = model_fit.forecast(steps=2)
            forecast_years = [2023, 2024]

            # Plot forecast
            plt.plot(forecast_years, forecast, '--', color=state_colors[state], linewidth=2,
                     alpha=0.7, label=f"{state} (Forecast)")

            # Add forecast points
            plt.scatter(forecast_years, forecast, marker='*', s=100, color=state_colors[state],
                        edgecolor='black', linewidth=1, zorder=5)

            # Add forecast values as text
            for year, value in zip(forecast_years, forecast):
                plt.text(year, value + max(state_yearly.values) * 0.05, f'{int(value):,}',
                         ha='center', color=state_colors[state], fontweight='bold', fontsize=9)

        except Exception as e:
            print(f"Could not fit ARIMA model for {state}: {e}")

    # Add styling
    plt.title(
        'ARIMA Forecast of Financial Fraud Cybercrimes for Top 5 States (2023-2024)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Set x-axis ticks
    plt.xticks(list(range(2020, 2025)))

    # Format y-axis
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add shaded region for forecast period
    plt.axvspan(2022.5, 2024.5, color='gray',
                alpha=0.1, label='Forecast Period')

    # Add vertical line separating actual from forecast
    plt.axvline(x=2022.5, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Add text annotation
    plt.figtext(0.5, 0.01, "Forecast based on ARIMA(1,1,0) model using limited historical data.\n"
                "Due to limited data points, forecasts should be interpreted with caution.",
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('arima_forecast_top_5_states.png', dpi=300)
    plt.close()

    return 'arima_forecast_top_5_states.png'

# Main function to execute all analysis


def main():
    """
    Main function to execute the complete analysis
    """
    print("\n===== Financial Fraud-related Cybercrime Analysis (2020-2022) =====")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    # Load data
    df = load_data()

    # Filter data for financial fraud
    fraud_df = filter_fraud_data(df)

    # Perform exploratory analysis
    total_crimes_by_year, fraud_by_year = exploratory_analysis(df, fraud_df)

    # Create visualizations
    print("\n--- Creating Visualizations ---")

    # 1. Time series trend of fraud cases
    trend_chart = plot_fraud_trend(fraud_df)
    print(f"Created visualization: {trend_chart}")

    # 2. Bar chart comparing top states
    bar_chart = plot_top_states_comparison(fraud_df)
    print(f"Created visualization: {bar_chart}")

    # 3. Stacked area plot of state distribution
    area_chart = plot_state_fraud_distribution(fraud_df)
    print(f"Created visualization: {area_chart}")

    # 4. Choropleth map of fraud cases
    choropleth = plot_fraud_choropleth(fraud_df)
    print(f"Created visualization: {choropleth}")

    # 5. Bubble plot comparing 2021 and 2022
    bubble_chart = plot_fraud_bubble(fraud_df)
    print(f"Created visualization: {bubble_chart}")

    # 6. ARIMA forecasting for top states
    forecast_chart = perform_arima_forecast(fraud_df)
    print(f"Created visualization: {forecast_chart}")

    print("\n===== Analysis Complete =====")
    print("All visualizations saved in the current directory.")


if __name__ == "__main__":
    main()
