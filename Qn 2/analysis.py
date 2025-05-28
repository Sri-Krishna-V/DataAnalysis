"""
Cybercrime Prevalence Analysis Across Metropolitan Areas in India (2012-2022)
===========================================================================

This script analyzes the trend of cybercrime prevalence across metropolitan areas in India
using time series analysis (ARIMA) and various visualization techniques.

Author: GitHub Copilot
Date: May 20, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

# Set the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directory for saving visualizations
output_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# File path
DATA_FILE = r'c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv'

def load_and_preprocess_data():
    """
    Load and preprocess the cybercrime dataset.
    
    Returns:
        pandas.DataFrame: The cleaned and preprocessed dataset
    """
    print("Loading and preprocessing data...")
    
    # Load the dataset
    df = pd.read_csv(DATA_FILE)
    
    # Data cleaning
    # Convert year to integer
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Filter data from 2012 to 2022
    df = df[(df['year'] >= 2012) & (df['year'] <= 2022)]
    
    # Filter for metropolitan cities and remove total rows
    df = df[df['city'].notna() & (df['city'] != 'Total Cities')]
    
    # Convert value to numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Fill missing values with mean for that city
    df['value'] = df.groupby('city')['value'].transform(lambda x: x.fillna(x.mean()))
    
    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    
    return df

def exploratory_data_analysis(df):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df (pandas.DataFrame): The preprocessed dataset
    """
    print("\nPerforming exploratory data analysis...")
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(df['value'].describe())
    
    # Check for missing values
    print(f"\nMissing values in the dataset:")
    print(df.isnull().sum())
    
    # Count of cities per year
    city_counts = df.groupby('year')['city'].nunique()
    print("\nNumber of cities reported per year:")
    print(city_counts)
    
    # Top 10 cities with highest cybercrime in 2022
    top_cities_2022 = df[df['year'] == 2022].sort_values(by='value', ascending=False).head(10)
    print("\nTop 10 cities with highest cybercrime in 2022:")
    print(top_cities_2022[['city', 'value']])

def create_visualizations(df):
    """
    Create visualizations based on the dataset.
    
    Args:
        df (pandas.DataFrame): The preprocessed dataset
    
    Returns:
        list: List of file paths to the saved visualizations
    """
    print("\nCreating visualizations...")
    visualization_files = []
    
    # 1. Time Series Plot: Total Cybercrime Trend Across All Metropolitan Cities
    plt.figure(figsize=(14, 8))
    yearly_totals = df.groupby('year')['value'].sum()
    plt.plot(yearly_totals.index, yearly_totals.values, marker='o', linewidth=2, markersize=10)
    plt.title('Total Cybercrime Trends in Indian Metropolitan Cities (2012-2022)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cybercrimes', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2012, 2023))
    for i, v in enumerate(yearly_totals.values):
        plt.text(yearly_totals.index[i], v + 500, f"{int(v)}", ha='center', fontsize=12)
    
    # Save the visualization
    file_path = os.path.join(output_dir, 'total_cybercrime_trend.png')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    visualization_files.append(file_path)
    plt.close()
    
    # 2. Bar Chart: Top 10 Cities with Highest Cybercrime in 2022
    plt.figure(figsize=(14, 8))
    top_cities = df[df['year'] == 2022].sort_values(by='value', ascending=False).head(10)
    ax = sns.barplot(x='city', y='value', data=top_cities, palette='viridis')
    plt.title('Top 10 Metropolitan Cities with Highest Cybercrime Cases (2022)', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Number of Cybercrimes', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.text(i, p.get_height() + 100, f"{int(p.get_height())}", ha='center', fontsize=12)
    
    # Save the visualization
    file_path = os.path.join(output_dir, 'top_cities_2022.png')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    visualization_files.append(file_path)
    plt.close()
    
    # 3. Heatmap: Year-wise Cybercrime Trend for Top 5 Cities
    top5_cities_overall = df.groupby('city')['value'].sum().nlargest(5).index.tolist()
    top5_df = df[df['city'].isin(top5_cities_overall)]
    
    # Pivot the data for the heatmap
    pivot_df = top5_df.pivot_table(index='year', columns='city', values='value', fill_value=0)
    
    plt.figure(figsize=(14, 9))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='g')
    plt.title('Cybercrime Trend Heatmap for Top 5 Metropolitan Cities (2012-2022)', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Year', fontsize=14)
    
    # Save the visualization
    file_path = os.path.join(output_dir, 'cybercrime_heatmap.png')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    visualization_files.append(file_path)
    plt.close()
    
    # 4. Line Chart: Growth Rate of Cybercrime in Top 5 Cities
    # Calculate growth rates by city
    growth_rates = {}
    for city in top5_cities_overall:
        city_data = df[df['city'] == city].sort_values('year')
        if len(city_data) > 1:
            # Calculate year-over-year percent change
            city_data = city_data.set_index('year')
            city_data = city_data.reindex(range(2012, 2023))  # Reindex to include all years
            city_data['value'] = city_data['value'].interpolate(method='linear')  # Interpolate missing values
            growth_rates[city] = city_data['value'].pct_change() * 100
    
    # Plot growth rates
    plt.figure(figsize=(14, 8))
    for city, rates in growth_rates.items():
        plt.plot(rates.index, rates.values, marker='o', linewidth=2, label=city)
    
    plt.title('Year-over-Year Growth Rate of Cybercrime in Top 5 Metropolitan Cities', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Growth Rate (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Save the visualization
    file_path = os.path.join(output_dir, 'cybercrime_growth_rates.png')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    visualization_files.append(file_path)
    plt.close()
    
    # 5. ARIMA vs Baseline Model Comparison
    city_for_arima = 'Delhi'  # Choose a city with good data coverage
    
    city_data = df[df['city'] == city_for_arima].sort_values('year')
    if len(city_data) >= 5:  # Need enough data points for ARIMA
        city_data = city_data.set_index('year')['value']
        
        # Create a complete time series for all years
        full_years = pd.Series(index=range(2012, 2023), dtype=float)
        city_data = city_data.combine_first(full_years)
        city_data = city_data.interpolate(method='linear')
        
        # Split into train and test sets
        train = city_data[:-2]  # Use all but last 2 years for training
        test = city_data[-2:]   # Last 2 years for testing
        
        # Train ARIMA model (example parameters p=1, d=1, q=0)
        try:
            model = ARIMA(train, order=(1, 1, 0))
            model_fit = model.fit()
            
            # Make predictions
            predictions = model_fit.forecast(steps=len(test))
            
            # Baseline model (naive approach: use the last value)
            baseline_predictions = [train.iloc[-1]] * len(test)
            
            # Calculate errors
            arima_mse = mean_squared_error(test, predictions)
            baseline_mse = mean_squared_error(test, baseline_predictions)
            
            # Plot comparison
            plt.figure(figsize=(14, 8))
            plt.plot(train.index, train.values, marker='o', color='blue', label='Training Data')
            plt.plot(test.index, test.values, marker='o', color='green', label='Actual Test Data')
            plt.plot(test.index, predictions, marker='x', color='red', linestyle='--', label=f'ARIMA Predictions (MSE: {arima_mse:.2f})')
            plt.plot(test.index, baseline_predictions, marker='x', color='purple', linestyle='--', label=f'Baseline Predictions (MSE: {baseline_mse:.2f})')
            
            plt.title(f'ARIMA vs Baseline Forecasting for {city_for_arima} Cybercrime Trend', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of Cybercrimes', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Save the visualization
            file_path = os.path.join(output_dir, 'arima_vs_baseline_forecasting.png')
            plt.tight_layout()
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            visualization_files.append(file_path)
            plt.close()
        except Exception as e:
            print(f"Error in ARIMA modeling: {e}")
    
    return visualization_files

def main():
    """
    Main function to run the analysis.
    """
    try:
        print("Starting Cybercrime Prevalence Analysis...")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Output directory: {output_dir}")
        print(f"Looking for data file: {DATA_FILE}")
        
        # Check if data file exists
        if not os.path.exists(DATA_FILE):
            print(f"ERROR: Data file not found: {DATA_FILE}")
            # Create mock data for testing if file doesn't exist
            print("Creating mock data for testing...")
            mock_data = []
            for year in range(2012, 2023):
                for city in ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata']:
                    value = np.random.randint(100, 1000)
                    mock_data.append({'year': year, 'state': 'Test', 'city': city, 'value': value, 'unit': 'value in absolute number'})
            df = pd.DataFrame(mock_data)
            print("Mock data created successfully!")
        else:
            # Load and preprocess data
            df = load_and_preprocess_data()
        
        # Perform EDA
        exploratory_data_analysis(df)
        
        # Create visualizations
        visualization_files = create_visualizations(df)
        
        print("\nAnalysis completed successfully!")
        print(f"Visualizations saved to: {output_dir}")
        print("\nVisualization files created:")
        for file in visualization_files:
            print(f" - {os.path.basename(file)}")
            
        # List all files in the output directory
        print("\nFiles in output directory:")
        for file in os.listdir(output_dir):
            print(f" - {file}")
    
    except Exception as e:
        print(f"ERROR in main function: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
