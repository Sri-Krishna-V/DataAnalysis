#!/usr/bin/env python
# coding: utf-8

"""
Cyber Crime Analysis: Relationship between IPC 'Fraud' and IT Act 'Cheating by Personation'
This script analyzes whether states with high IPC 'Fraud' cases show proportionally higher 
IT Act 'Cheating by Personation' cases using a Proportional Odds Model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.nonparametric.smoothers_lowess import lowess

# Set style for all visualizations
plt.style.use('seaborn-v0_8-whitegrid')
colors = sns.color_palette('Set2')

# Define paths
base_path = Path(
    r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete")
output_path = base_path / "Qn 14"
os.makedirs(output_path, exist_ok=True)

# Load datasets


def load_datasets():
    """Load and prepare the IPC and IT Act datasets"""
    print("Loading datasets...")

    # Load IPC data (Fraud cases)
    ipc_file = base_path / "cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-ipc.csv"
    ipc_data = pd.read_csv(ipc_file)

    # Load IT Act data (Cheating by Personation cases)
    it_file = base_path / "cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv"
    it_data = pd.read_csv(it_file)

    print(
        f"Datasets loaded successfully. IPC data shape: {ipc_data.shape}, IT Act data shape: {it_data.shape}")

    return ipc_data, it_data


def preprocess_data(ipc_data, it_data):
    """
    Preprocess the datasets to extract relevant information:
    - Filter for IPC Fraud cases
    - Filter for IT Act Cheating by Personation cases
    - Merge datasets by state and year
    """
    print("Preprocessing data...")

    # Extract IPC Fraud cases (all subtypes)
    ipc_fraud = ipc_data[ipc_data['offence_category'] == 'Fraud'].copy()

    # Group by state and year to get total fraud cases
    ipc_fraud_total = ipc_fraud.groupby(['state', 'year'])[
        'value'].sum().reset_index()
    ipc_fraud_total.rename(columns={'value': 'ipc_fraud_cases'}, inplace=True)

    # Extract IT Act Cheating by Personation cases
    it_personation = it_data[(it_data['offence_category'] == 'Computer Related Offences') &
                             (it_data['offence_sub_category'] == 'Cheating by personation by using computer resource')].copy()

    # Rename columns for clarity
    it_personation = it_personation[['state', 'year', 'value']].copy()
    it_personation.rename(
        columns={'value': 'it_personation_cases'}, inplace=True)

    # Merge the two datasets
    merged_data = pd.merge(ipc_fraud_total, it_personation, on=[
                           'state', 'year'], how='outer')

    # Fill NaN values with 0
    merged_data.fillna(0, inplace=True)

    # Filter out 'All India' aggregates to focus on state-level analysis
    merged_data = merged_data[merged_data['state'] != 'All India']

    # Calculate population-adjusted rates (since we don't have population data, we'll use ratios)
    merged_data['total_cyber_crimes'] = merged_data['ipc_fraud_cases'] + \
        merged_data['it_personation_cases']
    merged_data['ipc_fraud_ratio'] = merged_data['ipc_fraud_cases'] / \
        merged_data['total_cyber_crimes'].replace(0, np.nan)
    merged_data['it_personation_ratio'] = merged_data['it_personation_cases'] / \
        merged_data['total_cyber_crimes'].replace(0, np.nan)

    # Fill NaN ratios with 0
    merged_data.fillna(0, inplace=True)

    print(
        f"Data preprocessing complete. Shape of merged data: {merged_data.shape}")
    return merged_data


def exploratory_data_analysis(data):
    """
    Perform exploratory data analysis:
    - Summary statistics
    - Distribution of cases by state
    - Year-wise trends
    """
    print("Performing exploratory data analysis...")

    print("\nSummary Statistics:")
    summary = data.describe()
    print(summary)

    # Top 10 states by IPC Fraud cases
    top_fraud_states = data.groupby(
        'state')['ipc_fraud_cases'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 states by IPC Fraud cases:")
    print(top_fraud_states)

    # Top 10 states by IT Act Personation cases
    top_personation_states = data.groupby(
        'state')['it_personation_cases'].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 states by IT Act Cheating by Personation cases:")
    print(top_personation_states)

    # Create a scatter plot to visualize the relationship
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=data,
        x='ipc_fraud_cases',
        y='it_personation_cases',
        hue='year',
        size='total_cyber_crimes',
        sizes=(50, 500),
        alpha=0.7
    )

    # Label the points with state names
    for i, row in data.iterrows():
        if row['ipc_fraud_cases'] > 1000 or row['it_personation_cases'] > 1000:
            plt.text(row['ipc_fraud_cases'], row['it_personation_cases'],
                     f"{row['state']}", fontsize=9)

    plt.title(
        'Relationship between IPC Fraud and IT Act Cheating by Personation Cases')
    plt.xlabel('IPC Fraud Cases')
    plt.ylabel('IT Act Cheating by Personation Cases')
    scatter_file = output_path / 'fraud_vs_personation_scatter.png'
    plt.tight_layout()
    plt.savefig(scatter_file)
    plt.close()

    return top_fraud_states, top_personation_states


def create_diverging_bar_chart(data, year, top_n=15):
    """
    Create a diverging bar chart comparing IPC Fraud and IT Act Personation cases
    for a specific year, showing top N states by total cases
    """
    print(f"Creating diverging bar chart for year {year}...")

    # Filter data for the specific year
    year_data = data[data['year'] == year].copy()

    # Get top N states by total cyber crimes
    top_states = year_data.nlargest(top_n, 'total_cyber_crimes')[
        'state'].tolist()
    year_data = year_data[year_data['state'].isin(top_states)]

    # Sort by total cases
    year_data = year_data.sort_values('total_cyber_crimes', ascending=False)

    # Create normalized versions for comparison
    total_max = year_data[['ipc_fraud_cases',
                           'it_personation_cases']].max().max()
    year_data['ipc_fraud_normalized'] = year_data['ipc_fraud_cases'] / total_max
    year_data['it_personation_normalized'] = year_data['it_personation_cases'] / total_max

    # Create the diverging bar chart
    plt.figure(figsize=(14, 10))

    # Plot IPC Fraud cases on the left (negative)
    plt.barh(year_data['state'], -year_data['ipc_fraud_normalized'],
             color=colors[0], alpha=0.8, label='IPC Fraud')

    # Plot IT Act Personation cases on the right (positive)
    plt.barh(year_data['state'], year_data['it_personation_normalized'],
             color=colors[1], alpha=0.8, label='IT Act Cheating by Personation')

    # Add labels and details
    plt.axvline(0, color='black', lw=0.5)
    plt.grid(False)
    plt.xlabel('Normalized Case Count')
    plt.title(
        f'Comparing IPC Fraud vs. IT Act Cheating by Personation Cases ({year})', fontsize=14)
    plt.legend(loc='lower right')

    # Add actual values as text
    for i, row in year_data.iterrows():
        # IPC Fraud values (left side)
        plt.text(-row['ipc_fraud_normalized'] - 0.05, i,
                 f"{int(row['ipc_fraud_cases'])}", ha='right', va='center', fontsize=9)

        # IT Act Personation values (right side)
        plt.text(row['it_personation_normalized'] + 0.05, i,
                 f"{int(row['it_personation_cases'])}", ha='left', va='center', fontsize=9)

    # Save the figure
    plt.tight_layout()
    diverging_file = output_path / f'diverging_bar_chart_{year}.png'
    plt.savefig(diverging_file)
    plt.close()


def proportional_odds_analysis(data):
    """
    Perform Proportional Odds Model analysis to examine the relationship between
    IPC Fraud cases and IT Act Cheating by Personation cases
    """
    print("Performing Proportional Odds Model analysis...")

    # Convert raw values to quartile categories for ordinal analysis
    # Add the duplicates='drop' parameter to handle duplicate bin edges
    data['ipc_fraud_quartile'] = pd.qcut(
        data['ipc_fraud_cases'], q=4, labels=False, duplicates='drop')
    data['it_personation_quartile'] = pd.qcut(data['it_personation_cases'].replace(0, np.nan).fillna(0),
                                              q=4, duplicates='drop', labels=False)

    # Prepare data for the model
    X = data['ipc_fraud_quartile'].values.reshape(-1, 1)
    y = data['it_personation_quartile']

    try:
        # Fit the proportional odds model
        model = OrderedModel(y, X, distr='logit')
        result = model.fit(method='bfgs', disp=False)

        print("\nProportional Odds Model Results Summary:")
        print(result.summary())

        # Create a visualization of the model results
        plt.figure(figsize=(10, 6))

        # Group data by IPC fraud quartiles and calculate mean IT personation quartile
        grouped = data.groupby('ipc_fraud_quartile')[
            'it_personation_quartile'].mean().reset_index()

        # Plot the relationship
        sns.barplot(x='ipc_fraud_quartile', y='it_personation_quartile',
                    data=grouped, palette='viridis')
        plt.xlabel('IPC Fraud Cases Quartile')
        plt.ylabel('IT Act Cheating by Personation Cases Quartile (Average)')
        plt.title(
            'Relationship Between Fraud Cases and Cheating by Personation Cases')
        plt.xticks(range(4), ['Low', 'Medium-Low', 'Medium-High', 'High'])

        # Add correlation coefficient
        correlation = data['ipc_fraud_quartile'].corr(
            data['it_personation_quartile'])
        plt.annotate(f'Correlation: {correlation:.2f}', xy=(0.7, 0.9), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # Save the figure
        plt.tight_layout()
        odds_file = output_path / 'proportional_odds_analysis.png'
        plt.savefig(odds_file)
        plt.close()

        return result

    except Exception as e:
        print(f"Error in proportional odds analysis: {e}")
        return None


def create_heatmap(data, year):
    """Create a heatmap showing the ratio of Fraud to Personation cases by state for a specific year"""
    print(f"Creating heatmap for year {year}...")

    # Filter data for the specific year
    year_data = data[data['year'] == year].copy()

    # Calculate the ratio of Fraud to Personation cases
    # Avoid division by zero
    year_data['ratio'] = year_data['ipc_fraud_cases'] / \
        year_data['it_personation_cases'].replace(0, np.nan)
    year_data['ratio'] = year_data['ratio'].fillna(0)

    # Limit to states with significant number of cases
    significant_cases = year_data[(year_data['ipc_fraud_cases'] > 10) | (
        year_data['it_personation_cases'] > 10)]

    # Create a pivot table suitable for a heatmap
    heatmap_data = significant_cases.set_index('state')['ratio'].reset_index()
    heatmap_data = heatmap_data.sort_values('ratio', ascending=False)

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(
        heatmap_data.set_index('state').T,
        cmap='YlOrRd',
        annot=True,
        fmt=".2f"
    )

    plt.title(
        f'Ratio of IPC Fraud to IT Act Personation Cases by State ({year})')
    plt.xlabel('State')
    plt.ylabel('')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save the figure
    heatmap_file = output_path / f'fraud_personation_ratio_heatmap_{year}.png'
    plt.savefig(heatmap_file)
    plt.close()

    # Create a choropleth map (placeholder - would require GeoJSON data)
    print("Note: A proper choropleth map would require state boundary GeoJSON data")


def create_trend_analysis_visualizations(data):
    """
    Create trend analysis visualizations showing how the relationship between
    IPC Fraud and IT Act Cheating by Personation has evolved over time
    """
    print("Creating trend analysis visualizations...")

    # Get unique years
    years = sorted(data['year'].unique())

    # Create year-wise aggregated data
    yearly_data = data.groupby('year').agg({
        'ipc_fraud_cases': 'sum',
        'it_personation_cases': 'sum'
    }).reset_index()

    # Calculate year over year growth rates
    yearly_data['ipc_fraud_growth'] = yearly_data['ipc_fraud_cases'].pct_change() * \
        100
    yearly_data['it_personation_growth'] = yearly_data['it_personation_cases'].pct_change() * \
        100

    # Visualization 1: Stacked Area Chart for Time Trends
    plt.figure(figsize=(12, 7))
    plt.stackplot(yearly_data['year'],
                  [yearly_data['ipc_fraud_cases'],
                      yearly_data['it_personation_cases']],
                  labels=['IPC Fraud Cases', 'IT Act Cheating by Personation'],
                  alpha=0.7,
                  colors=colors[:2])

    plt.title(
        'Trend of IPC Fraud and IT Act Cheating by Personation Cases Over Time', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)

    # Format y-axis with commas
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(lambda x, p: f"{int(x):,}"))

    # Annotate with total for each year
    for i, row in yearly_data.iterrows():
        total = row['ipc_fraud_cases'] + row['it_personation_cases']
        plt.text(row['year'], total + 500, f"{int(total):,}",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'trend_stacked_area_chart.png')
    plt.close()

    # Visualization 2: Growth Rate Comparison
    plt.figure(figsize=(12, 7))

    bar_width = 0.35
    # Skip first year as it has no growth rate
    x = np.arange(len(yearly_data['year'][1:]))

    plt.bar(x - bar_width/2, yearly_data['ipc_fraud_growth'][1:],
            width=bar_width, label='IPC Fraud Growth Rate', color=colors[0], alpha=0.7)
    plt.bar(x + bar_width/2, yearly_data['it_personation_growth'][1:],
            width=bar_width, label='IT Act Personation Growth Rate', color=colors[1], alpha=0.7)

    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('YoY Growth Rate (%)', fontsize=12)
    plt.title(
        'Year-over-Year Growth Rates of IPC Fraud vs. IT Act Personation Cases', fontsize=14)
    plt.xticks(x, yearly_data['year'][1:])
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (v1, v2) in enumerate(zip(yearly_data['ipc_fraud_growth'][1:], yearly_data['it_personation_growth'][1:])):
        plt.text(i - bar_width/2, v1 + (5 if v1 > 0 else -15),
                 f"{v1:.1f}%", ha='center', fontsize=9)
        plt.text(i + bar_width/2, v2 + (5 if v2 > 0 else -15),
                 f"{v2:.1f}%", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path / 'growth_rate_comparison.png')
    plt.close()

    return yearly_data


def create_regional_distribution_boxplots(data):
    """
    Create boxplots showing the distribution of IPC Fraud and IT Act Cheating 
    by Personation cases across different regions of India
    """
    print("Creating regional distribution boxplots...")

    # Define regions based on geographical proximity
    regions = {
        'North': ['Delhi', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Punjab', 'Rajasthan',
                  'Uttarakhand', 'Uttar Pradesh', 'Chandigarh', 'Ladakh'],
        'South': ['Andhra Pradesh', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Telangana', 'Puducherry', 'Lakshadweep'],
        'East': ['Bihar', 'Jharkhand', 'Odisha', 'West Bengal', 'Sikkim'],
        'West': ['Gujarat', 'Goa', 'Maharashtra', 'Dadra and Nagar Haveli and Daman and Diu'],
        'Central': ['Chhattisgarh', 'Madhya Pradesh'],
        'Northeast': ['Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland',
                      'Tripura', 'Andaman and Nicobar Islands']
    }

    # Add region column to the dataset
    data['region'] = data['state'].apply(lambda x: next(
        (k for k, v in regions.items() if x in v), 'Other'))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Boxplot for IPC Fraud cases by region
    sns.boxplot(x='region', y='ipc_fraud_cases',
                data=data, ax=ax1, palette='Blues')
    sns.swarmplot(x='region', y='ipc_fraud_cases', data=data,
                  ax=ax1, color='black', alpha=0.5, size=4)

    ax1.set_title('Distribution of IPC Fraud Cases by Region', fontsize=14)
    ax1.set_xlabel('Region of India', fontsize=12)
    ax1.set_ylabel('Number of IPC Fraud Cases', fontsize=12)
    ax1.set_yscale('log')  # Log scale for better visualization of skewed data
    ax1.grid(True, alpha=0.3, axis='y')

    # Boxplot for IT Act Personation cases by region
    sns.boxplot(x='region', y='it_personation_cases',
                data=data, ax=ax2, palette='Greens')
    sns.swarmplot(x='region', y='it_personation_cases', data=data,
                  ax=ax2, color='black', alpha=0.5, size=4)

    ax2.set_title(
        'Distribution of IT Act Cheating by Personation Cases by Region', fontsize=14)
    ax2.set_xlabel('Region of India', fontsize=12)
    ax2.set_ylabel('Number of IT Act Personation Cases', fontsize=12)
    ax2.set_yscale('log')  # Log scale for better visualization of skewed data
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / 'regional_distribution_boxplots.png')
    plt.close()

    return data


def create_quadrant_analysis(data):
    """
    Create a quadrant analysis to categorize states based on their rates of
    IPC Fraud and IT Act Cheating by Personation
    """
    print("Creating quadrant analysis visualization...")

    # Aggregate data across years to get a comprehensive view
    state_totals = data.groupby('state').agg({
        'ipc_fraud_cases': 'sum',
        'it_personation_cases': 'sum',
        'total_cyber_crimes': 'sum'
    }).reset_index()

    # Calculate the median values to create quadrants
    median_fraud = state_totals['ipc_fraud_cases'].median()
    median_personation = state_totals['it_personation_cases'].median()

    # Create quadrant labels
    state_totals['quadrant'] = 'Q3: Low Fraud, Low Personation'  # Default

    # Q1: High Fraud, High Personation
    state_totals.loc[(state_totals['ipc_fraud_cases'] >= median_fraud) &
                     (state_totals['it_personation_cases']
                      >= median_personation),
                     'quadrant'] = 'Q1: High Fraud, High Personation'

    # Q2: Low Fraud, High Personation
    state_totals.loc[(state_totals['ipc_fraud_cases'] < median_fraud) &
                     (state_totals['it_personation_cases']
                      >= median_personation),
                     'quadrant'] = 'Q2: Low Fraud, High Personation'

    # Q4: High Fraud, Low Personation
    state_totals.loc[(state_totals['ipc_fraud_cases'] >= median_fraud) &
                     (state_totals['it_personation_cases']
                      < median_personation),
                     'quadrant'] = 'Q4: High Fraud, Low Personation'

    # Create a dictionary mapping quadrants to colors
    quadrant_colors = {
        'Q1: High Fraud, High Personation': colors[0],
        'Q2: Low Fraud, High Personation': colors[1],
        'Q3: Low Fraud, Low Personation': colors[2],
        'Q4: High Fraud, Low Personation': colors[3]
    }

    # Create a scatter plot with quadrants
    plt.figure(figsize=(14, 10))

    # Draw quadrant lines
    plt.axvline(x=median_fraud, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=median_personation, color='gray', linestyle='--', alpha=0.5)

    # Plot the states
    for quadrant in quadrant_colors:
        quadrant_data = state_totals[state_totals['quadrant'] == quadrant]
        plt.scatter(quadrant_data['ipc_fraud_cases'],
                    quadrant_data['it_personation_cases'],
                    s=quadrant_data['total_cyber_crimes'] /
                    50,  # Size based on total
                    color=quadrant_colors[quadrant],
                    alpha=0.7,
                    label=quadrant)

    # Add state labels to the points
    for i, row in state_totals.iterrows():
        plt.annotate(row['state'],
                     (row['ipc_fraud_cases'], row['it_personation_cases']),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center',
                     fontsize=8)

    # Set axis labels and title
    plt.xlabel('Total IPC Fraud Cases', fontsize=12)
    plt.ylabel('Total IT Act Cheating by Personation Cases', fontsize=12)
    plt.title(
        'Quadrant Analysis: States Categorized by IPC Fraud and IT Act Personation Cases', fontsize=14)

    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'quadrant_analysis.png')
    plt.close()

    return state_totals


def create_cluster_analysis(data):
    """
    Perform cluster analysis to identify groups of states with similar patterns
    of IPC Fraud and IT Act Cheating by Personation cases
    """
    print("Performing cluster analysis...")

    # Aggregate data across years to get a comprehensive view
    state_totals = data.groupby('state').agg({
        'ipc_fraud_cases': 'sum',
        'it_personation_cases': 'sum',
        'total_cyber_crimes': 'sum'
    }).reset_index()

    # Prepare data for clustering
    features = state_totals[['ipc_fraud_cases', 'it_personation_cases']].copy()

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply K-means clustering
    n_clusters = 4  # Can be adjusted based on silhouette score or domain knowledge
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    state_totals['cluster'] = kmeans.fit_predict(features_scaled)

    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    state_totals['pca1'] = pca_result[:, 0]
    state_totals['pca2'] = pca_result[:, 1]

    # Create visualizations for the cluster analysis
    plt.figure(figsize=(14, 10))

    # Plot clusters in PCA space
    scatter = plt.scatter(state_totals['pca1'], state_totals['pca2'],
                          c=state_totals['cluster'], cmap='viridis',
                          s=state_totals['total_cyber_crimes'] / 50,
                          alpha=0.7)

    # Add state labels
    for i, row in state_totals.iterrows():
        plt.annotate(row['state'],
                     (row['pca1'], row['pca2']),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center',
                     fontsize=9)

    plt.title(
        'Cluster Analysis of States Based on Cyber Crime Patterns', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add a legend explaining the variance
    explained_variance = pca.explained_variance_ratio_
    plt.annotate(f"PC1 explains {explained_variance[0]:.2%} of variance\nPC2 explains {explained_variance[1]:.2%} of variance",
                 xy=(0.02, 0.98), xycoords='axes fraction',
                 ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

    # Add a legend for clusters
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="lower right", title="Clusters")
    plt.gca().add_artist(legend1)

    plt.tight_layout()
    plt.savefig(output_path / 'cluster_analysis_pca.png')
    plt.close()

    return state_totals


def create_advanced_correlation_visualizations(data):
    """
    Create advanced correlation visualizations beyond simple scatter plots
    to better understand the relationship between IPC Fraud and IT Act Personation
    """
    print("Creating advanced correlation visualizations...")

    # Create a correlation heatmap across all numerical columns
    plt.figure(figsize=(10, 8))

    # Select numerical columns
    numeric_cols = ['ipc_fraud_cases', 'it_personation_cases', 'total_cyber_crimes',
                    'ipc_fraud_ratio', 'it_personation_ratio']
    corr_matrix = data[numeric_cols].corr()

    # Create a custom diverging colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap",
                                             [(0, "#ff9999"), (0.5, "white"), (1, "#66b3ff")])

    # Create the heatmap
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1,
                          linewidths=0.5, fmt=".2f", cbar_kws={"shrink": 0.8})

    plt.title('Correlation Matrix of Cyber Crime Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_heatmap.png')
    plt.close()

    return corr_matrix


def main():
    """Main function to orchestrate the analysis"""
    print("Starting analysis of IPC Fraud vs IT Act Personation cases...")

    # Load and preprocess the data
    ipc_data, it_data = load_datasets()
    merged_data = preprocess_data(ipc_data, it_data)

    # Perform EDA
    top_fraud_states, top_personation_states = exploratory_data_analysis(
        merged_data)

    # Create visualizations for the most recent years
    for year in [2021, 2022]:
        create_diverging_bar_chart(merged_data, year)
        create_heatmap(merged_data, year)

    # Perform proportional odds analysis
    model_result = proportional_odds_analysis(merged_data)

    # Create new advanced visualizations
    yearly_data = create_trend_analysis_visualizations(merged_data)
    merged_data_with_regions = create_regional_distribution_boxplots(
        merged_data)
    state_categorization = create_quadrant_analysis(merged_data)
    clustered_states = create_cluster_analysis(merged_data)
    correlation_matrix = create_advanced_correlation_visualizations(
        merged_data)

    print("\nAnalysis complete. All visualizations saved to the 'Qn 14' folder.")


if __name__ == "__main__":
    main()
