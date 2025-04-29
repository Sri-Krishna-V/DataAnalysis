#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of Cheating-by-Personation vs Identity Theft Crimes in India (2018-2022)
This script analyzes the ratio between cheating-by-personation and identity theft crimes
across Indian states from 2018 to 2022 using regression analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os
from matplotlib import colormaps

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create a consistent color palette for states
state_colors = {}

# Define file paths
DATA_PATH = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv"
OUTPUT_PATH = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\Qn 8"


def load_data():
    """Load and preprocess the cyber crime dataset."""
    # Read the dataset
    df = pd.read_csv(DATA_PATH)

    # Filter for only identity theft and cheating by personation crimes
    filtered_df = df[
        ((df['offence_category'] == 'Computer Related Offences') &
         ((df['offence_sub_category'] == 'Identity Theft') |
          (df['offence_sub_category'] == 'Cheating by personation by using computer resource')))
    ]

    # Convert to wide format for easier analysis
    pivot_df = filtered_df.pivot_table(
        index=['year', 'state'],
        columns='offence_sub_category',
        values='value',
        aggfunc='sum'
    ).reset_index()

    # Handle missing values with 0
    pivot_df = pivot_df.fillna(0)

    # Calculate the ratio of cheating by personation to identity theft
    # Adding small epsilon to prevent division by zero
    epsilon = 1e-10
    pivot_df['ratio'] = pivot_df['Cheating by personation by using computer resource'] / \
        (pivot_df['Identity Theft'] + epsilon)

    # Filter for years 2018-2022
    pivot_df = pivot_df[pivot_df['year'] >= 2018]

    # Sort by year
    pivot_df = pivot_df.sort_values(['year', 'state'])

    return pivot_df


def analyze_data(data):
    """Perform basic exploratory data analysis."""
    print("Data shape:", data.shape)
    print("\nSummary statistics for Cheating by personation:")
    print(data['Cheating by personation by using computer resource'].describe())
    print("\nSummary statistics for Identity Theft:")
    print(data['Identity Theft'].describe())
    print("\nSummary statistics for Ratio:")
    print(data['ratio'].describe())

    # Filter out extreme ratios for better visualization
    data_filtered = data[data['ratio'] < 100]  # Remove extreme outliers
    return data_filtered


def assign_state_colors(data):
    """Create a consistent color mapping for states."""
    global state_colors
    unique_states = data['state'].unique()
    cmap = colormaps['tab20']
    colors = [cmap(i % 20) for i in range(len(unique_states))]

    state_colors = {state: color for state,
                    color in zip(unique_states, colors)}
    return state_colors


def plot_bubble_chart(data, year1, year2, title=None):
    """Create bubble chart comparing two years with state labels."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define epsilon to prevent division by zero
    epsilon = 1e-10

    data_year1 = data[data['year'] == year1]
    data_year2 = data[data['year'] == year2]

    # Merge the data for the two years
    merged_data = pd.merge(
        data_year1,
        data_year2,
        on='state',
        suffixes=(f'_{year1}', f'_{year2}')
    )

    # Filter for states with some meaningful data
    merged_data = merged_data[
        (merged_data[f'Identity Theft_{year1}'] > 0) |
        (merged_data[f'Identity Theft_{year2}'] > 0) |
        (merged_data[f'Cheating by personation by using computer resource_{year1}'] > 0) |
        (merged_data[f'Cheating by personation by using computer resource_{year2}'] > 0)
    ]

    # Create bubble chart
    for _, row in merged_data.iterrows():
        state = row['state']
        # +1 to handle log scale with zeros
        x = row[f'Identity Theft_{year1}'] + 1
        y = row[f'Identity Theft_{year2}'] + 1
        size1 = row[f'Cheating by personation by using computer resource_{year1}'] + 1
        size2 = row[f'Cheating by personation by using computer resource_{year2}'] + 1
        ratio_change = (row[f'ratio_{year2}'] /
                        (row[f'ratio_{year1}'] + epsilon)) - 1

        color = state_colors.get(state, 'blue')

        # Use size of bubble to represent cheating by personation cases
        size = (size1 + size2) * 10
        ax.scatter(x, y, s=size, alpha=0.6, color=color,
                   edgecolor='black', linewidth=1)

        # Add state labels to bubbles
        ax.annotate(state, (x, y), fontsize=8)

    ax.set_xlabel(f'Identity Theft Cases ({year1})', fontsize=12)
    ax.set_ylabel(f'Identity Theft Cases ({year2})', fontsize=12)

    # Add a line where x = y
    max_val = max(merged_data[f'Identity Theft_{year1}'].max(),
                  merged_data[f'Identity Theft_{year2}'].max())
    ax.plot([0, max_val*1.1], [0, max_val*1.1], 'k--', alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Identity Theft Cases: {year1} vs {year2}\n'
                     f'(Bubble size represents Cheating by Personation cases)', fontsize=14)

    plt.xscale('log')
    plt.yscale('log')

    # Add grid lines
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'identity_theft_comparison_{year1}_vs_{year2}.png'),
                dpi=300, bbox_inches='tight')
    return fig


def plot_ratio_heatmap(data):
    """Create heatmap showing the ratio change across years."""
    # Filter for top states by total number of cases
    top_states = data.groupby('state')[
        ['Cheating by personation by using computer resource', 'Identity Theft']].sum().sum(axis=1).nlargest(15).index
    filtered_data = data[data['state'].isin(top_states)]

    # Pivot data to create year vs state matrix of ratios
    ratio_matrix = filtered_data.pivot_table(
        index='state',
        columns='year',
        values='ratio'
    )

    # Calculate percentage change from year to year
    pct_change = pd.DataFrame()
    years = sorted(filtered_data['year'].unique())
    for i in range(1, len(years)):
        year_prev = years[i-1]
        year_curr = years[i]
        col_name = f"{year_prev}-{year_curr}"
        pct_change[col_name] = (
            ratio_matrix[year_curr] - ratio_matrix[year_prev]) / ratio_matrix[year_prev] * 100

    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = pct_change.isnull()

    # Use a diverging colormap to show increase/decrease
    ax = sns.heatmap(pct_change, annot=True, fmt=".1f", cmap="RdBu_r",
                     mask=mask, center=0, linewidths=.5, cbar_kws={'label': '% Change'})

    plt.title(
        "Year-on-Year Percentage Change in Ratio of\nCheating-by-Personation to Identity Theft", fontsize=14)
    plt.ylabel("State", fontsize=12)
    plt.xlabel("Year Range", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'ratio_percentage_change_heatmap.png'),
                dpi=300, bbox_inches='tight')
    return ax


def perform_regression_analysis(data):
    """Perform regression analysis on the relationship between identity theft and cheating by personation."""
    # Prepare data for regression
    X = data['Identity Theft'].values.reshape(-1, 1)
    y = data['Cheating by personation by using computer resource'].values

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    # Create scatter plot with regression line
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.6, s=50)

    # Add regression line
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='red', linewidth=2)

    # Add labels and title
    plt.xlabel('Identity Theft Cases', fontsize=12)
    plt.ylabel('Cheating by Personation Cases', fontsize=12)
    plt.title(f'Linear Regression: Identity Theft vs. Cheating by Personation\n'
              f'R² = {r2:.3f}, RMSE = {rmse:.3f}, Slope = {model.coef_[0]:.3f}',
              fontsize=14)

    # Add equation text
    equation = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
    plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'identity_theft_personation_regression.png'),
                dpi=300, bbox_inches='tight')

    return {
        'r2': r2,
        'rmse': rmse,
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }


def plot_ratio_trends(data):
    """Plot trends of the cheating-by-personation to identity theft ratio across years."""
    # Get top 5 states by average ratio
    top_states = data.groupby('state')['ratio'].mean().nlargest(5).index

    # Filter for these states
    top_data = data[data['state'].isin(top_states)]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each state's trend
    for state in top_states:
        state_data = top_data[top_data['state'] == state]
        ax.plot(state_data['year'], state_data['ratio'], marker='o',
                linewidth=2, label=state, color=state_colors.get(state, None))

    # Add labels and title
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(
        'Ratio of Cheating-by-Personation to Identity Theft', fontsize=12)
    ax.set_title('Trends in Cheating-by-Personation to Identity Theft Ratio\n'
                 'Top 5 States by Average Ratio (2018-2022)', fontsize=14)

    # Add legend
    ax.legend(loc='best', fontsize=10)

    # Customize grid
    ax.grid(True, alpha=0.3)

    # Format y-axis to reduce extreme values
    ax.set_yscale('symlog')  # Symmetric log scale for better visualization

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'ratio_trends_top_states.png'),
                dpi=300, bbox_inches='tight')
    return fig


def create_state_choropleth(data, year):
    """Create a choropleth map showing the ratio across Indian states."""
    # This function requires additional packages like geopandas
    # and a shapefile of India states. For now, we'll create a
    # simpler visualization.

    # Get data for the specific year
    year_data = data[data['year'] == year].copy()

    # Sort by ratio
    year_data = year_data.sort_values('ratio', ascending=False)

    # Create a bar chart instead of a choropleth
    plt.figure(figsize=(14, 10))

    # Filter out extreme values for better visualization
    year_data = year_data[year_data['ratio'] < 50]

    # Select states with non-zero values
    year_data = year_data[(year_data['Cheating by personation by using computer resource'] > 0) |
                          (year_data['Identity Theft'] > 0)]

    bars = plt.bar(year_data['state'], year_data['ratio'],
                   color=[state_colors.get(state, 'blue') for state in year_data['state']])

    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('Ratio (Cheating-by-Personation / Identity Theft)', fontsize=12)
    plt.title(
        f'Ratio of Cheating-by-Personation to Identity Theft by State ({year})', fontsize=14)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}',
                 ha='center', va='bottom', rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'ratio_by_state_{year}.png'),
                dpi=300, bbox_inches='tight')
    return plt.gcf()


def main():
    # Load and preprocess the data
    data = load_data()

    # Analyze the data
    filtered_data = analyze_data(data)

    # Assign consistent colors for states
    assign_state_colors(filtered_data)

    # Create visualizations
    print("Creating visualizations...")

    # 1. Bubble chart comparing two years
    plot_bubble_chart(filtered_data, 2021, 2022,
                      "Identity Theft Cases: 2021 vs 2022\n(Bubble size represents Cheating by Personation cases)")

    # 2. Ratio heatmap showing percentage change year over year
    plot_ratio_heatmap(filtered_data)

    # 3. Perform regression analysis
    regression_results = perform_regression_analysis(filtered_data)
    print("Regression Results:")
    print(f"R²: {regression_results['r2']:.3f}")
    print(f"RMSE: {regression_results['rmse']:.3f}")
    print(f"Slope: {regression_results['slope']:.3f}")
    print(f"Intercept: {regression_results['intercept']:.3f}")

    # 4. Plot ratio trends for top states
    plot_ratio_trends(filtered_data)

    # 5. Create state choropleth/bar chart for 2022
    create_state_choropleth(filtered_data, 2022)

    print("All visualizations have been saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
