#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cybercrime Pattern Analysis: Bengaluru, Hyderabad, and Pune
---------------------------------------------------------
This script analyzes cybercrime patterns in three major Indian tech cities
for the year 2022 using K-means clustering and various visualization techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.dummy import DummyRegressor
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set the style for visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
colors = sns.color_palette("viridis", 3)

# Create output directory if it doesn't exist
output_dir = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\Qn 6"
os.makedirs(output_dir, exist_ok=True)


def load_and_preprocess_data():
    """
    Loads and preprocesses the cybercrime dataset for analysis.

    Returns:
        tuple: (complete_df, city_crime_df)
            - complete_df: Full dataset filtered for our cities of interest
            - city_crime_df: Aggregated data pivoted for city comparison
    """
    print("Loading and preprocessing data...")

    # Load the dataset
    file_path = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv"
    df = pd.read_csv(file_path)

    # Filter the data for the year 2022 and the three cities of interest
    cities_of_interest = ['Bengaluru', 'Hyderabad', 'Pune']
    df_2022 = df[df['year'] == 2022]
    df_filtered = df_2022[df_2022['city'].isin(cities_of_interest)]

    # Replace empty strings in offence_sub_category with the offence_category
    df_filtered['offence_sub_category'] = df_filtered['offence_sub_category'].fillna(
        df_filtered['offence_category']
    )

    # Create a category column that combines category and subcategory for detailed analysis
    df_filtered['crime_category'] = np.where(
        df_filtered['offence_sub_category'] == df_filtered['offence_category'],
        df_filtered['offence_category'],
        df_filtered['offence_category'] + " - " +
        df_filtered['offence_sub_category']
    )

    # Create a pivot table for city comparison
    city_crime_pivot = pd.pivot_table(
        df_filtered,
        values='value',
        index=['city'],
        columns=['crime_category'],
        aggfunc=np.sum,
        fill_value=0
    )

    # Reset the index to make city a column
    city_crime_df = city_crime_pivot.reset_index()

    print(
        f"Data preprocessing complete. Found {len(city_crime_df)} cities with {city_crime_df.shape[1]-1} crime categories.")

    return df_filtered, city_crime_df


def visualize_top_crimes_by_city(df_filtered, output_dir):
    """
    Creates a bar chart comparing top cybercrime categories across the three cities.

    Args:
        df_filtered (DataFrame): Filtered dataset with crime information
        output_dir (str): Directory to save the visualization
    """
    print("Creating bar chart visualization...")

    # Group by city and crime_category and sum the values
    city_crime_grouped = df_filtered.groupby(['city', 'crime_category'])[
        'value'].sum().reset_index()

    # Get the top 10 crime categories by total value across all cities
    top_crimes = city_crime_grouped.groupby(
        'crime_category')['value'].sum().nlargest(10).index

    # Filter for only the top 10 crime categories
    top_city_crimes = city_crime_grouped[city_crime_grouped['crime_category'].isin(
        top_crimes)]

    # Create the plot
    plt.figure(figsize=(14, 10))

    # Create the grouped bar chart
    sns.barplot(
        x='crime_category',
        y='value',
        hue='city',
        data=top_city_crimes,
        palette=colors
    )

    plt.title('Top 10 Cybercrime Categories Across Cities (2022)', fontsize=16)
    plt.xlabel('Crime Category', fontsize=14)
    plt.ylabel('Number of Cases', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='City')
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'top_crimes_comparison.png'), dpi=300)
    plt.close()

    print("Bar chart visualization completed and saved.")


def create_crime_heatmap(city_crime_df, output_dir):
    """
    Creates a heatmap showing crime intensity across cities and categories.

    Args:
        city_crime_df (DataFrame): Pivoted data with cities and crime categories
        output_dir (str): Directory to save the visualization
    """
    print("Creating crime intensity heatmap...")

    # Create a copy of the dataframe for the heatmap
    heatmap_df = city_crime_df.copy()
    heatmap_df.set_index('city', inplace=True)

    # Select top 15 crime categories by total occurrences for better visualization
    top_categories = heatmap_df.sum().nlargest(15).index
    heatmap_df = heatmap_df[top_categories]

    # Create the heatmap
    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_df,
        cmap='YlOrRd',
        annot=True,
        fmt='g',
        linewidths=.5,
        cbar_kws={'label': 'Number of Cases'}
    )

    plt.title('Cybercrime Intensity Heatmap (2022)', fontsize=16)
    plt.xlabel('Crime Category', fontsize=14)
    plt.ylabel('City', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(
        output_dir, 'crime_intensity_heatmap.png'), dpi=300)
    plt.close()

    print("Heatmap visualization completed and saved.")


def perform_clustering_analysis(city_crime_df, output_dir):
    """
    Performs K-means clustering on the cybercrime data and visualizes the results.

    Args:
        city_crime_df (DataFrame): Pivoted data with cities and crime categories
        output_dir (str): Directory to save the visualization
    """
    print("Performing K-means clustering analysis...")

    # Prepare the data for clustering
    X = city_crime_df.drop('city', axis=1)
    city_names = city_crime_df['city']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction and visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Since we only have 3 cities, we'll use 2 clusters to ensure we have enough data points per cluster
    # We'll skip the silhouette score calculation because it requires at least 2 samples per cluster
    optimal_n_clusters = 2

    # Apply K-means with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
    city_crime_df['cluster'] = kmeans.fit_predict(X_scaled)

    # Create the clustering visualization
    plt.figure(figsize=(12, 10))

    # Plot each city in the PCA space
    for i, city in enumerate(city_names):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=[colors[city_crime_df['cluster'].iloc[i]]],
                    s=300, label=city, edgecolors='black', linewidth=1.5)
        plt.annotate(city, (X_pca[i, 0]+0.1, X_pca[i, 1]), fontsize=14)

    # Add titles and labels
    plt.title(
        f'Clustering of Cities Based on Cybercrime Patterns (K={optimal_n_clusters})', fontsize=16)
    plt.xlabel(
        f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
    plt.ylabel(
        f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'cybercrime_clustering.png'), dpi=300)
    plt.close()

    # Create a radar chart to visualize feature importance
    plt.figure(figsize=(14, 12))

    # Get the cluster centers and transform back to original scale
    cluster_centers = kmeans.cluster_centers_
    centers_original_scale = scaler.inverse_transform(cluster_centers)

    # Get the top 8 features that contribute most to the variance
    pca_full = PCA()
    pca_full.fit(X_scaled)
    top_features_idx = np.argsort(-np.abs(pca_full.components_[0]))[:8]

    # Get the names of these features
    feature_names = X.columns[top_features_idx]

    # Prepare data for the radar chart
    angles = np.linspace(0, 2*np.pi, len(feature_names),
                         endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Set up the radar chart
    ax = plt.subplot(111, polar=True)

    # For each cluster
    for i in range(optimal_n_clusters):
        values = centers_original_scale[i, top_features_idx].tolist()
        values += values[:1]  # Close the polygon
        ax.plot(angles, values, linewidth=2,
                linestyle='solid', label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.25)

    # Set chart properties
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], feature_names, size=12)

    # Add city labels to the clusters
    for i in range(optimal_n_clusters):
        cities_in_cluster = city_names[city_crime_df['cluster'] == i].tolist()
        if cities_in_cluster:
            new_label = f"Cluster {i}: {', '.join(cities_in_cluster)}"
            handles, labels = ax.get_legend_handles_labels()
            labels[i] = new_label

    plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Cluster Centers by Top Crime Features', size=16)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(
        output_dir, 'cluster_feature_importance.png'), dpi=300)
    plt.close()

    print(f"Clustering analysis completed with {optimal_n_clusters} clusters.")
    return city_crime_df


def create_crime_distribution(df_filtered, output_dir):
    """
    Creates stacked bar charts showing the distribution of crime categories in each city.

    Args:
        df_filtered (DataFrame): Filtered dataset with crime information
        output_dir (str): Directory to save the visualization
    """
    print("Creating crime distribution visualization...")

    # Group by city and main crime category for a cleaner visualization
    city_category_grouped = df_filtered.groupby(['city', 'offence_category'])[
        'value'].sum().reset_index()

    # Pivot the data for plotting
    dist_pivot = city_category_grouped.pivot(
        index='city', columns='offence_category', values='value')

    # Normalize the data to show percentage distribution within each city
    dist_pivot_pct = dist_pivot.div(dist_pivot.sum(axis=1), axis=0) * 100

    # Create the stacked bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

    # Plot absolute values
    dist_pivot.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
    ax1.set_title(
        'Absolute Distribution of Cybercrime Categories by City (2022)', fontsize=16)
    ax1.set_xlabel('')
    ax1.set_ylabel('Number of Cases', fontsize=14)
    ax1.legend(title='Crime Category',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot percentage distribution
    dist_pivot_pct.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis')
    ax2.set_title(
        'Percentage Distribution of Cybercrime Categories by City (2022)', fontsize=16)
    ax2.set_xlabel('City', fontsize=14)
    ax2.set_ylabel('Percentage (%)', fontsize=14)
    ax2.legend(title='Crime Category',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(
        output_dir, 'crime_distribution_by_city.png'), dpi=300)
    plt.close()

    print("Crime distribution visualization completed and saved.")


def compare_models(city_crime_df, output_dir):
    """
    Compares baseline model with K-means clustering model for predicting crime patterns.

    Args:
        city_crime_df (DataFrame): Dataset with cities and their crime data
        output_dir (str): Directory to save the visualization
    """
    print("Comparing baseline model with K-means clustering model...")

    # Prepare the data
    X = city_crime_df.drop(['city', 'cluster'], axis=1, errors='ignore')

    # Choose a target variable for prediction - total cybercrime cases
    if 'Total' in X.columns:
        y = X['Total']
        X = X.drop('Total', axis=1)
    else:
        y = X.sum(axis=1)  # Use sum of all crime categories as target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Baseline Model: Mean prediction (DummyRegressor)
    baseline_model = DummyRegressor(strategy='mean')
    baseline_model.fit(X_scaled, y)
    baseline_preds = baseline_model.predict(X_scaled)
    baseline_mse = np.mean((baseline_preds - y) ** 2)
    baseline_rmse = np.sqrt(baseline_mse)

    # K-means based prediction
    # For this small dataset (3 cities), we'll use 2 clusters
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # For each cluster, use the cluster mean as the prediction
    kmeans_preds = np.zeros_like(y)
    for cluster in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_mean = y.iloc[cluster_indices].mean()
            kmeans_preds[cluster_indices] = cluster_mean

    kmeans_mse = np.mean((kmeans_preds - y) ** 2)
    kmeans_rmse = np.sqrt(kmeans_mse)

    # Create comparison visualization
    plt.figure(figsize=(14, 10))

    # Create bar chart for model comparison
    models = ['Baseline (Mean)', 'K-means Clustering']
    rmse_values = [baseline_rmse, kmeans_rmse]

    bars = plt.bar(models, rmse_values, color=['lightgray', colors[0]])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=12)

    plt.title(
        'Model Comparison: RMSE for Predicting Total Cybercrime Cases', fontsize=16)
    plt.ylabel('Root Mean Square Error (RMSE)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # Create a scatter plot of actual vs predicted values
    plt.figure(figsize=(12, 10))

    # Plot the baseline predictions
    plt.scatter(y, baseline_preds, alpha=0.7, s=200, label=f'Baseline (RMSE: {baseline_rmse:.2f})',
                color='lightgray', edgecolors='black')

    # Plot the K-means predictions
    plt.scatter(y, kmeans_preds, alpha=0.7, s=200, label=f'K-means (RMSE: {kmeans_rmse:.2f})',
                color=colors[0], edgecolors='black')

    # Plot the perfect prediction line
    max_val = max(max(y), max(baseline_preds), max(kmeans_preds))
    min_val = min(min(y), min(baseline_preds), min(kmeans_preds))
    plt.plot([min_val, max_val], [min_val, max_val],
             'k--', label='Perfect Prediction')

    # Add city labels
    for i, city in enumerate(city_crime_df['city']):
        plt.annotate(city, (y.iloc[i], kmeans_preds[i]), fontsize=12)

    plt.title('Actual vs Predicted Total Cybercrime Cases', fontsize=16)
    plt.xlabel('Actual Cases', fontsize=14)
    plt.ylabel('Predicted Cases', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    print(
        f"Model comparison completed. Baseline RMSE: {baseline_rmse:.2f}, K-means RMSE: {kmeans_rmse:.2f}")

    return {
        'baseline_rmse': baseline_rmse,
        'kmeans_rmse': kmeans_rmse,
        'improvement': (baseline_rmse - kmeans_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
    }


def main():
    """
    Main function to execute the cybercrime analysis pipeline.
    """
    print("\n=== Cybercrime Pattern Analysis: Bengaluru, Hyderabad, and Pune ===\n")

    # Load and preprocess the data
    df_filtered, city_crime_df = load_and_preprocess_data()

    # Generate visualizations
    visualize_top_crimes_by_city(df_filtered, output_dir)
    create_crime_heatmap(city_crime_df, output_dir)
    city_crime_df = perform_clustering_analysis(city_crime_df, output_dir)
    create_crime_distribution(df_filtered, output_dir)
    model_metrics = compare_models(city_crime_df, output_dir)

    print("\nAnalysis complete! All visualizations saved to:", output_dir)
    print(f"\nModel Comparison Results:")
    print(f"Baseline RMSE: {model_metrics['baseline_rmse']:.2f}")
    print(f"K-means RMSE: {model_metrics['kmeans_rmse']:.2f}")
    print(f"Improvement: {model_metrics['improvement']:.2f}%")


if __name__ == "__main__":
    main()
