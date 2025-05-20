"""
Analysis of Cybercrime Rate Changes in Indian Cities During COVID-19 (2019-2021)
Using Support Vector Machine (SVM) Classification

This script analyzes which Indian cities showed the most significant changes in 
cybercrime rates due to the COVID-19 pandemic (2019-2021).

Author: GitHub Copilot
Date: May 20, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import warnings
warnings.filterwarnings('ignore')

# Set the style and color palette for visualizations
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-pastel')
color_palette = sns.color_palette("viridis", 10)
sns.set_palette(color_palette)

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset


def load_data():
    """Load and preprocess the dataset."""
    file_path = "../cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv"
    df = pd.read_csv(file_path)

    # Filter for years 2019-2021 (COVID-19 pandemic period)
    df = df[(df['year'] >= 2019) & (df['year'] <= 2021)]

    # Remove 'Total Cities' rows
    df = df[df['city'] != 'Total Cities']

    # Drop rows with missing values
    df = df.dropna(subset=['city', 'value'])

    # Convert value column to numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    print(
        f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

# Perform Exploratory Data Analysis


def perform_eda(df):
    """Perform exploratory data analysis on the dataset."""
    print("\nExploratory Data Analysis:")
    print(f"Number of unique cities: {df['city'].nunique()}")
    print(f"Number of unique states: {df['state'].nunique()}")
    print(f"Years in the dataset: {sorted(df['year'].unique())}")

    # City-wise statistics for different years
    yearly_stats = df.groupby(['year'])['value'].agg(
        ['sum', 'mean', 'median', 'std']).reset_index()
    print("\nYearly Statistics:")
    print(yearly_stats)

    # City-wise statistics
    city_stats = df.groupby(['city'])['value'].agg(
        ['mean', 'max', 'min']).reset_index()
    city_stats['range'] = city_stats['max'] - city_stats['min']
    city_stats = city_stats.sort_values('range', ascending=False)
    print("\nTop 10 cities with highest range in cybercrime rates:")
    print(city_stats.head(10))

    return yearly_stats, city_stats

# Calculate percent change in cybercrime rates


def calculate_percent_change(df):
    """Calculate percentage change in cybercrime rates for each city between years."""
    # Pivot the data to have years as columns and cities as rows
    pivot_df = df.pivot_table(
        index=['state', 'city'], columns='year', values='value').reset_index()

    # Calculate percent change from 2019 to 2020 and 2020 to 2021
    pivot_df['pct_change_2019_2020'] = (
        (pivot_df[2020] - pivot_df[2019]) / pivot_df[2019]) * 100
    pivot_df['pct_change_2020_2021'] = (
        (pivot_df[2021] - pivot_df[2020]) / pivot_df[2020]) * 100

    # Calculate overall change from 2019 to 2021
    pivot_df['pct_change_2019_2021'] = (
        (pivot_df[2021] - pivot_df[2019]) / pivot_df[2019]) * 100

    # Classify cities based on the direction of change
    pivot_df['change_direction_2019_2020'] = np.where(
        pivot_df['pct_change_2019_2020'] > 0, 'Increase', 'Decrease')
    pivot_df['change_direction_2020_2021'] = np.where(
        pivot_df['pct_change_2020_2021'] > 0, 'Increase', 'Decrease')
    pivot_df['change_direction_2019_2021'] = np.where(
        pivot_df['pct_change_2019_2021'] > 0, 'Increase', 'Decrease')

    # Create a new classification based on change patterns
    conditions = [
        (pivot_df['change_direction_2019_2020'] == 'Increase') & (
            pivot_df['change_direction_2020_2021'] == 'Increase'),
        (pivot_df['change_direction_2019_2020'] == 'Decrease') & (
            pivot_df['change_direction_2020_2021'] == 'Decrease'),
        (pivot_df['change_direction_2019_2020'] == 'Increase') & (
            pivot_df['change_direction_2020_2021'] == 'Decrease'),
        (pivot_df['change_direction_2019_2020'] == 'Decrease') & (
            pivot_df['change_direction_2020_2021'] == 'Increase')
    ]

    choices = ['Continuous Increase', 'Continuous Decrease',
               'Increase then Decrease', 'Decrease then Increase']
    pivot_df['change_pattern'] = np.select(
        conditions, choices, default='Unknown')

    # Add a classification for overall significant change
    pivot_df['significant_change'] = np.where(
        abs(pivot_df['pct_change_2019_2021']) > 50, 'Significant', 'Moderate')

    return pivot_df

# Visualization 1: Percent change in cybercrime rates for top cities (2019-2021)


def visualize_percent_change(change_df):
    """Create a bar chart showing the percent change in cybercrime rates for top cities."""
    # Select top 15 cities with highest absolute percent change
    top_cities = change_df.sort_values(
        'pct_change_2019_2021', key=abs, ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.bar(top_cities['city'], top_cities['pct_change_2019_2021'],
                  color=sns.color_palette("RdYlGn_r", len(top_cities)))

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        label_height = height + 10 if height > 0 else height - 30
        ax.text(bar.get_x() + bar.get_width()/2., label_height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title(
        'Percent Change in Cybercrime Rates in Top Indian Cities (2019-2021)', fontsize=16)
    ax.set_xlabel('City', fontsize=14)
    ax.set_ylabel('Percent Change (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a color bar to indicate the magnitude of change
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Magnitude of Change', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'cybercrime_percent_change.png'), dpi=300)
    plt.close()

# Visualization 2: Year-wise cybercrime trends for top affected cities


def visualize_yearly_trends(df):
    """Create a line plot showing yearly cybercrime trends for top affected cities."""
    # Identify top 10 cities with highest cybercrime rates in 2021
    top_cities_2021 = df[df['year'] == 2021].nlargest(8, 'value')[
        'city'].unique()

    # Filter data for these cities
    top_cities_data = df[df['city'].isin(top_cities_2021)]

    plt.figure(figsize=(14, 10))

    # Create a line plot for each city
    sns.lineplot(data=top_cities_data, x='year', y='value',
                 hue='city', marker='o', linewidth=2.5, markersize=10)

    plt.title(
        'Yearly Cybercrime Trends for Top Affected Indian Cities (2019-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Cybercrimes', fontsize=14)
    plt.xticks(top_cities_data['year'].unique())
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='City', title_fontsize=12, fontsize=10,
               loc='upper left', bbox_to_anchor=(1, 1))

    # Annotate key points
    for city in top_cities_2021:
        city_data = top_cities_data[top_cities_data['city'] == city]
        for _, row in city_data.iterrows():
            plt.annotate(f"{int(row['value'])}",
                         (row['year'], row['value']),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=9,
                         fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'yearly_cybercrime_trends.png'), dpi=300)
    plt.close()

# Visualization 3: Heatmap showing cybercrime rates by city and year


def visualize_heatmap(df):
    """Create a heatmap showing cybercrime rates by city and year."""
    # Pivot the data to create a heatmap
    pivot_data = df.pivot_table(index='city', columns='year', values='value')

    # Select top 20 cities with highest cybercrime rates in 2021
    top_cities = df[df['year'] == 2021].nlargest(20, 'value')['city'].unique()
    pivot_data = pivot_data.loc[top_cities]

    plt.figure(figsize=(12, 14))
    ax = sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd',
                     linewidths=.5, cbar_kws={'label': 'Number of Cybercrimes'})

    plt.title(
        'Heatmap of Cybercrime Rates by City and Year (2019-2021)', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('City', fontsize=14)

    # Increase the font size of annotations
    for text in ax.texts:
        text.set_size(9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cybercrime_heatmap.png'), dpi=300)
    plt.close()

# Visualization 4: Classification of cities based on change patterns


def visualize_change_patterns(change_df):
    """Create a visualization showing the classification of cities based on change patterns."""
    # Count the number of cities in each change pattern
    pattern_counts = change_df['change_pattern'].value_counts().reset_index()
    pattern_counts.columns = ['Change Pattern', 'Count']

    plt.figure(figsize=(14, 10))

    # Create a pie chart
    plt.subplot(1, 2, 1)
    plt.pie(pattern_counts['Count'], labels=pattern_counts['Change Pattern'], autopct='%1.1f%%',
            colors=sns.color_palette("Set2", len(pattern_counts)),
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'fontsize': 12})
    plt.title(
        'Distribution of Cities by Cybercrime Change Pattern (2019-2021)', fontsize=14)

    # Create a bar chart for pattern distribution
    plt.subplot(1, 2, 2)
    sns.barplot(x='Change Pattern', y='Count',
                data=pattern_counts, palette="Set2")
    plt.title('Count of Cities by Cybercrime Change Pattern', fontsize=14)
    plt.xlabel('Change Pattern', fontsize=12)
    plt.ylabel('Number of Cities', fontsize=12)
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'cybercrime_change_patterns.png'), dpi=300)
    plt.close()

    # Create a scatter plot of cities showing 2019-2020 vs 2020-2021 percent changes
    plt.figure(figsize=(14, 10))

    # Create scatter plot
    scatter = plt.scatter(change_df['pct_change_2019_2020'],
                          change_df['pct_change_2020_2021'],
                          c=pd.factorize(change_df['change_pattern'])[0],
                          cmap='viridis',
                          s=100,
                          alpha=0.7)

    # Add a legend
    legend1 = plt.legend(scatter.legend_elements()[0],
                         change_df['change_pattern'].unique(),
                         title="Change Pattern",
                         loc="upper right")
    plt.gca().add_artist(legend1)

    # Highlight significant cities
    significant_cities = change_df[abs(
        change_df['pct_change_2019_2021']) > 100]
    for idx, row in significant_cities.iterrows():
        plt.annotate(row['city'],
                     (row['pct_change_2019_2020'],
                      row['pct_change_2020_2021']),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='left',
                     fontweight='bold')

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.title(
        'Change Patterns in Cybercrime Rates (2019-2020 vs 2020-2021)', fontsize=16)
    plt.xlabel('Percent Change 2019-2020 (%)', fontsize=14)
    plt.ylabel('Percent Change 2020-2021 (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'cybercrime_change_scatter.png'), dpi=300)
    plt.close()

# Prepare data for SVM classification


def prepare_model_data(change_df):
    """Prepare data for SVM classification."""
    # Feature engineering
    model_data = change_df.copy()

    # Create features from the data
    # We'll use the city's cybercrime values and percent changes
    model_data['pct_change_abs'] = abs(model_data['pct_change_2019_2021'])

    # Encode categorical variables
    le = LabelEncoder()
    model_data['city_encoded'] = le.fit_transform(model_data['city'])
    model_data['state_encoded'] = le.fit_transform(model_data['state'])
    model_data['pattern_encoded'] = le.fit_transform(
        model_data['change_pattern'])

    # Define features and target
    features = ['city_encoded', 'state_encoded', 'pattern_encoded',
                'pct_change_2019_2020', 'pct_change_2020_2021', 'pct_change_2019_2021', 'pct_change_abs']

    # Target: Classify cities as 'Significant' or 'Moderate' change
    target = model_data['significant_change']

    # Split data into training and testing sets
    X = model_data[features]
    y = target

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, model_data, features

# Train and evaluate SVM model


def train_svm_model(X_train, X_test, y_train, y_test):
    """Train and evaluate the SVM model."""
    # Initialize and train SVM model
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_svm = svm_model.predict(X_test)

    # Evaluate the model
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    class_report_svm = classification_report(y_test, y_pred_svm)

    print("\nSVM Model Evaluation:")
    print(f"Accuracy: {accuracy_svm:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix_svm)
    print("\nClassification Report:")
    print(class_report_svm)

    return svm_model, y_pred_svm, accuracy_svm

# Train and evaluate Logistic Regression model (baseline)


def train_baseline_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a baseline Logistic Regression model."""
    # Initialize and train Logistic Regression model
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    # Make predictions
    y_pred_lr = lr_model.predict(X_test)

    # Evaluate the model
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    class_report_lr = classification_report(y_test, y_pred_lr)

    print("\nLogistic Regression Model Evaluation (Baseline):")
    print(f"Accuracy: {accuracy_lr:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix_lr)
    print("\nClassification Report:")
    print(class_report_lr)

    return lr_model, y_pred_lr, accuracy_lr

# Visualization 5: Model comparison and feature importance


def visualize_model_comparison(svm_model, lr_model, X_test, y_test, y_pred_svm, y_pred_lr, accuracy_svm, accuracy_lr, model_data, features):
    """Create visualizations comparing SVM and baseline model performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Create subplot for model accuracy comparison
    models = ['SVM', 'Logistic Regression']
    accuracies = [accuracy_svm, accuracy_lr]

    bars = axes[0, 0].bar(models, accuracies, color=['#3498db', '#e74c3c'])

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

    axes[0, 0].set_title('Model Accuracy Comparison', fontsize=14)
    axes[0, 0].set_ylabel('Accuracy Score', fontsize=12)
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.3)

    # Create subplot for confusion matrix comparison
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Moderate', 'Significant'],
                yticklabels=['Moderate', 'Significant'],
                ax=axes[0, 1])
    axes[0, 1].set_title('SVM Confusion Matrix', fontsize=14)
    axes[0, 1].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 1].set_ylabel('True Label', fontsize=12)

    conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Moderate', 'Significant'],
                yticklabels=['Moderate', 'Significant'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Logistic Regression Confusion Matrix', fontsize=14)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 0].set_ylabel('True Label', fontsize=12)

    # Create a subplot for feature importances (for Logistic Regression only)
    if hasattr(lr_model, 'coef_'):
        importance = np.abs(lr_model.coef_[0])
        # Create a DataFrame with features and their importance scores
        feature_importance = pd.DataFrame(
            {'Feature': features, 'Importance': importance})
        feature_importance = feature_importance.sort_values(
            'Importance', ascending=False)

        # Plot feature importance
        sns.barplot(x='Importance', y='Feature',
                    data=feature_importance, palette='viridis', ax=axes[1, 1])
        axes[1, 1].set_title(
            'Feature Importance (Logistic Regression)', fontsize=14)
        axes[1, 1].set_xlabel('Importance Score', fontsize=12)
        axes[1, 1].set_ylabel('Feature', fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, "Feature importance not available",
                        ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

    # Create a visualization showing cities classified as having significant changes
    significant_cities = model_data[model_data['significant_change']
                                    == 'Significant']
    significant_cities = significant_cities.sort_values(
        'pct_change_abs', ascending=False)

    plt.figure(figsize=(14, 10))
    bars = plt.bar(significant_cities['city'], significant_cities['pct_change_2019_2021'],
                   color=sns.color_palette("RdYlGn_r", len(significant_cities)))

    # Add data labels
    for bar in bars:
        height = bar.get_height()
        label_height = height + 10 if height > 0 else height - 30
        plt.text(bar.get_x() + bar.get_width()/2., label_height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(
        'Cities with Significant Changes in Cybercrime Rates (2019-2021)', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Percent Change (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, 'significant_change_cities.png'), dpi=300)
    plt.close()

# Main function


def main():
    """Main function to run the analysis."""
    print("Starting analysis of cybercrime rate changes in Indian cities during COVID-19 (2019-2021)...")

    # Load and preprocess the data
    df = load_data()

    # Perform exploratory data analysis
    yearly_stats, city_stats = perform_eda(df)

    # Calculate percent change in cybercrime rates
    change_df = calculate_percent_change(df)

    # Create visualizations
    print("\nGenerating visualizations...")

    # Visualization 1: Percent change in cybercrime rates for top cities
    visualize_percent_change(change_df)
    print("Visualization 1: Percent change in cybercrime rates for top cities - Completed")

    # Visualization 2: Year-wise cybercrime trends for top affected cities
    visualize_yearly_trends(df)
    print("Visualization 2: Year-wise cybercrime trends for top affected cities - Completed")

    # Visualization 3: Heatmap showing cybercrime rates by city and year
    visualize_heatmap(df)
    print("Visualization 3: Heatmap showing cybercrime rates by city and year - Completed")

    # Visualization 4: Classification of cities based on change patterns
    visualize_change_patterns(change_df)
    print("Visualization 4: Classification of cities based on change patterns - Completed")

    # Prepare data for modeling
    X_train, X_test, y_train, y_test, model_data, features = prepare_model_data(
        change_df)

    # Train and evaluate SVM model
    print("\nTraining SVM model...")
    svm_model, y_pred_svm, accuracy_svm = train_svm_model(
        X_train, X_test, y_train, y_test)

    # Train and evaluate baseline model
    print("\nTraining baseline Logistic Regression model...")
    lr_model, y_pred_lr, accuracy_lr = train_baseline_model(
        X_train, X_test, y_train, y_test)

    # Visualization 5: Model comparison
    visualize_model_comparison(svm_model, lr_model, X_test, y_test, y_pred_svm, y_pred_lr,
                               accuracy_svm, accuracy_lr, model_data, features)
    print("Visualization 5: Model comparison and feature importance - Completed")

    print("\nAnalysis completed successfully.")
    print(f"All visualizations have been saved to: {output_dir}")


if __name__ == "__main__":
    main()
