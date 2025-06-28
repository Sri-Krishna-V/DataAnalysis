"""
Regional Cybercrime Analysis: Fraud-Related Cybercrimes in Western vs Eastern India (2018-2022)
Author: Data Analysis Team
Date: June 2025

This script analyzes regional variations in fraud-related cybercrimes between 
Western and Eastern India from 2018 to 2022 using NCRB cybercrime data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for consistent, professional visualizations
plt.style.use('default')
sns.set_palette("husl")
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']

class CybercrimeAnalysis:
    """
    A comprehensive analysis class for cybercrime data focusing on fraud-related crimes
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.western_states = [
            'Gujarat', 'Maharashtra', 'Rajasthan', 'Goa', 
            'Dadra and Nagar Haveli', 'Daman and Diu', 
            'Dadra and Nagar Haveli and Daman and Diu'
        ]
        self.eastern_states = [
            'West Bengal', 'Odisha', 'Jharkhand', 'Bihar', 
            'Sikkim', 'Assam', 'Arunachal Pradesh', 'Nagaland', 
            'Manipur', 'Mizoram', 'Tripura', 'Meghalaya'
        ]
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the cybercrime datasets
        """
        print("Loading and preprocessing data...")
        
        # Load the main datasets
        self.motives_df = pd.read_csv(f'{self.data_path}/cyber-crimes-from-ncrb-master-data-year-and-state-wise-number-of-cyber-crimes-committed-in-india-by-types-of-motives.csv')
        self.ipc_df = pd.read_csv(f'{self.data_path}/cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-ipc.csv')
        self.it_act_df = pd.read_csv(f'{self.data_path}/cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv')
        
        # Filter for target years (2018-2022)
        target_years = [2018, 2019, 2020, 2021, 2022]
        
        # Extract fraud data from motives dataset
        fraud_motives = self.motives_df[
            (self.motives_df['offence_category'] == 'Fraud or Illegal Gain') & 
            (self.motives_df['year'].isin(target_years)) &
            (self.motives_df['state'] != 'All India')
        ].copy()
        
        # Extract fraud data from IPC dataset
        fraud_ipc = self.ipc_df[
            (self.ipc_df['offence_category'] == 'Fraud') & 
            (self.ipc_df['year'].isin(target_years)) &
            (self.ipc_df['state'] != 'All India')
        ].copy()
        
        # Aggregate IPC fraud data by state and year
        fraud_ipc_agg = fraud_ipc.groupby(['year', 'state'])['value'].sum().reset_index()
        fraud_ipc_agg['offence_category'] = 'IPC_Fraud'
        
        # Combine fraud data
        fraud_motives_clean = fraud_motives[['year', 'state', 'value', 'offence_category']].copy()
        
        # Create combined dataset
        self.fraud_data = pd.concat([
            fraud_motives_clean.rename(columns={'offence_category': 'crime_type'}),
            fraud_ipc_agg.rename(columns={'offence_category': 'crime_type'})
        ], ignore_index=True)
        
        # Add regional classification
        self.fraud_data['region'] = self.fraud_data['state'].apply(self.classify_region)
        
        # Remove unclassified regions
        self.fraud_data = self.fraud_data[self.fraud_data['region'] != 'Other'].copy()
        
        # Handle missing values
        self.fraud_data['value'] = pd.to_numeric(self.fraud_data['value'], errors='coerce')
        self.fraud_data = self.fraud_data.dropna(subset=['value'])
        
        print(f"Processed dataset shape: {self.fraud_data.shape}")
        print(f"Date range: {self.fraud_data['year'].min()} to {self.fraud_data['year'].max()}")
        print(f"Regions covered: {self.fraud_data['region'].unique()}")
        
        return self.fraud_data
    
    def classify_region(self, state):
        """
        Classify states into Western or Eastern regions
        """
        if state in self.western_states:
            return 'Western'
        elif state in self.eastern_states:
            return 'Eastern'
        else:
            return 'Other'
    
    def exploratory_data_analysis(self):
        """
        Perform comprehensive exploratory data analysis
        """
        print("\nPerforming Exploratory Data Analysis...")
        
        # Basic statistics
        print("\n=== BASIC STATISTICS ===")
        print(self.fraud_data.describe())
        
        # Regional distribution
        print("\n=== REGIONAL DISTRIBUTION ===")
        regional_stats = self.fraud_data.groupby(['region', 'year'])['value'].agg(['sum', 'mean', 'count']).round(2)
        print(regional_stats)
        
        # Crime type distribution
        print("\n=== CRIME TYPE DISTRIBUTION ===")
        crime_stats = self.fraud_data.groupby(['crime_type', 'region'])['value'].agg(['sum', 'mean']).round(2)
        print(crime_stats)
        
        return regional_stats, crime_stats
    
    def create_visualizations(self):
        """
        Create 5 key visualizations to answer the research question
        """
        print("\nCreating visualizations...")
        
        # 1. Time Series Analysis: Regional Fraud Trends Over Time
        self.create_time_series_plot()
        
        # 2. Comparative Bar Chart: Total Fraud Cases by Region and Year
        self.create_comparative_bar_chart()
        
        # 3. Heatmap: State-wise Fraud Intensity by Year
        self.create_fraud_heatmap()
        
        # 4. Box Plot: Distribution of Fraud Cases by Region
        self.create_distribution_boxplot()
        
        # 5. Stacked Area Chart: Crime Type Composition Over Time
        self.create_crime_composition_chart()
    
    def create_time_series_plot(self):
        """
        Visualization 1: Time series plot showing fraud trends by region
        """
        plt.figure(figsize=(12, 8))
        
        # Aggregate data by region and year
        regional_trends = self.fraud_data.groupby(['year', 'region'])['value'].sum().reset_index()
        
        # Create line plot
        for i, region in enumerate(['Western', 'Eastern']):
            region_data = regional_trends[regional_trends['region'] == region]
            plt.plot(region_data['year'], region_data['value'], 
                    marker='o', linewidth=3, markersize=8, 
                    color=colors[i], label=f'{region} India')
        
        plt.title('Fraud-Related Cybercrime Trends: Western vs Eastern India (2018-2022)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14, fontweight='bold')
        plt.ylabel('Total Fraud Cases', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add trend annotations
        for i, region in enumerate(['Western', 'Eastern']):
            region_data = regional_trends[regional_trends['region'] == region]
            if len(region_data) > 1:
                growth_rate = ((region_data['value'].iloc[-1] - region_data['value'].iloc[0]) / 
                              region_data['value'].iloc[0] * 100)
                plt.annotate(f'{region}: {growth_rate:.1f}% growth', 
                           xy=(0.02, 0.95 - i*0.05), xycoords='axes fraction',
                           fontsize=11, bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=colors[i], alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('fraud_trends_regional_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparative_bar_chart(self):
        """
        Visualization 2: Comparative bar chart showing fraud cases by region and year
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Aggregate data
        yearly_regional = self.fraud_data.groupby(['year', 'region'])['value'].sum().reset_index()
        
        # Bar chart by year
        years = sorted(yearly_regional['year'].unique())
        western_values = []
        eastern_values = []
        
        for year in years:
            western_val = yearly_regional[(yearly_regional['year'] == year) & 
                                        (yearly_regional['region'] == 'Western')]['value'].sum()
            eastern_val = yearly_regional[(yearly_regional['year'] == year) & 
                                        (yearly_regional['region'] == 'Eastern')]['value'].sum()
            western_values.append(western_val)
            eastern_values.append(eastern_val)
        
        x = np.arange(len(years))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, western_values, width, label='Western India', 
                       color=colors[0], alpha=0.8)
        bars2 = ax1.bar(x + width/2, eastern_values, width, label='Eastern India', 
                       color=colors[1], alpha=0.8)
        
        ax1.set_title('Annual Fraud Cases by Region', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        # Crime type comparison
        crime_regional = self.fraud_data.groupby(['crime_type', 'region'])['value'].sum().reset_index()
        crime_pivot = crime_regional.pivot(index='crime_type', columns='region', values='value').fillna(0)
        
        crime_pivot.plot(kind='bar', ax=ax2, color=[colors[0], colors[1]], alpha=0.8)
        ax2.set_title('Total Fraud Cases by Crime Type and Region', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Crime Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Cases (2018-2022)', fontsize=12, fontweight='bold')
        ax2.legend(title='Region')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fraud_comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_fraud_heatmap(self):
        """
        Visualization 3: Heatmap showing state-wise fraud intensity
        """
        plt.figure(figsize=(14, 10))
        
        # Create state-year pivot table
        state_year_data = self.fraud_data.groupby(['state', 'year'])['value'].sum().reset_index()
        heatmap_data = state_year_data.pivot(index='state', columns='year', values='value').fillna(0)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Number of Fraud Cases'}, linewidths=0.5)
        
        plt.title('State-wise Fraud Case Intensity Heatmap (2018-2022)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year', fontsize=14, fontweight='bold')
        plt.ylabel('State', fontsize=14, fontweight='bold')
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(rotation=0, fontsize=10)
        
        # Add regional indicators
        y_pos = 0
        for state in heatmap_data.index:
            region = self.classify_region(state)
            if region != 'Other':
                color = colors[0] if region == 'Western' else colors[1]
                plt.axhspan(y_pos, y_pos + 1, xmin=-0.05, xmax=0, 
                           color=color, alpha=0.7, clip_on=False, linewidth=2)
            y_pos += 1
        
        plt.tight_layout()
        plt.savefig('fraud_intensity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_distribution_boxplot(self):
        """
        Visualization 4: Box plot showing distribution of fraud cases by region
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Overall regional distribution
        sns.boxplot(data=self.fraud_data, x='region', y='value', ax=ax1, 
                   palette=[colors[0], colors[1]])
        ax1.set_title('Distribution of Fraud Cases by Region', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Region', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        
        # Year-wise distribution
        sns.boxplot(data=self.fraud_data, x='year', y='value', hue='region', ax=ax2,
                   palette=[colors[0], colors[1]])
        ax2.set_title('Year-wise Distribution by Region', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax2.legend(title='Region')
        
        # Crime type distribution
        sns.boxplot(data=self.fraud_data, x='crime_type', y='value', hue='region', ax=ax3,
                   palette=[colors[0], colors[1]])
        ax3.set_title('Distribution by Crime Type and Region', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Crime Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='Region')
        
        # Violin plot for detailed distribution
        sns.violinplot(data=self.fraud_data, x='region', y='value', ax=ax4,
                      palette=[colors[0], colors[1]])
        ax4.set_title('Density Distribution by Region', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Region', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('fraud_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_crime_composition_chart(self):
        """
        Visualization 5: Stacked area chart showing crime composition over time
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Prepare data for stacked area chart
        for idx, region in enumerate(['Western', 'Eastern']):
            region_data = self.fraud_data[self.fraud_data['region'] == region]
            pivot_data = region_data.groupby(['year', 'crime_type'])['value'].sum().unstack(fill_value=0)
            
            # Create stacked area chart
            ax = ax1 if region == 'Western' else ax2
            pivot_data.plot.area(ax=ax, alpha=0.7, color=colors[:len(pivot_data.columns)])
            
            ax.set_title(f'{region} India: Crime Type Composition Over Time', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Year', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
            ax.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('crime_composition_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def prepare_modeling_data(self):
        """
        Prepare data for machine learning models
        """
        print("\nPreparing data for modeling...")
        
        # Create feature matrix
        modeling_data = self.fraud_data.copy()
        
        # Encode categorical variables
        le_region = LabelEncoder()
        le_crime_type = LabelEncoder()
        le_state = LabelEncoder()
        
        modeling_data['region_encoded'] = le_region.fit_transform(modeling_data['region'])
        modeling_data['crime_type_encoded'] = le_crime_type.fit_transform(modeling_data['crime_type'])
        modeling_data['state_encoded'] = le_state.fit_transform(modeling_data['state'])
        
        # Create additional features
        modeling_data['year_normalized'] = modeling_data['year'] - 2018  # Start from 0
        
        # Calculate state-level statistics as features
        state_stats = modeling_data.groupby('state')['value'].agg(['mean', 'std', 'max']).reset_index()
        state_stats.columns = ['state', 'state_mean', 'state_std', 'state_max']
        modeling_data = modeling_data.merge(state_stats, on='state')
        
        # Fill NaN values
        modeling_data['state_std'] = modeling_data['state_std'].fillna(0)
        
        self.modeling_data = modeling_data
        
        # Feature selection
        features = ['year_normalized', 'region_encoded', 'crime_type_encoded', 
                   'state_encoded', 'state_mean', 'state_std', 'state_max']
        target = 'value'
        
        X = modeling_data[features]
        y = modeling_data[target]
        
        return X, y
    
    def train_baseline_model(self, X, y):
        """
        Train a simple baseline model (mean predictor)
        """
        print("\nTraining baseline model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Simple baseline: predict the mean of training set
        baseline_pred = np.full(len(y_test), y_train.mean())
        
        # Calculate metrics
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_mse = mean_squared_error(y_test, baseline_pred)
        baseline_r2 = r2_score(y_test, baseline_pred)
        
        print(f"Baseline Model Performance:")
        print(f"MAE: {baseline_mae:.2f}")
        print(f"MSE: {baseline_mse:.2f}")
        print(f"R² Score: {baseline_r2:.4f}")
        
        return {
            'predictions': baseline_pred,
            'actual': y_test,
            'mae': baseline_mae,
            'mse': baseline_mse,
            'r2': baseline_r2,
            'model_name': 'Baseline (Mean Predictor)'
        }, X_train, X_test, y_train, y_test
    
    def train_advanced_model(self, X_train, X_test, y_train, y_test):
        """
        Train an advanced model (Random Forest)
        """
        print("\nTraining advanced model...")
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)
        
        # Predictions
        rf_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_mse = mean_squared_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        print(f"Random Forest Model Performance:")
        print(f"MAE: {rf_mae:.2f}")
        print(f"MSE: {rf_mse:.2f}")
        print(f"R² Score: {rf_r2:.4f}")
        
        # Feature importance
        feature_names = ['year_normalized', 'region_encoded', 'crime_type_encoded', 
                        'state_encoded', 'state_mean', 'state_std', 'state_max']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return {
            'model': rf_model,
            'predictions': rf_pred,
            'actual': y_test,
            'mae': rf_mae,
            'mse': rf_mse,
            'r2': rf_r2,
            'feature_importance': feature_importance,
            'model_name': 'Random Forest'
        }
    
    def create_model_comparison_plots(self, baseline_results, advanced_results):
        """
        Create visualizations comparing model performance
        """
        print("\nCreating model comparison visualizations...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Prediction vs Actual scatter plot
        ax1.scatter(baseline_results['actual'], baseline_results['predictions'], 
                   alpha=0.6, color=colors[0], label='Baseline', s=30)
        ax1.scatter(advanced_results['actual'], advanced_results['predictions'], 
                   alpha=0.6, color=colors[1], label='Random Forest', s=30)
        
        # Perfect prediction line
        min_val = min(min(baseline_results['actual']), min(advanced_results['actual']))
        max_val = max(max(baseline_results['actual']), max(advanced_results['actual']))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction vs Actual Values', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        baseline_residuals = baseline_results['actual'] - baseline_results['predictions']
        advanced_residuals = advanced_results['actual'] - advanced_results['predictions']
        
        ax2.scatter(baseline_results['predictions'], baseline_residuals, 
                   alpha=0.6, color=colors[0], label='Baseline', s=30)
        ax2.scatter(advanced_results['predictions'], advanced_residuals, 
                   alpha=0.6, color=colors[1], label='Random Forest', s=30)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Model performance comparison
        metrics = ['MAE', 'MSE', 'R² Score']
        baseline_metrics = [baseline_results['mae'], baseline_results['mse'], baseline_results['r2']]
        advanced_metrics = [advanced_results['mae'], advanced_results['mse'], advanced_results['r2']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, baseline_metrics, width, label='Baseline', 
               color=colors[0], alpha=0.8)
        ax3.bar(x + width/2, advanced_metrics, width, label='Random Forest', 
               color=colors[1], alpha=0.8)
        
        ax3.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (baseline_val, advanced_val) in enumerate(zip(baseline_metrics, advanced_metrics)):
            ax3.annotate(f'{baseline_val:.3f}', 
                        xy=(i - width/2, baseline_val), 
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
            ax3.annotate(f'{advanced_val:.3f}', 
                        xy=(i + width/2, advanced_val), 
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        # 4. Feature importance (if available)
        if 'feature_importance' in advanced_results:
            feature_imp = advanced_results['feature_importance']
            bars = ax4.barh(feature_imp['feature'], feature_imp['importance'], 
                           color=colors[2], alpha=0.8)
            ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Features', fontsize=12, fontweight='bold')
            ax4.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax4.annotate(f'{width:.3f}',
                           xy=(width, bar.get_y() + bar.get_height() / 2),
                           xytext=(3, 0), textcoords="offset points",
                           ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_insights(self, baseline_results, advanced_results):
        """
        Generate summary insights from the analysis
        """
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY AND KEY INSIGHTS")
        print("="*80)
        
        # Regional comparison
        regional_totals = self.fraud_data.groupby('region')['value'].sum()
        print(f"\n1. REGIONAL COMPARISON (2018-2022):")
        for region, total in regional_totals.items():
            print(f"   {region} India: {total:,} total fraud cases")
        
        # Yearly trends
        yearly_growth = self.fraud_data.groupby(['region', 'year'])['value'].sum().reset_index()
        print(f"\n2. YEARLY TRENDS:")
        for region in ['Western', 'Eastern']:
            region_data = yearly_growth[yearly_growth['region'] == region]
            if len(region_data) > 1:
                growth_rate = ((region_data['value'].iloc[-1] - region_data['value'].iloc[0]) / 
                              region_data['value'].iloc[0] * 100)
                print(f"   {region} India: {growth_rate:.1f}% growth from 2018 to 2022")
        
        # Model performance
        print(f"\n3. MODEL PERFORMANCE:")
        print(f"   Baseline Model R²: {baseline_results['r2']:.4f}")
        print(f"   Advanced Model R²: {advanced_results['r2']:.4f}")
        print(f"   Improvement: {((advanced_results['r2'] - baseline_results['r2']) / abs(baseline_results['r2']) * 100):.1f}%")
        
        # Top factors
        if 'feature_importance' in advanced_results:
            top_factors = advanced_results['feature_importance'].head(3)
            print(f"\n4. TOP PREDICTIVE FACTORS:")
            for _, row in top_factors.iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
        
        print("\n" + "="*80)

def main():
    """
    Main function to run the complete analysis
    """
    # Initialize analysis
    data_path = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete"
    analyzer = CybercrimeAnalysis(data_path)
    
    # Load and preprocess data
    fraud_data = analyzer.load_and_preprocess_data()
    
    # Perform EDA
    regional_stats, crime_stats = analyzer.exploratory_data_analysis()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Prepare modeling data
    X, y = analyzer.prepare_modeling_data()
    
    # Train models
    baseline_results, X_train, X_test, y_train, y_test = analyzer.train_baseline_model(X, y)
    advanced_results = analyzer.train_advanced_model(X_train, X_test, y_train, y_test)
    
    # Create model comparison plots
    analyzer.create_model_comparison_plots(baseline_results, advanced_results)
    
    # Generate summary insights
    analyzer.generate_summary_insights(baseline_results, advanced_results)
    
    print("\nAnalysis completed successfully!")
    print("All visualizations saved to 'Qn 12' folder.")

if __name__ == "__main__":
    main()
