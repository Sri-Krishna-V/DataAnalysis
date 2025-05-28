
"""
COVID-19 Impact on Cybercrime Reporting in Major Indian Cities (2018-2022)

This script analyzes how the COVID-19 pandemic (2020-2021) affected cybercrime reporting
patterns in major Indian cities using Random Forest classification and visualization techniques.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings

# Try to import additional packages, but gracefully handle if they're not available
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imblearn.over_sampling.SMOTE not available. Continuing without SMOTE.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not available. Continuing without XGBoost.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not available. Continuing without LightGBM.")

# Ignore warnings to keep output clean
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Define output directory
OUTPUT_DIR = 'c:/Users/srikr/Desktop/Studies/Self/Papers/Data Analysis/Complete/Qn 3'


def load_data():
    """Load and preprocess the cybercrime datasets."""
    # Load city-level total cybercrime data
    city_total_path = 'c:/Users/srikr/Desktop/Studies/Self/Papers/Data Analysis/Complete/cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-total-number-of-cyber-crimes-committed-in-india.csv'
    city_data = pd.read_csv(city_total_path)

    # Load city-level IT Act violation data for more detailed analysis
    city_it_path = 'c:/Users/srikr/Desktop/Studies/Self/Papers/Data Analysis/Complete/cyber-crimes-from-ncrb-master-data-year-state-and-city-wise-type-of-cyber-crimes-committed-in-violation-of-it-act.csv'
    city_it_data = pd.read_csv(city_it_path)

    # Filter out total rows and focus only on individual cities
    city_data = city_data[city_data['city'] != 'Total Cities']

    # Filter data for 2018-2022 (2 years pre-COVID, COVID years, and 1 year post-COVID)
    city_data = city_data[(city_data['year'] >= 2018) &
                          (city_data['year'] <= 2022)]

    # Add pandemic period classification column
    city_data['period'] = city_data['year'].apply(
        lambda x: 'Pre-COVID' if x < 2020 else ('COVID' if x <= 2021 else 'Post-COVID'))

    # Add binary classification target for Random Forest (Was this during COVID or not?)
    city_data['is_covid_period'] = city_data['year'].apply(
        lambda x: 1 if (x == 2020 or x == 2021) else 0)

    print(f"Data loaded successfully with {len(city_data)} records.")
    print(f"Years range: {min(city_data['year'])} - {max(city_data['year'])}")
    print(f"Cities included: {len(city_data['city'].unique())}")

    return city_data, city_it_data


def preprocess_data(city_data, city_it_data):
    """Preprocess and enrich the data for analysis with enhanced feature engineering."""

    # Create a more comprehensive dataset for the machine learning models
    # First, prepare IT Act violations summary by city and year
    city_it_summary = city_it_data.groupby(['year', 'city', 'offence_category'])[
        'value'].sum().reset_index()

    # Pivot to get offence categories as columns
    it_pivot = city_it_summary.pivot_table(
        index=['year', 'city'],
        columns='offence_category',
        values='value',
        fill_value=0
    ).reset_index()

    # Merge with the main city data
    merged_data = pd.merge(
        city_data,
        it_pivot,
        on=['year', 'city'],
        how='left'
    )

    # Fill NaN values with 0 for crime categories
    merged_data = merged_data.fillna(0)

    # Add population proxy by city (based on 2021 estimated data)
    # This is important because cybercrime rates might be population-dependent
    population_data = {
        'Bengaluru': 12000000,  # 12 million
        'Mumbai': 20000000,     # 20 million
        'Delhi': 30000000,      # 30 million
        'Hyderabad': 10000000,  # 10 million
        'Chennai': 8000000,     # 8 million
        'Kolkata': 15000000,    # 15 million
        'Pune': 6500000,        # 6.5 million
        'Ahmedabad': 8000000,   # 8 million
        'Jaipur': 4000000,      # 4 million
        'Lucknow': 3500000,     # 3.5 million
        'Surat': 7000000,       # 7 million
        'Kanpur': 3000000,      # 3 million
        'Nagpur': 2800000,      # 2.8 million
        'Patna': 2200000,       # 2.2 million
        'Indore': 2100000,      # 2.1 million
        'Kochi': 3000000,       # 3 million
        'Kozhikode': 2200000,   # 2.2 million
        'Ghaziabad': 1800000,   # 1.8 million
        'Coimbatore': 1800000   # 1.8 million
    }

    # Add population data to the DataFrame
    merged_data['population'] = merged_data['city'].map(population_data)

    # ENHANCED FEATURE ENGINEERING

    # Basic crime metrics
    merged_data['crime_rate'] = (
        merged_data['value'] / merged_data['population']) * 100000

    # Calculate digital literacy proxy (IT sector employment per capita - approximation)
    digital_literacy = {
        'Bengaluru': 0.12,    # High tech presence
        'Mumbai': 0.08,       # Financial hub
        'Delhi': 0.06,        # Government and mixed economy
        'Hyderabad': 0.10,    # Strong IT sector
        'Chennai': 0.09,      # Growing tech hub
        'Kolkata': 0.05,      # Traditional industries
        'Pune': 0.11,         # Education and IT
        'Ahmedabad': 0.07,    # Industrial
        'Jaipur': 0.04,       # Tourism and some tech
        'Lucknow': 0.03,      # Administrative center
        'Surat': 0.05,        # Diamond industry
        'Kanpur': 0.02,       # Manufacturing
        'Nagpur': 0.03,       # Central India hub
        'Patna': 0.02,        # Less developed tech sector
        'Indore': 0.04,       # Emerging tech
        'Kochi': 0.08,        # IT park
        'Kozhikode': 0.04,    # Traditional commerce
        'Ghaziabad': 0.05,    # NCR industrial
        'Coimbatore': 0.06    # Manufacturing and education
    }
    merged_data['digital_literacy'] = merged_data['city'].map(digital_literacy)

    # Internet penetration estimates
    internet_penetration = {
        'Bengaluru': 0.85,
        'Mumbai': 0.82,
        'Delhi': 0.80,
        'Hyderabad': 0.78,
        'Chennai': 0.76,
        'Kolkata': 0.72,
        'Pune': 0.80,
        'Ahmedabad': 0.74,
        'Jaipur': 0.68,
        'Lucknow': 0.65,
        'Surat': 0.70,
        'Kanpur': 0.60,
        'Nagpur': 0.65,
        'Patna': 0.58,
        'Indore': 0.70,
        'Kochi': 0.75,
        'Kozhikode': 0.68,
        'Ghaziabad': 0.73,
        'Coimbatore': 0.72
    }
    merged_data['internet_penetration'] = merged_data['city'].map(
        internet_penetration)

    # Feature interactions
    merged_data['tech_vulnerability'] = merged_data['digital_literacy'] * \
        merged_data['internet_penetration']
    merged_data['cybercrime_susceptibility'] = merged_data['tech_vulnerability'] * \
        merged_data['population'] / 1000000

    # Generate lag features (previous year's crime rates)
    city_year_crime = merged_data.groupby(['city', 'year'])[
        'value'].sum().reset_index()
    city_year_crime.columns = ['city', 'year', 'total_crimes']

    # Create lag features
    city_year_crime['prev_year'] = city_year_crime['year'] - 1
    lagged_data = city_year_crime[['city', 'prev_year', 'total_crimes']]
    lagged_data.columns = ['city', 'year', 'prev_year_crimes']

    # Merge lagged data
    merged_data = pd.merge(merged_data, lagged_data, on=[
                           'city', 'year'], how='left')
    merged_data['prev_year_crimes'].fillna(0, inplace=True)

    # Calculate growth rates and ratios
    merged_data['crime_growth'] = np.where(
        merged_data['prev_year_crimes'] > 0,
        (merged_data['value'] - merged_data['prev_year_crimes']) /
        merged_data['prev_year_crimes'],
        0
    )

    # Calculate ratios between different crime types
    crime_types = ['Computer Related Offences', 'Cyber Terrorism',
                   'Publication or transmission of Obscene or Sexually Explicit Act in Electronic Form',
                   'Tampering Computer Source documents']

    # Total of all cybercrime types for ratio calculation
    merged_data['total_cybercrimes'] = merged_data[crime_types].sum(axis=1)

    # Create proportion features for each crime type
    for crime_type in crime_types:
        if crime_type in merged_data.columns:
            col_name = f"{crime_type.replace(' ', '_')}_ratio"
            merged_data[col_name] = np.where(
                merged_data['total_cybercrimes'] > 0,
                merged_data[crime_type] / merged_data['total_cybercrimes'],
                0
            )

    # Encode categorical variables for machine learning
    le_city = LabelEncoder()
    le_state = LabelEncoder()
    merged_data['city_encoded'] = le_city.fit_transform(merged_data['city'])
    merged_data['state_encoded'] = le_state.fit_transform(merged_data['state'])

    # Create period-specific binary features (one-hot encoding of period)
    merged_data['is_pre_covid'] = (
        merged_data['period'] == 'Pre-COVID').astype(int)
    merged_data['is_post_covid'] = (
        merged_data['period'] == 'Post-COVID').astype(int)

    # Create interaction terms
    merged_data['covid_tech_interaction'] = merged_data['is_covid_period'] * \
        merged_data['tech_vulnerability']
    merged_data['covid_internet_interaction'] = merged_data['is_covid_period'] * \
        merged_data['internet_penetration']

    print(f"Enhanced dataset created with {merged_data.shape[1]} features")

    return merged_data


def visualize_total_crimes_by_period(data):
    """Create a bar chart comparing total cybercrime counts across pandemic periods."""

    # Group by period and calculate total crimes
    period_totals = data.groupby('period')['value'].sum().reset_index()

    plt.figure(figsize=(10, 7))
    sns.barplot(x='period', y='value', data=period_totals,
                order=['Pre-COVID', 'COVID', 'Post-COVID'])

    plt.title('Total Cybercrime Cases by Pandemic Period', fontsize=16)
    plt.xlabel('Period', fontsize=14)
    plt.ylabel('Number of Cybercrime Cases', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add value labels on top of bars
    for index, row in period_totals.iterrows():
        plt.text(index, row['value'] + 500, f'{int(row["value"]):,}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, 'total_crimes_by_period.png'), dpi=300)
    plt.close()


def visualize_city_comparison(data):
    """Create a visualization comparing cybercrime rates in major cities before and during COVID."""

    # Select top 10 cities by total cybercrime cases
    top_cities = data.groupby(
        'city')['value'].sum().nlargest(10).index.tolist()
    city_period_data = data[data['city'].isin(top_cities)]

    # Pivot data for easier plotting
    pivot_data = city_period_data.pivot_table(
        index='city',
        columns='period',
        values='value',
        aggfunc='sum'
    ).reset_index()

    # Melt data for seaborn
    melted_data = pd.melt(
        pivot_data,
        id_vars='city',
        value_vars=['Pre-COVID', 'COVID', 'Post-COVID'],
        var_name='period',
        value_name='cases'
    )

    # Calculate growth rate from Pre-COVID to COVID
    growth_data = pivot_data.copy()
    growth_data['growth_rate'] = (
        growth_data['COVID'] - growth_data['Pre-COVID']) / growth_data['Pre-COVID'] * 100

    # Sort by COVID cases
    melted_data = melted_data.merge(
        growth_data[['city', 'growth_rate']],
        on='city',
        how='left'
    )

    # Sort cities by COVID cases
    city_order = pivot_data.sort_values(
        'COVID', ascending=False)['city'].tolist()
    melted_data['city'] = pd.Categorical(
        melted_data['city'], categories=city_order, ordered=True)

    plt.figure(figsize=(14, 10))
    chart = sns.barplot(
        x='city',
        y='cases',
        hue='period',
        data=melted_data.sort_values('city'),
        palette=['#2c7fb8', '#e41a1c', '#4daf4a']
    )

    # Add growth rate labels
    for i, city in enumerate(city_order):
        growth = growth_data.loc[growth_data['city']
                                 == city, 'growth_rate'].values[0]
        plt.text(
            i,
            5,
            f'{growth:.1f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='black' if growth >= 0 else 'red'
        )

    plt.title(
        'Cybercrime Cases in Top 10 Cities: Pre-COVID vs. COVID vs. Post-COVID', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Number of Cases', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Period', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Add annotation for growth rate
    plt.figtext(0.5, 0.01, 'Percentage numbers show growth rate from Pre-COVID to COVID period',
                ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'city_comparison.png'), dpi=300)
    plt.close()


def visualize_crime_types_heatmap(data, city_it_data):
    """Create a heatmap showing how different cybercrime types changed during the pandemic."""

    # Filter IT Act data for the relevant periods
    filtered_it_data = city_it_data[(city_it_data['year'] >= 2018) & (
        city_it_data['year'] <= 2022)]

    # Add pandemic period
    filtered_it_data['period'] = filtered_it_data['year'].apply(
        lambda x: 'Pre-COVID' if x < 2020 else ('COVID' if x <= 2021 else 'Post-COVID'))

    # Focus on the top 10 cities by total cybercrimes
    top_cities = data.groupby(
        'city')['value'].sum().nlargest(10).index.tolist()
    top_city_data = filtered_it_data[filtered_it_data['city'].isin(top_cities)]

    # Group by period and crime category
    crime_period_data = top_city_data.groupby(['period', 'offence_category'])[
        'value'].sum().reset_index()

    # Convert to wide format for heatmap
    crime_pivot = crime_period_data.pivot(
        index='offence_category',
        columns='period',
        values='value'
    )

    # Calculate percentage change from Pre-COVID to COVID
    crime_pivot['change_pct'] = (
        (crime_pivot['COVID'] - crime_pivot['Pre-COVID']) / crime_pivot['Pre-COVID'] * 100)

    # Filter top crime categories by volume
    top_crimes = crime_pivot.sum(axis=1).nlargest(10).index.tolist()
    crime_pivot = crime_pivot.loc[top_crimes]

    # Sort by percentage change
    crime_pivot = crime_pivot.sort_values('change_pct', ascending=False)

    # Create the heatmap
    plt.figure(figsize=(14, 10))

    # Drop the change_pct column for the heatmap
    heatmap_data = crime_pivot.drop(columns=['change_pct'])

    # Create the main heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': 'Number of Cases'}
    )

    # Add a second column with percentage changes
    ax2 = plt.twinx()

    # Position for percentage change text
    for i, (idx, row) in enumerate(crime_pivot.iterrows()):
        change = row['change_pct']
        color = 'green' if change >= 0 else 'red'
        plt.text(
            3.1,
            i + 0.5,
            f"{change:.1f}%",
            ha='left',
            va='center',
            fontsize=11,
            fontweight='bold',
            color=color
        )

    # Hide the second y-axis
    ax2.set_yticks([])

    # Adjust the title and labels
    plt.title('Change in Cybercrime Types During COVID-19 Pandemic', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'crime_types_heatmap.png'), dpi=300)
    plt.close()


def train_models(data):
    """Train advanced ML models to identify factors influencing COVID period crimes.
    Implements proper cross-validation and regularization to prevent overfitting."""

    # Prepare features and target
    # Select all potentially useful features from our enhanced feature set
    exclude_cols = ['year', 'state', 'city', 'value', 'period', 'is_covid_period',
                    'total_cybercrimes', 'unit', 'notes']

    # Get numerical columns for features
    feature_cols = [col for col in data.columns if col not in exclude_cols
                    and data[col].dtype in ['int64', 'float64']]

    # Use time-based split instead of random split, as this is time-series data
    # Sort data by year to preserve temporal order
    data_sorted = data.sort_values('year')

    # Limit to a reasonable number of features to prevent overfitting
    # For a dataset with ~95 records, we should use much fewer features
    if len(feature_cols) > 10:
        print(f"Original number of features: {len(feature_cols)}")
        # Use correlation analysis to eliminate highly correlated features
        X_corr = data[feature_cols].corr().abs()
        # Upper triangle of correlation matrix
        upper = X_corr.where(np.triu(np.ones(X_corr.shape), k=1).astype(bool))
        # Find features with correlation greater than 0.8
        to_drop = [column for column in upper.columns if any(
            upper[column] > 0.8)]
        print(
            f"Dropping {len(to_drop)} highly correlated features: {', '.join(to_drop)}")
        feature_cols = [col for col in feature_cols if col not in to_drop]
        print(f"Reduced to {len(feature_cols)} features")

    print(f"Using {len(feature_cols)} features for model training")

    X = data_sorted[feature_cols]
    y = data_sorted['is_covid_period']

    # Use time-based splitting (earlier years for training, later for testing)
    # This is more realistic for time series data than random splitting
    n_samples = X.shape[0]
    train_size = int(0.7 * n_samples)  # Use 70% for training

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    print(
        f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    print(
        f"Train set years: {data_sorted['year'].iloc[:train_size].min()} to {data_sorted['year'].iloc[:train_size].max()}")
    print(
        f"Test set years: {data_sorted['year'].iloc[train_size:].min()} to {data_sorted['year'].iloc[train_size:].max()}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Check class distribution
    print(f"Target class distribution (train): {np.bincount(y_train)}")
    print(f"Target class distribution (test): {np.bincount(y_test)}")

    # Apply SMOTE to handle class imbalance if needed and available
    if SMOTE_AVAILABLE and len(np.unique(y_train)) > 1 and np.min(np.bincount(y_train)) < 10:
        print("Applying SMOTE for class balancing...")
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(
                5, np.min(np.bincount(y_train))-1))
            X_train_scaled, y_train = smote.fit_resample(
                X_train_scaled, y_train)
            print(f"After SMOTE, class distribution: {np.bincount(y_train)}")
        except Exception as e:
            print(f"SMOTE error: {e}. Continuing without SMOTE.")

    # Feature selection - use a simpler method with fewer features due to small dataset
    print("Performing more aggressive feature selection to prevent overfitting...")

    # Use a simple correlation-based selector for transparency
    if X_train.shape[1] > 5:
        # Calculate correlation with target
        corr_with_target = []
        for i, col in enumerate(feature_cols):
            corr = np.corrcoef(X_train[col], y_train)[0, 1]
            corr_with_target.append((i, abs(corr)))

        # Select top 5 features with highest correlation with target
        top_features_idx = [idx for idx, _ in sorted(
            corr_with_target, key=lambda x: x[1], reverse=True)[:5]]

        # Create selector mask
        selector_mask = np.zeros(len(feature_cols), dtype=bool)
        selector_mask[top_features_idx] = True

        # Get selected feature names
        selected_features = [feature_cols[i] for i in top_features_idx]
        print(
            f"Selected top 5 features by correlation: {', '.join(selected_features)}")

        # Filter features
        X_train_selected = X_train_scaled[:, selector_mask]
        X_test_selected = X_test_scaled[:, selector_mask]
    else:
        selector_mask = np.ones(len(feature_cols), dtype=bool)
        selected_features = feature_cols
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled

    # Set up cross-validation to get more reliable estimates
    cv = 5  # 5-fold cross-validation

    # Train models with proper regularization to prevent overfitting
    print("Training models with regularization to prevent overfitting...")

    # 1. Decision Tree with pruning as baseline
    dt_model = DecisionTreeClassifier(
        max_depth=3,  # Limit depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )

    # Initialize model dictionary
    models = {'Decision Tree (Baseline)': dt_model}

    # 2. Random Forest with regularization
    rf_model = RandomForestClassifier(
        n_estimators=100,  # Fewer estimators
        max_depth=4,  # Shallower trees to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',  # Feature subsampling
        bootstrap=True,
        random_state=42
    )
    models['Random Forest'] = rf_model

    # 3. Gradient Boosting with regularization
    gb_model = GradientBoostingClassifier(
        n_estimators=50,  # Fewer estimators
        learning_rate=0.05,  # Slower learning rate
        max_depth=3,  # Shallower trees
        subsample=0.8,  # Use 80% of samples
        random_state=42
    )
    models['Gradient Boosting'] = gb_model

    # 4. Support Vector Machine with proper regularization
    svm_model = SVC(
        C=1.0,  # Standard regularization (smaller C = more regularization)
        kernel='rbf',
        gamma='scale',
        probability=True,
        random_state=42
    )
    models['SVM'] = svm_model

    # 5. AdaBoost with limited iterations
    ada_model = AdaBoostClassifier(
        n_estimators=50,  # Fewer estimators
        learning_rate=0.05,  # Slower learning rate
        random_state=42
    )
    models['AdaBoost'] = ada_model

    # 6. Logistic Regression with L2 regularization
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression(
        C=1.0,  # Standard regularization
        penalty='l2',
        solver='liblinear',
        random_state=42
    )
    models['Logistic Regression'] = lr_model

    # Add XGBoost and LightGBM only if available
    if XGBOOST_AVAILABLE:
        xgb_model = XGBClassifier(
            learning_rate=0.05,  # Slower learning rate
            n_estimators=50,     # Fewer estimators
            max_depth=3,         # Shallower trees
            subsample=0.8,       # Use 80% of samples
            colsample_bytree=0.8,  # Use 80% of features
            reg_alpha=1.0,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        models['XGBoost'] = xgb_model

    if LIGHTGBM_AVAILABLE:
        lgbm_model = LGBMClassifier(
            learning_rate=0.05,  # Slower learning rate
            n_estimators=50,     # Fewer estimators
            max_depth=3,         # Shallower trees
            subsample=0.8,       # Use 80% of samples
            colsample_bytree=0.8,  # Use 80% of features
            reg_alpha=1.0,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
            random_state=42
        )
        models['LightGBM'] = lgbm_model

    # Use cross-validation to get more reliable performance estimates
    cv_results = {}
    for name, model in models.items():
        print(f"Training {name} with {cv}-fold cross-validation...")

        # Use cross_val_score to get accuracy across folds
        cv_scores = cross_val_score(
            model, X_train_selected, y_train,
            cv=cv, scoring='accuracy'
        )

        # Fit model on entire training set
        model.fit(X_train_selected, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test_selected)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # Store results
        cv_results[name] = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'model': model,
            'cv_scores': cv_scores
        }

        print(f"{name} - CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    # Check for signs of overfitting
    for name, results in cv_results.items():
        cv_acc = results['cv_accuracy_mean']
        test_acc = results['test_accuracy']
        diff = cv_acc - test_acc

        overfit_status = "No overfit" if abs(
            diff) < 0.1 else "Potential overfit" if diff > 0 else "Underfit"
        print(
            f"{name}: CV={cv_acc:.4f}, Test={test_acc:.4f}, Diff={diff:.4f} - {overfit_status}")

    # Find best model based on lowest overfitting (smallest gap between CV and test accuracy)
    best_model_name = min(cv_results.items(), key=lambda x: abs(
        x[1]['cv_accuracy_mean'] - x[1]['test_accuracy']))[0]

    # Find second best model
    sorted_models = sorted(cv_results.items(), key=lambda x: abs(
        x[1]['cv_accuracy_mean'] - x[1]['test_accuracy']))
    second_best_model_name = sorted_models[1][0] if len(
        sorted_models) > 1 else best_model_name

    print(f"Best model (least overfit): {best_model_name}")
    print(f"Second best model: {second_best_model_name}")

    # For the model comparison plot, use a dictionary with the same structure as before
    plot_results = {name: {
        'accuracy': results['test_accuracy'],
        'precision': results['test_precision'],
        'recall': results['test_recall'],
        'f1': results['test_f1'],
        'model': results['model']
    } for name, results in cv_results.items()}

    # Return results
    return (
        cv_results[best_model_name]['model'],
        cv_results[second_best_model_name]['model'],
        X, y, X_test_selected, y_test,
        cv_results[best_model_name]['test_accuracy'],
        cv_results[second_best_model_name]['test_accuracy'],
        selected_features,
        best_model_name,
        second_best_model_name,
        plot_results
    )


def visualize_feature_importance(model, feature_names, model_name):
    """Create a visualization of feature importance from the best model."""

    plt.figure(figsize=(14, 10))

    # Different models have different ways to access feature importance
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_[0])
    elif model_name == 'Voting Ensemble':
        # For voting classifier, use the first base estimator that has feature_importances_
        for _, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
                break
    elif model_name == 'Neural Network':
        # For neural networks, use the weights of the first layer as a proxy
        importances = np.abs(model.coefs_[0]).sum(axis=1)
    elif model_name == 'SVM':
        # For SVM, we can't directly get feature importances
        # Just create a uniform distribution as placeholder
        importances = np.ones(len(feature_names)) / len(feature_names)
    else:
        # Default placeholder if we can't determine importances
        importances = np.ones(len(feature_names)) / len(feature_names)

    # If lengths don't match, handle it gracefully
    if len(importances) != len(feature_names):
        print(
            f"Warning: Feature importance length ({len(importances)}) doesn't match feature names ({len(feature_names)})")
        # Use only the available feature names
        if len(importances) < len(feature_names):
            feature_names = feature_names[:len(importances)]
        else:
            importances = importances[:len(feature_names)]

    # Sort indices by importance
    indices = np.argsort(importances)

    # Get top 15 features for better visualization
    if len(indices) > 15:
        indices = indices[-15:]

    plt.barh(range(len(indices)),
             importances[indices], align='center', color='#1f77b4')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title(
        f'{model_name} Feature Importance for COVID Period Classification', fontsize=16)

    # Add value labels to the bars
    for i, v in enumerate(importances[indices]):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300)
    plt.close()


def visualize_decision_tree(dt_model, feature_names, X, y):
    """Visualize a decision tree to highlight key factors influencing cybercrime rates."""

    # If we got a tree-based model directly, use it
    if isinstance(dt_model, DecisionTreeClassifier):
        tree_model = dt_model
    else:
        # Otherwise create a new decision tree with limited depth for visualization
        tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree_model.fit(X, y)

    plt.figure(figsize=(20, 12))

    # Handle potential different number of features
    if hasattr(tree_model, 'n_features_in_'):
        n_features = tree_model.n_features_in_
        if n_features != len(feature_names):
            print(
                f"Warning: Tree model has {n_features} features but {len(feature_names)} names provided")
            if n_features < len(feature_names):
                feature_names = feature_names[:n_features]
            else:
                # Pad feature names if needed
                feature_names = list(
                    feature_names) + [f"Feature_{i}" for i in range(len(feature_names), n_features)]

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=['Non-COVID', 'COVID'],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=10
    )

    plt.title(
        'Decision Tree for COVID vs Non-COVID Period Classification', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'decision_tree.png'), dpi=300)
    plt.close()


def compare_model_performance(results, cv_results=None):
    """Create a bar chart comparing the performance of models with train/test metrics.
    Includes cross-validation scores when available to identify overfitting."""

    # Sort models by test accuracy
    sorted_results = dict(
        sorted(results.items(), key=lambda x: x[1]['accuracy']))

    model_names = list(sorted_results.keys())
    test_accuracies = [results[model]['accuracy'] for model in model_names]
    test_f1_scores = [results[model]['f1'] for model in model_names]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(
        16, 14), gridspec_kw={'height_ratios': [3, 1]})

    # Set up bar positions
    x = np.arange(len(model_names))
    width = 0.35

    # Create top subplot bars (main metrics)
    bars1 = ax1.bar(x - width/2, test_accuracies, width,
                    label='Test Accuracy', color='#1f77b4')
    bars2 = ax1.bar(x + width/2, test_f1_scores, width,
                    label='Test F1 Score', color='#2ca02c')

    # Add validation metrics if available
    if cv_results:
        # Extract CV accuracy means and standard deviations
        cv_means = []
        cv_stds = []

        for name in model_names:
            if name in cv_results and 'cv_accuracy_mean' in cv_results[name]:
                cv_means.append(cv_results[name]['cv_accuracy_mean'])
                cv_stds.append(cv_results[name]['cv_accuracy_std'])
            else:
                cv_means.append(0)
                cv_stds.append(0)

        # Add cross-validation accuracy with error bars
        ax1.errorbar(x - width/2, cv_means, yerr=cv_stds, fmt='o', color='black',
                     label='CV Accuracy (with std dev)')

    # Add target line
    ax1.axhline(y=0.75, color='red', linestyle='--',
                alpha=0.7, label='Target Accuracy (75%)')

    # Add value labels for test accuracy
    for i, (test_acc, f1) in enumerate(zip(test_accuracies, test_f1_scores)):
        ax1.text(i - width/2, test_acc + 0.02,
                 f'{test_acc:.2f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, f1 + 0.02,
                 f'{f1:.2f}', ha='center', va='bottom', fontsize=10)

    # Add labels and title to top subplot
    ax1.set_xlabel('Models', fontsize=14)
    ax1.set_ylabel('Score', fontsize=14)
    ax1.set_title('Model Performance Comparison', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Add a second subplot to show overfitting
    if cv_results:
        # Calculate the difference between CV and test accuracy (overfitting measure)
        overfit_scores = []
        for name in model_names:
            if name in cv_results and 'cv_accuracy_mean' in cv_results[name]:
                diff = cv_results[name]['cv_accuracy_mean'] - \
                    results[name]['accuracy']
                overfit_scores.append(diff)
            else:
                overfit_scores.append(0)

        # Create the overfitting bars
        bars = ax2.bar(x, overfit_scores, width, label='CV - Test Accuracy')

        # Color bars based on overfitting severity
        for i, score in enumerate(overfit_scores):
            if score > 0.2:  # Severe overfitting
                bars[i].set_color('red')
            elif score > 0.1:  # Moderate overfitting
                bars[i].set_color('orange')
            elif score > 0:  # Slight overfitting
                bars[i].set_color('yellow')
            else:  # No overfitting or underfitting
                bars[i].set_color('green')

        # Add value labels
        for i, v in enumerate(overfit_scores):
            ax2.text(i, v + 0.01 if v >= 0 else v - 0.05,
                     f'{v:.2f}', ha='center', va='bottom' if v >= 0 else 'top',
                     fontsize=10, color='black')

        # Add labels and title to bottom subplot
        ax2.set_xlabel('Models', fontsize=14)
        ax2.set_ylabel('Overfitting Score\n(CV - Test)', fontsize=12)
        ax2.set_title(
            'Overfitting Analysis: CV Accuracy - Test Accuracy', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=12)

        # Add a reference line at 0 (no overfitting)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add reference lines for different levels of overfitting
        ax2.axhline(y=0.1, color='orange', linestyle='--',
                    alpha=0.5, label='Moderate Overfitting')
        ax2.axhline(y=0.2, color='red', linestyle='--',
                    alpha=0.5, label='Severe Overfitting')

        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=300)
    plt.close()


def main():
    """Main function to execute the analysis pipeline."""
    print("Starting analysis of COVID-19 impact on cybercrime reporting in Indian cities...")

    # Load data
    city_data, city_it_data = load_data()

    # Preprocess data
    processed_data = preprocess_data(city_data, city_it_data)

    # Data visualizations
    print("Generating visualizations...")
    visualize_total_crimes_by_period(processed_data)
    visualize_city_comparison(processed_data)
    visualize_crime_types_heatmap(processed_data, city_it_data)

    # Train models with improved regularization to prevent overfitting
    print("Training classification models with proper regularization...")
    (
        best_model, second_best_model, X, y, X_test, y_test,
        best_accuracy, second_best_accuracy, feature_names,
        best_model_name, second_best_model_name, all_results
    ) = train_models(processed_data)

    # Extract CV results if they are included in model training output
    cv_results = {}
    for model_name, result in all_results.items():
        if hasattr(result, 'get') and isinstance(result, dict) and 'cv_accuracy_mean' in result:
            cv_results[model_name] = result

    # Model visualization and evaluation
    visualize_feature_importance(best_model, feature_names, best_model_name)

    # If second best model is a tree-based model, visualize its decision tree
    if second_best_model_name == 'Decision Tree (Baseline)':
        visualize_decision_tree(second_best_model, feature_names, X, y)
    else:
        # If it's not a decision tree, use the basic decision tree model for visualization
        dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        X_train = X.iloc[:int(0.7 * len(X))]
        y_train = y.iloc[:int(0.7 * len(y))]
        dt_model.fit(X_train, y_train)
        visualize_decision_tree(dt_model, feature_names, X, y)

    # Add learning curve visualization to diagnose bias-variance tradeoff
    # Import the function's code from the temporary file
    with open('c:/Users/srikr/Desktop/Studies/Self/Papers/Data Analysis/Complete/Qn 3/learning_curve_function.py', 'r') as f:
        exec(f.read())

    # Get training data
    X_train = X.iloc[:int(0.7 * len(X))]
    y_train = y.iloc[:int(0.7 * len(y))]

    # Compare all models with cross-validation results
    if cv_results:
        compare_model_performance(all_results, cv_results)
    else:
        compare_model_performance(all_results)

    print("Analysis complete. All visualizations saved to output directory.")


if __name__ == "__main__":
    main()
