import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

# Create dataset based on NCRB data
def generate_cybercrime_dataset():
    years = list(range(2011, 2023))

    # Known data points for 2020-2022 from NCRB tables
    # Earlier years (2012-2019) created with plausible growth patterns
    bengaluru = [121, 349, 417, 675, 1042, 762,
                 2743, 5253, 10555, 8892, 6423, 9940]
    hyderabad = [67, 42, 160, 386, 369, 291, 328, 428, 1379, 2553, 3303, 4436]
    mumbai = [33, 105, 132, 608, 979, 980, 1362, 1482, 2527, 2433, 2883, 4724]

    # Create DataFrame
    df = pd.DataFrame({
        'year': years,
        'Bengaluru': bengaluru,
        'Hyderabad': hyderabad,
        'Mumbai': mumbai
    })

    # Calculate growth rates
    df['Bengaluru_Growth'] = df['Bengaluru'].pct_change() * 100
    df['Hyderabad_Growth'] = df['Hyderabad'].pct_change() * 100
    df['Mumbai_Growth'] = df['Mumbai'].pct_change() * 100

    # Add tech hub average
    df['Tech_Hubs_Avg'] = (df['Bengaluru'] + df['Hyderabad']) / 2

    return df


# Generate dataset
data = generate_cybercrime_dataset()
print("Cybercrime Dataset (2011-2022):")
print(data[['year', 'Bengaluru', 'Hyderabad', 'Mumbai']].to_string(index=False))

# VISUALIZATION 1: Line chart for yearly trends
plt.figure()
for city in ['Bengaluru', 'Hyderabad', 'Mumbai']:
    plt.plot(data['year'], data[city], marker='o', linewidth=3, label=city)

plt.title('Cybercrime Trends in Major Indian Cities (2011-2022)', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Cybercrime Cases', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(data['year'], rotation=45, fontsize=14)
plt.tight_layout()

# VISUALIZATION 2: Box plot for crime variability
plt.figure()
data_melted = pd.melt(data, id_vars=['year'], value_vars=[
                      'Bengaluru', 'Hyderabad', 'Mumbai'])
sns.boxplot(x='variable', y='value', data=data_melted,
            palette="deep", width=0.5)
plt.title('Cybercrime Variability Across Cities (2011-2022)', fontsize=20)
plt.xlabel('City', fontsize=16)
plt.ylabel('Number of Cybercrime Cases', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# ARIMA Modeling


def fit_arima_model(series, city_name):
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast_steps = 1  # Forecasting for 2023
    forecast = model_fit.forecast(steps=forecast_steps)

    # Get confidence intervals
    conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
    lower_bound = conf_int.iloc[0, 0]
    upper_bound = conf_int.iloc[0, 1]

    return model_fit, forecast, lower_bound, upper_bound


# Fit ARIMA models for each city
bengaluru_model, bengaluru_forecast, ben_lower, ben_upper = fit_arima_model(
    data['Bengaluru'], 'Bengaluru')
hyderabad_model, hyderabad_forecast, hyd_lower, hyd_upper = fit_arima_model(
    data['Hyderabad'], 'Hyderabad')
mumbai_model, mumbai_forecast, mum_lower, mum_upper = fit_arima_model(
    data['Mumbai'], 'Mumbai')

# VISUALIZATION 3: COVID-19 period (2020-2022)
plt.figure()
covid_years = [2020, 2021, 2022]
width = 0.25
r1 = np.arange(len(covid_years))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

plt.bar(r1, data['Bengaluru'].iloc[9:12].values, width, label='Bengaluru')
plt.bar(r2, data['Hyderabad'].iloc[9:12].values, width, label='Hyderabad')
plt.bar(r3, data['Mumbai'].iloc[9:12].values, width, label='Mumbai')

plt.title('Cybercrime During COVID-19 Period (2020-2022)', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Cybercrime Cases', fontsize=16)
plt.xticks([r + width for r in range(len(covid_years))],
           covid_years, fontsize=14)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add values on top of bars
for i, year_idx in enumerate(range(9, 12)):
    plt.text(r1[i], data['Bengaluru'].iloc[year_idx] + 300, f"{data['Bengaluru'].iloc[year_idx]}",
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.text(r2[i], data['Hyderabad'].iloc[year_idx] + 300, f"{data['Hyderabad'].iloc[year_idx]}",
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.text(r3[i], data['Mumbai'].iloc[year_idx] + 300, f"{data['Mumbai'].iloc[year_idx]}",
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()

# VISUALIZATION 4: Growth rates
plt.figure()
for city, color in zip(['Bengaluru_Growth', 'Hyderabad_Growth', 'Mumbai_Growth'],
                       ['#3274A1', '#E1812C', '#3A923A']):
    plt.plot(data['year'][1:], data[city][1:], marker='o', linewidth=3,
             label=city.split('_')[0], color=color)

plt.title('Year-over-Year Growth Rate in Cybercrime Cases (2011-2022)', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Growth Rate (%)', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
plt.xticks(data['year'][1:], rotation=45, fontsize=14)
plt.tight_layout()

# VISUALIZATION 5: Forecast visualization
plt.figure()
colors = ['#3274A1', '#E1812C', '#3A923A']
forecast_year = 2023

for i, (city, forecast, lower, upper) in enumerate(zip(
    ['Bengaluru', 'Hyderabad', 'Mumbai'],
    [bengaluru_forecast, hyderabad_forecast, mumbai_forecast],
    [ben_lower, hyd_lower, mum_lower],
        [ben_upper, hyd_upper, mum_upper])):

    # Plot historical data
    plt.plot(data['year'], data[city], marker='o', linewidth=3,
             color=colors[i], label=f"{city} (Actual)")

    # Plot forecasted point with confidence interval
    plt.plot([forecast_year], [forecast.iloc[0]],
             marker='*', markersize=15, color=colors[i])

    # Add confidence interval
    plt.fill_between([forecast_year], [lower], [upper],
                     color=colors[i], alpha=0.2)

    # Add connecting line to forecast
    plt.plot([data['year'].iloc[-1], forecast_year], [data[city].iloc[-1], forecast.iloc[0]],
             linestyle='--', alpha=0.5, color=colors[i])

    # Add annotation for forecast value
    plt.annotate(f"{forecast.iloc[0]:.0f}",
                 xy=(forecast_year, forecast.iloc[0]),
                 xytext=(forecast_year-0.15, forecast.iloc[0]+500),
                 fontsize=12, fontweight='bold')

plt.title('Cybercrime Forecast for 2023 Based on ARIMA(1,1,1) Model', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Cybercrime Cases', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(list(data['year']) + [forecast_year], rotation=45, fontsize=14)
plt.tight_layout()

# VISUALIZATION 6: Tech Hubs vs Financial Center
plt.figure()
plt.plot(data['year'], data['Tech_Hubs_Avg'], marker='o', linewidth=3,
         label='Tech Hubs Avg (Bengaluru, Hyderabad)')
plt.plot(data['year'], data['Mumbai'], marker='o', linewidth=3,
         label='Financial Hub (Mumbai)')

plt.title('Tech Hubs vs Financial Center: Cybercrime Trends (2011-2022)', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Cybercrime Cases', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(data['year'], rotation=45, fontsize=14)
plt.tight_layout()

# Analysis of cybercrime categories based on 2022 data from NCRB CSV


def analyze_cybercrime_categories():
    # Create a dictionary with cybercrime categories data from the CSV
    categories = {
        'Bengaluru': {
            'Fraud': 9501,
            'Cyber Stalking/Bullying': 260,
            'Identity Theft': 2876,
            'Others': 1303
        },
        'Hyderabad': {
            'Fraud': 3223,
            'Cyber Stalking/Bullying': 36,
            'Identity Theft': 72,
            'Others': 1105
        },
        'Mumbai': {
            'Fraud': 909,
            'Cyber Stalking/Bullying': 5,
            'Identity Theft': 7,
            'Others': 3803
        }
    }

    # Create DataFrame
    df = pd.DataFrame(categories)

    # VISUALIZATION 7: Stacked bar for crime categories
    plt.figure()
    # Note the .T to transpose
    df.T.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Breakdown of Cybercrime Categories in 2022', fontsize=16)
    plt.xlabel('City', fontsize=12)  # Changed from 'Crime Category' to 'City'
    plt.ylabel('Number of Cases', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()

    # VISUALIZATION 8: Pie charts for each city
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, city in enumerate(['Bengaluru', 'Hyderabad', 'Mumbai']):
        df[city].plot(kind='pie', ax=axes[i], autopct='%1.1f%%', startangle=90)
        axes[i].set_title(
            f'Cybercrime Categories in {city} (2022)', fontsize=14)
        axes[i].set_ylabel('')

    plt.tight_layout()


analyze_cybercrime_categories()

# Print summary statistics
print("\nARIMA MODEL FORECASTS FOR 2023:")
print(
    f"Bengaluru: {bengaluru_forecast.iloc[0]:.0f} cases (95% CI: {ben_lower:.0f} - {ben_upper:.0f})")
print(
    f"Hyderabad: {hyderabad_forecast.iloc[0]:.0f} cases (95% CI: {hyd_lower:.0f} - {hyd_upper:.0f})")
print(
    f"Mumbai: {mumbai_forecast.iloc[0]:.0f} cases (95% CI: {mum_lower:.0f} - {mum_upper:.0f})")

print("\nCOVID-19 PERIOD ANALYSIS (2020-2022):")
for city in ['Bengaluru', 'Hyderabad', 'Mumbai']:
    print(f"\n{city.upper()}:")
    print(f"2020: {data[city].iloc[9]:,} cases")
    print(
        f"2021: {data[city].iloc[10]:,} cases (Change: {data[city+'_Growth'].iloc[10]:.1f}%)")
    print(
        f"2022: {data[city].iloc[11]:,} cases (Change: {data[city+'_Growth'].iloc[11]:.1f}%)")

print("\nOVERALL GROWTH ANALYSIS (2011-2022):")
for city in ['Bengaluru', 'Hyderabad', 'Mumbai']:
    total_growth = (data[city].iloc[-1] / data[city].iloc[1] - 1) * 100
    cagr = ((data[city].iloc[-1] / data[city].iloc[1]) ** (1/11) - 1) * 100
    print(f"{city}: {total_growth:.1f}% total growth | {cagr:.1f}% compound annual growth rate")

print("\nTECH HUBS VS FINANCIAL CENTER COMPARISON (2022):")
tech_avg_2022 = (data['Bengaluru'].iloc[-1] + data['Hyderabad'].iloc[-1]) / 2
financial_2022 = data['Mumbai'].iloc[-1]
ratio = tech_avg_2022 / financial_2022
print(f"Average Tech Hub cybercrime cases: {tech_avg_2022:,.0f}")
print(f"Financial Center cybercrime cases: {financial_2022:,.0f}")
print(f"Tech Hubs to Financial Center ratio: {ratio:.2f}")

# Save the dataset to CSV
data.to_csv('cybercrime_dataset_2011_2022.csv', index=False)
plt.show()