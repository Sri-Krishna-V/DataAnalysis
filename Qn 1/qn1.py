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


def generate_cybercrime_dataset():
    years = list(range(2011, 2023))

    # Actual data from NCRB tables for 2020-2022, with earlier years based on historical trends
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

plt.title(
    'Cybercrime Trends in Tech Hubs and Financial Captial (2011-2022)', fontsize=20)
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

# ARIMA Modeling with extended forecast to 2024


def fit_arima_model(series, city_name):
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast_steps = 2  # Forecasting for 2023 and 2024
    forecast = model_fit.forecast(steps=forecast_steps)

    # Get confidence intervals
    conf_int = model_fit.get_forecast(steps=forecast_steps).conf_int()
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values

    return model_fit, forecast, lower_bound, upper_bound


# Fit ARIMA models for each city
bengaluru_model, bengaluru_forecast, ben_lower, ben_upper = fit_arima_model(
    data['Bengaluru'], 'Bengaluru')
hyderabad_model, hyderabad_forecast, hyd_lower, hyd_upper = fit_arima_model(
    data['Hyderabad'], 'Hyderabad')
mumbai_model, mumbai_forecast, mum_lower, mum_upper = fit_arima_model(
    data['Mumbai'], 'Mumbai')

# VISUALIZATION 3: ARIMA Forecast visualization (extended to 2024)
plt.figure()
colors = ['#3274A1', '#E1812C', '#3A923A']
forecast_years = [2023, 2024]

for i, (city, forecast, lower, upper) in enumerate(zip(
    ['Bengaluru', 'Hyderabad', 'Mumbai'],
    [bengaluru_forecast, hyderabad_forecast, mumbai_forecast],
    [ben_lower, hyd_lower, mum_lower],
        [ben_upper, hyd_upper, mum_upper])):

    # Plot historical data
    plt.plot(data['year'], data[city], marker='o', linewidth=3,
             color=colors[i], label=f"{city} (Actual)")

    # Plot forecasted points with confidence intervals
    plt.plot(forecast_years, forecast, marker='*', markersize=15,
             color=colors[i], label=f"{city} (Forecast)" if i == 0 else "")

    # Add confidence intervals
    plt.fill_between(forecast_years, lower, upper, color=colors[i], alpha=0.2,
                     label=f"95% Confidence Interval" if i == 0 else "")

    # Add connecting line from actual to forecast
    plt.plot([data['year'].iloc[-1], forecast_years[0]],
             [data[city].iloc[-1], forecast.iloc[0]],
             linestyle='--', alpha=0.5, color=colors[i])

    # Add annotations for forecast values
    for j, year in enumerate(forecast_years):
        plt.annotate(f"{forecast.iloc[j]:.0f}",
                     xy=(year, forecast.iloc[j]),
                     xytext=(year-0.15, forecast.iloc[j]+500),
                     fontsize=12, fontweight='bold', color=colors[i])

plt.title(
    'Cybercrime Forecast for 2023-2024 Based on ARIMA(1,1,1) Model', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Number of Cybercrime Cases', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(list(data['year']) + forecast_years, rotation=45, fontsize=14)
plt.tight_layout()

# Print summary statistics
print("\nARIMA MODEL FORECASTS FOR 2023-2024:")
for i, city in enumerate(['Bengaluru', 'Hyderabad', 'Mumbai']):
    if city == 'Bengaluru':
        forecast = bengaluru_forecast
        lower = ben_lower
        upper = ben_upper
    elif city == 'Hyderabad':
        forecast = hyderabad_forecast
        lower = hyd_lower
        upper = hyd_upper
    else:  # Mumbai
        forecast = mumbai_forecast
        lower = mum_lower
        upper = mum_upper

    print(f"\n{city.upper()}:")
    print(
        f"2023: {forecast.iloc[0]:.0f} cases (95% CI: {lower[0]:.0f} - {upper[0]:.0f})")
    print(
        f"2024: {forecast.iloc[1]:.0f} cases (95% CI: {lower[1]:.0f} - {upper[1]:.0f})")
    print(
        f"Projected growth 2023-2024: {((forecast.iloc[1]/forecast.iloc[0])-1)*100:.1f}%")

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
    total_growth = (data[city].iloc[-1] / data[city].iloc[0] - 1) * 100
    cagr = ((data[city].iloc[-1] / data[city].iloc[0]) ** (1/11) - 1) * 100
    print(f"{city}: {total_growth:.1f}% total growth | {cagr:.1f}% compound annual growth rate")

# Save the dataset to CSV
data.to_csv('cybercrime_dataset_2011_2022.csv', index=False)
plt.show()
