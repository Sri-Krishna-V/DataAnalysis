import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the path to the dataset and output folder
DATASET_PATH = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv"
OUTPUT_FOLDER = r"c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\Qn 9"

# Ensure the output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Task 1: Load and Preprocess Data ---


def load_and_preprocess_data(filepath):
    """Loads and preprocesses the cybercrime data."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {filepath}")
        return None

    # Rename columns for easier access
    df.rename(columns={
        'year': 'Year',
        'state': 'State',
        'offence_sub_category': 'Crime_Type',
        'value': 'Cases'
    }, inplace=True)

    # Filter for relevant years (since 2018)
    df = df[df['Year'] >= 2018]

    # Filter for relevant crime types
    relevant_crimes = ['Identity Theft',
                       'Cheating by personation by using computer resource']
    df = df[df['Crime_Type'].isin(relevant_crimes)]

    # Convert 'Cases' to numeric, coercing errors to NaN
    df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce')

    # Handle missing values - assuming missing means zero cases reported
    df['Cases'].fillna(0, inplace=True)
    df['Cases'] = df['Cases'].astype(int)

    # Filter out summary rows like 'All India'
    df = df[~df['State'].str.contains('Total|All India', na=False, case=False)]

    # Standardize state/UT names if necessary (example)
    df['State'] = df['State'].replace(
        'Dadra and Nagar Haveli and Daman and Diu', 'DNH & DD')
    df['State'] = df['State'].replace(
        'Andaman and Nicobar Islands', 'A&N Islands')

    # Aggregate cases if multiple rows exist for the same year, state, and crime type
    df_agg = df.groupby(['Year', 'State', 'Crime_Type'])[
        'Cases'].sum().reset_index()

    return df_agg

# --- Task 2: Exploratory Data Analysis (EDA) and Visualizations ---


# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
# Lower DPI for potentially smaller file sizes
plt.rcParams['figure.dpi'] = 100


def plot_stacked_bar_state_vs_year(df, output_folder):
    """Plots stacked bar chart: State vs. Total Crime Count, faceted by Year."""
    df_total = df.groupby(['Year', 'State'])['Cases'].sum().reset_index()

    # Limit to top N states per year for clarity if too many states
    top_n = 15
    df_top_states = df_total.groupby('Year').apply(
        lambda x: x.nlargest(top_n, 'Cases')).reset_index(drop=True)

    g = sns.FacetGrid(df_top_states, col="Year", col_wrap=3,
                      height=4, aspect=1.5, sharex=False)
    g.map(sns.barplot, "Cases", "State", palette="viridis",
          order=df_top_states.groupby('State')['Cases'].sum().nlargest(top_n).index)
    g.set_titles("Year: {col_name}")
    g.set_axis_labels(
        "Total Cases (Identity Theft + Cheating by Personation)", "State")
    plt.suptitle(
        'Top 15 States: Total Identity Theft & Cheating Cases per Year (2018-2022)', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(output_folder, '1_stacked_bar_state_vs_year.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def plot_line_top_states_trend(df, output_folder, top_n=5):
    """Plots line chart: Year vs. Total Crime Count for Top N states."""
    df_total = df.groupby(['Year', 'State'])['Cases'].sum().reset_index()
    total_cases_per_state = df_total.groupby(
        'State')['Cases'].sum().nlargest(top_n).index
    df_top = df_total[df_total['State'].isin(total_cases_per_state)]

    plt.figure()
    sns.lineplot(data=df_top, x='Year', y='Cases',
                 hue='State', marker='o', palette='tab10')
    plt.title(
        f'Trend of Total Identity Theft & Cheating Cases for Top {top_n} States (2018-2022)')
    plt.xlabel('Year')
    plt.ylabel('Total Cases')
    plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(df_top['Year'].unique())
    plt.tight_layout()
    filepath = os.path.join(output_folder, '2_line_top_states_trend.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def plot_heatmap_state_year(df, output_folder):
    """Plots heatmap: State vs. Year, color intensity by Total Crime Count."""
    df_total = df.groupby(['Year', 'State'])['Cases'].sum().reset_index()
    heatmap_data = df_total.pivot(
        index='State', columns='Year', values='Cases').fillna(0)

    # Sort states by total cases over the period for better visualization
    heatmap_data['Total'] = heatmap_data.sum(axis=1)
    heatmap_data = heatmap_data.sort_values(
        'Total', ascending=False).drop('Total', axis=1)

    # Adjust size for better state label visibility
    plt.figure(figsize=(10, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f",
                cmap="viridis", linewidths=.5)
    plt.title(
        'Heatmap of Total Identity Theft & Cheating Cases (State vs. Year, 2018-2022)')
    plt.xlabel('Year')
    plt.ylabel('State')
    plt.tight_layout()
    filepath = os.path.join(output_folder, '3_heatmap_state_year.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def plot_grouped_bar_top_states_by_crime_type(df, output_folder, year=2022, top_n=10):
    """Plots grouped bar chart: Top N States vs. Crime Count, grouped by Crime Type for a specific year."""
    df_year = df[df['Year'] == year]

    # Find top N states based on total cases in that year
    top_states = df_year.groupby('State')['Cases'].sum().nlargest(top_n).index
    df_top_year = df_year[df_year['State'].isin(top_states)]

    # Sort states by total cases for the plot
    df_top_year = df_top_year.groupby('State').filter(
        lambda x: x['Cases'].sum() > 0)  # Ensure state has cases
    state_order = df_top_year.groupby(
        'State')['Cases'].sum().sort_values(ascending=False).index

    plt.figure()
    sns.barplot(data=df_top_year, x='Cases', y='State',
                hue='Crime_Type', palette='muted', order=state_order)
    plt.title(
        f'Identity Theft vs. Cheating by Personation Cases in Top {top_n} States ({year})')
    plt.xlabel('Number of Cases')
    plt.ylabel('State')
    plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filepath = os.path.join(
        output_folder, f'4_grouped_bar_top_{top_n}_states_{year}.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def plot_pie_top_states_distribution(df, output_folder, top_n=10):
    """Plots pie chart: Distribution of Total Crimes among Top N States (2018-2022)."""
    df_total = df.groupby('State')['Cases'].sum().reset_index()
    df_total = df_total.sort_values('Cases', ascending=False)

    top_states_data = df_total.head(top_n)
    other_cases = df_total.iloc[top_n:]['Cases'].sum()

    if other_cases > 0:
        # Create a DataFrame for 'Others'
        others_df = pd.DataFrame([{'State': 'Others', 'Cases': other_cases}])
        # Concatenate top_states_data and others_df
        pie_data = pd.concat([top_states_data, others_df], ignore_index=True)
    else:
        pie_data = top_states_data

    plt.figure()
    plt.pie(pie_data['Cases'], labels=pie_data['State'], autopct='%1.1f%%',
            startangle=140, colors=sns.color_palette('pastel'))
    plt.title(
        f'Distribution of Total Identity Theft & Cheating Cases among Top {top_n} States (2018-2022)')
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.tight_layout()
    filepath = os.path.join(
        output_folder, f'5_pie_top_{top_n}_states_distribution.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting analysis...")
    crime_data = load_and_preprocess_data(DATASET_PATH)

    if crime_data is not None:
        print("Data loaded and preprocessed successfully.")

        # Generate and save visualizations
        print("Generating visualizations...")
        plot_stacked_bar_state_vs_year(crime_data, OUTPUT_FOLDER)
        plot_line_top_states_trend(crime_data, OUTPUT_FOLDER, top_n=5)
        plot_heatmap_state_year(crime_data, OUTPUT_FOLDER)
        plot_grouped_bar_top_states_by_crime_type(
            crime_data, OUTPUT_FOLDER, year=2022, top_n=10)
        plot_pie_top_states_distribution(crime_data, OUTPUT_FOLDER, top_n=10)

        print(f"Analysis complete. Visualizations saved to: {OUTPUT_FOLDER}")
    else:
        print("Analysis aborted due to data loading issues.")
