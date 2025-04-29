# Analysis of Identity Theft and Cheating by Personation Cybercrimes in India (2018-2022)

## Project Objective
This project aims to identify which Indian states have shown the highest concentration of identity theft and cheating by personation cybercrime cases and analyze how these patterns have evolved since 2018.

## Datasets Used

*   **Dataset:** `cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv`
*   **Source:** National Crime Records Bureau (NCRB), India (assumed based on filename)
*   **Justification:** This dataset was chosen because it provides granular data on specific types of cybercrimes committed in violation of the IT Act, broken down by state and year. Crucially, it includes the specific categories required for this analysis: 'Identity Theft' and 'Cheating by personation by using computer resource'. The data spans the necessary time frame (2018 onwards).

## Analysis Summary

The analysis involves the following steps:
1.  **Loading Data:** The selected CSV dataset is loaded using Pandas.
2.  **Preprocessing:**
    *   Columns are renamed for clarity.
    *   Data is filtered to include only records from 2018 onwards.
    *   Records are filtered to include only 'Identity Theft' and 'Cheating by personation by using computer resource'.
    *   The 'Cases' column is converted to a numeric type, handling potential errors and filling missing values with 0.
    *   Summary rows (like 'All India', 'Total') are removed to focus on state-level data.
    *   State/UT names are standardized for consistency (e.g., 'DNH & DD', 'A&N Islands').
    *   Data is aggregated to ensure a single entry per state, year, and crime type.
3.  **Exploratory Data Analysis (EDA):** The preprocessed data is used to generate visualizations that reveal patterns in crime concentration and trends over time across different states.

## Visualizations Chosen

Five types of visualizations were generated to address the project objective:

1.  **Stacked Bar Chart (State vs. Total Crime Count, faceted by Year):**
    *   **Purpose:** To visualize the total number of combined identity theft and cheating cases for the top 15 states, year by year.
    *   **Justification:** Allows for easy comparison of crime magnitude across states within a specific year and observation of year-over-year changes for individual states. Faceting by year clearly shows the evolution.
2.  **Line Plot (Year vs. Total Crime Count, lines per Top 5 States):**
    *   **Purpose:** To illustrate the trend of total reported cases (identity theft + cheating) over the period 2018-2022 for the 5 states with the highest overall counts.
    *   **Justification:** Clearly highlights increasing, decreasing, or stable trends in the most affected states, addressing the "how patterns evolved" part of the question.
3.  **Heatmap (State vs. Year, color intensity by Total Crime Count):**
    *   **Purpose:** To provide a matrix overview of the concentration of total cases across all states and years (2018-2022).
    *   **Justification:** Excellent for quickly identifying hotspots (states and years with high crime counts) and observing broad temporal patterns or shifts in concentration.
4.  **Grouped Bar Chart (Top 10 States vs. Crime Count, grouped by Crime Type for 2022):**
    *   **Purpose:** To compare the prevalence of 'Identity Theft' versus 'Cheating by personation' within the top 10 states for the most recent year (2022).
    *   **Justification:** Helps understand if certain states have a higher concentration of one specific crime type over the other, providing a more nuanced view than just the total count.
5.  **Pie Chart (Distribution of Total Crimes among Top 10 States, 2018-2022):**
    *   **Purpose:** To show the percentage share of the total reported cases (combined crimes, 2018-2022) accounted for by the top 10 states and 'Others'.
    *   **Justification:** Effectively demonstrates the extent to which these cybercrimes are concentrated in a few key states versus being more evenly distributed.

## Execution Instructions

1.  **Ensure Dependencies:** Make sure you have Python installed along with the required libraries. You can install them using pip:
    ```bash
    pip install pandas matplotlib seaborn numpy
    ```
2.  **Place Dataset:** Ensure the dataset `cyber-crimes-from-ncrb-master-data-year-and-state-wise-types-of-cyber-crimes-committed-in-violation-of-it-act.csv` is located at the path specified in the script (`c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\`). Adjust the `DATASET_PATH` variable in the script if needed.
3.  **Run Script:** Execute the Python script from your terminal:
    ```bash
    python "c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\Qn 9\analysis.py"
    ```
4.  **Output:** The script will print status messages to the console. The generated visualization images (`.png` files) will be saved in the `c:\Users\srikr\Desktop\Studies\Self\Papers\Data Analysis\Complete\Qn 9\` folder.

## Dependencies

*   pandas
*   numpy
*   matplotlib
*   seaborn

