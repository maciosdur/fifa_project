# FIFA 22 Data Analysis Project

## 📌 Project Overview

This project analyzes key FIFA 22 player attributes to uncover performance patterns and market dynamics. We focus on these core statistics:

| Statistic             | Description                                            | Game Impact                                       |
| --------------------- | ------------------------------------------------------ | ------------------------------------------------- |
| **Overall (0-100)**   | Current skill level composite score                    | Determines player effectiveness in matches        |
| **Potential (0-100)** | Maximum achievable skill level                         | Guides long-term career development and transfers |
| **Age**               | Player's age in years                                  | Affects performance curves and transfer value     |
| **Height_cm**         | Height in centimeters                                  | Influences aerial duels and physical presence     |
| **Weight_kg**         | Weight in kilograms                                    | Impacts strength and stamina                      |
| **Nationality_name**  | Player's home country                                  | Reveals regional talent distributions             |
| **Club_name**         | Current professional team                              | Shows team roster strengths and financial power   |
| **Value_eur (€)**     | Estimated market transfer value                        | Reflects player demand and economic factors       |
| **Wage_eur (€/week)** | Weekly salary                                          | Indicates club investment and player status       |
| **Preferred_foot**    | Dominant foot (Left/Right)                             | Affects positioning and tactical options          |
| **Player_positions**  | Primary and secondary playing roles (e.g. ST, CAM, CB) | Determines team formation compatibility           |

---

## 🛠 Prerequisites

Make sure you have the following installed before running the project:

- **Anaconda**
- **Git** (optional, for cloning the repository)

## 🚀 How to Set Up and Run the Project

### 1️⃣ Clone the Repository

If you haven't already, clone the repository to your local machine:

```sh
git clone https://github.com/maciosdur/fifa_project.git
cd fifa_project
```

### 2️⃣ Set Up the Conda Environment

The project uses Conda to manage dependencies. Create and activate the environment using:

```sh
conda env create -f environment.yml
conda activate fifa_env
```

### 3️⃣ Run the Data Analysis Script

Execute the main script to start the analysis:

```sh
python data_analysis_main.py
```

# 📂 Project Structure

```
fifa_project
│   data_analysis_main.py
│   environment.yml
│   players_22.csv
│   README.md
│   requirements.txt
│
└───output
    │   categorical_distributions.txt
    │   categorical_stats.csv
    │   numerical_stats.csv
    │
    └───plots
            age_vs_value_distribution.png
            boxplot_foot_value.png
            boxplot_positions.png
            conditional_histogram_foot.png
            correlation_heatmap.png
            error_bars_age_groups.png
            histogram_age.png
            histogram_overall.png
            histogram_potential.png
            histogram_value_eur.png
            histogram_value_eur_trimmed.png
            histogram_wage_eur.png
            histogram_wage_eur_trimmed.png
            linear_regression_overall_value.png
            overall_distribution_clean.png
            regression_value_wage_by_foot.png
            simple_exp_regression.png
            simple_regression_value_wage.png
            violinplot_nationality_age.png
            violinplot_position_potential.png
```

## 📊 Implementation Details

To achieve a full analysis, the project includes:

- **Basic statistics**: Mean, median, min, max, standard deviation, percentiles, and missing values for numerical and categorical features.
- **Visualizations**:
  - Boxplots and violin plots for numerical data.
  - Error bars and histograms with conditional distributions.
  - Correlation heatmaps and linear regression analysis.
