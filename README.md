# County-Obesity-Risk-Intervention-System-C-ORIS-
C-ORIS: Advanced Streamlit dashboard. Uses XGBoost to predict US county obesity risk and a Monte Carlo simulator with guardrails to prescribe optimal, constrained policy interventions for maximum health equity gains.
# 📈 County Obesity Risk & Intervention System (C-ORIS)

C-ORIS is an advanced, AI-driven dashboard designed to combat public health disparities by analyzing the socioeconomic determinants of **Adult Obesity Rates** across US counties. Built on Streamlit, this tool provides policymakers, public health officials, and analysts with predictive insights and prescriptive recommendations for effective, targeted interventions.

-----

## 🌟 Features Overview

C-ORIS provides a comprehensive analytics workflow across 7 interconnected stages:

| Tab | Name | Functionality & Value |
| :--- | :--- | :--- |
| **1-2** | **Data Exploration / Geospatial Analysis** | Visualizes the distribution and correlation of health outcomes and risk factors, including choropleth maps of county-level obesity rates. |
| **3** | **Predictive Modeling** | Trains and hyperparameter-tunes a high-performance **XGBoost Regressor** to predict obesity rates based on user-selected socioeconomic features. |
| **4** | **SHAP & Policy Sandbox** | Interprets the trained model using **SHAP Values** to reveal how each feature (e.g., poverty, income) influences predictions. Includes a **Sandbox** for manual, counterfactual policy simulation. |
| **5** | **Comparative Analysis** | Benchmarks the risk profiles of two different counties side-by-side using a normalized **Radar Chart** based on top predictive features. |
| **6** | **Robust Policy Simulator (Prescriptive)** | Uses **Monte Carlo Simulation** with intelligent, directional guardrails to identify the single optimal combination of policy changes (within a budget) that yields the maximum possible reduction in the predicted obesity rate for a target county. |
| **7** | **Reporting & Collaboration** | Allows analysts to save field notes and generates an executive-ready **PDF Report** for any county, detailing risk factor percentile rankings and predicted vs. actual outcomes. |

## 🚀 Key Technologies

| Category | Technology | Role |
| :--- | :--- | :--- |
| **Framework** | **Streamlit** | Interactive web application frontend. |
| **ML Core** | **XGBoost Regressor** | State-of-the-art predictive model for tabular data. |
| **Interpretability** | **SHAP** | Model explainability (TreeExplainer). |
| **Prescriptive** | **Monte Carlo Simulation** | Optimization technique used in the Policy Simulator. |
| **Data Science** | **Pandas, NumPy, Scikit-learn** | Data handling, preprocessing (StandardScaler), and model utilities. |
| **Reporting** | **FPDF** | Generates structured PDF export reports. |

-----

## ⚙️ Installation and Setup

### Prerequisites

You must have Python 3.8+ installed.

1.  **Clone the repository:**

    ```bash
    git clone [Your Repository URL]
    cd C-ORIS
    ```

2.  **Install dependencies:**
    The application requires specific advanced libraries (`xgboost`, `shap`, `fpdf`, `scipy`).

    ```bash
    pip install streamlit pandas numpy plotly matplotlib xgboost scikit-learn shap fpdf scipy
    ```

3.  **Data Files:**
    Place the following two data files in the root directory of the project:

      * `StateAndCountyData.csv` (The main long-format data containing FIPS, county, and feature values).
      * `VariableList.csv` (The lookup table mapping variable codes to readable names).

### Running the App

Execute the following command from the root directory:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser (typically `http://localhost:8501`).

-----

## 👩‍💻 Usage Workflow

### Step 1: Data Review (Tabs 1 & 2)

Verify data distributions and geospatial trends of the **Adult Obesity Rate**.

### Step 2: Model Training (Tab 3)

Navigate to **3. Predictive Modeling**.

1.  Use the categorized checkboxes to select your desired set of socioeconomic features.
2.  Click **"🚀 Train Model"**. The app performs hyperparameter tuning and saves the best model, the scaler, and the feature list to the session state.

### Step 3: Analysis & Prescription (Tabs 4 & 6)

Once the model is trained, the analysis tabs are unlocked:

  * **For Explanation (Tab 4):** Calculate the  **SHAP Summary Plot** to understand which risk factors globally drive the prediction. Use the **Policy Sandbox** for quick, manual testing of interventions.
  * **For Recommendation (Tab 6):** Use the **Robust Policy Simulator**. Select a high-risk county, define your policy levers (e.g., income, poverty rate), and set a maximum change constraint (e.g., $\pm 10\%$). The Monte Carlo engine will return the **optimal policy mix** for maximum health gain.

### Step 4: Reporting (Tab 7)

Use the **Reporting & Collaboration** tab to save detailed analyst notes and generate a comprehensive **PDF Report** for executive stakeholders.
