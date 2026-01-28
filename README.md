# C-ORIS: County Obesity Risk & Intervention System
**Prescriptive AI for Strategic Resource Allocation & Policy Optimization**

##  Project Executive Summary
Public health interventions often fail due to a lack of "Prescriptive" clarity. **C-ORIS** is a decision-support engine designed to move beyond identifying risk into **optimizing interventions**. 

By integrating **XGBoost** for risk prediction and **Monte Carlo Simulations** for constraint-based optimization, C-ORIS identifies the most cost-effective socioeconomic "levers" (e.g., median income, poverty rates, education access) to reduce obesity. The system provides a robust "Policy Simulator" that prescribes the single most impactful combination of changes within user-defined budget and feasibility guardrails.

---

## The Technical Architecture

### 1. High-Performance Predictive Modeling
* **XGBoost Regressor:** Engineered a gradient-boosted tree model to predict county-level obesity rates based on a high-dimensional feature set of socioeconomic drivers.
* **Hyperparameter Tuning:** Implemented K-Fold cross-validation to ensure model generalizability across diverse US geographic regions.
* **Geospatial Hotspot Analysis:** Built interactive choropleth maps using Plotly to visualize "at-risk" clusters and geographic disparities.

### 2. Explainable AI (XAI) with SHAP
To ensure transparency for policymakers, I integrated **SHAP (SHapley Additive exPlanations)** to de-risk the "Black Box" nature of machine learning:
* **Global Interpretation:** Quantified the overall impact of factors like poverty and median income on national obesity trends.
* **Local Explanation:** Developed a **SHAP TreeExplainer** to provide a "diagnostic" for individual counties—showing exactly which local features are driving the specific risk score.



### 3. Prescriptive Monte Carlo Optimization
The core innovation is the **Robust Policy Simulator**, which transforms predictions into specific action plans:
* **Stochastic Simulation:** Uses **Monte Carlo methods** with intelligent directional guardrails to test thousands of potential policy combinations simultaneously.
* **Constrained Prescription:** Utilizes `scipy.optimize` to identify the optimal mix of socioeconomic interventions that yields the maximum possible reduction in predicted risk within a defined "budget" or feasibility limit (e.g., "What is the best we can do with a 10% change limit?").
* **Counterfactual Sandbox:** Allows stakeholders to run "What-If" scenarios to validate the sensitivity of health outcomes to specific socioeconomic shifts.



### 4. Enterprise Reporting Pipeline
* **Automated PDF Architect:** Built an export engine using **FPDF** that generates executive-ready reports, including risk percentile rankings and optimized policy recommendations.
* **Production UI:** Designed a 7-stage interactive dashboard workflow in **Streamlit** to facilitate cross-functional collaboration between data teams and policymakers.

---

##  Technical Stack
* **Languages:** Python (Pandas, NumPy, Scipy).
* **ML & XAI:** XGBoost, Scikit-Learn, SHAP.
* **Optimization:** Monte Carlo Simulation, SciPy Optimize.
* **UI/Deployment:** Streamlit.
* **Reporting:** FPDF (Automated Executive Reports).

---

##  Key Strategic Value
* **Actionable Prescription:** Shifts the conversation from "Where is the problem?" to "What is the most effective way to solve it with our current budget?"
* **Policy Benchmarking:** Uses **Radar Charts** to compare high-risk counties against national averages, identifying specific socioeconomic gaps.
* **Scalable Framework:** While applied to public health, this **Constrained Optimization Engine** is directly applicable to business problems like marketing budget allocation and supply chain risk mitigation.

---
**Author:** Mohit Phulwani  
**Focus:** Bridging Behavioral Economics and Cloud-Native Data Science
