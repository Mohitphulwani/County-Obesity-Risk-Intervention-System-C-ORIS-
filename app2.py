import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import time
import requests
from typing import Dict, Any, List, Tuple
from io import BytesIO

# ML imports
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# --- ENTERPRISE FEATURES IMPORTS ---
from scipy.optimize import minimize
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Optional advanced module
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
# ==============================================================================
# --- CONFIGURATION & SETUP ---
# ==============================================================================

# 🟢 GOOGLE DRIVE LINKS (Updated)
DATA_URL = "https://drive.google.com/file/d/1zzDPjSvOvGt34zA45gUkaT7X7giXtZvn/view?usp=sharing"
VAR_URL = "https://drive.google.com/file/d/1D-5XzYTPvv4CT5u3xqZIQ_v2f2H0055A/view?usp=sharing"

st.set_page_config(
    page_title="C-ORIS | County Obesity Risk System", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- STRICT UI STYLING FIXES (GLOBAL) ---
st.markdown("""
<style>
    /* 1. GLOBAL BACKGROUND & TEXT */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* 2. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #262730;
        border-right: 1px solid #333333;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* 3. WIDGET LABELS */
    .stNumberInput label, .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }

    /* 4. DROPDOWN MENUS */
    div[data-baseweb="select"] > div {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
        border: 1px solid #4A4A4A !important;
    }
    div[data-baseweb="select"] span {
        color: #FFFFFF !important;
    }
    ul[data-baseweb="menu"] {
        background-color: #262730 !important; 
        border: 1px solid #4A4A4A !important;
    }
    li[data-baseweb="option"] {
        color: #FFFFFF !important;
        background-color: #262730 !important;
    }
    li[data-baseweb="option"]:hover, li[data-baseweb="option"][aria-selected="true"] {
        background-color: #FF4B4B !important; 
        color: #FFFFFF !important;
    }
    
    /* 5. METRICS & BUTTONS */
    div[data-testid="stMetric"] {
        background-color: #262730;
        border: 1px solid #4A4A4A;
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stMetric"] label {
        color: #DDDDDD !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    div.stButton > button {
        background-color: #262730 !important;
        color: #FFFFFF !important;
        border: 1px solid #FFFFFF !important;
    }
    div.stButton > button:hover {
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 C-ORIS: County Obesity Risk & Intervention System")
st.subheader("Advanced Health Disparities Analysis & Robust Policy Simulation")

st.markdown("""
### Project Overview
This dashboard leverages **advanced machine learning (XGBoost)** to analyze the social, economic, and environmental drivers of **Obesity Rates** across US counties.
""")
st.markdown("---")

# ==============================================================================
# 1. CORE FUNCTIONS (Data Prep & Helpers)
# ==============================================================================

def format_drive_url(url: str) -> str:
    """
    Converts a Google Drive 'view' URL into a direct download URL.
    """
    if not url:
        return None
    if "drive.google.com" in url and "/view" in url:
        # Extract ID
        try:
            file_id = url.split('/d/')[1].split('/')[0]
            return f"https://drive.google.com/uc?id={file_id}&export=download"
        except IndexError:
            return url
    return url

@st.cache_data
def generate_demo_data():
    """Generates synthetic data so the app runs if CSVs are missing."""
    st.warning("⚠️ Using Synthetic Demo Data (Live data URL not provided or failed).")
    np.random.seed(42)
    n_counties = 500
    
    data = {
        'FIPS': [f"{i:05d}" for i in range(1000, 1000 + n_counties)],
        'State': np.random.choice(['AL', 'TX', 'CA', 'NY', 'FL', 'IL'], n_counties),
        'County': [f"County {i}" for i in range(n_counties)],
        'PCT_OBESE_ADULTS22': np.random.normal(38, 6, n_counties), 
        'POVRATE21': np.random.normal(15, 5, n_counties),
        'MEDHHINC21': np.random.normal(60000, 15000, n_counties),
        'PCT_LACCESS_POP19': np.random.normal(20, 10, n_counties),
        'PCT_65OLDER20': np.random.normal(18, 4, n_counties),
        'CHILDPOVRATE21': np.random.normal(20, 8, n_counties),
        'PCT_DIABETES_ADULTS20': np.random.normal(10, 3, n_counties),
        'Latitude': np.random.uniform(25, 48, n_counties),
        'Longitude': np.random.uniform(-120, -75, n_counties)
    }
    df = pd.DataFrame(data)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].abs()
    
    var_map = {
        'PCT_OBESE_ADULTS22': 'Adult Obesity Rate (%)',
        'POVRATE21': 'Poverty Rate (%)',
        'MEDHHINC21': 'Median Household Income ($)',
        'PCT_LACCESS_POP19': '% Low Access to Stores',
        'PCT_65OLDER20': '% Population 65+',
        'CHILDPOVRATE21': 'Child Poverty Rate (%)',
        'PCT_DIABETES_ADULTS20': 'Adult Diabetes Rate (%)'
    }
    return df, 'PCT_OBESE_ADULTS22', var_map

@st.cache_data
def load_and_prepare_data(data_url: str, var_url: str) -> Tuple[pd.DataFrame, str, Dict]:
    st.info("Attempting to load data from Cloud Storage...")
    
    # 1. Format URLs
    dl_data_url = format_drive_url(data_url)
    dl_var_url = format_drive_url(var_url)

    # 2. Try Loading
    try:
        if not dl_data_url:
            raise ValueError("No URL provided")
            
        nan_values = ['nan', 'NaN', '00nan', '00NaN', -9999.0, -8888.0]
        
        # Load Main Data
        df_long = pd.read_csv(dl_data_url, dtype={'FIPS': str}, na_values=nan_values)
        
        # Load Variable Map (Optional, fallback if missing)
        try:
            df_vars = pd.read_csv(dl_var_url)
            var_map = df_vars.set_index('Variable_Code')['Variable_Name'].to_dict()
        except:
            var_map = {}

    except Exception as e:
        st.error(f"Could not load data from URL: {e}")
        return generate_demo_data()

    # 3. Pivot and Process
    try:
        df_wide = df_long.groupby(['FIPS', 'State', 'County', 'Variable_Code'])['Value'].mean().unstack().reset_index()

        if 'PCT_OBESE_ADULTS22' in df_wide.columns:
            target_col = 'PCT_OBESE_ADULTS22'
        elif 'PCT_OBESE_ADULTS17' in df_wide.columns:
            target_col = 'PCT_OBESE_ADULTS17'
        else:
            st.warning("Target variable (Obesity Rate) not found in dataset.")
            return generate_demo_data()

        df_wide[target_col] = pd.to_numeric(df_wide[target_col], errors='coerce')
        df_clean = df_wide.dropna(subset=[target_col]).copy()
        
        # Ensure key predictors are numeric
        key_predictors = ['POVRATE21', 'MEDHHINC21', 'PCT_LACCESS_POP19', 'PCT_65OLDER20', 'CHILDPOVRATE21']
        for col in key_predictors:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col].fillna(df_clean[col].median(), inplace=True)

        st.success(f"✅ Data loaded successfully: {len(df_clean)} counties.")
        return df_clean.reset_index(drop=True), target_col, var_map
        
    except Exception as e:
        st.error(f"Error processing data structure: {e}")
        return generate_demo_data()

# Initialize Session State
if 'df_wide' not in st.session_state:
    df, target, vmap = load_and_prepare_data(DATA_URL, VAR_URL)
    st.session_state['df_wide'] = df
    st.session_state['target_var'] = target
    st.session_state['variable_map'] = vmap
    st.session_state['comments_db'] = {}

df_wide = st.session_state['df_wide']
TARGET_VAR = st.session_state['target_var']
var_map = st.session_state['variable_map']

def map_code_to_name(code: str) -> str:
    return var_map.get(code, code) 

def map_codes_to_names_list(codes: List[str]) -> List[str]:
    return [var_map.get(code, code) for code in codes]

def categorize_feature(code: str) -> str:
    code_u = code.upper()
    if 'OBESE' in code_u or 'DIABETES' in code_u or 'HEALTH' in code_u:
        return "🏥 Health Outcomes"
    elif 'POVERTY' in code_u or 'INCOME' in code_u or 'MEDHHINC' in code_u or 'UNEMP' in code_u:
        return "💰 Economic Indicators"
    elif 'ACCESS' in code_u or 'SODA' in code_u or 'FASTFOOD' in code_u or 'FARM' in code_u:
        return "🍎 Food Environment & Access"
    elif code_u.startswith('PCT_') and 'RACE' not in code_u and 'POP' not in code_u:
        return "🏠 Social & Housing"
    elif 'POP' in code_u or 'RACE' in code_u or 'AGE' in code_u:
        return "🧍 Demographics"
    else:
        return "📊 Other/General"

# --- PDF CLASS ---
class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Health Equity AI - Comprehensive Analysis Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()} - Generated by Health Equity AI', 0, 0, 'C')

def create_download_pdf(county_name, analysis_data, notes, target_name, target_val, predicted_val):
    pdf = ReportPDF()
    pdf.add_page()
    
    # --- Title & Header ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"County Profile: {county_name}", ln=True, align='L')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 5, txt=f"Report Date: {time.strftime('%Y-%m-%d')}", ln=True, align='L')
    pdf.ln(10)
    
    # --- Section 1: Outcomes ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, txt="1. Health Outcome Assessment", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", size=12)
    diff = target_val - predicted_val
    status = "Higher than expected" if diff > 0 else "Lower than expected"
    
    pdf.cell(0, 8, txt=f"Actual {target_name}: {target_val:.1f}%", ln=True)
    pdf.cell(0, 8, txt=f"AI Predicted Value: {predicted_val:.1f}%", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt=f"Discrepancy: {abs(diff):.1f} points ({status})", ln=True)
    pdf.ln(8)
    
    # --- Section 2: Drivers ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="2. Socioeconomic Drivers", ln=True, fill=True)
    pdf.ln(2)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(85, 8, "Feature Name", 1, 0, 'L', True)
    pdf.cell(30, 8, "Local", 1, 0, 'C', True)
    pdf.cell(30, 8, "Natl Avg", 1, 0, 'C', True)
    pdf.cell(45, 8, "Status", 1, 1, 'C', True)
    
    pdf.set_font("Arial", size=10)
    for item in analysis_data:
        # Safe string truncation
        name = str(item['name'])
        name = (name[:45] + '...') if len(name) > 45 else name
        
        pct = item['percentile'] * 100
        status_txt = "High" if pct >= 75 else "Low" if pct <= 25 else "Avg"
        
        pdf.cell(85, 8, name, 1)
        pdf.cell(30, 8, f"{item['value']:.1f}", 1, 0, 'C')
        pdf.cell(30, 8, f"{item['mean']:.1f}", 1, 0, 'C')
        pdf.cell(45, 8, status_txt, 1, 1, 'C')
        
    pdf.ln(10)
    
    # --- Section 3: Commentary ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="3. Analyst Commentary", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Arial", size=12)
    
    # Sanitize notes
    safe_notes = (notes if notes else "No specific notes.").encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=safe_notes)
    
    pdf_output = pdf.output(dest='S')
    
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin-1')
    else:
        return bytes(pdf_output)

# ==============================================================================
# 2. CORE FUNCTIONS (ML Modeling)
# ==============================================================================

def run_xgboost_tuning(X, y, test_size, n_splits, random_state, param_grid):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_score = float('inf')
    best_params = {}
    best_model = None
    results = []
    
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            for depth in param_grid['max_depth']:
                cv_scores = []
                for train_index, val_index in kf.split(X_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
                    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
                    model = xgb.XGBRegressor(n_estimators=n_est, learning_rate=lr, max_depth=depth, random_state=random_state, verbosity=0, n_jobs=-1)
                    model.fit(X_fold_train, y_fold_train)
                    val_preds = model.predict(X_fold_val)
                    cv_scores.append(mean_squared_error(y_fold_val, val_preds))
                avg_mse = np.mean(cv_scores)
                results.append({'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth, 'CV_MSE': avg_mse})
                if avg_mse < best_score:
                    best_score = avg_mse
                    best_params = {'n_estimators': n_est, 'learning_rate': lr, 'max_depth': depth}
                    best_model = model 

    if best_model:
        final_best_model = xgb.XGBRegressor(**best_params, random_state=random_state, verbosity=0, n_jobs=-1)
        final_best_model.fit(X_train, y_train)
        y_pred = final_best_model.predict(X_test)
        final_mse = mean_squared_error(y_test, y_pred)
        final_r2 = r2_score(y_test, y_pred)
        
        st.session_state['best_model'] = final_best_model
        st.session_state['X_test_scaled'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['model_features'] = X.columns.tolist() 
        st.session_state['tuning_results'] = pd.DataFrame(results).sort_values('CV_MSE').reset_index(drop=True)
        st.session_state['test_metrics'] = {'MSE': final_mse, 'R2': final_r2}
        return final_best_model, final_mse, final_r2
    return None, 0, 0

if df_wide.empty:
    st.stop()

# Sidebar for Navigation
st.sidebar.header("Navigation")
menu = [
    "1. Health Data Exploration", 
    "2. Geospatial Analysis", 
    "3. Predictive Modeling", 
    "4. SHAP & Policy Sandbox", 
    "5. Comparative Analysis", 
    "6. Prescriptive Optimization", 
    "7. Reporting & Collaboration"
]
choice = st.sidebar.radio("Analysis Stage", menu)

# ==============================================================================
# ML CONFIGURATION (Sidebar)
# ==============================================================================
if choice == "3. Predictive Modeling":
    st.sidebar.markdown("---")
    st.sidebar.header("Model Settings")
    st.session_state['test_size'] = st.sidebar.slider("Test Set Split (%)", 10, 50, 20, 5) / 100.0
    st.session_state['random_state'] = st.sidebar.number_input("Random Seed", 1, 100, 42)
    st.session_state['cv_folds'] = st.sidebar.number_input("CV Folds", 2, 10, 3)
    st.sidebar.subheader("Hyperparameters")
    st.session_state['n_estimators'] = st.sidebar.multiselect("Trees", [100, 300, 500], [100, 300])
    st.session_state['learning_rate'] = st.sidebar.multiselect("Learning Rate", [0.01, 0.05, 0.1], [0.05, 0.1])
    st.session_state['max_depth'] = st.sidebar.multiselect("Max Depth", [3, 5, 7], [3, 5])

# ==============================================================================
# TAB 1: EXPLORATORY DATA ANALYSIS
# ==============================================================================
if choice == "1. Health Data Exploration":
    st.header("1. Health Data Exploration")
    st.markdown(f"**Target Variable:** `{TARGET_VAR}` ({map_code_to_name(TARGET_VAR)})")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of Obesity Rates")
        fig = px.histogram(df_wide, x=TARGET_VAR, title='Distribution of County Obesity Rates', marginal='box', labels={TARGET_VAR: 'Adult Obesity Rate (%)'}, color_discrete_sequence=['#1f77b4'], template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Correlation with Social Determinants")
        numeric_cols = df_wide.select_dtypes(include=np.number).columns.tolist()
        analysis_cols_codes = [c for c in numeric_cols if c not in ['FIPS', 'Latitude', 'Longitude', TARGET_VAR]]
        code_to_name = {code: map_code_to_name(code) for code in analysis_cols_codes}
        name_to_code = {v: k for k, v in code_to_name.items()}
        display_names = sorted(list(code_to_name.values()))
        default_name = code_to_name.get('POVRATE21', display_names[0])
        feature_name_x = st.selectbox("Select Determinant (X-axis)", options=display_names, index=display_names.index(default_name) if default_name in display_names else 0)
        feature_code_x = name_to_code.get(feature_name_x)
        fig_scatter = px.scatter(df_wide, x=feature_code_x, y=TARGET_VAR, hover_name='County', title=f'Obesity Rate vs. {feature_name_x}', color=TARGET_VAR, labels={TARGET_VAR: 'Obesity Rate (%)', feature_code_x: feature_name_x}, color_continuous_scale='RdYlGn_r', template="plotly_dark")
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TAB 2: GEOSPATIAL ANALYSIS
# ==============================================================================
elif choice == "2. Geospatial Analysis":
    st.header("2. Geospatial Analysis")
    st.markdown("---")
    st.subheader(f"US Map: {map_code_to_name(TARGET_VAR)}")
    
    try:
        # Check if FIPS needs leading zero fix
        df_plot = df_wide.copy()
        df_plot['FIPS'] = df_plot['FIPS'].astype(str).str.zfill(5)
        
        fig = px.choropleth(
            df_plot, 
            geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json", 
            locations='FIPS', 
            color=TARGET_VAR, 
            hover_name='County', 
            hover_data={'FIPS': False, 'State': True, TARGET_VAR: ':.1f%'}, 
            scope="usa", 
            color_continuous_scale="Reds", 
            title="County-Level Adult Obesity Rates", 
            template="plotly_dark"
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', geo=dict(bgcolor='rgba(0,0,0,0)'), font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render map. Network error accessing GeoJSON or data mismatch. Details: {e}")

# ==============================================================================
# TAB 3: PREDICTIVE MODELING
# ==============================================================================
elif choice == "3. Predictive Modeling":
    st.header("3. Predictive Modeling (XGBoost)")
    st.markdown("Train a model to enable advanced features (Optimization, Comparison, SHAP).")
    
    # Feature prep
    numeric_cols = df_wide.select_dtypes(include=np.number).columns.tolist()
    all_features_codes = [col for col in numeric_cols if col not in [TARGET_VAR, 'FIPS', 'Latitude', 'Longitude', 'VULNERABILITY_INDEX']]
    feature_code_to_name = {code: map_code_to_name(code) for code in all_features_codes}
    feature_name_to_code = {v: k for k, v in feature_code_to_name.items()}
    
    feature_groups = {}
    for code, name in feature_code_to_name.items():
        group = categorize_feature(code)
        if group not in feature_groups: feature_groups[group] = []
        feature_groups[group].append(name)
        
    default_codes_list = ['POVRATE21', 'MEDHHINC21', 'PCT_LACCESS_POP19', 'CHILDPOVRATE21', 'PCT_65OLDER20']
    default_names = [map_code_to_name(c) for c in default_codes_list if c in all_features_codes]

    st.markdown("#### **Feature Selection**")
    st.info("Select features using the checkboxes below.")
    
    selected_names = []
    for group, names in feature_groups.items():
        st.markdown(f"**{group}**")
        cols = st.columns(3)
        for i, name in enumerate(sorted(names)):
            is_default = name in default_names
            if cols[i % 3].checkbox(name, value=is_default, key=f"chk_{name}"):
                selected_names.append(name)
                
    features = [feature_name_to_code[name] for name in selected_names]

    if not features:
        st.warning("Please select at least one feature to train the model.")
    else:
        X = df_wide[features].copy()
        y = df_wide[TARGET_VAR]
        st.session_state['original_features_data'] = X.copy()
        
        scaler = StandardScaler()
        X_filled = X.fillna(X.mean())
        X_scaled = pd.DataFrame(scaler.fit_transform(X_filled), columns=X.columns, index=X.index)
        st.session_state['data_scaler'] = scaler

        st.markdown(f"**Status:** Ready to train on {len(X)} counties using {len(features)} features.")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training XGBoost Model..."):
                param_grid = {
                    'n_estimators': st.session_state['n_estimators'],
                    'learning_rate': st.session_state['learning_rate'],
                    'max_depth': st.session_state['max_depth']
                }
                final_model, final_mse, final_r2 = run_xgboost_tuning(X_scaled, y, st.session_state['test_size'], st.session_state['cv_folds'], st.session_state['random_state'], param_grid)

            if final_model:
                st.success("✅ Training Complete!")
                col1, col2, col3 = st.columns(3)
                col1.metric("Test MSE", f"{final_mse:.2f}")
                col2.metric("Test R²", f"{final_r2:.3f}")
                col3.metric("Explained Variance", f"{(final_r2 * 100):.1f}%")
                
                fig_res = px.scatter(x=st.session_state['y_test'], y=st.session_state['y_pred'], title='Predicted vs. Actual', labels={'x': 'Actual', 'y': 'Predicted'}, opacity=0.6, template="plotly_dark")
                fig_res.add_shape(type="line", line=dict(dash='dash', color='red'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
                fig_res.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_res, use_container_width=True)

# ==============================================================================
# TAB 4: SHAP & POLICY SANDBOX
# ==============================================================================
elif choice == "4. SHAP & Policy Sandbox":
    st.header("4. Interpretability & Policy Sandbox")
    st.markdown("---")
    final_model = st.session_state.get('best_model')
    X_test_scaled = st.session_state.get('X_test_scaled')
    model_features = st.session_state.get('model_features')

    if final_model is None:
        st.warning("⚠️ Please train the model in Tab 3 first.")
        st.stop()

    full_feature_names = map_codes_to_names_list(model_features)
    test_indices = X_test_scaled.index
    df_test_original = df_wide.iloc[test_indices].reset_index(drop=True)
    county_list = (df_test_original['County'] + ", " + df_test_original['State']).tolist()

    st.subheader("A. Feature Importance (SHAP)")
    if st.button("Calculate SHAP Values"):
        with st.spinner("Computing SHAP values..."):
            if SHAP_AVAILABLE:
                # Subsample for speed
                X_shap = X_test_scaled.sample(n=min(500, len(X_test_scaled)), random_state=42)
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(X_shap)
                
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_alpha(0)
                ax.patch.set_alpha(0)
                shap.summary_plot(shap_values, X_shap, feature_names=full_feature_names, show=False)
                st.pyplot(fig)
            else:
                st.error("SHAP library not installed.")

    st.markdown("---")
    st.subheader("B. 🧪 Health Policy Sandbox")
    sb_county = st.selectbox("Select Target County", options=county_list)
    if sb_county:
        idx = df_test_original[df_test_original['County'] + ", " + df_test_original['State'] == sb_county].index[0]
        raw_X = st.session_state['original_features_data'].iloc[[idx]].copy()
        baseline_pred = st.session_state['y_pred'][idx]
        actual_val = st.session_state['y_test'].iloc[idx]
        
        col_a, col_b = st.columns(2)
        col_a.metric("Actual Obesity Rate", f"{actual_val:.1f}%")
        col_b.metric("Baseline Predicted Rate", f"{baseline_pred:.1f}%")
        
        st.markdown("#### Adjust Policy Variables")
        modified_X_raw = raw_X.copy()
        cols = st.columns(min(3, len(model_features)))
        for i, feature_code in enumerate(model_features[:6]):
            col = cols[i % 3]
            feat_name = map_code_to_name(feature_code)
            current_val = float(raw_X[feature_code].iloc[0])
            min_val = float(st.session_state['original_features_data'][feature_code].min())
            max_val = float(st.session_state['original_features_data'][feature_code].max())
            with col:
                new_val = st.slider(f"{feat_name}", min_value=min_val, max_value=max_val, value=current_val, format="%.1f")
                modified_X_raw[feature_code] = new_val

        if st.button("Simulate Intervention"):
            scaler = st.session_state['data_scaler']
            modified_X_scaled = pd.DataFrame(scaler.transform(modified_X_raw), columns=modified_X_raw.columns)
            new_pred = final_model.predict(modified_X_scaled)[0]
            diff = new_pred - baseline_pred
            st.markdown("### Simulation Results")
            c1, c2 = st.columns(2)
            c1.metric("New Predicted Obesity Rate", f"{new_pred:.2f}%", delta=f"{diff:.2f}%", delta_color="inverse")

# ==============================================================================
# TAB 5: COMPARATIVE ANALYSIS
# ==============================================================================
elif choice == "5. Comparative Analysis":
    st.header("5. Comparative Analysis")
    st.markdown("Compare the risk profiles of two different counties side-by-side.")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train the model in Tab 3 first.")
        st.stop()
        
    county_list = (df_wide['County'] + ", " + df_wide['State']).tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        c1_name = st.selectbox("Select County A (Baseline)", county_list, index=0)
    with col2:
        c2_name = st.selectbox("Select County B (Comparison)", county_list, index=1)
        
    if c1_name and c2_name:
        idx1 = df_wide[df_wide['County'] + ", " + df_wide['State'] == c1_name].index[0]
        idx2 = df_wide[df_wide['County'] + ", " + df_wide['State'] == c2_name].index[0]
        
        row1 = df_wide.iloc[idx1]
        row2 = df_wide.iloc[idx2]
        
        st.subheader("Health Outcome Comparison")
        m1, m2, m3 = st.columns(3)
        val1, val2 = row1[TARGET_VAR], row2[TARGET_VAR]
        diff = val1 - val2
        
        m1.metric(f"{c1_name}", f"{val1:.1f}%")
        m2.metric("Difference (A - B)", f"{diff:.1f}%", delta_color="inverse")
        m3.metric(f"{c2_name}", f"{val2:.1f}%")
        
        st.divider()
        st.subheader("Risk Factor Radar Chart")
        
        feats = st.session_state['model_features'][:6] 
        feat_names = [map_code_to_name(f) for f in feats]
        
        orig_X = st.session_state['original_features_data']
        vals1_norm = []
        vals2_norm = []
        
        for f in feats:
            mx, mn = orig_X[f].max(), orig_X[f].min()
            denom = mx - mn if mx != mn else 1
            vals1_norm.append((row1[f] - mn) / denom)
            vals2_norm.append((row2[f] - mn) / denom)
            
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=vals1_norm, theta=feat_names, fill='toself', name=c1_name))
        fig.add_trace(go.Scatterpolar(r=vals2_norm, theta=feat_names, fill='toself', name=c2_name))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500, template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', polar=dict(bgcolor='rgba(0,0,0,0)'), font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 6: ROBUST POLICY SIMULATOR (WITH GUARDRAILS)
# ==============================================================================
elif choice == "6. Prescriptive Optimization":
    st.header("6. Robust Policy Simulator")
    st.info("d Monte Carlo Simulation: Testing thousands of policy combinations with logical guardrails.")

    if 'best_model' not in st.session_state:
        st.warning("Please train the model in Tab 3 first.")
        st.stop()

    # 1. Select Target
    county_list = (df_wide['County'] + ", " + df_wide['State']).tolist()
    target_county = st.selectbox("Select Target County", county_list)
    
    if target_county:
        idx = df_wide[df_wide['County'] + ", " + df_wide['State'] == target_county].index[0]
        feats = st.session_state['model_features']
        
        # Current Data
        current_row = df_wide.iloc[idx][feats].fillna(0)
        current_obesity = df_wide.iloc[idx][TARGET_VAR]
        
        scaler = st.session_state['data_scaler']
        model = st.session_state['best_model']
        
        # Calculate Baseline
        x0 = current_row.values.astype(float)
        x0_df = pd.DataFrame([x0], columns=feats)
        x0_scaled = scaler.transform(x0_df)
        baseline_pred = float(model.predict(x0_scaled)[0])
        
        col1, col2 = st.columns(2)
        col1.metric("Current Actual", f"{current_obesity:.2f}%")
        col1.caption(f"Model sees: {baseline_pred:.2f}%")
        
        # 2. Select Levers
        nice_feat_names = [map_code_to_name(f) for f in feats]
        default_levers = [n for n in nice_feat_names if "POVERTY" in n.upper() or "INCOME" in n.upper() or "ACCESS" in n.upper()]
        selected_levers = st.multiselect("Select Policy Levers to Test", nice_feat_names, default=default_levers[:5])
        
        # 3. Simulation Settings
        sim_count = st.slider("Simulation Runs", 100, 5000, 2000)
        max_change_pct = st.slider("Max Allowed Change (%)", 5, 50, 20) / 100.0
        
        if st.button("🎲 Run Simulation"):
            # Map selected names back to codes and indices
            feat_indices = []
            selected_codes = []
            var_map_rev = {v:k for k,v in st.session_state['variable_map'].items()}
            
            for name in selected_levers:
                code = var_map_rev.get(name, name)
                if code in feats:
                    feat_indices.append(feats.index(code))
                    selected_codes.append(code)
            
            if not feat_indices:
                st.error("Select at least one lever.")
                st.stop()

            # --- MONTE CARLO LOGIC WITH GUARDRAILS ---
            X_sim = np.tile(x0, (sim_count, 1))
            
            for i, code in zip(feat_indices, selected_codes):
                code_u = code.upper()
                
                # DIRECTIONAL LOGIC
                # 1. Variables we want to DECREASE (Poverty, Unemployment, Low Access, etc.)
                if any(x in code_u for x in ['POV', 'UNEMP', 'LACCESS', 'INACTIVE', 'SMOK']):
                    # Generate changes between -Max% and 0% (Only improvement)
                    changes = np.random.uniform(-max_change_pct, 0.0, sim_count)
                    
                # 2. Variables we want to INCREASE (Income, Education, etc.)
                elif any(x in code_u for x in ['INCOME', 'GRAD', 'COLLEGE', 'MEDHHINC']):
                    # Generate changes between 0% and +Max%
                    changes = np.random.uniform(0.0, max_change_pct, sim_count)
                    
                # 3. Neutral/Unknown Variables (Allow both directions)
                else:
                    changes = np.random.uniform(-max_change_pct, max_change_pct, sim_count)

                # Apply the change
                X_sim[:, i] = X_sim[:, i] * (1 + changes)

            # Predict
            X_sim_df = pd.DataFrame(X_sim, columns=feats)
            X_sim_scaled = scaler.transform(X_sim_df)
            preds = model.predict(X_sim_scaled)
            
            # Find Best Result
            best_idx = np.argmin(preds)
            best_pred = preds[best_idx]
            best_inputs = X_sim[best_idx]
            
            imp_total = baseline_pred - best_pred
            new_actual = current_obesity - imp_total
            
            # --- RESULTS DISPLAY ---
            if imp_total <= 0.001:
                st.warning("⚠️ No improvement found within these constraints.")
            else:
                st.success(f"✅ Strategy Found! Potential Reduction: -{imp_total:.2f}%")
                
                # Visuals
                c1, c2 = st.columns(2)
                c1.metric("New Predicted Rate", f"{new_actual:.2f}%", delta=f"-{imp_total:.2f}%", delta_color="inverse")
                
                # Table Formatting
                res_data = []
                for i in feat_indices:
                    orig = x0[i]
                    new = best_inputs[i]
                    change_val = new - orig
                    pct_change = (change_val / orig) if orig != 0 else 0
                    
                    res_data.append({
                        "Policy Lever": nice_feat_names[i],
                        "Current Value": orig,
                        "Proposed Value": new,
                        "Change": change_val,
                        "% Change": pct_change
                    })
                
                res_df = pd.DataFrame(res_data)

                # Use Streamlit Column Config for beautiful alignment and formatting
                st.dataframe(
                    res_df,
                    column_config={
                        "Policy Lever": st.column_config.TextColumn("Policy Lever", width="medium"),
                        "Current Value": st.column_config.NumberColumn("Current", format="%.2f"),
                        "Proposed Value": st.column_config.NumberColumn("Target", format="%.2f"),
                        "Change": st.column_config.NumberColumn("Abs Change", format="%.2f"),
                        "% Change": st.column_config.ProgressColumn(
                            "Relative Impact", 
                            format="%.1f%%", 
                            min_value=-0.5, 
                            max_value=0.5
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )
# ==============================================================================
# TAB 7: REPORTING & COLLABORATION
# ==============================================================================
elif choice == "7. Reporting & Collaboration":
    st.header("7. Reporting & Collaboration")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train the model in Tab 3 first.")
        st.stop()
        
    county_list = (df_wide['County'] + ", " + df_wide['State']).tolist()
    r_county = st.selectbox("Select County for Report", county_list)
    
    if r_county:
        idx = df_wide[df_wide['County'] + ", " + df_wide['State'] == r_county].index[0]
        target_val = df_wide.iloc[idx][TARGET_VAR]
        
        st.subheader("📝 Analyst Commentary")
        db_key = f"comment_{r_county}"
        existing = st.session_state['comments_db'].get(db_key, "")
        new_comment = st.text_area("Field Notes / Observations", value=existing, height=150)
        
        if st.button("Save Note to Database"):
            st.session_state['comments_db'][db_key] = new_comment
            st.success("Note saved successfully.")
            
        st.markdown("---")
        st.subheader("📄 Export Executive Report")
        
        if PDF_AVAILABLE:
            if st.button("Generate Full Analysis PDF"):
                feats = st.session_state['model_features']
                top_feats = feats[:10] 
                
                analysis_data = []
                orig_X = st.session_state['original_features_data'] 
                current_row = df_wide.iloc[idx][top_feats]
                
                for f in top_feats:
                    val = current_row[f]
                    avg = orig_X[f].mean()
                    pct = (orig_X[f] < val).mean()
                    
                    analysis_data.append({
                        'name': map_code_to_name(f),
                        'value': float(val),
                        'mean': float(avg),
                        'percentile': float(pct)
                    })

                scaler = st.session_state['data_scaler']
                model = st.session_state['best_model']
                full_row = df_wide.iloc[[idx]][feats].fillna(df_wide[feats].mean())
                row_scaled = pd.DataFrame(scaler.transform(full_row), columns=feats)
                predicted_val = float(model.predict(row_scaled)[0])

                try:
                    pdf_bytes = create_download_pdf(
                        r_county, 
                        analysis_data, 
                        st.session_state['comments_db'].get(db_key, ""), 
                        "Obesity Rate", 
                        float(target_val),
                        predicted_val
                    )
                    st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name=f"Full_Analysis_{r_county.replace(' ', '_')}.pdf", mime='application/pdf')
                except Exception as e:
                    st.error(f"PDF Generation failed. Error: {e}")
        else:
            st.error("FPDF library not installed. Please pip install fpdf.")

st.sidebar.markdown("---")
