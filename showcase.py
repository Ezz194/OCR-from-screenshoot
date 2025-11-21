import streamlit as st
import streamlit.components.v1 as components  # Import for iFrame
import pandas as pd
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Project",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("🔍 Project Navigation")
    
    # Main Navigation
    page = st.radio(
        "Go to:",
        [
            "Project Overview", 
            "Data Engineering (ETL)", 
            "Feature Engineering", 
            "Model Training", 
            "Live Prediction Demo",
            "Visualization Dashboard"  # <--- ADDED NEW PAGE HERE
        ]
    )
    
    st.markdown("---")
    
    # THE REQUESTED EXPANDABLE SECTION
    with st.expander("👨‍💻 About the Developer", expanded=False):
        st.write("""
        **Project Status:** Completed
        
        This project demonstrates a full End-to-End Big Data pipeline using:
        - **Apache Airflow** (Orchestration)
        - **PySpark** (Distributed Processing)
        - **Google Cloud Storage** (Data Lake)
        - **Spark ML** (Machine Learning)
        """)
        st.info("Connect with me on LinkedIn / GitHub")

# --- PAGE 1: PROJECT OVERVIEW ---
if page == "Project Overview":
    st.title("📉 Big Data Customer Churn Prediction")
    
    st.markdown("""
    ### Executive Summary
    This project aims to predict customer churn using a distributed big data pipeline. 
    It leverages the power of **PySpark** to process large datasets stored in a **Google Cloud Data Lake**, 
    orchestrated by **Apache Airflow**.
    """)

    # Architecture Diagram
    st.subheader("System Architecture")
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#f0f2f6"];
            
            Raw [label="Raw Data (CSV)\nGCS Bucket", shape=cylinder, fillcolor="#FFCDD2"];
            Airflow [label="Airflow DAGs", fillcolor="#BBDEFB"];
            Spark [label="PySpark Cluster", fillcolor="#FFF9C4"];
            Cleaned [label="Cleaned Data\n(Parquet)", shape=cylinder, fillcolor="#C8E6C9"];
            Enriched [label="Enriched Features\n(Parquet)", shape=cylinder, fillcolor="#C8E6C9"];
            Model [label="Random Forest\nModel", shape=component, fillcolor="#D1C4E9"];
            
            Raw -> Spark [label="Ingest"];
            Airflow -> Spark [label="Trigger"];
            Spark -> Cleaned [label="ETL"];
            Cleaned -> Spark [label="Feature Eng"];
            Spark -> Enriched [label="Transform"];
            Enriched -> Spark [label="Train"];
            Spark -> Model [label="Save"];
        }
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**ETL Layer**")
        st.write("Handles missing data, duplicate removal, and schema validation using PySpark.")
    with col2:
        st.warning("**ML Pipeline**")
        st.write("String Indexing, One-Hot Encoding, and Vector Assembly for feature preparation.")
    with col3:
        st.error("**Modeling**")
        st.write("Random Forest Classifier with Grid Search Cross-Validation (CV) for hyperparameter tuning.")

# --- PAGE 2: DATA ENGINEERING (ETL) ---
elif page == "Data Engineering (ETL)":
    st.title("🛠️ Data Engineering Layer")
    st.write("The ETL process is managed by `air-lake-spark.py`.")
    
    st.subheader("Key Responsibilities")
    st.markdown("""
    1. **Ingestion:** Downloads raw CSV data from GCS to a temporary spark context.
    2. **Cleaning:** - Drops duplicates.
        - Imputes numeric missing values (Age, Monthly Charges) with **Mean/Median**.
        - Imputes categorical missing values (Gender, Internet Service) with **Mode**.
    3. **Storage:** Saves the processed data back to GCS in **Parquet** format for optimized querying.
    """)

    with st.expander("View ETL Code (Snippet)", expanded=True):
        st.code("""
def clean_and_process_data(**kwargs):
    # ... (Spark Session creation) ...
    
    # 1. Drop duplicates
    df_no_duplicates = df.dropDuplicates()
    
    # 2. Calculate Statistics for Imputation
    median_age = df_no_duplicates.approxQuantile("age", [0.5], 0.0)[0]
    mean_monthly_charges = df_no_duplicates.select(F.mean("monthly_charges")).first()[0]
    mode_gender = df_no_duplicates.groupBy("gender").count().orderBy(F.desc("count")).first()[0]
    
    # 3. Fill missing values
    df_filled = df_no_duplicates.fillna({
        "age": median_age,
        "gender": mode_gender,
        "monthly_charges": mean_monthly_charges,
        "internet_service": mode_internet_service,
        "tech_support": "No"
    })
    
    # 4. Write to Parquet
    df_processed.write.mode("overwrite").parquet(processed_temp_path)
        """, language="python")

# --- PAGE 3: FEATURE ENGINEERING ---
elif page == "Feature Engineering":
    st.title("⚙️ Feature Engineering Pipeline")
    st.write("Handled by `Mal.py`, this DAG prepares the data for machine learning algorithms.")
    
    st.info("The pipeline converts raw business logic into mathematical vectors.")
    
    st.subheader("Pipeline Stages")
    
    stages = [
        "**String Indexer**: Converts strings (e.g., 'Male', 'Female') into indices (0, 1).",
        "**One Hot Encoder**: Converts indices into binary vectors to prevent ordinal bias.",
        "**Vector Assembler**: Combines all features (Age, Tenure, Encoded Columns) into a single `features` vector.",
        "**Standard Scaler**: Normalizes numeric features so large numbers don't dominate the model."
    ]
    
    for stage in stages:
        st.markdown(f"- {stage}")

    st.subheader("Implementation Details")
    st.code("""
# Define pipeline stages
stages = []

# Stage 1: Index target variable
label_indexer = StringIndexer(inputCol="churn", outputCol="label")

# Stage 2-3: Index & Encode Categoricals
cat_indexers = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cat_cols)
cat_encoder = OneHotEncoder(inputCols=indexed_cat_cols, outputCols=ohe_cat_cols)

# Stage 4: Vector Assemble
final_assembler = VectorAssembler(
    inputCols=ohe_cat_cols + ["scaled_numeric_features"], 
    outputCol="features"
)

# Stage 5: Fit Pipeline
pipeline = Pipeline(stages=stages)
model = pipeline.fit(df)
    """, language="python")

# --- PAGE 4: MODEL TRAINING ---
elif page == "Model Training":
    st.title("🧠 Model Training")
    st.write("The model logic is found in `model_creation.py`.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Algorithm", "Random Forest Classifier")
        st.metric("Library", "Spark ML")
    with col2:
        st.metric("Tuning Strategy", "Grid Search CV")
        st.metric("Target Metric", "Accuracy / F1-Score")
        
    st.subheader("Hyperparameter Tuning")
    st.write("We used `CrossValidator` and `ParamGridBuilder` to find the best model parameters:")
    st.json({
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30],
        "min_impurity_decrease": [0.0, 0.01, 0.05]
    })
    
    st.subheader("Code Snippet")
    st.code("""
rf = RandomForestClassifier(featuresCol="features", labelCol="label")

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

grid.fit(X, y)
best_model = grid.best_estimator_
    """, language="python")

# --- PAGE 5: LIVE PREDICTION DEMO ---
elif page == "Live Prediction Demo":
    st.title("🔮 Churn Prediction Demo")
    st.markdown("""
    *Note: This interface mimics the model's behavior. In production, this would send an API request to the deployed Spark model.*
    """)
    
    st.subheader("Customer Profile")
    
    # Input Form
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 90, 30)
            senior = st.checkbox("Senior Citizen")
            
        with col2:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            
        submit = st.form_submit_button("Predict Churn Probability")
        
    if submit:
        # SIMULATED PREDICTION LOGIC
        risk_score = 0
        
        # High risk factors
        if contract == "Month-to-month": risk_score += 40
        if internet == "Fiber optic": risk_score += 20
        if tenure < 12: risk_score += 20
        if tech_support == "No": risk_score += 15
        if monthly_charges > 80: risk_score += 10
        
        # Low risk factors
        if contract == "Two year": risk_score -= 40
        if tenure > 48: risk_score -= 30
        if tech_support == "Yes": risk_score -= 10
        
        # Normalize
        prob = max(0, min(100, risk_score + np.random.randint(-5, 5)))
        
        with st.spinner("Calling Model Endpoint..."):
            time.sleep(1) # Simulate API latency
        
        st.markdown("### Prediction Result")
        
        col_res1, col_res2 = st.columns([1, 3])
        
        with col_res1:
            if prob > 50:
                st.error(f"Churn: YES")
            else:
                st.success(f"Churn: NO")
                
        with col_res2:
            st.progress(prob / 100)
            st.caption(f"Probability of Churning: {prob}%")
            
        if prob > 50:
            st.warning("⚠️ Recommendation: Offer this customer a 1-year contract discount immediately.")
        else:
            st.success("✅ This customer is stable.")

# --- PAGE 6: VISUALIZATION DASHBOARD ---
elif page == "Visualization Dashboard":
    st.title("📊 Interactive Dashboard")
    st.write("Embed your Looker Studio, Tableau, or PowerBI dashboard here to show business metrics.")
    
    # ---------------------------------------------------------
    # 🔗 PASTE YOUR DASHBOARD URL BELOW
    # ---------------------------------------------------------
    dashboard_url = "https://app.powerbi.com/view?r=eyJrIjoiMTNlM2ExMmUtODc3YS00MzkxLWJmNjMtYmFhY2M1NTMxNjEwIiwidCI6ImVhZjYyNGM4LWEwYzQtNDE5NS04N2QyLTQ0M2U1ZDc1MTZjZCIsImMiOjh9"
    
    # Optional: UI Override for testing
    use_manual = st.checkbox("Use Manual URL Input", value=False)
    if use_manual:
        dashboard_url = st.text_input("Paste Dashboard URL here:", dashboard_url)

    st.markdown("---")
    
    if dashboard_url and "..." not in dashboard_url:
        # Adjust height as needed for your specific dashboard
        components.iframe(dashboard_url, height=800, scrolling=True)
    else:
        st.info("👆 Please replace the `dashboard_url` variable in the code with your actual Dashboard link.")
        st.image("https://placehold.co/800x400?text=Dashboard+Placeholder", caption="Dashboard will appear here")
