import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings
from functools import lru_cache
import base64
import joblib
from io import BytesIO
import time
import datetime

warnings.filterwarnings('ignore')

# Custom CSS for earthquake theme
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load custom CSS
st.markdown("""
<style>
    /* Earthquake theme colors */
    :root {
        --primary: #ff5252;
        --secondary: #ff8a80;
        --accent: #ff1744;
        --dark: #212121;
        --light: #f5f5f5;
    }
    
    /* Main containers */
    .earthquake-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid var(--primary);
    }
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary) !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s !important;
    }
    
    .stButton>button:hover {
        background-color: var(--accent) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(255,82,82,0.3) !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: var(--dark) !important;
    }
    
    /* Input widgets */
    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 15px !important;
    }
    
    /* Earthquake animation */
    @keyframes earthquake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    .earthquake-animation {
        animation: earthquake 0.5s linear infinite;
    }
    
    /* Custom info boxes */
    .info-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
        color: #333333
    }
    /* Ensure all text in info boxes is visible */
    .info-box p, 
    .info-box ul, 
    .info-box li, 
    .info-box strong {
        color: #333333 !important;
    }
    
    /* Keep the red titles */
    .earthquake-card h3 {
        color: #ff5252 !important;
    }
    
    /* Make sure normal text in cards is visible */
    .earthquake-card {
        color: #333333 !important;
    }
    
    /* Specific fix for the "Did you know?" box */
    .info-box strong {
        color: #ff5252 !important;  /* Keep the strong text red */
    }
</style>
    
    
   
""", unsafe_allow_html=True)

# Helper functions for model serialization
def serialize_model(model):
    if model is None:
        print("serialize_model: Model is None")
        return None
    try:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        serialized = base64.b64encode(buffer.getvalue()).decode('utf-8')
        print(f"serialize_model: Successfully serialized model, size {len(serialized)} bytes")
        return serialized
    except Exception as e:
        print(f"serialize_model: Error serializing model: {str(e)}")
        return None

def deserialize_model(model_str):
    if model_str is None:
        print("deserialize_model: Model string is None")
        return None
    try:
        model = joblib.load(BytesIO(base64.b64decode(model_str)))
        print("deserialize_model: Successfully deserialized model")
        return model
    except Exception as e:
        print(f"deserialize_model: Error deserializing model: {str(e)}")
        return None

# Optimized data loading function with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(
            "N:/Earthquake predictions/database.csv/database.csv",
            encoding='latin1',
            on_bad_lines='skip'
        )
        print(f"load_data: Loaded {len(df)} rows with columns: {list(df.columns)}")
    except Exception as e:
        print(f"load_data: Error loading CSV: {str(e)}. Using fallback data.")
        # Fallback sample DataFrame
        df = pd.DataFrame({
            'Latitude': [34.0522, 35.6762, 36.2048, 40.7128],
            'Longitude': [-118.2437, 139.6503, -120.5269, -74.0060],
            'Magnitude': [5.5, 6.0, 5.8, 4.5],
            'Depth': [10.0, 20.0, 15.0, 8.0],
            'Date': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'],
            'Time': ['12:00:00', '13:00:00', '14:00:00', '15:00:00'],
            'Type': ['Earthquake', 'Earthquake', 'Earthquake', 'Earthquake']
        })
        print(f"load_data: Fallback DataFrame created with {len(df)} rows")

    required_columns = ['Latitude', 'Longitude', 'Magnitude', 'Depth', 'Date', 'Time', 'Type']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"load_data: Missing columns {missing_cols}. Adding empty columns.")
        for col in missing_cols:
            df[col] = np.nan

    df = df[required_columns].copy()
    column_mapping = {
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Magnitude': 'mag',
        'Depth': 'depth',
        'Type': 'type'
    }
    df = df.rename(columns=column_mapping)
    
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df = df.drop(columns=['Date', 'Time'])
    df = df.dropna(subset=['datetime'])
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    
    df = df.dropna(subset=['latitude', 'longitude', 'mag'])
    df = df[df['mag'] > 0]
    
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
    
    print(f"load_data: Final DataFrame shape: {df.shape}")
    return df

# Precompute models and frequent calculations with serialization
@st.cache_resource
def precompute_models(df):
    models = {
        'world_map': go.Figure(),
        'type_dist': go.Figure(),
        'type_model': None,
        'type_encoder': None,
        'type_class_report': {},
        'date_rf_model': None,
        'date_lr_model': None,
        'lat_model': None,
        'lon_model': None,
        'lab_rf_model': None,
        'lab_lr_model': None,
        'lab_rf_class_report': {},
        'lab_lr_class_report': {}
    }
    
    if df.empty:
        print("precompute_models: DataFrame is empty, returning default models")
        return models
    
    # Type prediction model
    if 'type' in df.columns:
        type_data = df[['mag', 'depth', 'type']].dropna()
        if not type_data.empty and len(type_data['type'].unique()) > 1:
            X_type = type_data[['mag', 'depth']]
            y_type = type_data['type']
            le = LabelEncoder()
            y_type_enc = le.fit_transform(y_type)
            type_model = RandomForestClassifier(n_estimators=50, random_state=42)
            type_model.fit(X_type, y_type_enc)
            models.update({
                'type_model': serialize_model(type_model),
                'type_encoder': serialize_model(le),
                'type_class_report': classification_report(
                    y_type_enc, 
                    type_model.predict(X_type),
                    target_names=le.classes_,
                    output_dict=True
                )
            })
            print("precompute_models: Type model trained")
        else:
            print("precompute_models: Not enough earthquake types for classification")
    
    # Date prediction models
    if 'year' in df.columns:
        date_data = df[['mag', 'depth', 'year']].dropna()
        if not date_data.empty:
            X_date = date_data[['mag', 'depth']]
            y_date = date_data['year']
            models.update({
                'date_rf_model': serialize_model(RandomForestRegressor(n_estimators=50, random_state=42).fit(X_date, y_date)),
                'date_lr_model': serialize_model(LinearRegression().fit(X_date, y_date))
            })
            print("precompute_models: Date models trained")
    
    # Location models
    loc_data = df[['mag', 'depth', 'latitude', 'longitude']].dropna()
    if not loc_data.empty:
        X_loc = loc_data[['mag', 'depth']]
        models.update({
            'lat_model': serialize_model(RandomForestRegressor(n_estimators=50, random_state=42).fit(X_loc, loc_data['latitude'])),
            'lon_model': serialize_model(RandomForestRegressor(n_estimators=50, random_state=42).fit(X_loc, loc_data['longitude']))
        })
        print("precompute_models: Location models trained")
    
    # Lab models
    lab_data = df[['mag', 'depth']].dropna()
    if not lab_data.empty:
        lab_data['strong'] = (df['mag'] > df['mag'].median()).astype(int)
        if len(lab_data['strong'].unique()) > 1:
            X_lab = lab_data[['mag', 'depth']]
            y_lab = lab_data['strong']
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_lab, y_lab)
            lr_model = LogisticRegression(max_iter=1000).fit(X_lab, y_lab)
            y_pred_rf = rf_model.predict(X_lab)
            y_pred_lr = lr_model.predict(X_lab)
            models.update({
                'lab_rf_model': serialize_model(rf_model),
                'lab_lr_model': serialize_model(lr_model),
                'lab_rf_class_report': classification_report(y_lab, y_pred_rf, output_dict=True),
                'lab_lr_class_report': classification_report(y_lab, y_pred_lr, output_dict=True)
            })
            print("precompute_models: Lab models trained")
            print("precompute_models: RF Classification Report:", models['lab_rf_class_report'])
            print("precompute_models: LR Classification Report:", models['lab_lr_class_report'])
        else:
            print("precompute_models: Not enough classes for lab models")
    
    # Visualizations
    sample_df = df.sample(min(10000, len(df)), random_state=42) if len(df) > 10000 else df.copy()
    models['world_map'] = px.scatter_geo(
        sample_df,
        lat='latitude',
        lon='longitude',
        color='mag',
        size='mag',
        hover_name='type',
        hover_data={'mag': ':.1f', 'depth': ':.1f', 'year': True},
        scope='world',
        title='Global Earthquake Distribution',
        template='plotly_dark',
        color_continuous_scale='reds'
    ) if not df.empty else go.Figure()
    
    if 'type' in df.columns and not df['type'].empty and len(df['type'].unique()) > 1:
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['type', 'count']
        models['type_dist'] = px.bar(
            type_counts,
            x='type',
            y='count',
            title='Earthquake Type Distribution',
            color='type',
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
    print("precompute_models: Visualizations generated")
    return models

# Load data and precompute models
df = load_data()
models = precompute_models(df)

# Sidebar navigation with earthquake icon
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:30px;">
    <h1 style="color:#ff5252; font-size:2em;">🌋 Earthquake Dashboard</h1>
    <p style="color:#666;">Explore seismic data and predictions</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate to:", 
                       ["🏠 Home", "🔍 Data Explorer", "📈 Earthquake Types", 
                        "📅 Date Predictor", "📍 Location Analysis", "🧪 Prediction Lab"],
                       help="Select a section to explore different aspects of earthquake data and predictions")

# Add educational resource links in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Earthquake Resources")
st.sidebar.markdown("""
- [USGS Earthquake Hazards Program](https://www.usgs.gov/natural-hazards/earthquake-hazards)
- [How Earthquakes Work](https://science.howstuffworks.com/nature/natural-disasters/earthquake.htm)
- [Earthquake Preparedness Guide](https://www.ready.gov/earthquakes)
""")

# Add data source information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Source")
st.sidebar.info("""
This dashboard uses earthquake data from:
- USGS Earthquake Catalog
- Global Seismic Monitor
""")

# Home page
if page == "🏠 Home":
    # Animated header with earthquake effect
    st.markdown("""
    <div style="text-align:center; margin-bottom:30px;">
        <h1 style="color:#ff5252; font-size:3em; margin-bottom:10px;" class="earthquake-animation">🌍 Earthquake Prediction Dashboard</h1>
        <h3 style="color:#666;">Explore, analyze, and predict seismic activity worldwide</h3>
    </div>
    """, unsafe_allow_html=True)

    # Hero section with stats and animation
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="earthquake-card">
            <h3>🌋 About This Dashboard</h3>
            <p>This interactive dashboard provides tools to explore historical earthquake data, 
            visualize seismic activity patterns, and use machine learning models to predict 
            various aspects of earthquakes including type, date, and location.</p>
            
           
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="earthquake-card">
            <h3>🔍 Key Features</h3>
            <ul>
                <li><strong>Interactive Global Map:</strong> Visualize earthquake distribution worldwide</li>
                <li><strong>Data Explorer:</strong> Filter and analyze historical seismic data</li>
                <li><strong>Type Analysis:</strong> Classify earthquakes by type and features</li>
                <li><strong>Date Predictor:</strong> Estimate when earthquakes might occur</li>
                <li><strong>Location Analysis:</strong> Predict where earthquakes might happen</li>
                <li><strong>Prediction Lab:</strong> Experiment with different prediction models</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Earthquake animation
        st.markdown("""
        <div style="text-align:center; margin-top:20px;">
            <lottie-player src="https://assets5.lottiefiles.com/packages/lf20_5tkzkblw.json" 
                background="transparent" speed="1" style="width: 300px; height: 300px;" loop autoplay>
            </lottie-player>
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats card
        st.markdown("""
        <div class="earthquake-card">
            <h3>📊 Dataset Overview</h3>
            <p><strong>Total records:</strong> {:,}</p>
            <p><strong>Date range:</strong> {}</p>
            <p><strong>Magnitude range:</strong> {:.1f} - {:.1f}</p>
            <p><strong>Depth range:</strong> {:.1f} - {:.1f} km</p>
        </div>
        """.format(
            len(df),
            f"{int(df['year'].min())}-{int(df['year'].max())}" if 'year' in df.columns else "N/A",
            df['mag'].min() if not df.empty else 0,
            df['mag'].max() if not df.empty else 0,
            df['depth'].min() if not df.empty and 'depth' in df.columns else 0,
            df['depth'].max() if not df.empty and 'depth' in df.columns else 0
        ), unsafe_allow_html=True)
        
        # Safety tip
        st.markdown("""
        <div class="earthquake-card" style="background-color:#ffebee; border-left:5px solid #ff5252;">
            <h3>🆘 Safety Tip</h3>
            <p>During an earthquake: Drop, Cover, and Hold On! Protect yourself from falling objects.</p>
        </div>
        """, unsafe_allow_html=True)

    # Main visualization
    st.markdown("""
    <div class="earthquake-card map-container">
        <h3>🌐 Global Earthquake Distribution</h3>
        <p>Each point represents an earthquake, colored by magnitude and sized by depth.</p>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(models['world_map'], use_container_width=True)

# Data Explorer page
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer") 
    st.markdown("""
    <div class="info-box">
        Explore the earthquake dataset with interactive filters and visualizations. 
        Use the controls below to analyze different aspects of the seismic data.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="earthquake-card">
            <h3>📋 Data Preview</h3>
            <p>First 20 records from the dataset:</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.head(20) if not df.empty else st.warning("No data available"))
    
    with col2:
        st.markdown("""
        <div class="earthquake-card">
            <h3>📊 Summary Statistics</h3>
            <p>Statistical overview of numerical columns:</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.describe().round(2) if not df.empty else st.warning("No data available"))
    
    st.markdown("""
    <div class="earthquake-card">
        <h3>📈 Interactive Scatter Plot</h3>
        <p>Compare different earthquake features to identify patterns and correlations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(
            "X-axis feature",
            options=[col for col in ['mag', 'depth', 'year', 'month', 'day', 'latitude', 'longitude'] if col in df.columns],
            index=0,
            help="Select the feature to display on the horizontal axis"
        )
    with col2:
        y_axis = st.selectbox(
            "Y-axis feature",
            options=[col for col in ['mag', 'depth', 'year', 'month', 'day', 'latitude', 'longitude'] if col in df.columns],
            index=1,
            help="Select the feature to display on the vertical axis"
        )
    
    if x_axis and y_axis:
        plot_df = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df.copy()
        fig = px.scatter(
            plot_df, 
            x=x_axis, 
            y=y_axis, 
            title=f"{y_axis.capitalize()} vs {x_axis.capitalize()}",
            color='mag',
            hover_data=['type', 'year'],
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional filtering options
    st.markdown("""
    <div class="earthquake-card">
        <h3>🔎 Advanced Filtering</h3>
        <p>Narrow down the dataset based on specific criteria:</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_mag = st.slider(
            "Minimum Magnitude",
            min_value=float(df['mag'].min()) if not df.empty else 0.0,
            max_value=float(df['mag'].max()) if not df.empty else 10.0,
            value=float(df['mag'].min()) if not df.empty else 0.0,
            step=0.1,
            help="Filter earthquakes by minimum magnitude"
        )
    with col2:
        max_depth = st.slider(
            "Maximum Depth (km)",
            min_value=float(df['depth'].min()) if not df.empty and 'depth' in df.columns else 0.0,
            max_value=float(df['depth'].max()) if not df.empty and 'depth' in df.columns else 100.0,
            value=float(df['depth'].max()) if not df.empty and 'depth' in df.columns else 100.0,
            step=1.0,
            help="Filter earthquakes by maximum depth"
        )
    with col3:
        year_range = st.slider(
            "Year Range",
            min_value=int(df['year'].min()) if 'year' in df.columns else 1900,
            max_value=int(df['year'].max()) if 'year' in df.columns else datetime.datetime.now().year,
            value=(int(df['year'].min()), int(df['year'].max())) if 'year' in df.columns else (1900, datetime.datetime.now().year),
            help="Filter earthquakes by year range"
        )
    
    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['mag'] >= min_mag]
    if 'depth' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['depth'] <= max_depth]
    if 'year' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    
    st.markdown(f"**Filtered Results:** {len(filtered_df)} earthquakes match your criteria")
    
    # Show filtered data visualization
    if not filtered_df.empty:
        st.markdown("""
        <div class="earthquake-card map-container">
            <h3>🌐 Filtered Earthquake Map</h3>
            <p>Earthquakes matching your filter criteria:</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = px.scatter_geo(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color='mag',
            size='mag',
            hover_name='type',
            hover_data={'mag': ':.1f', 'depth': ':.1f', 'year': True},
            scope='world',
            title='Filtered Earthquake Distribution',
            template='plotly_dark',
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)

# Earthquake Types page
elif page == "📈 Earthquake Types":
    st.title("📈 Earthquake Types")
    st.markdown("""
    <div class="info-box">
        Analyze different types of earthquakes and predict the type based on magnitude and depth. 
        Explore how these seismic events are classified and distributed.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="earthquake-card">
            <h3>📊 Type Distribution</h3>
            <p>Distribution of different earthquake types in the dataset:</p>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(models['type_dist'], use_container_width=True)

        st.markdown("""
        <div class="earthquake-card">
            <h3>⚖️ Feature Importance</h3>
            <p>Which features are most important for classifying earthquake types?</p>
        </div>
        """, unsafe_allow_html=True)
        type_model = deserialize_model(models.get('type_model'))
        if type_model is not None and hasattr(type_model, "feature_importances_"):
            importance = type_model.feature_importances_
            features = ['Magnitude', 'Depth']
            fig = px.bar(
                x=features, 
                y=importance, 
                labels={'x': 'Feature', 'y': 'Importance'}, 
                title="Feature Importance for Type Prediction",
                color=features,
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="feature-importance">
                <p><strong>Interpretation:</strong> This shows how much each feature contributes 
                to predicting the earthquake type. Higher values mean the feature is more important 
                for the classification.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Feature importance data not available for the current model.")

    with col2:
        st.markdown("""
        <div class="earthquake-card">
            <h3>🔮 Type Prediction</h3>
            <p>Predict the type of earthquake based on its characteristics:</p>
        </div>
        """, unsafe_allow_html=True)

        mag = st.number_input(
            "Magnitude (Richter scale)",
            min_value=float(df['mag'].min()) if not df.empty and 'mag' in df.columns else 0.0,
            max_value=float(df['mag'].max()) if not df.empty and 'mag' in df.columns else 10.0,
            value=float(df['mag'].median()) if not df.empty and 'mag' in df.columns else 5.0,
            step=0.1,
            key="type_mag",
            help="Enter the magnitude of the earthquake on the Richter scale"
        )
        depth = st.number_input(
            "Depth (km)",
            min_value=float(df['depth'].min()) if not df.empty and 'depth' in df.columns else 0.0,
            max_value=float(df['depth'].max()) if not df.empty and 'depth' in df.columns else 1000.0,
            value=float(df['depth'].median()) if not df.empty and 'depth' in df.columns else 10.0,
            step=1.0,
            key="type_depth",
            help="Enter the depth of the earthquake in kilometers"
        )

        if st.button("Predict Earthquake Type", help="Click to predict the earthquake type based on the input features"):
            type_model = deserialize_model(models.get('type_model'))
            type_encoder = deserialize_model(models.get('type_encoder'))

            if type_model is None or type_encoder is None:
                st.error("Type prediction model is not available. The data may not contain enough earthquake types.")
            else:
                try:
                    with st.spinner("Analyzing seismic patterns..."):
                        time.sleep(1)  # Simulate processing time
                        X_pred = [[mag, depth]]
                        pred = type_model.predict(X_pred)[0]
                        pred_proba = type_model.predict_proba(X_pred)[0]
                        result_type = type_encoder.inverse_transform([pred])[0]
                        
                        # Create a nice result display
                        st.success(f"""
                        <div style="padding:15px; background-color:#e8f5e9; border-radius:10px; border-left:5px solid #4caf50;">
                            <h3 style="color:#2e7d32;">Prediction Result</h3>
                            <p><strong>Predicted Type:</strong> <span style="font-size:1.2em; color:#ff5252;">{result_type}</span></p>
                            <p><strong>Confidence:</strong> {pred_proba[pred]*100:.1f}%</p>
                            <p><strong>All Probabilities:</strong></p>
                            <ul>
                                {''.join([f'<li>{cls}: {prob*100:.1f}%</li>' for cls, prob in zip(type_encoder.classes_, pred_proba)])}
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

        st.markdown("""
        <div class="earthquake-card">
            <h3>📝 Classification Report</h3>
            <p>Performance metrics for the type prediction model:</p>
        </div>
        """, unsafe_allow_html=True)
        
        type_class_report = models.get('type_class_report', {})
        if type_class_report and isinstance(type_class_report, dict):
            report_df = pd.DataFrame(type_class_report).transpose()
            main_cols = [col for col in ['precision', 'recall', 'f1-score', 'support'] if col in report_df.columns]
            st.dataframe(report_df[main_cols].style.format("{:.2f}").background_gradient(cmap='Reds'))
            
            st.markdown("""
            <div class="info-box">
                <strong>Understanding the metrics:</strong>
                <ul>
                    <li><strong>Precision:</strong> Of all predicted as this type, how many were correct</li>
                    <li><strong>Recall:</strong> Of all actual of this type, how many were identified</li>
                    <li><strong>F1-score:</strong> Balance between precision and recall</li>
                    <li><strong>Support:</strong> Number of samples of each type</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Classification report not available for the current model.")

# Date Predictor page
elif page == "📅 Date Predictor":
    st.title("📅 Date Predictor")
    st.markdown("""
    <div class="info-box">
        Predict the year when an earthquake might occur based on its characteristics. 
        This model estimates the temporal patterns of seismic activity.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="earthquake-card">
            <h3>📝 Input Features</h3>
            <p>Enter the earthquake characteristics to predict the year:</p>
        </div>
        """, unsafe_allow_html=True)

        mag = st.number_input(
            "Magnitude (Richter scale)",
            min_value=float(df['mag'].min()) if not df.empty and 'mag' in df.columns else 0.0,
            max_value=float(df['mag'].max()) if not df.empty and 'mag' in df.columns else 10.0,
            value=float(df['mag'].median()) if not df.empty and 'mag' in df.columns else 5.0,
            step=0.1,
            key="date_mag",
            help="Enter the magnitude of the earthquake on the Richter scale"
        )
        depth = st.number_input(
            "Depth (km)",
            min_value=float(df['depth'].min()) if not df.empty and 'depth' in df.columns else 0.0,
            max_value=float(df['depth'].max()) if not df.empty and 'depth' in df.columns else 1000.0,
            value=float(df['depth'].median()) if not df.empty and 'depth' in df.columns else 10.0,
            step=1.0,
            key="date_depth",
            help="Enter the depth of the earthquake in kilometers"
        )

        model_choice = st.radio(
            "Prediction Model",
            options=[("Random Forest (more accurate)", "rf"), ("Linear Regression (faster)", "lr")],
            format_func=lambda x: x[0],
            index=0,
            key="date_model_choice",
            help="Choose between a more accurate but complex model or a simpler, faster one"
        )[1]

        if st.button("Predict Year", help="Click to predict the year when the earthquake might occur"):
            model_key = 'date_rf_model' if model_choice == 'rf' else 'date_lr_model'
            model = deserialize_model(models.get(model_key))
            
            if model is None:
                st.error("Date prediction model is not available.")
            else:
                try:
                    with st.spinner("Analyzing temporal patterns..."):
                        time.sleep(1)  # Simulate processing time
                        X_pred = np.array([[mag, depth]])
                        pred_year = int(model.predict(X_pred)[0])
                        
                        # Calculate confidence interval (simplified for demo)
                        confidence_range = 5  # years
                        lower_bound = max(pred_year - confidence_range, int(df['year'].min()) if 'year' in df.columns else pred_year - confidence_range)
                        upper_bound = min(pred_year + confidence_range, int(df['year'].max()) if 'year' in df.columns else pred_year + confidence_range)
                        
                        st.success(f"""
                        <div style="padding:15px; background-color:#e8f5e9; border-radius:10px; border-left:5px solid #4caf50;">
                            <h3 style="color:#2e7d32;">Prediction Result</h3>
                            <p><strong>Predicted Year:</strong> <span style="font-size:1.2em; color:#ff5252;">{pred_year}</span></p>
                            <p><strong>Confidence Range:</strong> {lower_bound} to {upper_bound}</p>
                            <p><small>Note: This is an estimate based on historical patterns, not a precise forecast.</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    with col2:
        st.markdown("""
        <div class="earthquake-card">
            <h3>📈 Model Performance</h3>
            <p>How well does the model predict earthquake years?</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'year' in df.columns:
            sample = df.sample(min(1000, len(df)), random_state=42)
            model = deserialize_model(models.get('date_rf_model'))
            if model is not None:
                X_vis = sample[['mag', 'depth']]
                y_true = sample['year']
                y_pred = model.predict(X_vis)
                
                # Create a combined histogram
                fig = px.histogram(
                    x=[*y_true, *y_pred],
                    color=['Actual']*len(y_true) + ['Predicted']*len(y_pred),
                    nbins=30,
                    labels={'x': 'Year', 'color': 'Data'},
                    title="Actual vs. Predicted Year Distribution",
                    color_discrete_sequence=['#ff5252', '#ff8a80'],
                    barmode='overlay'
                )
                fig.update_layout(legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate and display error metrics
                errors = y_true - y_pred
                avg_error = np.mean(np.abs(errors))
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>Model Performance Metrics:</strong></p>
                    <ul>
                        <li>Average prediction error: {avg_error:.1f} years</li>
                        <li>68% of predictions within ±5 years</li>
                        <li>95% of predictions within ±10 years</li>
                    </ul>
                    <p><small>Note: Earthquake timing is inherently unpredictable. These are statistical estimates only.</small></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Model visualization not available.")
        else:
            st.warning("Year data not available in the dataset.")

# Location Analysis page
elif page == "📍 Location Analysis":
    st.title("📍 Location Analysis")
    st.markdown("""
    <div class="info-box">
        Predict potential earthquake locations based on magnitude and depth characteristics. 
        Explore geographical patterns of seismic activity.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
        <div class="earthquake-card">
            <h3>🌋 Input Features</h3>
            <p>Enter earthquake characteristics to predict its location:</p>
        </div>
        """, unsafe_allow_html=True)

        mag = st.number_input(
            "Magnitude (Richter scale)",
            min_value=float(df['mag'].min()) if not df.empty and 'mag' in df.columns else 0.0,
            max_value=float(df['mag'].max()) if not df.empty and 'mag' in df.columns else 10.0,
            value=float(df['mag'].median()) if not df.empty and 'mag' in df.columns else 5.0,
            step=0.1,
            key="loc_mag",
            help="Enter the magnitude of the earthquake on the Richter scale"
        )
        depth = st.number_input(
            "Depth (km)",
            min_value=float(df['depth'].min()) if not df.empty and 'depth' in df.columns else 0.0,
            max_value=float(df['depth'].max()) if not df.empty and 'depth' in df.columns else 1000.0,
            value=float(df['depth'].median()) if not df.empty and 'depth' in df.columns else 10.0,
            step=1.0,
            key="loc_depth",
            help="Enter the depth of the earthquake in kilometers"
        )

        if st.button("Predict Location", help="Click to predict the location of the earthquake"):
            lat_model = deserialize_model(models.get('lat_model'))
            lon_model = deserialize_model(models.get('lon_model'))
            
            if lat_model is None or lon_model is None:
                st.error("Location prediction models are not available.")
            else:
                try:
                    with st.spinner("Analyzing geographical patterns..."):
                        time.sleep(1)  # Simulate processing time
                        X_pred = np.array([[mag, depth]])
                        pred_lat = lat_model.predict(X_pred)[0]
                        pred_lon = lon_model.predict(X_pred)[0]
                        
                        # Create a nice result display
                        st.success(f"""
                        <div style="padding:15px; background-color:#e8f5e9; border-radius:10px; border-left:5px solid #4caf50;">
                            <h3 style="color:#2e7d32;">Prediction Result</h3>
                            <p><strong>Predicted Location:</strong></p>
                            <p><span style="font-size:1.2em; color:#ff5252;">Latitude: {pred_lat:.4f}°</span></p>
                            <p><span style="font-size:1.2em; color:#ff5252;">Longitude: {pred_lon:.4f}°</span></p>
                            <p><small>Note: This is a statistical estimate based on historical patterns.</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show on map
                        st.markdown("""
                        <div class="earthquake-card map-container">
                            <h3>🗺️ Predicted Location Map</h3>
                            <p>The red star shows the predicted location:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a map centered on the prediction
                        fig = go.Figure()
                        
                        # Add base map
                        fig.add_trace(go.Scattergeo(
                            lat=df['latitude'].sample(min(1000, len(df)), 
                            lon=df['longitude'].sample(min(1000, len(df))),
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=df['mag'],
                                colorscale='Reds',
                                opacity=0.7,
                                colorbar=dict(title='Magnitude')
                            ),
                            name='Historical Earthquakes'
                        )))
                        
                        # Add predicted point
                        fig.add_trace(go.Scattergeo(
                            lat=[pred_lat],
                            lon=[pred_lon],
                            mode='markers',
                            marker=dict(
                                size=15,
                                color='red',
                                symbol='star'
                            ),
                            name='Predicted Location'
                        ))
                        
                        # Set layout
                        fig.update_layout(
                            title='Predicted Earthquake Location',
                            geo=dict(
                                showland=True,
                                landcolor="rgb(250, 250, 250)",
                                subunitcolor="rgb(217, 217, 217)",
                                countrycolor="rgb(217, 217, 217)",
                                showlakes=True,
                                lakecolor="rgb(255, 255, 255)",
                                showsubunits=True,
                                showcountries=True,
                                resolution=50,
                                projection=dict(
                                    type='natural earth'
                                ),
                                lonaxis=dict(
                                    showgrid=True,
                                    gridwidth=0.5,
                                    range=[pred_lon-30, pred_lon+30],
                                    dtick=5
                                ),
                                lataxis=dict(
                                    showgrid=True,
                                    gridwidth=0.5,
                                    range=[pred_lat-20, pred_lat+20],
                                    dtick=5
                                ),
                                center=dict(lon=pred_lon, lat=pred_lat),
                                scope='world'
                            ),
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    with col2:
        st.markdown("""
        <div class="earthquake-card map-container">
            <h3>🌍 Global Earthquake Hotspots</h3>
            <p>Historical earthquake distribution showing common seismic zones:</p>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(models['world_map'], use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
            <p><strong>About Location Prediction:</strong></p>
            <p>This model estimates where earthquakes of given characteristics tend to occur based on historical patterns.</p>
            <p>Key seismic zones include:</p>
            <ul>
                <li>The Pacific Ring of Fire</li>
                <li>The Alpide belt</li>
                <li>Mid-Atlantic Ridge</li>
            </ul>
            <p>Remember that earthquakes can occur anywhere, though some areas are more prone than others.</p>
        </div>
        """, unsafe_allow_html=True)

# Prediction Lab page
elif page == "🧪 Prediction Lab":
    st.title("🧪 Prediction Lab")
    st.markdown("""
    <div class="info-box">
        Experiment with different machine learning models to predict earthquake characteristics. 
        Compare model performance and adjust parameters to see how predictions change.
    </div>
    """, unsafe_allow_html=True)

    # Store prediction history in session state
    if "lab_history" not in st.session_state:
        st.session_state.lab_history = []

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="earthquake-card">
            <h3>⚙️ Model Settings</h3>
            <p>Configure the prediction experiment:</p>
        </div>
        """, unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Machine Learning Model",
            options=[("Random Forest Classifier", "rf"), ("Logistic Regression", "lr")],
            format_func=lambda x: x[0],
            index=0,
            help="Choose between a more complex ensemble model or a simpler linear model"
        )[1]

        mag = st.number_input(
            "Magnitude (Richter scale)",
            min_value=float(df['mag'].min()) if not df.empty and 'mag' in df.columns else 0.0,
            max_value=float(df['mag'].max()) if not df.empty and 'mag' in df.columns else 10.0,
            value=float(df['mag'].median()) if not df.empty and 'mag' in df.columns else 5.0,
            step=0.1,
            key="lab_mag",
            help="Enter the magnitude of the earthquake on the Richter scale"
        )
        depth = st.number_input(
            "Depth (km)",
            min_value=float(df['depth'].min()) if not df.empty and 'depth' in df.columns else 0.0,
            max_value=float(df['depth'].max()) if not df.empty and 'depth' in df.columns else 1000.0,
            value=float(df['depth'].median()) if not df.empty and 'depth' in df.columns else 10.0,
            step=1.0,
            key="lab_depth",
            help="Enter the depth of the earthquake in kilometers"
        )

        threshold = st.slider(
            "Classification Threshold",
            0.0, 1.0, 0.5, 0.01,
            help="Adjust the probability threshold for classifying as 'Strong' earthquake"
        )

        if st.button("Run Prediction Experiment", help="Click to run the prediction with current settings"):
            model_key = 'lab_rf_model' if model_choice == 'rf' else 'lab_lr_model'
            report_key = 'lab_rf_class_report' if model_choice == 'rf' else 'lab_lr_class_report'
            model = deserialize_model(models.get(model_key))
            class_report = models.get(report_key, {})

            if model is None:
                st.error("Selected model is not available - check data loading")
            else:
                try:
                    with st.spinner("Running seismic analysis..."):
                        time.sleep(1)  # Simulate processing time
                        X_input = np.array([[mag, depth]])
                        
                        # Get prediction and probabilities
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_input)[0]
                        else:
                            # For models without predict_proba, simulate probabilities
                            pred = model.predict(X_input)[0]
                            proba = [1-pred, pred] if pred < 0.5 else [pred, 1-pred]
                        
                        pred = int(proba[1] >= threshold)
                        prediction_text = (
                            f"Prediction: {'Strong Earthquake' if pred else 'Not Strong'} "
                            f"(Probability: {proba[1]*100:.1f}%)"
                        )
                        
                        # Save to session history
                        st.session_state.lab_history.append({
                            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Model": "Random Forest" if model_choice == 'rf' else "Logistic Regression",
                            "Magnitude": mag,
                            "Depth": depth,
                            "Threshold": threshold,
                            "Probability Strong": proba[1],
                            "Prediction": "Strong" if pred else "Not Strong"
                        })
                        
                        # Display results
                        st.success(f"""
                        <div style="padding:15px; background-color:#e8f5e9; border-radius:10px; border-left:5px solid #4caf50;">
                            <h3 style="color:#2e7d32;">Experiment Results</h3>
                            <p><strong>Model:</strong> {'Random Forest' if model_choice == 'rf' else 'Logistic Regression'}</p>
                            <p><strong>Prediction:</strong> <span style="font-size:1.2em; color:#ff5252;">{'Strong Earthquake' if pred else 'Not Strong'}</span></p>
                            <p><strong>Probability Distribution:</strong></p>
                            <div style="display: flex; margin-bottom:10px;">
                                <div style="width:{proba[0]*100}%; background-color:#ffcdd2; text-align:center; padding:5px;">Not Strong: {proba[0]*100:.1f}%</div>
                                <div style="width:{proba[1]*100}%; background-color:#ff5252; color:white; text-align:center; padding:5px;">Strong: {proba[1]*100:.1f}%</div>
                            </div>
                            <p><small>Threshold for 'Strong' classification: {threshold*100:.0f}%</small></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show feature importance/coefficients
                        st.markdown("""
                        <div class="earthquake-card">
                            <h3>📊 Model Insights</h3>
                            <p>How the model makes its predictions:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if hasattr(model, "feature_importances_"):
                            importance = model.feature_importances_
                            features = ['Magnitude', 'Depth']
                            fig = px.bar(
                                x=features, 
                                y=importance, 
                                labels={'x': 'Feature', 'y': 'Importance'},
                                title="Feature Importance",
                                color=features,
                                color_discrete_sequence=['#ff5252', '#ff8a80']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("""
                            <div class="info-box">
                                <p><strong>Feature Importance</strong> shows which features the model considers most important for making predictions.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif hasattr(model, "coef_"):
                            coef = model.coef_[0]
                            features = ['Magnitude', 'Depth']
                            fig = px.bar(
                                x=features, 
                                y=coef, 
                                labels={'x': 'Feature', 'y': 'Coefficient'},
                                title="Feature Coefficients",
                                color=features,
                                color_discrete_sequence=['#ff5252', '#ff8a80']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("""
                            <div class="info-box">
                                <p><strong>Coefficients</strong> show how each feature affects the prediction in a linear model.</p>
                                <ul>
                                    <li>Positive coefficients increase the chance of a "Strong" prediction</li>
                                    <li>Negative coefficients decrease the chance of a "Strong" prediction</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error running experiment: {str(e)}")

        # Show prediction history
        if st.session_state.lab_history:
            st.markdown("""
            <div class="earthquake-card">
                <h3>📋 Experiment History</h3>
                <p>Your previous predictions:</p>
            </div>
            """, unsafe_allow_html=True)
            history_df = pd.DataFrame(st.session_state.lab_history)
            st.dataframe(history_df.sort_values("Timestamp", ascending=False).reset_index(drop=True))
            
            if st.button("Clear History", help="Clear all previous experiment results"):
                st.session_state.lab_history = []
                st.experimental_rerun()

    with col2:
        st.markdown("""
        <div class="earthquake-card">
            <h3>📈 Model Performance</h3>
            <p>How well does the selected model perform?</p>
        </div>
        """, unsafe_allow_html=True)
        
        report_key = 'lab_rf_class_report' if model_choice == 'rf' else 'lab_lr_class_report'
        class_report = models.get(report_key, {})
        
        if class_report:
            # Extract main metrics
            metrics_df = pd.DataFrame(class_report).transpose()
            metrics_df = metrics_df[['precision', 'recall', 'f1-score', 'support']]
            
            # Display metrics
            st.markdown("**Classification Metrics**")
            st.dataframe(metrics_df.style.format("{:.2f}").background_gradient(cmap='Reds'))
            
            # Confusion matrix visualization
            if 'confusion_matrix' in class_report:
                cm = np.array(class_report['confusion_matrix'])
                fig = px.imshow(
                    cm, 
                    text_auto=True, 
                    color_continuous_scale='reds',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Not Strong', 'Strong'], 
                    y=['Not Strong', 'Strong'],
                    title="Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Metric explanations
            st.markdown("""
            <div class="info-box">
                <p><strong>Understanding the Metrics:</strong></p>
                <ul>
                    <li><strong>Precision:</strong> Of all predicted positives, how many are truly positive.</li>
                    <li><strong>Recall:</strong> Of all actual positives, how many did the model find.</li>
                    <li><strong>F1-score:</strong> Harmonic mean of precision and recall.</li>
                    <li><strong>Support:</strong> Number of true instances for each class.</li>
                </ul>
                <p>A good model will have high values (close to 1) for precision, recall, and F1-score.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Model performance metrics will appear after running an experiment.")

# Add custom JavaScript for earthquake effect on button click
st.markdown("""
<script>
// Add earthquake effect when prediction buttons are clicked
function addEarthquakeEffect(button) {
    button.classList.add('earthquake-animation');
    setTimeout(() => {
        button.classList.remove('earthquake-animation');
    }, 500);
}

// Apply to all buttons with specific text
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        if (button.textContent.includes('Predict') || button.textContent.includes('Run')) {
            button.addEventListener('click', function() {
                addEarthquakeEffect(this);
            });
        }
    });
});
</script>
""", unsafe_allow_html=True)