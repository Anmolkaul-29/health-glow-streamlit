import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
# from geopy.distance import geodesic
from shapely.geometry import Point, Polygon
# import geopandas as gpd
# from scipy.spatial.distance import cdist
import json
from io import StringIO
import joblib
import os
IS_HF = os.getenv("SPACE_ID") is not None


# Page Configuration
st.set_page_config(
    page_title="Health & Glow Store Location Analysis",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Navigation Pills */
    .nav-pills {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
        flex-wrap: wrap;
    }
    
    .nav-pill {
        background: white;
        border: 2px solid #667eea;
        color: #667eea;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-decoration: none;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-pill.active {
        background: #667eea;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Card Styles */
    .analysis-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        transition: transform 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-card h3 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Status Cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .status-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .status-warning {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .status-info {
        background: #d1ecf1;
        border-color: #17a2b8;
        color: #0c5460;
    }
    
    .status-error {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .css-1v0mbdj {
        color: white;
    }
    
    /* Map Container */
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 1rem 0;
    }
    
    /* Instructions */
    .instructions {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .instructions h4 {
        color: #495057;
        margin-bottom: 1rem;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .nav-pills {
            flex-direction: column;
            align-items: center;
        }
        
        .analysis-card {
            padding: 1rem;
        }
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5a6fd8;
    }
</style>
""", unsafe_allow_html=True)

# Define bounds for Bangalore and Hyderabad
bangalore_bounds = (12.8, 13.2, 77.4, 77.8)  # (min_lat, max_lat, min_lon, max_lon)
hyderabad_bounds = (17.3, 17.5, 78.35, 78.6)  # Adjust these values as needed
chennai_bounds = (12.7, 13.2, 80.13, 80.4)

def is_within_bounds(lat, lon, city):
    """Check if the coordinates are within the bounds of the selected city."""
    if city == "Bangalore":
        return (bangalore_bounds[0] <= lat <= bangalore_bounds[1] and 
                bangalore_bounds[2] <= lon <= bangalore_bounds[3])
    elif city == "Hyderabad":
        return (hyderabad_bounds[0] <= lat <= hyderabad_bounds[1] and 
                hyderabad_bounds[2] <= lon <= hyderabad_bounds[3])
    elif city == "Chennai":
        return (chennai_bounds[0] <= lat <= chennai_bounds[1] and 
                chennai_bounds[2] <= lon <= chennai_bounds[3])
    return False

# Initialize Session State
# def initialize_session_state():
#     """Initialize all session state variables"""
#     default_values = {
#         'selected_lat': 12.9716,
#         'selected_lon': 77.5946,
#         'boundary_width_km': 6,
#         'boundary_height_km': 6,
#         'analysis_results': None,
#         'datasets_loaded': False,
#         'map_key': 0,  # This will help with map rerendering
#         'city_selection': 'Bangalore'
#     }
    
#     for key, value in default_values.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# initialize_session_state()

@st.cache_data
def load_all_datasets():
    """Load all required datasets for analysis"""
    datasets = {}

    try:
        # ================================================
        # 1️⃣ Load Housing + Population + Competitors + IT
        # ================================================
        datasets['housing'] = pd.read_csv("data/NoBrokerdata.csv")
        datasets['housing_hyderabad'] = pd.read_csv("data/hyderabad_real.csv")
        datasets['housing_chennai'] = pd.read_csv("data/chennai_real.csv")

        datasets['housing']['rent_per_sqft'] = (
            datasets['housing']['total_rent'] / datasets['housing']['total_sqft']
        )

        datasets['population'] = pd.read_csv("data/Population density of bangalore.csv")
        datasets['population_hyderabad'] = pd.read_csv("data/hyderabad_population_density.csv")
        datasets['population_chennai'] = pd.read_csv("data/chennai_population_density.csv")

        datasets['it_companies'] = pd.read_csv("data/bangalore-corporate-offices-2025-07-23.csv")
        datasets['competitors'] = pd.read_csv("data/extracted_competitor_data_final.csv")

        # ================================================
        # 2️⃣ Load Health & Glow Store Data
        # ================================================
        bangalore_hg = pd.read_csv("data/bangalore_health_glow_stores_2025-07-23.csv")
        hyd_che_hg = pd.read_csv("data/hydandcheLocation.csv")
        datasets['healthglow'] = pd.concat([bangalore_hg, hyd_che_hg], ignore_index=True)

        # ==============================================================
        # 3️⃣ LOAD MASTER LAT/LONG (STORE COORDINATES)
        # ==============================================================
        df_master_latlon = pd.read_excel("data/latlong.xlsx")

        df_master_latlon = df_master_latlon.rename(columns={
            "location_code": "LocationCode",
            "latitude": "Store_Lat",
            "longitude": "Store_Lng"
        })

        # ==============================================================
        # 4️⃣ LOAD PIN CODES (no lat/long inside this file)
        # ==============================================================
        df_pins = pd.read_csv("data/pin_codes.csv")
        

        df_pins = df_pins.rename(columns={
            "Locationcode": "LocationCode",
            "LocationName": "LocationName",
            "StoreLocationPinCode": "StoreLocationPinCode",
            "ServiceablePinCode": "ServiceablePinCode"
        })
        df_pins = df_pins[df_pins["ServiceablePinCode"] != 562130]
        # ==============================================================
        # 5️⃣ MERGE STORE LAT/LNG (correct join)
        # ==============================================================
        df_pins = df_pins.merge(
            df_master_latlon[["LocationCode", "Store_Lat", "Store_Lng"]],
            on="LocationCode",
            how="left"
        )

        # ==============================================================
        # 6️⃣ GEOCODE SERVICEABLE PINCODES
        # ==============================================================
        # from geopy.geocoders import Nominatim
        # geolocator = Nominatim(user_agent="h_and_g_geocoder")

        # @st.cache_data(show_spinner=False)
        # def geocode_pin(pin):
        #     try:
        #         if pd.isna(pin): 
        #             return None, None
        #         loc = geolocator.geocode(f"{pin}, India")
        #         return (loc.latitude, loc.longitude) if loc else (None, None)
        #     except:
        #         return None, None

        # df_pins["Serv_Lat"], df_pins["Serv_Lng"] = zip(
        #     *df_pins["ServiceablePinCode"].apply(geocode_pin)
        df_pins["Serv_Lat"] = None
        df_pins["Serv_Lng"] = None

        

        # ==============================================================
        # 7️⃣ VALIDATE SERVICEABLE PIN LAT/LONG (remove outliers)
        # ==============================================================
        def is_valid_lat_lon(lat, lon):
            return (
                lat is not None and lon is not None and
                6 <= float(lat) <= 38 and
                68 <= float(lon) <= 98
            )

        invalid_rows = df_pins[
            ~df_pins.apply(lambda r: is_valid_lat_lon(r["Serv_Lat"], r["Serv_Lng"]), axis=1)
        ]

        if len(invalid_rows) > 0:
            # st.warning("⚠ Invalid serviceable pins removed:")
            # st.dataframe(invalid_rows)

            df_pins = df_pins[
            df_pins.apply(lambda r: is_valid_lat_lon(r["Serv_Lat"], r["Serv_Lng"]), axis=1)
        ].copy()

        # ==============================================================
        # 8️⃣ SAVE CLEANED PIN MAPPING
        # ==============================================================
        datasets["pin_mapping"] = df_pins

        return datasets

    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None





def combine_housing_data(datasets):
    """Combine housing data from Bangalore, Hyderabad, and Chennai"""
    
    # Combine all housing data into a single DataFrame
    bangalore_data = datasets['housing']
    hyderabad_data = datasets['housing_hyderabad']
    chennai_data = datasets['housing_chennai']

    # Add a new column to indicate the city
    bangalore_data['city'] = 'Bangalore'
    hyderabad_data['city'] = 'Hyderabad'
    chennai_data['city'] = 'Chennai'

    # Concatenate the DataFrames
    combined_housing_data = pd.concat([bangalore_data, hyderabad_data, chennai_data], ignore_index=True)
    return combined_housing_data

# def standardize_healthglow_data(datasets):
    # """Standardize Health & Glow store data from different datasets"""
    # # Process Bangalore data
    # bangalore_data = datasets['healthglow'][['place_id', 'name', 'Latitude', 'Longitude', 'address', 'rating']]
    # bangalore_data.columns = ['id', 'name', 'Latitude', 'Longitude', 'address', 'rating']
    # bangalore_data['city'] = 'Bangalore'

    # # Process combined Chennai and Hyderabad data
    # combined_data = datasets['healthglow']
    # combined_data.columns = ['Region_Name', 'location_code', 'location_name', 'Latitude', 'Longitude']  # Original column names
    # combined_data['address'] = combined_data['location_name']  # Assuming location_name serves as address
    # combined_data['rating'] = None  # No rating available in this dataset
    # combined_data['city'] = combined_data['Region_Name']  # Using Region_Name to indicate the city

    # # Rename columns to match the Bangalore data for consistency
    # combined_data = combined_data[['location_name', 'Latitude', 'Longitude', 'address', 'rating', 'city']]
    # combined_data.columns = ['name', 'Latitude', 'Longitude', 'address', 'rating', 'city']

    # # Combine all data into a single DataFrame
    # standardized_data = pd.concat([bangalore_data, combined_data], ignore_index=True)
    # return standardized_data

def standardize_healthglow_data(datasets):
    """Standardize Health & Glow store data from different datasets"""
    # Process Bangalore data
    bangalore_data = datasets['healthglow'][datasets['healthglow']['Latitude'].notnull()].copy()

    # If Region_Name exists, it's Hyd/Chennai data
    hyd_che_data = datasets['healthglow'][datasets['healthglow'].get('Region_Name').notna()].copy() if 'Region_Name' in datasets['healthglow'].columns else pd.DataFrame()

    # --- Standardize Bangalore Data ---
    bangalore_data = bangalore_data[['place_id', 'name', 'Latitude', 'Longitude', 'address', 'rating']].rename(
        columns={'place_id': 'id'}
    )
    bangalore_data['city'] = 'Bangalore'

    # --- Standardize Hyderabad & Chennai Data (if any) ---
    if not hyd_che_data.empty:
        hyd_che_data = hyd_che_data.rename(columns=lambda x: x.strip())
        hyd_che_data = hyd_che_data[['Region_Name', 'location_code', 'location_name', 'Latitude', 'Longitude']]
        hyd_che_data['address'] = hyd_che_data['location_name']
        hyd_che_data['rating'] = None
        hyd_che_data['city'] = hyd_che_data['Region_Name']
        hyd_che_data = hyd_che_data[['location_name', 'Latitude', 'Longitude', 'address', 'rating', 'city']]
        hyd_che_data.columns = ['name', 'Latitude', 'Longitude', 'address', 'rating', 'city']

        # Combine both datasets
        standardized_data = pd.concat([bangalore_data, hyd_che_data], ignore_index=True)
    else:
        standardized_data = bangalore_data

    return standardized_data


# datasets = load_all_datasets()
# if datasets:
#     healthglow_data = standardize_healthglow_data(datasets)
#     housing_data = combine_housing_data(datasets)
#     st.session_state['all_housing_data'] = housing_data
#     st.session_state['all_healthglow_data'] = healthglow_data

    # You can now use healthglow_data for your analysis

    # st.write("Loaded datasets:", list(datasets.keys()))
    # st.write("Housing data sample:", datasets['housing'].head())
    # st.write("Hyderabad data sample:", datasets['housing_hyderabad'].head())
    # st.write("Chennai data sample:", datasets['housing_chennai'].head())
    # st.write("Combined Health & Glow sample:", datasets['healthglow_combined'].head())

# datasets['healthglow_hyd_che'] = datasets['healthglow']

# @st.cache_resource
# def load_prediction_assets():
#     return joblib.load("retail_store_prediction_assets.pkl")

# try:
#     assets = load_prediction_assets()
#     stacking_model = assets['model']
#     scaler = assets['scaler']
#     nn_model = assets['nn_model']
#     city_mapping = assets['city_mapping']
#     X_similarity_columns = assets['X_similarity_columns']
#     model_feature_columns = assets['model_feature_columns']
#     df_similarity = assets['df']
# except Exception as e:
#     assets = None
    # st.sidebar.error("⚠️ Could not load prediction model assets: " + str(e))

city_ranking = {
    'Bangalore': 9, 'Hyderabad': 8, 'Chennai': 7, 'Kolkata': 6,
    'Coimbatore': 5, 'Vizag': 4, 'Bhubaneshwar': 3, 'Trichy': 2, 'Patna': 1
}
visibility_ranking = {
    'Fully Visible': 6, 'Right and Front Visible': 5, 'Left and Front Visible': 5,
    'Front Visible': 4, 'Right Visible': 3, 'Left Visible': 3, 'Low Visible': 2
}
similarity_features = [
    "Store carpet area", "store trading area", "Visibility Score", "City Name",
    "Type of Store", "Compititors", "Supermarket", "Shoe store", "Jewelry store",
    "Drugstore", "Mobile Stores", "Bakery", "Fruit & Veg store", "Meat store",
    "College Near by", "Near by Mall if any", "Baby stores near by",
    "Store Density", "Accessibility Score"
]
cols_to_standardize = ['Store carpet area', 'store trading area', 'Compititors']

# def predict_sales_for_new_store(new_store_features, profit_threshold=500000):
#     # City mapping
#     new_store_features['City Name'] = city_mapping.get(
#         new_store_features.get('City Name'), new_store_features.get('City Name')
#     )
#     # Rankings
#     new_store_features['City Rank'] = city_ranking.get(
#         new_store_features.get('City Name'), 0
#     )
#     new_store_features['Visibility Score'] = visibility_ranking.get(
#         new_store_features.get('Visibility Score'), 1
#     )
#     # One-hot encode to match training
#     new_store_encoded = pd.get_dummies(pd.DataFrame([new_store_features])[similarity_features])
#     # Align columns
#     missing_cols = set(X_similarity_columns) - set(new_store_encoded.columns)
#     for col in missing_cols:
#         new_store_encoded[col] = 0
#     new_store_encoded = new_store_encoded.reindex(columns=X_similarity_columns, fill_value=0)
#     # Find most similar store
#     distances, indices = nn_model.kneighbors(new_store_encoded)
#     closest_store_index = indices[0][0]
#     closest_store = df_similarity.iloc[closest_store_index]
#     # Use proxy values if needed
#     proxy_values = {
#         'FY25NO_OF_DATE': closest_store.get('FY25NO_OF_DATE', 0),
#         'FY25SAL_QTY': closest_store.get('FY25SAL_QTY', 0)
#     }
#     new_store_features.update(proxy_values)
#     # Prepare for model: one-hot encode and align columns
#     new_store_input = pd.DataFrame([new_store_features])
#     new_store_input = pd.get_dummies(new_store_input)
#     missing_pred_cols = set(model_feature_columns) - set(new_store_input.columns)
#     for col in missing_pred_cols:
#         new_store_input[col] = 0
#     new_store_input = new_store_input[model_feature_columns]
#     new_store_input[cols_to_standardize] = scaler.transform(new_store_input[cols_to_standardize])
#     # Predict
#     predicted_sales = stacking_model.predict(new_store_input)[0]
#     is_profitable = predicted_sales > profit_threshold
#     return {
#         'predicted_sales': predicted_sales,
#         'matched_store_city': closest_store['City Name'],
#         'is_profitable': is_profitable
#     }

def create_rectangular_grid(center_lat, center_lon, width_km, height_km):
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))

    cell_size_lat = lat_per_km
    cell_size_lon = lon_per_km

    rows = int(round(height_km))
    cols = int(round(width_km))

    start_lat = center_lat - (rows / 2) * cell_size_lat
    start_lon = center_lon - (cols / 2) * cell_size_lon

    grid_cells = []
    cell_id = 0

    for r in range(rows):
        for c in range(cols):
            min_lat = start_lat + r * cell_size_lat
            min_lon = start_lon + c * cell_size_lon

            grid_cells.append({
                "id": cell_id,
                "center_lat": min_lat + cell_size_lat / 2,
                "center_lon": min_lon + cell_size_lon / 2,
                "min_lat": min_lat,
                "max_lat": min_lat + cell_size_lat,
                "min_lon": min_lon,
                "max_lon": min_lon + cell_size_lon,
            })

            cell_id += 1

    return grid_cells


def analyze_rectangular_boundary_complete(datasets, center_lat, center_lon, width_km, height_km):
    """Complete analysis with rectangular boundary"""
    
    try:
        # Convert km to degrees for filtering
        lat_per_km = 1.0 / 111.0
        lon_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))
        
        lat_offset = height_km * lat_per_km / 2
        lon_offset = width_km * lon_per_km / 2
        
        min_lat = center_lat - lat_offset
        max_lat = center_lat + lat_offset
        min_lon = center_lon - lon_offset
        max_lon = center_lon + lon_offset
        
        # Filter all datasets within boundary
        filtered_data = {}
        
        # Filter housing data
        if is_within_bounds(center_lat, center_lon, "Bangalore"):
            city = "Bangalore"
            housing_df = datasets['housing']
        elif is_within_bounds(center_lat, center_lon, "Hyderabad"):
            city = "Hyderabad"
            housing_df = datasets['housing_hyderabad']
        elif is_within_bounds(center_lat, center_lon, "Chennai"):
            city = "Chennai"
            housing_df = datasets['housing_chennai']
        else:
            city = "Unknown"
            st.warning("⚠️ Selected location is outside known city bounds.")
            housing_df = datasets['housing']  # default fallback
        filtered_data['housing'] = housing_df[
            (housing_df['Latitude'] >= min_lat) &
            (housing_df['Latitude'] <= max_lat) &
            (housing_df['Longitude'] >= min_lon) &
            (housing_df['Longitude'] <= max_lon)
        ].copy()

        # Filter housing data for Hyderabad
        # filtered_data['housing_hyderabad'] = datasets['housing_hyderabad'][
        #     (datasets['housing_hyderabad']['Latitude'] >= min_lat) &
        #     (datasets['housing_hyderabad']['Latitude'] <= max_lat) &
        #     (datasets['housing_hyderabad']['Longitude'] >= min_lon) &
        #     (datasets['housing_hyderabad']['Longitude'] <= max_lon)
        # ].copy()

        # Initialize comprehensive_metrics
        comprehensive_metrics = {}

        # Calculate average rent per sqft in the filtered area
        avg_rent_per_sqft = filtered_data['housing']['rent_per_sqft'].mean() if not filtered_data['housing'].empty else 0

        # Determine if the area is expensive
        expensive_threshold = 30  # Set a threshold for rent per sqft
        is_expensive = avg_rent_per_sqft > expensive_threshold

        # Add this information to the results
        comprehensive_metrics['avg_rent_per_sqft'] = avg_rent_per_sqft
        comprehensive_metrics['is_expensive'] = is_expensive
        
        # Filter population data
        # ---------------------- POPULATION SELECT -----------------------
        if city == "Bangalore":
            pop_df = datasets['population']
        elif city == "Hyderabad":
            pop_df = datasets['population_hyderabad']
        elif city == "Chennai":
            pop_df = datasets['population_chennai']
        else:
            pop_df = datasets['population']  # fallback

        filtered_data['population'] = pop_df[
        (pop_df['Latitude'] >= min_lat) &
        (pop_df['Latitude'] <= max_lat) &
        (pop_df['Longitude'] >= min_lon) &
        (pop_df['Longitude'] <= max_lon)
        ].copy()

        
        # Filter IT companies
        it_df = datasets['it_companies']
        filtered_data['it_companies'] = it_df[
            (it_df['Latitude'] >= min_lat) &
            (it_df['Latitude'] <= max_lat) &
            (it_df['Longitude'] >= min_lon) &
            (it_df['Longitude'] <= max_lon)
        ].copy()
        
        # Filter competitors
        comp_df = datasets['competitors']
        filtered_data['competitors'] = comp_df[
            (comp_df['Latitude'] >= min_lat) &
            (comp_df['Latitude'] <= max_lat) &
            (comp_df['Longitude'] >= min_lon) &
            (comp_df['Longitude'] <= max_lon)
        ].copy()
        
        # --- Health & Glow filtering fix ---
        if city == "Bangalore":
            hg_df = datasets['healthglow']
        elif city == "Hyderabad":
            hg_df = datasets['healthglow'][datasets['healthglow']['Region_Name'].str.contains("Hyderabad", case=False, na=False)]
        elif city == "Chennai":
            hg_df = datasets['healthglow'][datasets['healthglow']['Region_Name'].str.contains("Chennai", case=False, na=False)]
        else:
            hg_df = datasets['healthglow']  # default fallback

        filtered_data['healthglow'] = hg_df[
            (hg_df['Latitude'] >= min_lat) &
            (hg_df['Latitude'] <= max_lat) &
            (hg_df['Longitude'] >= min_lon) &
            (hg_df['Longitude'] <= max_lon)
]

        
        # Create rectangular grid
        grid_cells = create_rectangular_grid(center_lat, center_lon, width_km, height_km)
        
        # Analyze each grid cell
        grid_with_data = []
        for cell in grid_cells:
            # Count data points in this cell
            cell_housing = filtered_data['housing'][
                (filtered_data['housing']['Latitude'] >= cell['min_lat']) &
                (filtered_data['housing']['Latitude'] <= cell['max_lat']) &
                (filtered_data['housing']['Longitude'] >= cell['min_lon']) &
                (filtered_data['housing']['Longitude'] <= cell['max_lon'])
            ]
            
            cell_population = filtered_data['population'][
                (filtered_data['population']['Latitude'] >= cell['min_lat']) &
                (filtered_data['population']['Latitude'] <= cell['max_lat']) &
                (filtered_data['population']['Longitude'] >= cell['min_lon']) &
                (filtered_data['population']['Longitude'] <= cell['max_lon'])
            ]
            
            cell_it = filtered_data['it_companies'][
                (filtered_data['it_companies']['Latitude'] >= cell['min_lat']) &
                (filtered_data['it_companies']['Latitude'] <= cell['max_lat']) &
                (filtered_data['it_companies']['Longitude'] >= cell['min_lon']) &
                (filtered_data['it_companies']['Longitude'] <= cell['max_lon'])
            ]
            
            cell_competitors = filtered_data['competitors'][
                (filtered_data['competitors']['Latitude'] >= cell['min_lat']) &
                (filtered_data['competitors']['Latitude'] <= cell['max_lat']) &
                (filtered_data['competitors']['Longitude'] >= cell['min_lon']) &
                (filtered_data['competitors']['Longitude'] <= cell['max_lon'])
            ]
            
            cell_hg = filtered_data['healthglow'][
                (filtered_data['healthglow']['Latitude'] >= cell['min_lat']) &
                (filtered_data['healthglow']['Latitude'] <= cell['max_lat']) &
                (filtered_data['healthglow']['Longitude'] >= cell['min_lon']) &
                (filtered_data['healthglow']['Longitude'] <= cell['max_lon'])
            ]
            
            # Calculate metrics for this cell
            avg_rent = cell_housing['total_rent'].mean() if len(cell_housing) > 0 else 0
            if pd.isna(avg_rent):
                avg_rent = 0
            
            pop_density = cell_population['population_density'].sum() if len(cell_population) > 0 else 0
            
            # Calculate investment attractiveness score
            housing_density = len(cell_housing)
            it_density = len(cell_it)
            competitor_density = len(cell_competitors)
            hg_density = len(cell_hg)
            
            # Determine if it's a green zone
            is_green_zone = False
            
            # Check for performance criteria
            performance_threshold = 500000  # Example performance threshold for sales
            avg_sales = cell_housing['total_rent'].mean() if len(cell_housing) > 0 else 0
            
            if hg_density > 0 and competitor_density > 0:
                is_green_zone = True  # Green zone if there are competitors and Health & Glow stores
            elif hg_density > 0 and (housing_density >= 10 or it_density >= 5) and avg_sales > performance_threshold:
                is_green_zone = True  # Green zone based on housing or IT presence and performance
            
            # Market opportunity score (0-100)
            customer_base_score = min(50, housing_density * 2 + it_density * 5)
            affordability_score = max(0, 25 - (avg_rent - 15000) / 1000) if avg_rent > 0 else 15
            competition_score = max(0, 25 - competitor_density * 5 - hg_density * 3)
            
            market_opportunity_score = customer_base_score + affordability_score + competition_score
            
            # Investment attractiveness (can be negative for high competition areas)
            investment_attractiveness = (
                market_opportunity_score + 
                housing_density * 2 + 
                it_density * 3 - 
                competitor_density * 10 - 
                hg_density * 8
            )
            
            grid_with_data.append({
                'cell_id': cell['id'],
                'center_lat': cell['center_lat'],
                'center_lon': cell['center_lon'],
                'housing_count': housing_density,
                'population_points': len(cell_population),
                'it_companies_count': it_density,
                'competitors_count': competitor_density,
                'hg_stores_count': hg_density,
                'avg_rent': avg_rent,
                'population_density': pop_density,
                'housing_density': housing_density,
                'it_density': it_density,
                'competitor_density': competitor_density,
                'hg_density': hg_density,
                'market_opportunity_score': market_opportunity_score,
                'investment_attractiveness': investment_attractiveness,
                'is_green_zone': is_green_zone  # Track green zone status
            })
        
        # Convert to DataFrame
        complete_grid = pd.DataFrame(grid_with_data)
        
        # Calculate comprehensive metrics
        total_area = width_km * height_km
        
        comprehensive_metrics = {
            'total_area_km2': total_area,
            'total_grid_cells': len(grid_cells),
            'total_properties': len(filtered_data['housing']),
            'total_population_points': len(filtered_data['population']),
            'total_it_companies': len(filtered_data['it_companies']),
            'total_competitors': len(filtered_data['competitors']),
            'total_hg_stores': len(filtered_data['healthglow']),
            'avg_rent': filtered_data['housing']['total_rent'].mean() if len(filtered_data['housing']) > 0 else 0,
            'max_housing_density': complete_grid['housing_density'].max(),
            'max_it_density': complete_grid['it_density'].max(),
            'max_competitor_density': complete_grid['competitor_density'].max(),
            'avg_population_density': complete_grid['population_density'].mean(),
            'high_opportunity_cells': len(complete_grid[complete_grid['investment_attractiveness'] >= 100]),
            'medium_opportunity_cells': len(complete_grid[
                (complete_grid['investment_attractiveness'] >= 20) & 
                (complete_grid['investment_attractiveness'] < 50)
            ])
        }
        
        return {
            'filtered_data': filtered_data,
            'complete_grid': complete_grid,
            'comprehensive_metrics': comprehensive_metrics,
            'boundary_info': {
                'center_lat': center_lat,
                'center_lon': center_lon,
                'width_km': width_km,
                'height_km': height_km,
                'area_km2': total_area
            }
        }
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        print(f"Analysis error: {str(e)}")
        return None
    

def create_selection_map(center_lat, center_lon, width_km, height_km, city_selection=None):
    """Create map for location selection — rectangle sized by width_km × height_km (not full city)."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)

    # Convert km to degrees (preserve correct sizing independent of city bounds)
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))

    lat_offset = height_km * lat_per_km / 2
    lon_offset = width_km * lon_per_km / 2

    boundary_coords = [
        [center_lat - lat_offset, center_lon - lon_offset],
        [center_lat - lat_offset, center_lon + lon_offset],
        [center_lat + lat_offset, center_lon + lon_offset],
        [center_lat + lat_offset, center_lon - lon_offset],
        [center_lat - lat_offset, center_lon - lon_offset]
    ]

    folium.Polygon(
        locations=boundary_coords,
        color='#667eea',
        weight=3,
        fill=True,
        fillColor='#667eea',
        fillOpacity=0.2,
        popup=f'Analysis Area ({width_km}km × {height_km}km)'
    ).add_to(m)

    folium.Marker(
        location=[center_lat, center_lon],
        popup=f'Analysis Center<br>Lat: {center_lat:.6f}<br>Lon: {center_lon:.6f}',
        icon=folium.Icon(color='red', icon='crosshairs', prefix='fa')
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

def create_analysis_map(analysis_results, datasets):
    """Create map with analysis results"""
    
    boundary_info = analysis_results['boundary_info']
    complete_grid = analysis_results['complete_grid']
    
    center_lat = boundary_info['center_lat']
    center_lon = boundary_info['center_lon']
    width_km = boundary_info['width_km']
    height_km = boundary_info['height_km']
    
    # Base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )

        # ===========================
    # HYPERLOCAL PIN CODE LAYER
    # ===========================
    # if "pin_mapping" in datasets:
    #     df_pins = datasets["pin_mapping"]
    #     hyperlocal_layer = folium.FeatureGroup(name="Store & Serviceable Pin Codes")

    #     # Group by each store location code
    #     grouped = df_pins.groupby("StoreLocationPinCode")

    #     for store_pin, group in grouped:
    #         store_lat = group["Store_Lat"].iloc[0]
    #         store_lon = group["Store_Lng"].iloc[0]

    #         if pd.isna(store_lat) or pd.isna(store_lon):
    #             continue

    #         # 1️⃣ BLUE marker for store
    #         folium.CircleMarker(
    #             location=[store_lat, store_lon],
    #             radius=7,
    #             color="blue",
    #             fill=True,
    #             fill_color="blue",
    #             popup=f"Store {store_pin}"
    #         ).add_to(hyperlocal_layer)

    #         polygon_points = []

    #         # Loop through serviceable pins belonging only to THIS store
    #         for _, row in group.iterrows():
    #             serv_lat = row["Serv_Lat"]
    #             serv_lon = row["Serv_Lng"]
    #             service_pin = row["ServiceablePinCode"]

    #             if pd.isna(serv_lat) or pd.isna(serv_lon):
    #                 continue

    #             # 2️⃣ ORANGE marker — serviceable
    #             folium.CircleMarker(
    #                 location=[serv_lat, serv_lon],
    #                 radius=4,
    #                 color="orange",
    #                 fill=True,
    #                 fill_color="orange",
    #                 popup=f"Serviceable Pin: {service_pin}"
    #             ).add_to(hyperlocal_layer)

    #             # 3️⃣ Line connecting this store → this service pin
    #             folium.PolyLine(
    #                 locations=[[store_lat, store_lon], [serv_lat, serv_lon]],
    #                 color="black",
    #                 weight=2,
    #                 dash_array="5,5",
    #             ).add_to(hyperlocal_layer)

    #             polygon_points.append([serv_lat, serv_lon])

    #         # 4️⃣ Polygon ONLY around this store's serviceable pins
    #         if len(polygon_points) >= 3:
    #             folium.Polygon(
    #                 locations=polygon_points,
    #                 color="orange",
    #                 fill=True,
    #                 fill_opacity=0.15,
    #                 weight=2,
    #                 dash_array="5,5"
    #             ).add_to(hyperlocal_layer)

    #     hyperlocal_layer.add_to(m)

    # ===========================
    # END HYPERLOCAL LAYER
    # ===========================
    
    # Extra tile layers
    folium.TileLayer('CartoDB positron', name='Light Mode').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
    
    # Boundary rectangle
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * np.cos(np.radians(center_lat)))
    
    lat_offset = height_km * lat_per_km / 2
    lon_offset = width_km * lon_per_km / 2
    
    boundary_coords = [
        [center_lat - lat_offset, center_lon - lon_offset],
        [center_lat - lat_offset, center_lon + lon_offset],
        [center_lat + lat_offset, center_lon + lon_offset],
        [center_lat + lat_offset, center_lon - lon_offset],
        [center_lat - lat_offset, center_lon - lon_offset]
    ]
    
    folium.Polygon(
        locations=boundary_coords,
        color='#667eea',
        weight=3,
        fill=True,
        fillColor='#667eea',
        fillOpacity=0.1,
        popup=f'Analysis Boundary ({width_km}km × {height_km}km)'
    ).add_to(m)
    
    # GRID CELLS — investment heatmap
    if len(complete_grid) > 0:
        max_score = complete_grid['investment_attractiveness'].max()
        min_score = complete_grid['investment_attractiveness'].min()
        
        for _, cell in complete_grid.iterrows():
            cell_lat = cell['center_lat']
            cell_lon = cell['center_lon']
            
            cell_size_lat = lat_per_km
            cell_size_lon = lon_per_km
            
            cell_coords = [
                [cell_lat - cell_size_lat/2, cell_lon - cell_size_lon/2],
                [cell_lat - cell_size_lat/2, cell_lon + cell_size_lon/2],
                [cell_lat + cell_size_lat/2, cell_lon + cell_size_lon/2],
                [cell_lat + cell_size_lat/2, cell_lon - cell_size_lon/2],
                [cell_lat - cell_size_lat/2, cell_lon - cell_size_lon/2]
            ]
            
            score = cell['investment_attractiveness']
            if max_score > min_score:
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                normalized_score = 0.5
            
            if cell['is_green_zone']:
                color = 'darkgreen'
                fill_color = 'green'
                opacity = 0.8
            else:
                if normalized_score >= 0.7:
                    color = 'darkgreen'
                    fill_color = 'green'
                    opacity = 0.8
                elif normalized_score >= 0.4:
                    color = 'orange'
                    fill_color = 'yellow'
                    opacity = 0.6
                else:
                    color = 'darkred'
                    fill_color = 'red'
                    opacity = 0.4
            
            folium.Polygon(
                locations=cell_coords,
                color=color,
                weight=1,
                fill=True,
                fillColor=fill_color,
                fillOpacity=opacity,
                popup=f'''
                <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0; color: #333;">Grid Cell {cell["cell_id"]}</h4>
                <hr style="margin: 5px 0;">
                <b>Investment Score:</b> {score:.1f}<br>
                <b>Market Score:</b> {cell["market_opportunity_score"]:.1f}<br>
                <hr style="margin: 5px 0;">
                <b>Housing:</b> {cell["housing_count"]} properties<br>
                <b>IT Companies:</b> {cell["it_companies_count"]}<br>
                <b>Competitors:</b> {cell["competitors_count"]}<br>
                <b>H&G Stores:</b> {cell["hg_stores_count"]}<br>
                <b>Avg Rent:</b> ₹{cell["avg_rent"]:,.0f}
                </div>
                '''
            ).add_to(m)
    
    # Center marker
    folium.Marker(
        location=[center_lat, center_lon],
        popup='Analysis Center',
        icon=folium.Icon(color='red', icon='crosshairs', prefix='fa')
    ).add_to(m)
    
    # Single LayerControl at the end
    folium.LayerControl().add_to(m)
    
    return m


def display_analysis_results(analysis_results, datasets):

    """Display comprehensive analysis results with professional UI"""
    
    complete_grid = analysis_results['complete_grid']
    metrics = analysis_results['comprehensive_metrics']
    boundary_info = analysis_results['boundary_info']

    # Display rent information
    if 'avg_rent_per_sqft' in metrics:
        st.write(f"**Average Rent per Sqft:** ₹{metrics['avg_rent_per_sqft']:.2f}")
    else:
        st.warning("⚠️ Average Rent per Sqft data is not available.")
    
    if metrics.get('is_expensive', False):
        st.success("💰 This area is considered EXPENSIVE!")
    else:
        st.warning("🏡 This area is considered AFFORDABLE.")
    
    # Executive Summary
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['total_area_km2']}</h3>
            <p>Analysis Area (km²)</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['total_grid_cells']}</h3>
            <p>Grid Cells</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['total_properties']:,}</h3>
            <p>Total Properties</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <h3>₹{metrics['avg_rent']:,.0f}</h3>
            <p>Average Rent</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['total_it_companies']:,}</h3>
            <p>IT Companies</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['total_competitors']:,}</h3>
            <p>Competitors</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['total_hg_stores']:,}</h3>
            <p>H&G Stores</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="metric-card">
            <h3>{metrics['high_opportunity_cells']}</h3>
            <p>High Opportunity Cells</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Investment Map", "📊 Grid Analysis", "🎯 Recommendations", "📈 Market Insights"])
    
    with tab1:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Investment Opportunity Heatmap")
        
        # Create and display analysis map
        analysis_map = create_analysis_map(analysis_results, datasets)


        
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(analysis_map, width=700, height=600, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Map legend
        st.markdown('''
        <div class="status-card status-info">
        <h4>🗺️ Investment Opportunity Legend</h4>
        <ul>
        <li>🟢 <strong>Green Cells</strong>: High investment attractiveness (Score ≥ 70%) or areas with H&G stores and competitors</li>
        <li>🟡 <strong>Yellow Cells</strong>: Medium investment attractiveness (Score 40-70%)</li>
        <li>🔴 <strong>Red Cells</strong>: Low investment attractiveness (Score < 40%)</li>
        <li>🎯 <strong>Red Marker</strong>: Analysis center point</li>
        <li>📐 <strong>Blue Rectangle</strong>: Analysis boundary</li>
        </ul>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Detailed Grid Analysis")
        
        # Grid controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sort_metric = st.selectbox(
                "📈 Sort by:",
                ['investment_attractiveness', 'market_opportunity_score', 
                 'housing_density', 'it_density', 'competitor_density', 'population_density'],
                help="Choose metric to sort grid cells"
            )
        
        with col2:
            ascending_order = st.checkbox("📈 Ascending order", value=False)
        
        with col3:
            show_top_n = st.slider("📋 Show top N cells", min_value=5, max_value=min(50, len(complete_grid)), value=10)
        
        # Display sorted grid data
        if len(complete_grid) > 0:
            sorted_grid = complete_grid.sort_values(by=sort_metric, ascending=ascending_order).head(show_top_n)
            
            st.dataframe(
                sorted_grid[[
                    'cell_id', 'center_lat', 'center_lon', 'investment_attractiveness',
                    'market_opportunity_score', 'housing_density', 'it_density', 
                    'competitor_density', 'population_density', 'avg_rent', 'is_green_zone'
                ]].round(2),
                use_container_width=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Top Investment Recommendations")
        
        if len(complete_grid) > 0:
            # Prioritize low competition areas
            zero_competitor_cells = complete_grid[complete_grid['competitor_density'] == 0]
            
            if len(zero_competitor_cells) >= 3:
                top_3_locations = zero_competitor_cells.nlargest(3, 'investment_attractiveness')
                st.markdown('<div class="status-card status-success"><h4>🎯 Found locations with ZERO competition!</h4></div>', unsafe_allow_html=True)
            else:
                low_competition_cells = complete_grid[complete_grid['competitor_density'] <= 2]
                if len(low_competition_cells) >= 3:
                    top_3_locations = low_competition_cells.nlargest(3, 'investment_attractiveness')
                    st.markdown('<div class="status-card status-info"><h4>✅ Found locations with LOW competition (≤2 competitors)</h4></div>', unsafe_allow_html=True)
                else:
                    top_3_locations = complete_grid.nlargest(3, 'investment_attractiveness')
                    st.markdown('<div class="status-card status-warning"><h4>⚠️ Limited low-competition options available</h4></div>', unsafe_allow_html=True)
            
            for idx, (_, location) in enumerate(top_3_locations.iterrows(), 1):
                with st.expander(f"🏅 Recommendation #{idx} - Cell {location['cell_id']} | Score: {location['investment_attractiveness']:.1f}"):
                    
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        st.markdown("**📍 Location Details:**")
                        st.write(f"• **Coordinates:** {location['center_lat']:.6f}, {location['center_lon']:.6f}")
                        st.write(f"• **Investment Score:** {location['investment_attractiveness']:.1f}")
                        st.write(f"• **Market Opportunity:** {location['market_opportunity_score']:.1f}")
                        
                        st.markdown("**🏠 Customer Base:**")
                        st.write(f"• **Housing Density:** {location['housing_density']} properties/km²")
                        st.write(f"• **IT Companies:** {location['it_companies_count']} companies")
                        st.write(f"• **Population Density:** {location['population_density']:.0f}")
                    
                    with rec_col2:
                        st.markdown("**⚔️ Competition Analysis:**")
                        comp_status = '✅' if location['competitors_count'] == 0 else '⚠️' if location['competitors_count'] <= 2 else '❌'
                        hg_status = '✅' if location['hg_stores_count'] == 0 else '⚠️' if location['hg_stores_count'] <= 1 else '❌'
                        
                        st.write(f"• **Competitors:** {location['competitors_count']} in cell {comp_status}")
                        st.write(f"• **H&G Stores:** {location['hg_stores_count']} in cell {hg_status}")
                        st.write(f"• **Average Rent:** ₹{location['avg_rent']:,.0f}")
                        
                        # Investment rationale
                        st.markdown("**💡 Why this location:**")
                        rationale = []
                        if location['competitor_density'] == 0:
                            rationale.append("✅ **ZERO COMPETITORS** - Market gap opportunity!")
                        elif location['competitor_density'] <= 2:
                            rationale.append(f"✅ Low competition ({location['competitor_density']} competitors)")
                        
                        if location['hg_density'] == 0:
                            rationale.append("✅ **NO H&G STORES** - First mover advantage!")
                        elif location['hg_density'] <= 1:
                            rationale.append("✅ Minimal H&G presence")
                        
                        if location['housing_density'] >= 10:
                            rationale.append("✅ High housing density")
                        if location['it_density'] >= 3:
                            rationale.append("✅ Strong IT presence")
                        if location['population_density'] >= 1000:
                            rationale.append("✅ High population density")
                        
                        for reason in rationale:
                            st.write(f"• {reason}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Market Insights & Strategic Analysis")
        
        # Overall assessment
        total_cells = metrics['total_grid_cells']
        opportunity_ratio = metrics['high_opportunity_cells'] / total_cells if total_cells > 0 else 0
        
        if opportunity_ratio >= 0.3:
            assessment = "🟢 **EXCELLENT** - High opportunity area with multiple viable locations"
            assessment_class = "status-success"
        elif opportunity_ratio >= 0.15:
            assessment = "🟡 **GOOD** - Moderate opportunities available"
            assessment_class = "status-warning"
        else:
            assessment = "🔴 **CHALLENGING** - Limited high-opportunity locations"
            assessment_class = "status-error"
        
        st.markdown(f'<div class="status-card {assessment_class}"><h4>🎯 Overall Investment Assessment</h4><p>{assessment}</p></div>', unsafe_allow_html=True)
        
        # Competition analysis
        zero_comp_cells = len(complete_grid[complete_grid['competitor_density'] == 0])
        low_comp_cells = len(complete_grid[complete_grid['competitor_density'] <= 2])
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("#### ⚔️ Competition Landscape")
            st.write(f"• **Zero Competition Cells:** {zero_comp_cells} ({zero_comp_cells/total_cells*100:.1f}%)")
            st.write(f"• **Low Competition Cells:** {low_comp_cells} ({low_comp_cells/total_cells*100:.1f}%)")
            st.write(f"• **Total Competitors in Area:** {metrics['total_competitors']}")
            st.write(f"• **Total H&G Stores:** {metrics['total_hg_stores']}")
        
        with comp_col2:
            st.markdown("#### 💰 Market Potential")
            avg_rent = metrics['avg_rent']
            if avg_rent > 0:
                if avg_rent < 15000:
                    rent_assessment = "✅ Affordable market segment"
                elif avg_rent < 25000:
                    rent_assessment = "⚠️ Mid-range market segment"
                else:
                    rent_assessment = "❌ Premium market segment"
                
                st.write(f"• **Average Rent:** ₹{avg_rent:,.0f}")
                st.write(f"• **Market Segment:** {rent_assessment}")
            
            st.write(f"• **Total Properties:** {metrics['total_properties']:,}")
            st.write(f"• **IT Companies:** {metrics['total_it_companies']:,}")
        
        # Strategic recommendations
        st.markdown("#### 🎯 Strategic Recommendations")
        
        recommendations = []
        
        if zero_comp_cells >= 3:
            recommendations.append("🎯 **PRIORITY**: Focus on zero-competition cells for maximum market advantage")
        elif low_comp_cells >= 5:
            recommendations.append("✅ **OPPORTUNITY**: Target low-competition areas for easier market entry")
        else:
            recommendations.append("⚠️ **STRATEGY**: Consider differentiation strategies due to high competition")
        
        if metrics['total_it_companies'] >= 20:
            recommendations.append("💼 **TARGET**: High IT company density - ideal for professional customer base")
        elif metrics['total_it_companies'] >= 10:
            recommendations.append("💼 **MODERATE**: Decent IT presence - good customer potential")
        
        if metrics['avg_rent'] < 20000:
            recommendations.append("💰 **PRICING**: Affordable market - focus on value-oriented products")
        else:
            recommendations.append("💰 **PRICING**: Premium market - opportunity for high-end products")
        
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Data visualization
        st.markdown("#### 📊 Investment Score Distribution")
        
        if len(complete_grid) > 0:
            fig = px.histogram(
                complete_grid,
                x='investment_attractiveness',
                nbins=20,
                title="Distribution of Investment Attractiveness Scores",
                labels={'investment_attractiveness': 'Investment Score', 'count': 'Number of Grid Cells'},
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.markdown("#### 📥 Export Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Summary Report", use_container_width=True):
                export_summary_report(analysis_results)
        
        with col2:
            if st.button("📋 Export Detailed Grid Data", use_container_width=True):
                export_detailed_grid_data(analysis_results)
        
        st.markdown('</div>', unsafe_allow_html=True)

def export_summary_report(analysis_results):
    """Export summary report to CSV"""
    try:
        metrics = analysis_results['comprehensive_metrics']
        boundary_info = analysis_results['boundary_info']
        
        summary_data = {
            'Metric': [
                'Analysis Center Latitude',
                'Analysis Center Longitude',
                'Analysis Width (km)',
                'Analysis Height (km)',
                'Analysis Area (km²)',
                'Grid Cells',
                'Total Properties',
                'Total IT Companies',
                'Total Competitors',
                'Total H&G Stores',
                'Average Rent (₹)',
                'High Opportunity Cells',
                'Medium Opportunity Cells',
                'Max Housing Density',
                'Max IT Density',
                'Max Competitor Density'
            ],
            'Value': [
                boundary_info['center_lat'],
                boundary_info['center_lon'],
                boundary_info['width_km'],
                boundary_info['height_km'],
                metrics['total_area_km2'],
                metrics['total_grid_cells'],
                metrics['total_properties'],
                metrics['total_it_companies'],
                metrics['total_competitors'],
                metrics['total_hg_stores'],
                metrics['avg_rent'],
                metrics['high_opportunity_cells'],
                metrics['medium_opportunity_cells'],
                metrics['max_housing_density'],
                metrics['max_it_density'],
                metrics['max_competitor_density']
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Summary Report CSV",
            data=csv,
            file_name=f"hg_analysis_summary_{boundary_info['center_lat']:.4f}_{boundary_info['center_lon']:.4f}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.success("✅ Summary report ready for download!")
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def export_detailed_grid_data(analysis_results):
    """Export detailed grid analysis to CSV"""
    try:
        complete_grid = analysis_results['complete_grid']
        boundary_info = analysis_results['boundary_info']
        
        csv = complete_grid.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Detailed Grid Data CSV",
            data=csv,
            file_name=f"hg_grid_analysis_{boundary_info['center_lat']:.4f}_{boundary_info['center_lon']:.4f}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.success("✅ Detailed grid data ready for download!")
        
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def main():
    """Main application function with working map click functionality"""

    # ✅ GUARANTEED session init (Railway-safe)
    defaults = {
        "selected_lat": 12.9716,
        "selected_lon": 77.5946,
        "boundary_width_km": 6,
        "boundary_height_km": 6,
        "analysis_results": None,
        "datasets_loaded": False,
        "map_key": 0,
        "city_selection": "Bangalore",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 🚨 STOP app until session state is ready (Railway-safe)
    if st.session_state.selected_lat is None or st.session_state.selected_lon is None:
        st.info("📍 Please select a location to begin")
        return


    
    # Header
    st.markdown('''
    <div class="main-header">
        <h1> Health & Glow Store Location Analysis</h1>
        <p>Interactive Map Selection with Advanced Grid Analysis</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load datasets
    # if "datasets" not in st.session_state:
    #     with st.spinner("🔄 Loading datasets (first time)..."):
    #         st.session_state.datasets = load_all_datasets()

    # datasets = st.session_state.datasets

    
    # if not datasets:
    #     st.error("❌ Failed to load required datasets. Please check file availability.")
    #     return
    
    st.session_state.datasets_loaded = True
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("# 🎯 Analysis Controls")
        
        # Step 1: Select City
        city_selection = st.selectbox(
            "Select City",
            ["Bangalore", "Hyderabad", "Chennai"],
            index=0
        )

        # Save city selection in session
        st.session_state["city_selection"] = city_selection

        # Step 2: Set default coordinates per city if none selected yet
        default_coords = {
            "Bangalore": (12.9716, 77.5946),
            "Hyderabad": (17.3850, 78.4867),
            "Chennai": (13.0827, 80.2707),
        }

        if "selected_lat" not in st.session_state:
            st.session_state.selected_lat, st.session_state.selected_lon = default_coords[city_selection]

        # Step 3: Use latest session coordinates
        center_lat = st.session_state.selected_lat
        center_lon = st.session_state.selected_lon

        # # Step 4: Create selection map centered on latest point
        # selection_map = create_selection_map(
        #     center_lat,
        #     center_lon,
        #     st.session_state.boundary_width_km,
        #     st.session_state.boundary_height_km,
        #     city_selection
        # )

        # # Step 5: Display map
        # st.markdown('<div class="map-container">', unsafe_allow_html=True)
        # map_data = st_folium(
        #     selection_map,
        #     width=700,
        #     height=500,
        #     returned_objects=["last_clicked"],
        #     key=f"location_map_{st.session_state.map_key}"
        # )
        # st.markdown('</div>', unsafe_allow_html=True)

        # # Step 6: When user clicks anywhere on map
        # if map_data and map_data.get("last_clicked"):
        #     clicked_lat = map_data["last_clicked"]["lat"]
        #     clicked_lon = map_data["last_clicked"]["lng"]

        #     # Always update coordinates to clicked point
        #     st.session_state.selected_lat = clicked_lat
        #     st.session_state.selected_lon = clicked_lon
        #     st.session_state.analysis_results = None
        #     st.session_state.map_key += 1

        #     st.success(f"✅ Selected new location: {clicked_lat:.6f}, {clicked_lon:.6f}")
        #     st.rerun()
        
        # Quick location selector (city-aware) with persistent state
        st.markdown("## 🏙️ Popular Areas")

        # Define popular areas per city (example lists; add more for other cities)
        popular_by_city = {
            "Bangalore": {
                "Koramangala": (12.9279, 77.6271),
                "Indiranagar": (12.9784, 77.6408),
                "Whitefield": (12.9698, 77.7500),
                "Electronic City": (12.8456, 77.6603),
                "Hebbal": (13.0358, 77.5970),
                "Jayanagar": (12.9237, 77.5838),
                "BTM Layout": (12.9165, 77.6101),
                "Marathahalli": (12.9591, 77.6974),
                "Banashankari": (12.9081, 77.5571),
                "HSR Layout": (12.9116, 77.6370)
            },
            "Hyderabad": {
                "Hitech City": (17.4474, 78.3910),
                "Gachibowli": (17.4323, 78.4009),
                "Banjara Hills": (17.4176, 78.4420),
                "Madhapur": (17.4360, 78.3987)
            },
            "Chennai": {
                "T. Nagar": (13.0370, 80.2333),
                "Adyar": (12.9850, 80.2560),
                "OMR": (12.9216, 80.2251),
                "Anna Nagar": (13.0739, 80.2137)
            }
        }

        # ensure the selectbox state is preserved across reruns by providing a key
        selected_area = st.selectbox(
            "Choose area:",
            ["Select area..."] + list(popular_by_city.get(city_selection, {}).keys()),
            key=f"popular_area_{city_selection}"
        )

        # single action button to move
        if selected_area != "Select area...":
            if st.button("🚀 Go To Selected Area", use_container_width=True):
                lat, lon = popular_by_city[city_selection][selected_area]
                st.session_state.selected_lat = lat
                st.session_state.selected_lon = lon
                st.session_state.analysis_results = None
                st.session_state.map_key += 1
                st.success(f"✅ Moved to {selected_area}")
                st.rerun()

        # Boundary size controls
        st.markdown("## 📏 Analysis Area")
        
        col1, col2 = st.columns(2)
        with col1:
            width_km = st.slider(
                "Width (km):",
                min_value=2,
                max_value=12,
                value=st.session_state.boundary_width_km,
                step=1
            )
        
        with col2:
            height_km = st.slider(
                "Height (km):",
                min_value=2,
                max_value=12,
                value=st.session_state.boundary_height_km,
                step=1
            )
        
        st.session_state.boundary_width_km = width_km
        st.session_state.boundary_height_km = height_km
        
        # Current selection display
        st.markdown("## 📊 Current Selection")
        st.markdown(f'''
        <div class="status-card status-info">
        <strong>Center:</strong> {st.session_state.selected_lat:.6f}, {st.session_state.selected_lon:.6f}<br>
        <strong>Area:</strong> {width_km}km × {height_km}km<br>
        <strong>Total:</strong> {width_km * height_km} km²
        </div>
        ''', unsafe_allow_html=True)

    # Create and display map for location selection
    col1, col2 = st.columns([4, 1])

    with col1:
        # 🔧 FIX 1: Always center the map on the latest selected coordinates
        center_lat = st.session_state.get("selected_lat", 12.9716)
        center_lon = st.session_state.get("selected_lon", 77.5946)

        # Create map for location selection
        selection_map = create_selection_map(
            center_lat,
            center_lon,
            st.session_state.boundary_width_km,
            st.session_state.boundary_height_km,
            st.session_state.get("city_selection", "Bangalore")  # passing city_selection
        )

        st.markdown('<div class="map-container">', unsafe_allow_html=True)

        # Display the map and capture clicks
        map_key_str = f"location_map_{st.session_state.get('city_selection', 'Bangalore')}_{st.session_state.get('map_key', 0)}"
        map_data = st_folium(
            selection_map,
            width=700,
            height=500,
            returned_objects=["last_clicked"],
            key=map_key_str
        )

        # Handle map clicks
        if map_data and map_data.get('last_clicked') is not None:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']

            # Check if coordinates are within the selected city's bounds
            if is_within_bounds(clicked_lat, clicked_lng, st.session_state.get("city_selection", "Bangalore")):

                # Update only if coordinates differ significantly
                if (abs(clicked_lat - st.session_state.selected_lat) > 0.0001 or
                    abs(clicked_lng - st.session_state.selected_lon) > 0.0001):
                    st.session_state.selected_lat = clicked_lat
                    st.session_state.selected_lon = clicked_lng
                    st.session_state.analysis_results = None
                    st.session_state.map_key += 1

                    st.success(f"✅ New location selected: {clicked_lat:.6f}, {clicked_lng:.6f}")
                    st.rerun()
            else:
                st.warning("⚠️ Clicked location is outside the valid analysis area.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 Actions")

        st.markdown("#### 📍 Manual Coordinates")
        c1, c2 = st.columns(2)
    with c1:
        manual_lat = st.number_input("Lat", value=st.session_state.get("selected_lat", 12.9716),
                                     format="%.6f", label_visibility="collapsed")
    with c2:
        manual_lon = st.number_input("Lon", value=st.session_state.get("selected_lon", 77.5946),
                                     format="%.6f", label_visibility="collapsed")

    # Compact 'Set' button without rerun
        if st.button("🔘 Set", use_container_width=True):
                st.session_state.selected_lat = manual_lat
                st.session_state.selected_lon = manual_lon
                st.session_state.analysis_results = None
                st.session_state.map_key += 1
                st.session_state.manual_set = True
                st.success(f"Set to: {manual_lat:.4f}, {manual_lon:.4f}")

                st.divider()

    # Always visible buttons
                # st.button("🚀 START ANALYSIS", use_container_width=True)
                # st.button("🔄 Reset", use_container_width=True)
                # st.button("📍 Go to Selected Area", use_container_width=True)


        

        # 🚀 Start analysis button
        if st.button("🚀 START ANALYSIS", type="primary", use_container_width=True):
            # ✅ LAZY LOAD DATASETS HERE (only when button is clicked)
            if "datasets" not in st.session_state or st.session_state.datasets is None:
                with st.spinner("🔄 Loading datasets (first time)..."):
                    st.session_state.datasets = load_all_datasets()
                    
            if st.session_state.datasets is None:
                st.error("❌ Datasets failed to load. Check data files.")
                st.stop()

            datasets = st.session_state.datasets
        
            with st.spinner("🔄 Analyzing selected area..."):
                analysis_results = analyze_rectangular_boundary_complete(
                    datasets,
                    st.session_state.selected_lat,
                    st.session_state.selected_lon,
                    st.session_state.boundary_width_km,
                    st.session_state.boundary_height_km
                )

                if analysis_results:
                    st.session_state.analysis_results = analysis_results
                    st.success("✅ Analysis completed!")
                    st.rerun()
                else:
                    st.error("❌ Analysis failed. Please try a different area.")

        # 🔁 Reset button
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.selected_lat = 12.9716
            st.session_state.selected_lon = 77.5946
            st.session_state.analysis_results = None
            st.session_state.map_key += 1
            st.success("✅ Reset to default location")
            st.rerun()

        # 📍 FIX 2: Add 'Go to Selected Area' button
        if st.button("📍 Go to Selected Area", use_container_width=True):
            st.session_state.map_key += 1
            st.rerun()

        # Analysis status
        if st.session_state.analysis_results:
            st.markdown('<div class="status-card status-success"><strong>✅ Analysis Complete</strong><br>Results displayed below</div>',
            unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="status-card status-info"><strong>👆 Click on map</strong><br>Select a location to analyze</div>',
            unsafe_allow_html=True
            )

    # Display analysis results if available
    if st.session_state.analysis_results:
        st.markdown("---")
        if "datasets" in st.session_state:
            display_analysis_results(
                st.session_state.analysis_results,
                st.session_state.datasets
    )
        else:
            st.error("❌ Datasets not loaded. Please re-run analysis.")



    # Footer
    st.markdown("---")
    st.markdown(
        '''
        <div style="text-align: center; padding: 2rem; color: #666;">
            <strong>🏪 Health & Glow Store Location Analysis Tool</strong><br>
            Professional Edition with Interactive Map Selection & Advanced Analytics<br>
            Built with Streamlit, Folium & Advanced Data Science
        </div>
        ''',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

