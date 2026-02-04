import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import os
import sklearn.compose._column_transformer
import warnings
warnings.filterwarnings('ignore')

# FIX FOR MISSING CLASS - Add this at the VERY TOP
class _RemainderColsList(list):
    pass

# Patch the missing class
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# Set page config
st.set_page_config(
    page_title="Flood Risk Prediction Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Alert Banner on Top
st.warning("⚠️ Always follow official guidelines during severe weather. This dashboard is only a prediction tool.", icon="🚨")

st.title("🌊 Flood Risk Prediction Dashboard")
st.success("✅ Version compatibility fix applied")

# Feature definitions
NUM_FEATURES = [
    "rainfall_mm", "rain_7d", "rain_30d",
    "elevation", "dist_to_river", 
    "population_density", "impervious_frac", "land_cover_index"
]

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}

# USD to INR conversion rate (approximate)
USD_TO_INR = 83.0

# Load models with enhanced error handling
@st.cache_resource
def load_models():
    try:
        # Show loading status
        with st.spinner("Loading machine learning models..."):
            clf = joblib.load('models/clf_pipeline.joblib')
            reg = joblib.load('models/reg_pipeline.joblib')
        
        st.success("✅ Models loaded successfully!")
        return clf, reg
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        
        # Show specific installation instructions
        if "lightgbm" in str(e).lower():
            st.info("""
            **Missing LightGBM package. Install it with:**
            ```bash
            pip install lightgbm
            ```
            or
            ```bash
            conda install lightgbm
            ```
            """)
        else:
            st.info("""
            **Common solutions:**
            1. Install missing packages: `pip install lightgbm`
            2. Check model files exist in 'models/' folder
            3. Ensure all dependencies are installed
            """)
        
        return None, None

def create_interactive_map(lat, lon, risk_level, p_high, pred_damage_inr, city_name=""):
    """Create accurate interactive map"""
    m = folium.Map(
        location=[lat, lon],
        zoom_start=13,  # Increased zoom for better detail
        tiles="OpenStreetMap"
    )
    
    # Risk-based marker color
    if risk_level == "High":
        icon_color = "red"
        risk_radius = 3000
    elif risk_level == "Medium":
        icon_color = "orange" 
        risk_radius = 2000
    else:
        icon_color = "green"
        risk_radius = 1000
    
    # Add precise marker
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(
            f"""
            <div style="font-family: Arial; font-size: 14px;">
                <h4 style="margin: 5px 0; color: #333;">{city_name}</h4>
                <hr style="margin: 8px 0;">
                <b>Risk Level:</b> <span style="color: {RISK_COLORS[risk_level]}">{risk_level}</span><br>
                <b>High Risk Probability:</b> {p_high:.1%}<br>
                <b>Estimated Damage:</b> ₹{pred_damage_inr:,.0f}<br>
                <b>Coordinates:</b> {lat:.6f}, {lon:.6f}<br>
            </div>
            """,
            max_width=300
        ),
        tooltip=f"📍 {city_name} - {risk_level} Risk",
        icon=folium.Icon(color=icon_color, icon="info-sign")
    ).add_to(m)
    
    # Add risk area circle
    folium.Circle(
        location=[lat, lon],
        radius=risk_radius,
        popup=f"Risk Impact Area (~{risk_radius}m radius)",
        color=RISK_COLORS[risk_level],
        fill=True,
        fillOpacity=0.2,
        weight=3
    ).add_to(m)
    
    return m

def create_coordinate_picker():
    """Create a LARGE map for users to click and get coordinates"""
    st.sidebar.subheader("🎯 Click on Map to Get Coordinates")
    
    st.sidebar.markdown("""
    **Instructions:**
    1. **Zoom** to your exact area using mouse wheel
    2. **Click** precisely on your location  
    3. **Copy** the coordinates from popup
    4. **Paste** in Latitude/Longitude fields below
    """)
    
    # Create LARGE map centered on India
    m = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles="OpenStreetMap"
    )
    
    # Add click functionality
    m.add_child(folium.LatLngPopup())
    
    # Display the LARGE map in sidebar
    folium_static(m, width=350, height=400)  # Increased size for better usability

def plot_risk_probabilities(probas):
    """Plot risk probability chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(RISK_LABELS.values())
    colors = [RISK_COLORS[label] for label in labels]
    
    bars = ax.bar(labels, probas, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Flood Risk Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, prob in zip(bars, probas):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.1%}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig

def plot_damage_estimate(pred_damage_inr):
    """Plot damage estimate visualization in Rupees"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    ax.barh(['Estimated Damage'], [pred_damage_inr], 
            color='#ff6b6b', alpha=0.7, height=0.6)
    
    ax.set_xlabel('Damage Estimate (₹)', fontweight='bold')
    ax.set_title('Predicted Flood Damage', fontweight='bold')
    
    ax.text(pred_damage_inr, 0, f'₹{pred_damage_inr:,.0f}', 
            ha='left', va='center', fontweight='bold', 
            fontsize=16, color='#333')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def convert_usd_to_inr(usd_amount):
    """Convert USD amount to Indian Rupees"""
    return usd_amount * USD_TO_INR

def format_rupees(amount):
    """Format amount in Indian Rupees with proper formatting"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f} crore"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f} lakh"
    else:
        return f"₹{amount:,.0f}"

def display_emergency_contacts():
    """Display emergency contacts panel"""
    st.sidebar.markdown("---")
    st.sidebar.header("🆘 Emergency Contacts")
    
    contacts = [
        {"name": "National Disaster Response Force (NDRF)", "number": "011-24363260", "icon": "🚨"},
        {"name": "State Disaster Management Authority", "number": "1070", "icon": "🏛️"},
        {"name": "Rescue Services", "number": "101", "icon": "🚑"},
        {"name": "Ambulance (Medical Emergency)", "number": "108", "icon": "🏥"},
        {"name": "Police", "number": "100", "icon": "👮"},
        {"name": "Fire Department", "number": "102", "icon": "🚒"}
    ]
    
    for contact in contacts:
        with st.sidebar.container():
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #dc3545;'>
                <strong style='color: black;'>{contact['icon']} {contact['name']}</strong><br>
                <span style='color: black;'>📞 {contact['number']}</span>
            </div>
            """, unsafe_allow_html=True)

def display_safety_guidelines():
    """Display safety guidelines in expandable sections"""
    st.markdown("---")
    st.header("🛡️ Flood Safety Guidelines")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("📋 **Before Flood**", expanded=False):
            st.markdown("""
            - Know evacuation routes
            - Pack emergency kit
            - Keep phone charged
            - Secure important documents
            - Move valuables to higher floors
            - Monitor weather alerts
            """)
    
    with col2:
        with st.expander("🌊 **During Flood**", expanded=False):
            st.markdown("""
            - Move to higher ground immediately
            - Don't touch electric appliances
            - Don't go near rivers/streams
            - Avoid driving in waterlogged areas
            - Keep listening to official alerts
            - Switch off electricity at main
            """)
    
    with col3:
        with st.expander("🏠 **After Flood**", expanded=False):
            st.markdown("""
            - Avoid contaminated water
            - Clean and disinfect surroundings
            - Check for structural damages
            - Document damages for insurance
            - Boil drinking water
            - Watch for electrical hazards
            """)

def display_emergency_kit():
    """Display emergency kit checklist"""
    st.markdown("---")
    st.header("🎒 Emergency Kit Checklist")
    
    kit_items = [
        "💧 Water bottles (3+ days supply)",
        "🔦 Flashlight with extra batteries",
        "💊 Basic medicines & first aid kit",
        "🔋 Power bank & charger",
        "🍫 Dry snacks & non-perishable food",
        "📄 Important documents (waterproof)",
        "💰 Cash & emergency funds",
        "🧥 Warm clothes & blankets",
        "🧴 Hygiene & sanitation items",
        "📱 Emergency contact list"
    ]
    
    cols = st.columns(2)
    for i, item in enumerate(kit_items):
        cols[i % 2].checkbox(item, key=f"kit_{i}")

def display_affected_guidelines():
    """Display guidelines for affected people"""
    st.markdown("---")
    st.header("🚨 If You Are Affected")
    
    with st.container():
        st.error("""
        **IMMEDIATE ACTIONS REQUIRED:**
        
        🚶 **Move to higher ground immediately** - Don't wait for instructions
        ⚡ **Switch off electricity** at main switch
        📄 **Carry essential documents** in waterproof bag
        🚗 **Avoid driving** in waterlogged areas
        📞 **Call emergency helpline** if trapped: 108 or 101
        📻 **Stay tuned** to local news and alerts
        🏠 **Evacuate** if advised by authorities
        """)

def display_help_others():
    """Display information for helping others"""
    st.markdown("---")
    st.header("🤝 How You Can Help Others")
    
    organizations = [
        {"name": "Red Cross India", "focus": "Emergency relief & medical aid", "url": "https://www.indianredcross.org/"},
        {"name": "Goonj", "focus": "Material relief & rehabilitation", "url": "https://goonj.org/"},
        {"name": "Hemkunt Foundation", "focus": "Disaster relief & community support", "url": "https://hemkuntfoundation.com/"},
        {"name": "Local Municipal Relief Fund", "focus": "Direct local assistance", "url": "#"}
    ]
    
    st.info("""
    **Volunteer or support these organizations during floods:**
    """)
    
    for org in organizations:
        st.markdown(f"""
        **{org['name']}**  
        *{org['focus']}*  
        🔗 [Learn More]({org['url']})
        """)

def display_donation_section():
    """Display donation links"""
    st.markdown("---")
    st.header("❤️ Support Relief Efforts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <a href="https://www.indianredcross.org/donate" target="_blank">
            <button style="background-color: #dc3545; color: white; padding: 10px 20px; border: none; border-radius: 5px; width: 100%;">
                🟥 Donate to Red Cross India
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <a href="https://hemkuntfoundation.com/donate/" target="_blank">
            <button style="background-color: #fd7e14; color: white; padding: 10px 20px; border: none; border-radius: 5px; width: 100%;">
                🟧 Donate to Hemkunt Foundation
            </button>
        </a>
        """, unsafe_allow_html=True)

def display_awareness_facts():
    """Display educational awareness facts"""
    st.markdown("---")
    st.header("💡 Flood Awareness Facts")
    
    facts = [
        "🌧️ **1 cm increase** in rainfall in low-elevation areas can raise flood chances by 10–15%",
        "🏞️ **Urban areas** with >40% impervious surfaces are 3x more likely to experience flash floods",
        "⏰ **Early warning** systems can reduce flood fatalities by up to 35%",
        "🌳 **Natural vegetation** can absorb up to 90% of rainfall, reducing flood risk significantly",
        "📈 **Climate change** has increased extreme rainfall events by 20% in the last decade"
    ]
    
    for fact in facts:
        st.info(fact)

def main():
    # Load models first
    clf, reg = load_models()
    if clf is None or reg is None:
        # Show installation instructions
        st.info("""
        ## 📋 Installation Instructions
        
        Please install the required packages:
        ```bash
        pip install lightgbm streamlit pandas numpy matplotlib folium streamlit-folium scikit-learn joblib
        ```
        
        Then restart the app.
        """)
        
        # Show emergency contacts even if models aren't loaded
        display_emergency_contacts()
        display_safety_guidelines()
        display_emergency_kit()
        display_affected_guidelines()
        display_help_others()
        display_donation_section()
        display_awareness_facts()
        return
    
    # Input form
    st.sidebar.header("🔧 Prediction Inputs")
    
    features = {}
    
    st.sidebar.subheader("🌧️ Rainfall Data")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        features["rainfall_mm"] = st.number_input("Today (mm)", 
                                                 value=15.0, min_value=0.0, step=1.0,
                                                 help="Today's rainfall in millimeters")
    with col2:
        features["rain_7d"] = st.number_input("7-day (mm)", 
                                             value=65.0, min_value=0.0, step=5.0,
                                             help="Total rainfall last 7 days")
    with col3:
        features["rain_30d"] = st.number_input("30-day (mm)", 
                                              value=180.0, min_value=0.0, step=10.0,
                                              help="Total rainfall last 30 days")
    
    st.sidebar.subheader("🏔️ Geographic Features")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        features["elevation"] = st.number_input("Elevation (m)", 
                                               value=250.0, min_value=0.0, step=10.0,
                                               help="Area elevation in meters")
    with col2:
        features["dist_to_river"] = st.number_input("River Distance (km)", 
                                                   value=1.5, min_value=0.0, step=0.1,
                                                   help="Distance to nearest river in km")
    
    st.sidebar.subheader("🏙️ Urban Features")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        features["population_density"] = st.number_input("Population Density", 
                                                       value=1200.0, min_value=0.0, step=100.0,
                                                       help="People per square km")
        features["impervious_frac"] = st.slider("Impervious Surface %", 
                                               0.0, 1.0, 0.4, 0.05,
                                               help="Fraction of impervious surfaces")
    with col2:
        features["land_cover_index"] = st.slider("Land Cover Index", 
                                               0.0, 1.0, 0.6, 0.05,
                                               help="Vegetation/land cover index")
    
    st.sidebar.subheader("🗺️ Location Details")
    
    # Add the LARGE coordinate picker map
    create_coordinate_picker()
    
    st.sidebar.markdown("**Enter coordinates manually or use map above:**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=18.5204, format="%.6f",
                             help="Latitude coordinate for mapping")
    with col2:
        lon = st.number_input("Longitude", value=73.8567, format="%.6f",
                             help="Longitude coordinate for mapping")
    
    # Show current coordinates being used
    st.sidebar.info(f"📍 Using coordinates: {lat:.6f}, {lon:.6f}")
    
    city_name = st.sidebar.text_input("Location Name:", value="Prediction Location")
    
    # Feature summary
    with st.sidebar.expander("📋 Feature Summary"):
        st.write("**Location:**", f"{lat:.6f}, {lon:.6f}")
        for feature, value in features.items():
            st.write(f"**{feature}:** {value:.2f}")
    
    # PREDICTION BUTTON MOVED BEFORE EMERGENCY CONTACTS
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Predict Flood Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing flood risk..."):
            try:
                # Prepare input data
                X_input = pd.DataFrame([features], columns=NUM_FEATURES)
                
                # Make predictions
                probas = clf.predict_proba(X_input)[0]
                pred_idx = int(clf.predict(X_input)[0])
                pred_label = RISK_LABELS[pred_idx]
                
                # Predict damage in USD and convert to INR
                pred_log = reg.predict(X_input)[0]
                pred_damage_usd = float(np.expm1(pred_log))
                pred_damage_inr = convert_usd_to_inr(pred_damage_usd)
                
                # Display results
                st.header("📊 Prediction Results")
                
                # Results in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Risk Level", pred_label)
                
                with col2:
                    st.metric("High Risk Probability", f"{probas[2]:.1%}")
                
                with col3:
                    st.metric("Estimated Damage", format_rupees(pred_damage_inr))
                
                with col4:
                    confidence = max(probas)
                    st.metric("Model Confidence", f"{confidence:.1%}")
                
                # Show coordinates used
                st.info(f"📍 **Analysis for:** {city_name} at {lat:.6f}, {lon:.6f}")
                
                # Alert system
                st.subheader("🚨 Risk Assessment")
                if pred_label == "High" or probas[2] > 0.6:
                    st.error("""
                    **CRITICAL ALERT - HIGH FLOOD RISK DETECTED**
                    - Immediate evacuation preparation recommended
                    - Notify local authorities and emergency services
                    - Monitor weather forecasts continuously
                    - Activate emergency response plans
                    """)
                elif pred_label == "Medium" or probas[2] > 0.3:
                    st.warning("""
                    **MEDIUM RISK - INCREASED VIGILANCE REQUIRED**
                    - Monitor water levels and weather updates
                    - Prepare emergency supplies
                    - Review evacuation routes
                    - Stay informed about local alerts
                    """)
                else:
                    st.success("""
                    **LOW RISK - NORMAL CONDITIONS**
                    - Continue routine monitoring
                    - Maintain standard preparedness
                    - Stay informed about weather changes
                    """)
                
                # Visualizations
                st.subheader("📈 Risk Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.pyplot(plot_risk_probabilities(probas))
                
                with col2:
                    st.pyplot(plot_damage_estimate(pred_damage_inr))
                
                # Interactive Map - FULL SCREEN
                st.subheader("🗺️ Location Map - Full Screen View")
                
                # Map features description
                st.info("""
                **🗺️ Map Features:**
                
                📍 **Marker**: Your location with risk details (click to see details)  
                🎯 **Colored Circle**: Flood risk impact area  
                🔍 **Zoom in/out**: Use mouse wheel or +/- buttons
                """)
                
                # Create FULL WIDTH map container
                map_container = st.container()
                with map_container:
                    map_obj = create_interactive_map(
                        lat, lon, pred_label, probas[2], pred_damage_inr, city_name
                    )
                    # FULL SCREEN MAP - uses maximum available width
                    folium_static(map_obj, width=1200, height=600)
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Please check all input values and try again.")

    # Emergency Contacts (now comes after prediction button)
    display_emergency_contacts()

    # Always show safety information (even without prediction)
    display_safety_guidelines()
    display_emergency_kit()
    display_affected_guidelines()
    display_help_others()
    display_donation_section()
    display_awareness_facts()

    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 How to use:**
    1. Enter feature values
    2. Click on map OR enter coordinates manually
    3. Click 'Predict Flood Risk'
    4. View results and map
    
    **📍 Getting coordinates:**
    - Click anywhere on the map above
    - Copy coordinates from popup
    - Paste in Latitude/Longitude fields
    """)

if __name__ == "__main__":
    main()