import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Energy Analysis Application",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = pd.DataFrame()
if 'energy_data' not in st.session_state:
    st.session_state.energy_data = pd.DataFrame()

# Title
st.title("⚡ Energy Analysis Application")
st.markdown("*Testing Phase - Basic Functionality*")

# Simple navigation
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Dashboard", "🌤️ Weather Data", "⚡ Energy Data"]
)

# Dashboard
if page == "🏠 Dashboard":
    st.header("📊 Dashboard")
    st.write("Dashboard is working!")
    
    if st.button("Generate Sample Data"):
        # Simple sample data
        dates = pd.date_range('2024-01-01', periods=24, freq='H')
        sample_weather = pd.DataFrame({
            'Date_Time': dates,
            'Temperature (°C)': np.random.normal(20, 5, 24),
            'Humidity (%)': np.random.normal(60, 10, 24),
            'Location': 'Test Location'
        })
        st.session_state.weather_data = sample_weather
        st.success("Sample data generated!")
        st.rerun()

# Weather Data
elif page == "🌤️ Weather Data":
    st.header("🌤️ Weather Data")
    
    # Simple manual entry
    st.subheader("Add Weather Data")
    
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Temperature (°C)", value=20.0)
        humidity = st.number_input("Humidity (%)", value=60.0)
    
    with col2:
        location = st.text_input("Location", value="Test Location")
        date_time = st.datetime_input("Date & Time", datetime.now())
    
    if st.button("Add Record"):
        new_record = pd.DataFrame({
            'Date_Time': [date_time],
            'Temperature (°C)': [temp],
            'Humidity (%)': [humidity],
            'Location': [location]
        })
        
        if st.session_state.weather_data.empty:
            st.session_state.weather_data = new_record
        else:
            st.session_state.weather_data = pd.concat([st.session_state.weather_data, new_record], ignore_index=True)
        
        st.success("Record added!")
        st.rerun()
    
    # Display data
    if not st.session_state.weather_data.empty:
        st.subheader("Weather Data")
        st.dataframe(st.session_state.weather_data)
        
        # Simple plot
        if len(st.session_state.weather_data) > 1:
            fig = px.line(st.session_state.weather_data, 
                         x='Date_Time', 
                         y='Temperature (°C)', 
                         title='Temperature Over Time')
            st.plotly_chart(fig, use_container_width=True)

# Energy Data
elif page == "⚡ Energy Data":
    st.header("⚡ Energy Data")
    st.write("Energy data section - basic functionality")
    
    # Simple energy data entry
    equipment = st.text_input("Equipment ID", value="HVAC_01")
    power = st.number_input("Power (kW)", value=1500.0)
    
    if st.button("Add Energy Record"):
        new_record = pd.DataFrame({
            'Timestamp': [datetime.now()],
            'Equipment ID': [equipment],
            'Power (kW)': [power]
        })
        
        if st.session_state.energy_data.empty:
            st.session_state.energy_data = new_record
        else:
            st.session_state.energy_data = pd.concat([st.session_state.energy_data, new_record], ignore_index=True)
        
        st.success("Energy record added!")
        st.rerun()
    
    # Display energy data
    if not st.session_state.energy_data.empty:
        st.dataframe(st.session_state.energy_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("🔧 Basic Version - Testing")
