import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import requests
from dateutil import rrule
from io import StringIO

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

# Weather API Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_environment_canada_stations():
    """Get list of Environment Canada weather stations"""
    stations = {
        "Vancouver (YVR)": 51442,
        "Toronto (YYZ)": 51459,
        "Montreal (YUL)": 30165,
        "Calgary (YYC)": 2205,
        "Ottawa (YOW)": 49568,
        "Edmonton (YEG)": 1867,
        "Winnipeg (YWG)": 3698,
        "Halifax (YHZ)": 50620,
        "Quebec City": 26892,
        "Victoria": 51337,
        "Saskatoon": 3328,
        "St. John's": 50089,
        "Pitt Meadows": 6830,
        "Surrey": 1108,
        "Burnaby": 888,
        "Richmond": 6831
    }
    return stations

@st.cache_data(ttl=3600)  # Cache for 1 hour
def getHourlyData(stationID, year, month):
    """Get hourly weather data from Environment Canada API"""
    base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    query_url = f"format=csv&stationID={stationID}&Year={year}&Month={month}&timeframe=1"
    api_endpoint = base_url + query_url
    
    try:
        response = requests.get(api_endpoint, verify=False, timeout=30 )
        response.raise_for_status()
        
        # Read the CSV content
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, skiprows=0)
        
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

def get_weather_data_range(stationID, start_date, end_date):
    """Get weather data for a date range"""
    frames = []
    
    # Create progress bar
    total_months = len(list(rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date)))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Loop over each month within the date range
    for i, dt in enumerate(rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date)):
        status_text.text(f"Fetching data for {dt.strftime('%B %Y')}...")
        
        df = getHourlyData(stationID, dt.year, dt.month)
        if not df.empty:
            frames.append(df)
        
        # Update progress
        progress_bar.progress((i + 1) / total_months)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Concatenate all data frames
    if frames:
        weather_data = pd.concat(frames, ignore_index=True)
        
        # Clean and process the data
        weather_data['Date/Time (LST)'] = pd.to_datetime(weather_data['Date/Time (LST)'], errors='coerce')
        weather_data['Temp (°C)'] = pd.to_numeric(weather_data['Temp (°C)'], errors='coerce')
        weather_data['Dew Point Temp (°C)'] = pd.to_numeric(weather_data['Dew Point Temp (°C)'], errors='coerce')
        weather_data['Rel Hum (%)'] = pd.to_numeric(weather_data['Rel Hum (%)'], errors='coerce')
        weather_data['Wind Spd (km/h)'] = pd.to_numeric(weather_data['Wind Spd (km/h)'], errors='coerce')
        
        # Select and rename columns
        if 'Date/Time (LST)' in weather_data.columns:
            specific_wd = weather_data[['Date/Time (LST)', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)']].copy()
            specific_wd.columns = ['Date_Time', 'Temperature (°C)', 'Dew_Point_Temp_C', 'Humidity (%)', 'Wind Speed (km/h)']
            
            # Handle missing data with interpolation
            missing_before = specific_wd['Temperature (°C)'].isnull().sum()
            if missing_before > 0:
                st.info(f"Interpolating {missing_before} missing temperature values...")
                specific_wd['Temperature (°C)'] = specific_wd['Temperature (°C)'].interpolate()
                specific_wd['Dew_Point_Temp_C'] = specific_wd['Dew_Point_Temp_C'].interpolate()
                specific_wd['Humidity (%)'] = specific_wd['Humidity (%)'].interpolate()
                specific_wd['Wind Speed (km/h)'] = specific_wd['Wind Speed (km/h)'].interpolate()
            
            # Add location info
            stations = get_environment_canada_stations()
            station_name = next((name for name, id in stations.items() if id == stationID), f"Station {stationID}")
            specific_wd['Location'] = station_name
            
            return specific_wd
        else:
            st.error("Expected columns not found in weather data")
            return pd.DataFrame()
    else:
        st.warning("No weather data retrieved for the specified period")
        return pd.DataFrame()

# Title
st.title("⚡ Energy Analysis Application")
st.markdown("*Powered by Streamlit Cloud + Environment Canada API*")

# Navigation
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Dashboard", "🌤️ Weather Data", "⚡ Energy Data", "📊 Analysis", "📈 Energy Fits", "📅 Scheduler"]
)

# Dashboard
if page == "🏠 Dashboard":
    st.header("📊 Energy Analysis Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        weather_count = len(st.session_state.weather_data)
        st.metric("🌤️ Weather Records", f"{weather_count:,}")
    
    with col2:
        energy_count = len(st.session_state.energy_data)
        st.metric("⚡ Energy Records", f"{energy_count:,}")
    
    with col3:
        if not st.session_state.energy_data.empty and 'Power (kW)' in st.session_state.energy_data.columns:
            avg_power = st.session_state.energy_data['Power (kW)'].mean()
            st.metric("🔌 Avg Power", f"{avg_power:.1f} kW")
        else:
            st.metric("🔌 Avg Power", "No data")
    
    with col4:
        if not st.session_state.energy_data.empty and 'Equipment ID' in st.session_state.energy_data.columns:
            equipment_count = st.session_state.energy_data['Equipment ID'].nunique()
            st.metric("🏭 Equipment Count", f"{equipment_count}")
        else:
            st.metric("🏭 Equipment Count", "0")
    
    # Sample data generation
    if st.session_state.energy_data.empty:
        if st.button("🎲 Generate Sample Data", type="primary"):
            # Generate sample energy data
            dates = pd.date_range('2024-01-01', periods=48, freq='H')
            equipment_types = ['HVAC_01', 'LIGHTING_01', 'MOTORS_01']
            
            sample_data = []
            for equipment in equipment_types:
                base_power = {'HVAC_01': 1500, 'LIGHTING_01': 800, 'MOTORS_01': 1200}
                power_data = np.random.normal(base_power[equipment], base_power[equipment] * 0.2, 48)
                power_data = np.maximum(power_data, 0)
                
                for i, date in enumerate(dates):
                    sample_data.append({
                        'Timestamp': date,
                        'Equipment ID': equipment,
                        'Power (kW)': power_data[i],
                        'Energy (kWh)': power_data[i] * 1
                    })
            
            st.session_state.energy_data = pd.DataFrame(sample_data)
            st.success("✅ Sample energy data generated!")
            st.rerun()
    else:
        # Plot energy data
        fig = px.line(st.session_state.energy_data, 
                     x='Timestamp', 
                     y='Power (kW)', 
                     color='Equipment ID',
                     title='Energy Consumption Over Time')
        st.plotly_chart(fig, use_container_width=True)

# Weather Data Section with Environment Canada API
elif page == "🌤️ Weather Data":
    st.header("🌤️ Weather Data Management")
    
    # API Data Fetching Section
    st.subheader("🌐 Get Real Weather Data from Environment Canada")
    
    # Get available stations
    stations = get_environment_canada_stations()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_station = st.selectbox(
            "📍 Select Weather Station", 
            list(stations.keys()),
            index=list(stations.keys()).index("Pitt Meadows") if "Pitt Meadows" in stations else 0
        )
        station_id = stations[selected_station]
    
    with col2:
        start_date = st.date_input(
            "📅 Start Date", 
            value=datetime.now().date() - timedelta(days=90),
            max_value=datetime.now().date()
        )
    
    with col3:
        end_date = st.date_input(
            "📅 End Date", 
            value=datetime.now().date() - timedelta(days=1),
            max_value=datetime.now().date()
        )
    
    # Validate date range
    if start_date >= end_date:
        st.error("Start date must be before end date")
    elif (end_date - start_date).days > 365:
        st.warning("Large date ranges may take longer to fetch. Consider smaller ranges for faster results.")
    else:
        if st.button("🔄 Fetch Weather Data from Environment Canada", type="primary"):
            try:
                # Convert dates to datetime objects
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                
                # Fetch weather data
                weather_data = get_weather_data_range(station_id, start_dt, end_dt)
                
                if not weather_data.empty:
                    # Store in session state
                    if st.session_state.weather_data.empty:
                        st.session_state.weather_data = weather_data
                    else:
                        # Append new data
                        st.session_state.weather_data = pd.concat([st.session_state.weather_data, weather_data], ignore_index=True)
                        # Remove duplicates based on Date_Time and Location
                        st.session_state.weather_data = st.session_state.weather_data.drop_duplicates(subset=['Date_Time', 'Location'])
                    
                    st.success(f"✅ Successfully retrieved {len(weather_data)} weather records from {selected_station}")
                    
                    # Show data preview
                    st.subheader("📊 Data Preview")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Data Summary:**")
                        st.write(f"- Records: {len(weather_data):,}")
                        st.write(f"- Date range: {weather_data['Date_Time'].min().strftime('%Y-%m-%d')} to {weather_data['Date_Time'].max().strftime('%Y-%m-%d')}")
                        st.write(f"- Station: {selected_station}")
                        
                        # Basic statistics
                        if 'Temperature (°C)' in weather_data.columns:
                            temp_stats = weather_data['Temperature (°C)'].describe()
                            st.write(f"- Avg Temperature: {temp_stats['mean']:.1f}°C")
                            st.write(f"- Min Temperature: {temp_stats['min']:.1f}°C")
                            st.write(f"- Max Temperature: {temp_stats['max']:.1f}°C")
                    
                    with col2:
                        st.write("**Sample Data:**")
                        st.dataframe(weather_data.head(10), use_container_width=True)
                    
                    # Quick visualization
                    if len(weather_data) > 0:
                        st.subheader("📈 Temperature Trend")
                        fig = px.line(weather_data, x='Date_Time', y='Temperature (°C)', 
                                     title=f'Temperature Trend - {selected_station}')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.rerun()
                else:
                    st.error("No weather data retrieved. Please check the station ID and date range.")
                    
            except Exception as e:
                st.error(f"Error fetching weather data: {str(e)}")
    
    # Manual data entry section
    st.subheader("📝 Manual Data Entry")
    
    with st.expander("Add Weather Data Manually"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input("📍 Location", value="Custom Location")
            temperature = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.1)
        
        with col2:
            humidity = st.number_input("💧 Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, step=0.1)
            wind_speed = st.number_input("💨 Wind Speed (km/h)", value=15.0, min_value=0.0, step=0.1)
        
        with col3:
            # FIXED: Use separate date and time inputs
            manual_date = st.date_input("📅 Date", datetime.now().date(), key="manual_date")
            manual_time = st.time_input("🕐 Time", datetime.now().time(), key="manual_time")
        
        if st.button("💾 Add Manual Weather Record", type="secondary"):
            # Combine date and time
            date_time = datetime.combine(manual_date, manual_time)
            
            new_record = pd.DataFrame({
                'Date_Time': [date_time],
                'Location': [location],
                'Temperature (°C)': [temperature],
                'Humidity (%)': [humidity],
                'Wind Speed (km/h)': [wind_speed],
                'Dew_Point_Temp_C': [temperature - 5]  # Rough estimate
            })
            
            if st.session_state.weather_data.empty:
                st.session_state.weather_data = new_record
            else:
                st.session_state.weather_data = pd.concat([st.session_state.weather_data, new_record], ignore_index=True)
            
            st.success("✅ Manual weather record added!")
            st.rerun()
    
    # Display existing weather data
    if not st.session_state.weather_data.empty:
        st.subheader("📊 Weather Data Analysis")
        
        weather_df = st.session_state.weather_data
        
        # Location filter
        if 'Location' in weather_df.columns:
            locations = weather_df['Location'].unique()
            selected_location = st.selectbox("Filter by location:", ["All"] + list(locations))
            
            if selected_location != "All":
                weather_df = weather_df[weather_df['Location'] == selected_location]
        
        if not weather_df.empty:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'Temperature (°C)' in weather_df.columns:
                    avg_temp = weather_df['Temperature (°C)'].mean()
                    st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C")
            
            with col2:
                if 'Humidity (%)' in weather_df.columns:
                    avg_humidity = weather_df['Humidity (%)'].mean()
                    st.metric("💧 Avg Humidity", f"{avg_humidity:.1f}%")
            
            with col3:
                if 'Wind Speed (km/h)' in weather_df.columns:
                    avg_wind = weather_df['Wind Speed (km/h)'].mean()
                    st.metric("💨 Avg Wind Speed", f"{avg_wind:.1f} km/h")
            
            with col4:
                st.metric("📊 Total Records", f"{len(weather_df):,}")
            
            # Visualization
            st.subheader("📈 Weather Trends")
            
            # Select variable to plot
            numeric_cols = weather_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                plot_var = st.selectbox("Select variable to plot:", numeric_cols)
                
                if 'Date_Time' in weather_df.columns:
                    fig = px.line(weather_df, x='Date_Time', y=plot_var, 
                                 color='Location' if selected_location == "All" and 'Location' in weather_df.columns else None,
                                 title=f'{plot_var} Over Time')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("📋 Weather Data Table")
            st.dataframe(weather_df, use_container_width=True)
            
            # Download option
            csv = weather_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Weather Data as CSV",
                data=csv,
                file_name=f"weather_data_{selected_location}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No weather data found for the selected location.")
    else:
        st.info("No weather data available. Use the options above to fetch or add weather data.")

# Energy Data Section
elif page == "⚡ Energy Data":
    st.header("⚡ Energy Data Management")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Energy Data CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            energy_data = pd.read_csv(uploaded_file)
            st.session_state.energy_data = energy_data
            st.success(f"✅ Energy data uploaded! ({len(energy_data)} records)")
            st.dataframe(energy_data.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Manual data entry
    st.subheader("📝 Add Energy Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        equipment_id = st.text_input("🏭 Equipment ID", value="HVAC_01")
        power = st.number_input("🔌 Power (kW)", value=1500.0, min_value=0.0, step=0.1)
    
    with col2:
        energy = st.number_input("⚡ Energy (kWh)", value=1500.0, min_value=0.0, step=0.1)
        # FIXED: Use separate date and time inputs
        energy_date = st.date_input("📅 Date", datetime.now().date(), key="energy_date")
        energy_time = st.time_input("🕐 Time", datetime.now().time(), key="energy_time")
    
    if st.button("💾 Add Energy Record", type="primary"):
        # Combine date and time
        timestamp = datetime.combine(energy_date, energy_time)
        
        new_record = pd.DataFrame({
            'Timestamp': [timestamp],
            'Equipment ID': [equipment_id],
            'Power (kW)': [power],
            'Energy (kWh)': [energy]
        })
        
        if st.session_state.energy_data.empty:
            st.session_state.energy_data = new_record
        else:
            st.session_state.energy_data = pd.concat([st.session_state.energy_data, new_record], ignore_index=True)
        
        st.success("✅ Energy record added!")
        st.rerun()
    
    # Display energy data
    if not st.session_state.energy_data.empty:
        st.subheader("📊 Energy Data Visualization")
        
        # Plot energy data
        fig = px.line(st.session_state.energy_data, x='Timestamp', y='Power (kW)', 
                     color='Equipment ID', title='Power Consumption Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(st.session_state.energy_data, use_container_width=True)

# Analysis Section
elif page == "📊 Analysis":
    st.header("📊 Energy Analysis")
    
    if not st.session_state.energy_data.empty:
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        
        numeric_cols = st.session_state.energy_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(st.session_state.energy_data[numeric_cols].describe(), use_container_width=True)
        
        # Equipment comparison
        if 'Equipment ID' in st.session_state.energy_data.columns:
            st.subheader("🏭 Equipment Comparison")
            
            equipment_stats = st.session_state.energy_data.groupby('Equipment ID')['Power (kW)'].agg(['mean', 'max', 'std']).round(2)
            st.dataframe(equipment_stats, use_container_width=True)
            
            # Bar chart
            avg_power = st.session_state.energy_data.groupby('Equipment ID')['Power (kW)'].mean()
            fig = px.bar(x=avg_power.index, y=avg_power.values, 
                        title='Average Power Consumption by Equipment')
            fig.update_xaxis(title='Equipment ID')
            fig.update_yaxis(title='Average Power (kW)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No energy data available for analysis.")

# Energy Fits Section
elif page == "📈 Energy Fits":
    st.header("📈 Energy Fits & Modeling")
    
    if not st.session_state.energy_data.empty:
        energy_df = st.session_state.energy_data.copy()
        numeric_cols = energy_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.subheader("🔧 Simple Linear Regression")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X variable:", numeric_cols)
            
            with col2:
                y_var = st.selectbox("Y variable:", [col for col in numeric_cols if col != x_var])
            
            if st.button("🔄 Perform Linear Fit", type="primary"):
                # Get clean data
                clean_data = energy_df[[x_var, y_var]].dropna()
                
                if len(clean_data) > 1:
                    x = clean_data[x_var].values
                    y = clean_data[y_var].values
                    
                    # Linear fit
                    coeffs = np.polyfit(x, y, 1)
                    y_fit = np.polyval(coeffs, x)
                    
                    # Calculate R-squared
                    ss_res = np.sum((y - y_fit) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("📊 R-squared", f"{r_squared:.4f}")
                    
                    with col2:
                        st.metric("📈 Data Points", len(clean_data))
                    
                    equation = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"
                    st.write(f"**Equation:** {equation}")
                    
                    # Plot
                    fig = px.scatter(clean_data, x=x_var, y=y_var, title=f'{y_var} vs {x_var}')
                    
                    # Add trend line
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = np.polyval(coeffs, x_line)
                    
                    fig.add_scatter(x=x_line, y=y_line, mode='lines', name='Linear Fit', 
                                   line=dict(color='red', width=2))
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Not enough data points for fitting.")
        else:
            st.info("Need at least 2 numeric columns for fitting analysis.")
    else:
        st.info("No energy data available for fitting.")

# Scheduler Section
elif page == "📅 Scheduler":
    st.header("📅 Equipment Scheduler")
    
    st.subheader("⏰ Equipment Runtime Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        equipment_id = st.text_input("🏭 Equipment ID", value="HVAC_01")
        power_rating = st.number_input("🔌 Power Rating (kW)", value=1500.0, min_value=0.0, step=0.1)
    
    with col2:
        operating_hours = st.number_input("⏱️ Operating Hours/Day", value=8.0, min_value=0.0, max_value=24.0, step=0.5)
        efficiency = st.slider("⚡ Efficiency (%)", 0, 100, 85)
    
    # Days selection
    st.subheader("📅 Operating Days")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    operating_days = []
    
    cols = st.columns(7)
    for i, day in enumerate(days):
        with cols[i]:
            if st.checkbox(day, value=(i < 5), key=f"day_{i}"):
                operating_days.append(day)
    
    if st.button("📊 Calculate Energy Consumption", type="primary"):
        # Calculate consumption
        daily_energy = power_rating * operating_hours * (efficiency / 100)
        weekly_energy = daily_energy * len(operating_days)
        monthly_energy = weekly_energy * 4.33
        annual_energy = weekly_energy * 52
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Daily Energy", f"{daily_energy:.1f} kWh")
        
        with col2:
            st.metric("📅 Weekly Energy", f"{weekly_energy:.1f} kWh")
        
        with col3:
            st.metric("📅 Monthly Energy", f"{monthly_energy:.1f} kWh")
        
        with col4:
            st.metric("📅 Annual Energy", f"{annual_energy:.0f} kWh")
        
        # Schedule summary
        st.subheader("📋 Schedule Summary")
        schedule_data = {
            'Parameter': ['Equipment ID', 'Operating Days', 'Daily Hours', 'Power Rating', 'Efficiency', 'Daily Energy'],
            'Value': [
                equipment_id,
                f"{len(operating_days)} days",
                f"{operating_hours} hours",
                f"{power_rating} kW",
                f"{efficiency}%",
                f"{daily_energy:.1f} kWh"
            ]
        }
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🔧 Powered by Streamlit Cloud + Environment Canada**")

# Data management
if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.weather_data = pd.DataFrame()
    st.session_state.energy_data = pd.DataFrame()
    st.sidebar.success("All data cleared!")
    st.rerun()

# Debug section
if st.sidebar.checkbox("🐛 Debug Mode"):
    st.sidebar.write("Session State Keys:", list(st.session_state.keys()))
    if 'weather_data' in st.session_state:
        st.sidebar.write("Weather Data Shape:", st.session_state.weather_data.shape)
    if 'energy_data' in st.session_state:
        st.sidebar.write("Energy Data Shape:", st.session_state.energy_data.shape)
