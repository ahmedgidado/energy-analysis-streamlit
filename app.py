import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import requests
from datetime import datetime, timedelta
from dateutil import rrule
from io import StringIO

# Weather API Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_environment_canada_stations():
    """Get list of Environment Canada weather stations"""
    # Common Canadian weather stations
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
        "Pitt Meadows": 6830,  # Your example station
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
        with st.spinner(f"Fetching weather data for {year}-{month:02d}..." ):
            response = requests.get(api_endpoint, verify=False, timeout=30)
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

# Updated Weather Data Section for your app
def weather_data_section():
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
        return
    
    if (end_date - start_date).days > 365:
        st.warning("Large date ranges may take longer to fetch. Consider smaller ranges for faster results.")
    
    if st.button("🔄 Fetch Weather Data from Environment Canada", type="primary"):
        try:
            # Convert dates to datetime objects
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            
            # Fetch weather data
            weather_data = get_weather_data_range(station_id, start_dt, end_dt)
            
            if not weather_data.empty:
                # Store in session state
                if 'weather_data' not in st.session_state:
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
    
    # Manual data entry section (keep existing functionality)
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
            date_time = st.datetime_input("📅 Date & Time", datetime.now())
        
        if st.button("💾 Add Manual Weather Record", type="secondary"):
            new_record = pd.DataFrame({
                'Date_Time': [date_time],
                'Location': [location],
                'Temperature (°C)': [temperature],
                'Humidity (%)': [humidity],
                'Wind Speed (km/h)': [wind_speed],
                'Dew_Point_Temp_C': [temperature - 5]  # Rough estimate
            })
            
            if 'weather_data' not in st.session_state or st.session_state.weather_data.empty:
                st.session_state.weather_data = new_record
            else:
                st.session_state.weather_data = pd.concat([st.session_state.weather_data, new_record], ignore_index=True)
            
            st.success("✅ Manual weather record added!")
            st.rerun()
    
    # Display existing weather data
    if 'weather_data' in st.session_state and not st.session_state.weather_data.empty:
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
            
            # Data table with pagination
            st.subheader("📋 Weather Data Table")
            
            # Add search functionality
            search_term = st.text_input("🔍 Search in data:", "")
            if search_term:
                mask = weather_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                filtered_df = weather_df[mask]
            else:
                filtered_df = weather_df
            
            # Show data with pagination
            rows_per_page = st.selectbox("Rows per page:", [10, 25, 50, 100], index=1)
            
            if len(filtered_df) > rows_per_page:
                page = st.selectbox("Page:", range(1, len(filtered_df) // rows_per_page + 2))
                start_idx = (page - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                display_df = filtered_df.iloc[start_idx:end_idx]
            else:
                display_df = filtered_df
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download options
            col1, col2 = st.columns(2)
            
            with col1:
                csv = weather_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Weather Data as CSV",
                    data=csv,
                    file_name=f"weather_data_{selected_location}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Create summary statistics file
                if numeric_cols:
                    summary_stats = weather_df[numeric_cols].describe()
                    summary_csv = summary_stats.to_csv()
                    st.download_button(
                        label="📊 Download Summary Statistics",
                        data=summary_csv,
                        file_name=f"weather_summary_{selected_location}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No weather data found for the selected location.")
    else:
        st.info("No weather data available. Use the options above to fetch or add weather data.")
