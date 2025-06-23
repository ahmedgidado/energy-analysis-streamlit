import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

# Configure the page
st.set_page_config(
    page_title="Energy Analysis Application",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data persistence
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = pd.DataFrame()
if 'energy_data' not in st.session_state:
    st.session_state.energy_data = pd.DataFrame()

# Title and header
st.title("⚡ Energy Analysis Application")
st.markdown("*Powered by Streamlit Cloud*")

# Sidebar for navigation
st.sidebar.title("🔧 Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["🏠 Dashboard", "🌤️ Weather Data", "⚡ Energy Data", "📊 Analysis", "📈 Energy Fits", "📅 Scheduler"]
)

# Helper function to get weather data from API
def get_weather_data(city, days=7):
    """Get weather data from a free weather API"""
    try:
        # Using OpenWeatherMap API (you can get a free API key)
        # For demo purposes, we'll simulate the API response
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        
        # Simulate realistic weather data based on city
        city_temps = {
            "Toronto": 15, "Vancouver": 12, "Montreal": 10, 
            "Calgary": 8, "Ottawa": 12, "New York": 18,
            "London": 14, "Paris": 16, "Berlin": 13
        }
        
        base_temp = city_temps.get(city, 15)
        
        weather_data = pd.DataFrame({
            'Date': dates,
            'Location': city,
            'Temperature (°C)': np.random.normal(base_temp, 5, days),
            'Humidity (%)': np.random.normal(65, 15, days).clip(20, 100),
            'Wind Speed (km/h)': np.random.exponential(12, days).clip(0, 50),
            'Solar Radiation (W/m²)': np.random.normal(400, 150, days).clip(0, 1000)
        })
        
        return weather_data
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return pd.DataFrame()

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
    
    # Recent data overview
    st.subheader("📈 Recent Energy Consumption")
    
    if not st.session_state.energy_data.empty:
        # Plot recent energy consumption
        fig = px.line(st.session_state.energy_data, 
                     x='Timestamp', 
                     y='Power (kW)', 
                     color='Equipment ID' if 'Equipment ID' in st.session_state.energy_data.columns else None,
                     title='Energy Consumption Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.subheader("📋 Recent Data")
        st.dataframe(st.session_state.energy_data.tail(10), use_container_width=True)
    else:
        st.info("No energy data available. Please add some data in the Energy Data section.")
        
        # Quick start buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎲 Generate Sample Energy Data", type="primary"):
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
        
        with col2:
            if st.button("🌤️ Get Sample Weather Data", type="secondary"):
                sample_weather = get_weather_data("Toronto", 7)
                if not sample_weather.empty:
                    st.session_state.weather_data = sample_weather
                    st.success("✅ Sample weather data loaded!")
                    st.rerun()

# Weather Data Section
elif page == "🌤️ Weather Data":
    st.header("🌤️ Weather Data Management")
    
    # Weather API section
    st.subheader("🌐 Get Weather Data from API")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        city = st.selectbox("📍 Select City", 
                           ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", 
                            "New York", "London", "Paris", "Berlin"])
    
    with col2:
        days = st.slider("📅 Number of Days", 1, 30, 7)
    
    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🔄 Fetch Weather Data", type="primary"):
            with st.spinner("Fetching weather data..."):
                weather_data = get_weather_data(city, days)
                if not weather_data.empty:
                    # Append to existing data or replace
                    if st.session_state.weather_data.empty:
                        st.session_state.weather_data = weather_data
                    else:
                        st.session_state.weather_data = pd.concat([st.session_state.weather_data, weather_data], ignore_index=True)
                    
                    st.success(f"✅ Retrieved {len(weather_data)} days of weather data for {city}")
                    st.rerun()
    
    # Manual data entry
    st.subheader("📝 Manual Data Entry")
    
    with st.expander("Add Weather Data Manually"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input("📍 Location", value="Toronto")
            temperature = st.number_input("🌡️ Temperature (°C)", value=20.0, step=0.1)
        
        with col2:
            humidity = st.number_input("💧 Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, step=0.1)
            wind_speed = st.number_input("💨 Wind Speed (km/h)", value=15.0, min_value=0.0, step=0.1)
        
        with col3:
            solar_radiation = st.number_input("☀️ Solar Radiation (W/m²)", value=500.0, min_value=0.0, step=1.0)
            date = st.date_input("📅 Date", datetime.now().date())
        
        if st.button("💾 Add Weather Record", type="secondary"):
            new_record = pd.DataFrame({
                'Date': [pd.to_datetime(date)],
                'Location': [location],
                'Temperature (°C)': [temperature],
                'Humidity (%)': [humidity],
                'Wind Speed (km/h)': [wind_speed],
                'Solar Radiation (W/m²)': [solar_radiation]
            })
            
            if st.session_state.weather_data.empty:
                st.session_state.weather_data = new_record
            else:
                st.session_state.weather_data = pd.concat([st.session_state.weather_data, new_record], ignore_index=True)
            
            st.success("✅ Weather record added!")
            st.rerun()
    
    # Display existing weather data
    if not st.session_state.weather_data.empty:
        st.subheader("📊 Weather Data Analysis")
        
        # Location filter
        locations = st.session_state.weather_data['Location'].unique()
        selected_location = st.selectbox("Filter by location:", ["All"] + list(locations))
        
        # Filter data
        if selected_location == "All":
            filtered_data = st.session_state.weather_data
        else:
            filtered_data = st.session_state.weather_data[st.session_state.weather_data['Location'] == selected_location]
        
        if not filtered_data.empty:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_temp = filtered_data['Temperature (°C)'].mean()
                st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C")
            
            with col2:
                avg_humidity = filtered_data['Humidity (%)'].mean()
                st.metric("💧 Avg Humidity", f"{avg_humidity:.1f}%")
            
            with col3:
                avg_wind = filtered_data['Wind Speed (km/h)'].mean()
                st.metric("💨 Avg Wind Speed", f"{avg_wind:.1f} km/h")
            
            with col4:
                avg_solar = filtered_data['Solar Radiation (W/m²)'].mean()
                st.metric("☀️ Avg Solar Radiation", f"{avg_solar:.0f} W/m²")
            
            # Visualization
            st.subheader("📈 Weather Trends")
            plot_var = st.selectbox("Select variable to plot:", 
                                   ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Solar Radiation (W/m²)'])
            
            fig = px.line(filtered_data, x='Date', y=plot_var, 
                         color='Location' if selected_location == "All" else None,
                         title=f'{plot_var} Over Time')
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("📋 Weather Data Table")
            st.dataframe(filtered_data, use_container_width=True)
            
            # Download button
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Weather Data as CSV",
                data=csv,
                file_name=f"weather_data_{selected_location}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No weather data found for the selected location.")
    else:
        st.info("No weather data available. Use the options above to add weather data.")

# Energy Data Section
elif page == "⚡ Energy Data":
    st.header("⚡ Energy Data Management")
    
    # File upload section
    st.subheader("📁 Upload Energy Data")
    uploaded_file = st.file_uploader(
        "Upload Energy Data CSV file", 
        type=['csv'],
        help="Upload a CSV file containing energy consumption data"
    )
    
    if uploaded_file is not None:
        try:
            energy_data = pd.read_csv(uploaded_file)
            st.session_state.energy_data = energy_data
            st.success(f"✅ Energy data uploaded successfully! ({len(energy_data)} records)")
            
            # Data preview
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("👀 Data Preview")
                st.dataframe(energy_data.head(10), use_container_width=True)
            
            with col2:
                st.subheader("📊 Data Info")
                st.write(f"**Rows:** {len(energy_data)}")
                st.write(f"**Columns:** {len(energy_data.columns)}")
                st.write("**Column Names:**")
                for col in energy_data.columns:
                    st.write(f"- {col}")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Manual data entry
    st.subheader("📝 Manual Data Entry")
    
    with st.expander("Add Energy Data Manually"):
        col1, col2 = st.columns(2)
        
        with col1:
            equipment_id = st.text_input("🏭 Equipment ID", value="HVAC_01")
            power = st.number_input("🔌 Power (kW)", value=1500.0, min_value=0.0, step=0.1)
        
        with col2:
            energy = st.number_input("⚡ Energy (kWh)", value=1500.0, min_value=0.0, step=0.1)
            timestamp = st.datetime_input("📅 Timestamp", datetime.now())
        
        if st.button("💾 Add Energy Record", type="secondary"):
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
    
    # Display existing energy data
    if not st.session_state.energy_data.empty:
        st.subheader("📊 Energy Data Analysis")
        
        # Equipment filter
        if 'Equipment ID' in st.session_state.energy_data.columns:
            equipment_list = st.session_state.energy_data['Equipment ID'].unique()
            selected_equipment = st.selectbox("Filter by equipment:", ["All"] + list(equipment_list))
            
            # Filter data
            if selected_equipment == "All":
                filtered_data = st.session_state.energy_data
            else:
                filtered_data = st.session_state.energy_data[st.session_state.energy_data['Equipment ID'] == selected_equipment]
        else:
            filtered_data = st.session_state.energy_data
            selected_equipment = "All"
        
        if not filtered_data.empty:
            # Summary statistics
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                col1, col2, col3 = st.columns(3)
                
                if 'Energy (kWh)' in filtered_data.columns:
                    with col1:
                        total_energy = filtered_data['Energy (kWh)'].sum()
                        st.metric("⚡ Total Energy", f"{total_energy:.1f} kWh")
                
                if 'Power (kW)' in filtered_data.columns:
                    with col2:
                        avg_power = filtered_data['Power (kW)'].mean()
                        st.metric("🔌 Avg Power", f"{avg_power:.1f} kW")
                    
                    with col3:
                        max_power = filtered_data['Power (kW)'].max()
                        st.metric("📈 Peak Power", f"{max_power:.1f} kW")
            
            # Visualization
            st.subheader("📈 Energy Visualization")
            
            # Select columns for plotting
            if len(numeric_cols) > 0:
                y_column = st.selectbox("Y-axis (numeric)", numeric_cols)
                
                if 'Timestamp' in filtered_data.columns:
                    x_column = 'Timestamp'
                else:
                    x_column = st.selectbox("X-axis", filtered_data.columns.tolist())
                
                # Create plot
                if 'Equipment ID' in filtered_data.columns and selected_equipment == "All":
                    fig = px.line(filtered_data, x=x_column, y=y_column, color='Equipment ID',
                                 title=f'{y_column} vs {x_column}')
                else:
                    fig = px.line(filtered_data, x=x_column, y=y_column,
                                 title=f'{y_column} vs {x_column}')
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("📋 Energy Data Table")
            st.dataframe(filtered_data, use_container_width=True)
            
            # Download button
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Energy Data as CSV",
                data=csv,
                file_name=f"energy_data_{selected_equipment}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No energy data found for the selected equipment.")
    else:
        st.info("No energy data available. Upload a file or add data manually above.")

# Analysis Section
elif page == "📊 Analysis":
    st.header("📊 Energy Analysis")
    
    if not st.session_state.energy_data.empty:
        # Time-based analysis
        st.subheader("📈 Time-based Analysis")
        
        energy_df = st.session_state.energy_data.copy()
        
        # Ensure timestamp column exists and is datetime
        if 'Timestamp' in energy_df.columns:
            energy_df['Timestamp'] = pd.to_datetime(energy_df['Timestamp'])
            energy_df['Date'] = energy_df['Timestamp'].dt.date
            energy_df['Hour'] = energy_df['Timestamp'].dt.hour
            
            # Daily analysis
            if 'Energy (kWh)' in energy_df.columns:
                daily_energy = energy_df.groupby('Date')['Energy (kWh)'].sum().reset_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Daily Energy Consumption")
                    fig = px.bar(daily_energy, x='Date', y='Energy (kWh)',
                                title='Daily Energy Consumption')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("⏰ Hourly Energy Pattern")
                    hourly_energy = energy_df.groupby('Hour')['Energy (kWh)'].mean().reset_index()
                    fig = px.line(hourly_energy, x='Hour', y='Energy (kWh)',
                                 title='Average Energy Consumption by Hour')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Equipment comparison
        if 'Equipment ID' in energy_df.columns and 'Power (kW)' in energy_df.columns:
            st.subheader("🏭 Equipment Comparison")
            
            equipment_stats = energy_df.groupby('Equipment ID').agg({
                'Power (kW)': ['mean', 'max', 'std'],
                'Energy (kWh)': 'sum' if 'Energy (kWh)' in energy_df.columns else 'count'
            }).round(2)
            
            equipment_stats.columns = ['Avg Power (kW)', 'Max Power (kW)', 'Power Std Dev', 'Total Energy (kWh)']
            st.dataframe(equipment_stats, use_container_width=True)
            
            # Equipment power comparison chart
            avg_power = energy_df.groupby('Equipment ID')['Power (kW)'].mean().reset_index()
            fig = px.bar(avg_power, x='Equipment ID', y='Power (kW)',
                        title='Average Power Consumption by Equipment')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis with weather data
        if not st.session_state.weather_data.empty:
            st.subheader("🌤️ Weather vs Energy Correlation")
            
            # Try to merge weather and energy data by date
            weather_df = st.session_state.weather_data.copy()
            weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.date
            
            if 'Timestamp' in energy_df.columns:
                energy_daily = energy_df.groupby('Date').agg({
                    'Power (kW)': 'mean',
                    'Energy (kWh)': 'sum' if 'Energy (kWh)' in energy_df.columns else 'count'
                }).reset_index()
                
                # Merge datasets
                merged_df = pd.merge(weather_df, energy_daily, on='Date', how='inner')
                
                if not merged_df.empty:
                    # Correlation matrix
                    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        corr_matrix = merged_df[numeric_cols].corr()
                        
                        fig = px.imshow(corr_matrix, 
                                       title="Correlation Matrix: Weather vs Energy",
                                       color_continuous_scale="RdBu",
                                       aspect="auto")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Scatter plots
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'Temperature (°C)' in merged_df.columns and 'Power (kW)' in merged_df.columns:
                                fig = px.scatter(merged_df, x='Temperature (°C)', y='Power (kW)',
                                               title='Temperature vs Power Consumption')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if 'Solar Radiation (W/m²)' in merged_df.columns and 'Power (kW)' in merged_df.columns:
                                fig = px.scatter(merged_df, x='Solar Radiation (W/m²)', y='Power (kW)',
                                               title='Solar Radiation vs Power Consumption')
                                st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No matching dates found between weather and energy data.")
    else:
        st.info("No energy data available for analysis. Please add energy data first.")

# Energy Fits Section
elif page == "📈 Energy Fits":
    st.header("📈 Energy Fits & Modeling")
    
    if not st.session_state.energy_data.empty:
        energy_df = st.session_state.energy_data.copy()
        numeric_cols = energy_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.subheader("🔧 Data Fitting")
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X variable:", numeric_cols)
                fit_type = st.selectbox("Fit type:", ["Linear", "Polynomial", "Exponential"])
                
                if fit_type == "Polynomial":
                    degree = st.slider("Polynomial degree:", 1, 5, 2)
            
            with col2:
                y_var = st.selectbox("Y variable:", [col for col in numeric_cols if col != x_var])
            
            if st.button("🔄 Perform Fit", type="primary"):
                # Get clean data (no NaN values)
                clean_data = energy_df[[x_var, y_var]].dropna()
                
                if len(clean_data) > 1:
                    x = clean_data[x_var].values
                    y = clean_data[y_var].values
                    
                    # Perform fitting
                    if fit_type == "Linear":
                        coeffs = np.polyfit(x, y, 1)
                        y_fit = np.polyval(coeffs, x)
                        equation = f"y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}"
                    
                    elif fit_type == "Polynomial":
                        coeffs = np.polyfit(x, y, degree)
                        y_fit = np.polyval(coeffs, x)
                        equation = f"Polynomial degree {degree}"
                    
                    elif fit_type == "Exponential":
                        # Simple exponential fit: y = a * exp(b * x)
                        try:
                            log_y = np.log(np.maximum(y, 1e-10))  # Avoid log(0)
                            coeffs = np.polyfit(x, log_y, 1)
                            y_fit = np.exp(coeffs[1]) * np.exp(coeffs[0] * x)
                            equation = f"y = {np.exp(coeffs[1]):.3f} * exp({coeffs[0]:.3f} * x)"
                        except:
                            st.error("Exponential fit failed. Try linear or polynomial.")
                             st.stop()  # Stop execution here instead of continue
                    
                    # Calculate R-squared
                    ss_res = np.sum((y - y_fit) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((y - y_fit) ** 2))
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📊 R-squared", f"{r_squared:.4f}")
                    
                    with col2:
                        st.metric("📏 RMSE", f"{rmse:.4f}")
                    
                    with col3:
                        st.metric("📈 Data Points", len(clean_data))
                    
                    st.write(f"**Equation:** {equation}")
                    
                    # Plot results
                    fig = go.Figure()
                    
                    # Original data
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode='markers', name='Data',
                        marker=dict(color='blue', size=6)
                    ))
                    
                    # Fitted line
                    x_sorted = np.sort(x)
                    if fit_type == "Linear":
                        y_line = np.polyval(coeffs, x_sorted)
                    elif fit_type == "Polynomial":
                        y_line = np.polyval(coeffs, x_sorted)
                    elif fit_type == "Exponential":
                        y_line = np.exp(coeffs[1]) * np.exp(coeffs[0] * x_sorted)
                    
                    fig.add_trace(go.Scatter(
                        x=x_sorted, y=y_line, mode='lines', name='Fit',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f'{fit_type} Fit: {y_var} vs {x_var}',
                        xaxis_title=x_var,
                        yaxis_title=y_var
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Residuals plot
                    residuals = y - y_fit
                    fig_res = px.scatter(x=y_fit, y=residuals, 
                                        title='Residuals Plot',
                                        labels={'x': 'Fitted Values', 'y': 'Residuals'})
                    fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_res, use_container_width=True)
                
                else:
                    st.error("Not enough data points for fitting.")
        else:
            st.info("Need at least 2 numeric columns for fitting analysis.")
    else:
        st.info("No energy data available for fitting. Please add energy data first.")

# Scheduler Section
elif page == "📅 Scheduler":
    st.header("📅 Equipment Scheduler")
    
    st.subheader("⏰ Equipment Runtime Optimization")
    
    # Equipment selection
    if not st.session_state.energy_data.empty and 'Equipment ID' in st.session_state.energy_data.columns:
        equipment_list = st.session_state.energy_data['Equipment ID'].unique()
        selected_eq = st.selectbox("🏭 Select Equipment", equipment_list)
        
        # Get average power for selected equipment
        avg_power = st.session_state.energy_data[
            st.session_state.energy_data['Equipment ID'] == selected_eq
        ]['Power (kW)'].mean()
        suggested_power = avg_power if not np.isnan(avg_power) else 1500.0
    else:
        selected_eq = st.text_input("🏭 Equipment ID", value="HVAC_01")
        suggested_power = 1500.0
    
    # Schedule parameters
    col1, col2 = st.columns(2)
    
    with col1:
        start_time = st.time_input("⏰ Start Time", value=datetime.strptime("08:00", "%H:%M").time())
        operating_hours = st.number_input("⏱️ Operating Hours/Day", value=8.0, min_value=0.0, max_value=24.0, step=0.5)
    
    with col2:
        power_rating = st.number_input("🔌 Power Rating (kW)", value=float(suggested_power), min_value=0.0, step=0.1)
        efficiency = st.slider("⚡ Efficiency (%)", 0, 100, 85)
    
    # Days of operation
    st.subheader("📅 Operating Schedule")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    operating_days = []
    
    cols = st.columns(7)
    for i, day in enumerate(days):
        with cols[i]:
            if st.checkbox(day, value=(i < 5), key=f"day_{i}"):  # Default to weekdays
                operating_days.append(day)
    
    # Cost parameters
    st.subheader("💰 Cost Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        electricity_rate = st.number_input("💡 Electricity Rate ($/kWh)", value=0.12, min_value=0.0, step=0.01)
    
    with col2:
        carbon_factor = st.number_input("🌱 Carbon Factor (kg CO₂/kWh)", value=0.5, min_value=0.0, step=0.01)
    
    if st.button("📊 Calculate Schedule Optimization", type="primary"):
        # Calculate energy consumption
        daily_energy = power_rating * operating_hours * (efficiency / 100)
        weekly_energy = daily_energy * len(operating_days)
        monthly_energy = weekly_energy * 4.33  # Average weeks per month
        annual_energy = weekly_energy * 52
        
        # Calculate costs
        daily_cost = daily_energy * electricity_rate
        monthly_cost = monthly_energy * electricity_rate
        annual_cost = annual_energy * electricity_rate
        
        # Calculate carbon footprint
        daily_carbon = daily_energy * carbon_factor
        annual_carbon = annual_energy * carbon_factor
        
        # Display results
        st.subheader("📊 Energy Consumption Forecast")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📅 Daily Energy", f"{daily_energy:.1f} kWh")
            st.metric("💰 Daily Cost", f"${daily_cost:.2f}")
        
        with col2:
            st.metric("📅 Weekly Energy", f"{weekly_energy:.1f} kWh")
            st.metric("🌱 Daily Carbon", f"{daily_carbon:.1f} kg CO₂")
        
        with col3:
            st.metric("📅 Monthly Energy", f"{monthly_energy:.1f} kWh")
            st.metric("💰 Monthly Cost", f"${monthly_cost:.2f}")
        
        with col4:
            st.metric("📅 Annual Energy", f"{annual_energy:.0f} kWh")
            st.metric("💰 Annual Cost", f"${annual_cost:.2f}")
        
        # Optimization suggestions
        st.subheader("🎯 Optimization Suggestions")
        
        # Simple optimization scenarios
        scenarios = {
            "Current Schedule": {
                "hours": operating_hours,
                "days": len(operating_days),
                "efficiency": efficiency,
                "annual_energy": annual_energy,
                "annual_cost": annual_cost
            },
            "Reduced Hours (-1h/day)": {
                "hours": max(0, operating_hours - 1),
                "days": len(operating_days),
                "efficiency": efficiency,
                "annual_energy": annual_energy * (max(0, operating_hours - 1) / operating_hours) if operating_hours > 0 else 0,
                "annual_cost": annual_cost * (max(0, operating_hours - 1) / operating_hours) if operating_hours > 0 else 0
            },
            "Improved Efficiency (+10%)": {
                "hours": operating_hours,
                "days": len(operating_days),
                "efficiency": min(100, efficiency + 10),
                "annual_energy": annual_energy * (min(100, efficiency + 10) / efficiency) if efficiency > 0 else 0,
                "annual_cost": annual_cost * (min(100, efficiency + 10) / efficiency) if efficiency > 0 else 0
            },
            "Weekdays Only": {
                "hours": operating_hours,
                "days": 5,
                "efficiency": efficiency,
                "annual_energy": annual_energy * (5 / len(operating_days)) if len(operating_days) > 0 else 0,
                "annual_cost": annual_cost * (5 / len(operating_days)) if len(operating_days) > 0 else 0
            }
        }
        
        scenario_df = pd.DataFrame(scenarios).T
        scenario_df['Annual Savings (kWh)'] = scenario_df['annual_energy'].iloc[0] - scenario_df['annual_energy']
        scenario_df['Annual Savings ($)'] = scenario_df['annual_cost'].iloc[0] - scenario_df['annual_cost']
        
        # Format the dataframe for display
        display_df = scenario_df[['hours', 'days', 'efficiency', 'annual_energy', 'annual_cost', 'Annual Savings ($)']].round(1)
        display_df.columns = ['Hours/Day', 'Days/Week', 'Efficiency (%)', 'Annual Energy (kWh)', 'Annual Cost ($)', 'Annual Savings ($)']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Visualization of scenarios
        fig = px.bar(scenario_df.reset_index(), x='index', y='annual_cost',
                    title='Annual Cost Comparison by Scenario')
        fig.update_xaxis(title='Scenario')
        fig.update_yaxis(title='Annual Cost ($)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Schedule summary
        st.subheader("📋 Current Schedule Summary")
        schedule_data = {
            'Parameter': [
                'Equipment ID', 'Operating Days', 'Daily Hours', 'Start Time', 
                'Power Rating', 'Efficiency', 'Daily Energy', 'Annual Carbon Footprint'
            ],
            'Value': [
                selected_eq,
                f"{len(operating_days)} days ({', '.join(operating_days)})",
                f"{operating_hours} hours",
                start_time.strftime("%H:%M"),
                f"{power_rating} kW",
                f"{efficiency}%",
                f"{daily_energy:.1f} kWh",
                f"{annual_carbon:.1f} kg CO₂"
            ]
        }
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🔧 Powered by**")
st.sidebar.markdown("🎈 Streamlit Cloud")
st.sidebar.markdown("📊 Plotly")

# Data management
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Data Management**")

if st.sidebar.button("🗑️ Clear All Data"):
    st.session_state.weather_data = pd.DataFrame()
    st.session_state.energy_data = pd.DataFrame()
    st.sidebar.success("All data cleared!")
    st.rerun()

if st.sidebar.button("📥 Download All Data"):
    if not st.session_state.weather_data.empty or not st.session_state.energy_data.empty:
        # Create a zip file with all data
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            if not st.session_state.weather_data.empty:
                weather_csv = st.session_state.weather_data.to_csv(index=False)
                zip_file.writestr("weather_data.csv", weather_csv)
            
            if not st.session_state.energy_data.empty:
                energy_csv = st.session_state.energy_data.to_csv(index=False)
                zip_file.writestr("energy_data.csv", energy_csv)
        
        st.sidebar.download_button(
            label="📦 Download ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"energy_analysis_data_{datetime.now().strftime('%Y%m%d')}.zip",
            mime="application/zip"
        )

# Debug info
if st.sidebar.checkbox("🐛 Debug Mode"):
    st.sidebar.write("**Session State:**")
    st.sidebar.write(f"Weather records: {len(st.session_state.weather_data)}")
    st.sidebar.write(f"Energy records: {len(st.session_state.energy_data)}")
