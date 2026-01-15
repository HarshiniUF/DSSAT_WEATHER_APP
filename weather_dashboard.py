# enhanced_weather_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from io import StringIO, BytesIO
import requests
import folium
from streamlit_folium import st_folium
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable

# Page configuration
st.set_page_config(
    page_title="Weather Data Explorer",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== WEATHER DATA GENERATION FUNCTIONS ====================

def get_location_details(latitude: float, longitude: float):
    """
    Performs reverse geocoding to get address details from coordinates using geopy.
    Returns a dictionary with address, country name, state name, and country code.
    """
    try:
        # Initialize the geolocator with a unique user_agent
        geolocator = Nominatim(user_agent="dssat_weather_script_by_harshini")
        
        # Perform the reverse geocoding lookup
        location = geolocator.reverse((latitude, longitude), exactly_one=True, language='en')
        
        if location and location.address:
            address_details = location.raw.get('address', {})
            city = address_details.get('city', address_details.get('town', address_details.get('village')))
            state = address_details.get('state')
            country = address_details.get('country')
            country_code = address_details.get('country_code', 'XX').upper()
            
            # Construct a clean address string
            parts = [part for part in [city, state, country] if part]
            address = ', '.join(parts)
            
            return {'address': address, 'country': country, 'state': state, 'country_code': country_code}
            
    except Exception:
        pass
        
    # Fallback dictionary
    return {'address': f"Site at {latitude:.2f}, {longitude:.2f}", 'country': None, 'state': None, 'country_code': 'NOGEO'}

def normalize_longitude(lng):
    """Normalize longitude to -180 to 180 range"""
    while lng > 180:
        lng -= 360
    while lng < -180:
        lng += 360
    return lng

@st.cache_data(ttl=3600)
def generate_wth_from_open_meteo(latitude, longitude, start_date, end_date):
    """
    Fetches data from Open-Meteo's Historical Archive API and formats it into
    DSSAT-compliant .WTH file string using Geopy for site naming.
    """
    # Get detailed location information
    location_info = get_location_details(latitude, longitude)
    site_name = location_info['address']
    country_name = location_info.get('country')
    state_name = location_info.get('state')

    # Generate Dynamic INSI code
    if country_name:
        country_part = country_name[:2].upper()
        state_part = state_name[:2].upper() if state_name else country_part
        insi = country_part + state_part
    else:
        insi = "OMET"

    api_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "shortwave_radiation_sum,temperature_2m_max,temperature_2m_min,precipitation_sum,dewpoint_2m_mean,wind_speed_10m_mean",
        "timezone": "auto"
    }
    
    try:
        with st.spinner("Fetching weather data from Open-Meteo API..."):
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Request Failed: {e}")
        return None, None, "API request failed."
    
    if 'daily' not in data or not data['daily']:
        st.error("⚠️ The API did not return any daily data.")
        return None, None, "API returned no data."

    # Create DataFrame
    df = pd.DataFrame(data['daily']).rename(columns={
        "time": "date",
        "shortwave_radiation_sum": "SRAD",
        "temperature_2m_max": "TMAX",
        "temperature_2m_min": "TMIN",
        "precipitation_sum": "RAIN",
        "dewpoint_2m_mean": "DEWP",
        "wind_speed_10m_mean": "WIND"
    })
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.fillna(-99.0, inplace=True)
    
    df['daily_mean'] = (df['TMAX'] + df['TMIN']) / 2
    valid_temps = df[df['TMAX'] != -99.0]
    TAV = valid_temps['daily_mean'].mean() if len(valid_temps) > 0 else -99.0
    
    df['month'] = df.index.month
    monthly_means = df.groupby('month')['daily_mean'].mean()
    AMP = monthly_means.max() - monthly_means.min() if len(monthly_means) > 0 else -99.0
    
    try:
        elev_response = requests.get(
            f"https://api.open-elevation.com/api/v1/lookup?locations={latitude},{longitude}",
            timeout=10
        )
        elev = elev_response.json()['results'][0]['elevation']
    except Exception:
        elev = -99.0
    
    df['PAR'] = (df['SRAD'] * 0.45).where(df['SRAD'] != -99.0, -99.0)
    
    for col in ['SRAD', 'TMAX', 'TMIN', 'RAIN', 'DEWP', 'WIND', 'PAR']:
        if col in df.columns:
            df[col] = df[col].round(1)
    
    # Create DSSAT weather file content
    # Header logic corrected to use geopy site_name and dynamic INSI
    header_lines = [
        f"$WEATHER DATA : {site_name}",
        "",
        "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT",
        f"  {insi:<4} {latitude:8.3f} {longitude:8.3f} {elev:5.0f} {TAV:5.1f} {AMP:5.1f}   2.0  10.0",
        "",
        "@  DATE  SRAD  TMAX  TMIN  RAIN  DEWP  WIND   PAR"
    ]
    
    df['year'] = df.index.year
    df['doy'] = df.index.dayofyear
    df['DATE'] = df['year'] * 1000 + df['doy']

    data_lines = []
    for idx, row in df.iterrows():
        data_lines.append(
            f"{row['DATE']:7.0f} {row['SRAD']:5.1f} {row['TMAX']:5.1f} {row['TMIN']:5.1f} "
            f"{row['RAIN']:5.1f} {row['DEWP']:5.1f} {row['WIND']:5.1f} {row['PAR']:5.1f}"
        )

    # Add the Data Source Note here
    footer_note = [
        "",
        "! Note: Weather data sourced from Open-Meteo Historical Weather API",
        "! Data includes reanalysis from ERA5 and other global meteorological models.",
        f"! Generated on: {date.today()}"
    ]
    
    wth_content = "\n".join(header_lines + data_lines + footer_note)
    
    report_lines = ["🔍 QUALITY ASSURANCE REPORT", "=" * 50]
    issues = []
    warnings = []
    
    temp_issues = ((df['TMAX'] < df['TMIN']) & (df['TMAX'] != -99.0) & (df['TMIN'] != -99.0)).sum()
    if temp_issues > 0:
        issues.append(f"❌ Found {temp_issues} days where TMAX < TMIN")
    else:
        report_lines.append("✅ TMAX >= TMIN check passed")
    
    for col in ['SRAD', 'TMAX', 'TMIN', 'RAIN']:
        missing = (df[col] == -99.0).sum()
        if missing > 0:
            pct = (missing / len(df)) * 100
            if pct > 10:
                issues.append(f"❌ {col}: {missing} missing values ({pct:.1f}%)")
            else:
                warnings.append(f"⚠️ {col}: {missing} missing values ({pct:.1f}%)")
    
    if not issues and not warnings:
        report_lines.append("✅ No missing values detected")
    
    report_lines.extend(issues)
    report_lines.extend(warnings)
    report_lines.append(f"\n📊 Total records: {len(df)}")
    report_lines.append(f"📅 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    report = "\n".join(report_lines)
    
    return wth_content, df, report

# ==================== DATA LOADING AND PROCESSING ====================

def load_data_from_wth(file_like_object):
    """Load and parse DSSAT .WTH file"""
    content = file_like_object.getvalue().decode("utf-8")
    stringio = StringIO(content)
    data_records = []
    is_data_section = False
    
    for line in stringio:
        if line.strip().startswith('@') and 'DATE' in line:
            is_data_section = True
            continue
        
        if not is_data_section or not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 5:
            try:
                date_str = parts[0]
                if len(date_str) == 5:
                    year_short = int(date_str[:2])
                    year = 2000 + year_short if year_short < 70 else 1900 + year_short
                    day_of_year = int(date_str[2:])
                elif len(date_str) == 7:
                    year = int(date_str[:4])
                    day_of_year = int(date_str[4:])
                else:
                    continue

                date = pd.to_datetime(f'{year}-01-01') + pd.to_timedelta(day_of_year - 1, unit='d')
                
                record = {
                    'date': date,
                    'SRAD': float(parts[1]),
                    'TMAX': float(parts[2]),
                    'TMIN': float(parts[3]),
                    'RAIN': float(parts[4])
                }
                
                if len(parts) > 5:
                    record['DEWP'] = float(parts[5])
                if len(parts) > 6:
                    record['WIND'] = float(parts[6])
                if len(parts) > 7:
                    record['PAR'] = float(parts[7])
                
                data_records.append(record)
            except (ValueError, IndexError):
                continue
    
    if not data_records:
        return None
    
    df = pd.DataFrame(data_records)
    df.set_index('date', inplace=True)
    return df

def get_filtered_data(df, selected_year, selected_month, selected_day):
    """Filter data based on user selections"""
    filtered_df = df.copy()
    
    if selected_year != "All Years":
        filtered_df = filtered_df[filtered_df.index.year == int(selected_year)]
    
    if selected_month != "All Months":
        month_num = list(calendar.month_abbr).index(selected_month)
        filtered_df = filtered_df[filtered_df.index.month == month_num]
    
    if selected_day != "All Days":
        filtered_df = filtered_df[filtered_df.index.day == int(selected_day)]
    
    return filtered_df

# ==================== VISUALIZATION FUNCTIONS ====================

def create_monthly_boxplot(df, variable, title, color):
    """Create interactive monthly box plot using Plotly with dynamic scaling"""
    df_copy = df.copy()
    df_copy['month'] = df_copy.index.month
    df_copy['month_name'] = df_copy['month'].apply(lambda x: calendar.month_abbr[x])
    
    df_plot = df_copy[df_copy[variable] != -99.0]
    
    fig = go.Figure()
    
    for month_num in range(1, 13):
        month_name = calendar.month_abbr[month_num]
        month_data = df_plot[df_plot['month'] == month_num][variable]
        
        if len(month_data) > 0:
            fig.add_trace(go.Box(
                y=month_data,
                name=month_name,
                marker_color=color,
                boxmean='sd'
            ))
    
    # Dynamic Y-axis: max value + 10
    max_val = df_plot[variable].max() if not df_plot.empty else 40
    y_limit = max_val + 10

    fig.update_layout(
        title=title,
        yaxis_title="Temperature (°C)" if variable in ['TMAX', 'TMIN'] else "Value",
        xaxis_title="Month",
        height=400,
        showlegend=False,
        hovermode='closest',
        # yaxis=dict(range=[0, y_limit], dtick=10)
        yaxis=dict(range=[df_plot[variable].min() - 5, df_plot[variable].max() + 5])
    )
    
    return fig

def create_daily_line_chart(df, title):
    """Create interactive daily temperature line chart with outlier detection and dynamic scaling"""
    df_plot = df[(df['TMAX'] != -99.0) & (df['TMIN'] != -99.0)].copy()
    
    fig = go.Figure()
    
    # Outlier Detection Logic (IQR Method)
    def get_outlier_mask(series):
        if len(series) < 4: return pd.Series([False]*len(series))
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))

    # TMAX Line
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['TMAX'],
        mode='lines+markers', name='TMAX',
        line=dict(color='#FF6347', width=2),
        marker=dict(size=4)
    ))
    
    # TMAX Outliers
    outliers_max = df_plot[get_outlier_mask(df_plot['TMAX'])]
    if not outliers_max.empty:
        fig.add_trace(go.Scatter(
            x=outliers_max.index, y=outliers_max['TMAX'],
            mode='markers', name='TMAX Outlier',
            marker=dict(color='red', size=8, symbol='circle-open', line=dict(width=2))
        ))

    # TMIN Line
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['TMIN'],
        mode='lines+markers', name='TMIN',
        line=dict(color='#4682B4', width=2),
        marker=dict(size=4)
    ))
    
    # TMIN Outliers
    outliers_min = df_plot[get_outlier_mask(df_plot['TMIN'])]
    if not outliers_min.empty:
        fig.add_trace(go.Scatter(
            x=outliers_min.index, y=outliers_min['TMIN'],
            mode='markers', name='TMIN Outlier',
            marker=dict(color='darkred', size=8, symbol='x-open', line=dict(width=2))
        ))
    
    # Dynamic Y-axis: overall max + 10
    overall_max = df_plot[['TMAX', 'TMIN']].max().max() if not df_plot.empty else 40
    y_limit = overall_max + 10

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        height=400,
        hovermode='x unified',
        # yaxis=dict(range=[0, y_limit], dtick=10)
        yaxis=dict(range=[df_plot[['TMAX', 'TMIN']].min().min() - 5, y_limit])
    )
    
    return fig

def create_rainfall_chart(df, selected_month):
    """Create rainfall bar chart with dynamic scaling"""
    df_plot = df[df['RAIN'] != -99.0].copy()
    
    fig = go.Figure()
    if selected_month == "All Months":
        monthly_rain = df_plot['RAIN'].resample('M').sum().reset_index()
        monthly_rain['month_label'] = monthly_rain['date'].dt.strftime('%Y-%b')
        
        fig.add_trace(go.Bar(
            x=monthly_rain['month_label'], y=monthly_rain['RAIN'],
            marker_color='#00CED1', name='Rainfall'
        ))
        title_text = 'Total Monthly Rainfall'
        max_val = monthly_rain['RAIN'].max() if not monthly_rain.empty else 0
    else:
        fig.add_trace(go.Bar(
            x=df_plot.index, y=df_plot['RAIN'],
            marker_color='#00CED1', name='Rainfall'
        ))
        title_text = 'Daily Rainfall'
        max_val = df_plot['RAIN'].max() if not df_plot.empty else 0

    y_limit = max_val + 10
    fig.update_layout(
        title=title_text,
        xaxis_title='Date',
        yaxis_title='Rainfall (mm)',
        height=400,
        # yaxis=dict(range=[0, y_limit])
        yaxis=dict(range=[0, y_limit]) # 0 because rainfall cannot be negative!
    )
    
    return fig

# ==================== MAIN APP ====================

def main():

    
    st.markdown("<h1 style='text-align: center; color: #1e3a8a;'>🌦️ Interactive Weather Data Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>This application was developed as part of the Generative AI for Agriculture (GAIA) project, funded by the Gates Foundation and the UK International Development from the UK government, in collaboration with CGIAR and the University of Florida.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: #94a3b8;'>Data Source: Open-Meteo Historical Weather API (CC BY 4.0)</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    gen_tab, upload_tab = st.tabs(["📍 Generate New Weather File", "📤 Upload Existing .WTH File"])
    
    # ==================== GENERATION TAB ====================
    with gen_tab:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🗺️ Select Location on Map")
            
            if 'lat' not in st.session_state:
                st.session_state['lat'] = 0.0
            if 'lon' not in st.session_state:
                st.session_state['lon'] = 0.0
            
            m = folium.Map(location=[st.session_state['lat'], st.session_state['lon']], zoom_start=2)
            
            if st.session_state['lat'] != 0.0 or st.session_state['lon'] != 0.0:
                folium.Marker(
                    [st.session_state['lat'], st.session_state['lon']],
                    popup=f"Lat: {st.session_state['lat']:.4f}, Lon: {st.session_state['lon']:.4f}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            
            map_data = st_folium(m, use_container_width=True, height=400)
            
            if map_data and map_data.get('last_clicked'):
                st.session_state['lat'] = round(map_data['last_clicked']['lat'], 4)
                st.session_state['lon'] = round(normalize_longitude(map_data['last_clicked']['lng']), 4)
                st.rerun()
        
        with col2:
            st.subheader("📍 Location & Date Settings")
            
            lat_input = st.number_input(
                "Latitude",
                min_value=-90.0, max_value=90.0,
                value=st.session_state['lat'], format="%.4f"
            )
            
            lon_input = st.number_input(
                "Longitude",
                min_value=-180.0, max_value=180.0,
                value=st.session_state['lon'], format="%.4f"
            )
            
            if lat_input != st.session_state['lat']:
                st.session_state['lat'] = lat_input
            if lon_input != st.session_state['lon']:
                st.session_state['lon'] = lon_input
            
            st.markdown("---")
            
            start_date = st.date_input(
                "Start Date",
                value=date.today() - timedelta(days=366),
                max_value=date.today() - timedelta(days=1)
            )
            
            end_date = st.date_input(
                "End Date",
                value=date.today() - timedelta(days=1),
                max_value=date.today() - timedelta(days=1)
            )
            
            st.markdown("---")
            
            if st.button("🚀 Generate Weather Data", type="primary", use_container_width=True):
                if st.session_state['lat'] == 0.0 and st.session_state['lon'] == 0.0:
                    st.error("⚠️ Please select a location on the map first!")
                elif start_date >= end_date:
                    st.error("⚠️ Start date must be before end date!")
                else:
                    wth_content, df, report = generate_wth_from_open_meteo(
                        st.session_state['lat'],
                        st.session_state['lon'],
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if wth_content and df is not None:
                        st.session_state['wth_content'] = wth_content
                        st.session_state['df'] = df
                        st.session_state['report'] = report
                        # Get address for filename
                        loc_info = get_location_details(st.session_state['lat'], st.session_state['lon'])
                        st.session_state['site_name'] = loc_info['address'].replace(", ", "_").replace(" ", "_")
                        st.success("✅ Weather data generated successfully!")
                        st.rerun()
        
        if 'wth_content' in st.session_state and st.session_state['wth_content'] is not None:
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text_area("📋 Quality Assurance Report", st.session_state['report'], height=200)
            with col2:
                st.download_button(
                    label="📥 Download .WTH File",
                    data=st.session_state['wth_content'].encode('utf-8'),
                    file_name=f"{st.session_state['site_name']}.WTH",
                    mime="text/plain",
                    use_container_width=True
                )
                st.metric("Total Records", len(st.session_state['df']))
                st.metric("Location", f"{st.session_state['lat']:.2f}°, {st.session_state['lon']:.2f}°")
            
            display_analysis_dashboard(st.session_state['df'], st.session_state['site_name'])
    
    # ==================== UPLOAD TAB ====================
    with upload_tab:
        st.subheader("📤 Upload DSSAT Weather File")
        uploaded_file = st.file_uploader("Choose a .WTH file", type=['wth', 'WTH'])
        
        if uploaded_file is not None:
            df = load_data_from_wth(uploaded_file)
            if df is not None:
                site_name = uploaded_file.name.replace('.WTH', '').replace('.wth', '').replace('_', ' ')
                st.success(f"✅ Successfully loaded {len(df)} records from {uploaded_file.name}")
                display_analysis_dashboard(df, site_name)
            else:
                st.error("❌ Failed to parse the weather file.")

def display_analysis_dashboard(df, site_name):
    """Display the interactive analysis dashboard"""
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>📊 Analysis Dashboard: {site_name}</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    chart_type = st.selectbox(
        "📈 Visualization Type",
        ["Temperature Analysis", "Rainfall Analysis"],
        key=f"{site_name}_chart"
    )
    col1, col2, col3 = st.columns(3)
    
    with col1:
        years = ["All Years"] + sorted([str(y) for y in df.index.year.unique()])
        selected_year = st.selectbox("🗓️ Filter by Year", years, key=f"{site_name}_year")
    with col2:
        months = ["All Months"] + list(calendar.month_abbr[1:])
        selected_month = st.selectbox("📅 Filter by Month", months, key=f"{site_name}_month")
    with col3:
        days = ["All Days"] + [str(i) for i in range(1, 32)]
        selected_day = st.selectbox("📆 Filter by Day", days, key=f"{site_name}_day")
    
    filtered_df = get_filtered_data(df, selected_year, selected_month, selected_day)
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filter combination.")
        return
    
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    valid_tmax = filtered_df[filtered_df['TMAX'] != -99.0]['TMAX']
    valid_tmin = filtered_df[filtered_df['TMIN'] != -99.0]['TMIN']
    valid_rain = filtered_df[filtered_df['RAIN'] != -99.0]['RAIN']
    
    with col1:
        avg_tmax = valid_tmax.mean() if len(valid_tmax) > 0 else 0
        st.metric("🌡️ Avg Max Temp", f"{avg_tmax:.1f}°C")
    with col2:
        avg_tmin = valid_tmin.mean() if len(valid_tmin) > 0 else 0
        st.metric("❄️ Avg Min Temp", f"{avg_tmin:.1f}°C")
    
    with col3:
    # Count how many unique years are in the current view
    num_years = filtered_df.index.year.nunique()
    
    if selected_year == "All Years" and num_years > 1:
        # Calculate Average Annual Total for the professor
        avg_annual_rain = valid_rain.sum() / num_years
        st.metric("🌧️ Avg Annual Rainfall", f"{avg_annual_rain:.1f} mm/yr")
        st.caption(f"Calculated as average over {num_years} years")
    else:
        # Show total for a single year (which is correct/not misleading)
        total_rain = valid_rain.sum()
        st.metric("🌧️ Total Rainfall", f"{total_rain:.1f} mm")
        st.caption("Total for the selected period")
    with col4:
        st.metric("📅 Days Analyzed", len(filtered_df))
    
    st.markdown("---")
    
    if chart_type == "Temperature Analysis":
        if selected_month == "All Months":
            st.markdown("### 🌡️ Monthly Temperature Distribution (Outliers highlighted)")
            fig_tmax = create_monthly_boxplot(filtered_df, 'TMAX', 'Maximum Temperature (TMAX)', '#FF6347')
            st.plotly_chart(fig_tmax, use_container_width=True)
            fig_tmin = create_monthly_boxplot(filtered_df, 'TMIN', 'Minimum Temperature (TMIN)', '#4682B4')
            st.plotly_chart(fig_tmin, use_container_width=True)
        else:
            st.markdown("### 📈 Daily Temperature Trend (Red markers = Outliers)")
            fig = create_daily_line_chart(filtered_df, 'Daily Maximum and Minimum Temperature')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("### 🌧️ Rainfall Analysis")
        fig = create_rainfall_chart(filtered_df, selected_month)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("🔍 View Raw Data Preview"):
        st.dataframe(filtered_df.head(50), use_container_width=True)

if __name__ == "__main__":
    main()
