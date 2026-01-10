# Weather Data Explorer

Weather Data Explorer is an interactive Streamlit-based web application designed to fetch historical climate data from the **Open-Meteo API** and convert it into **DSSAT-compliant (.WTH)** weather files. It also provides a comprehensive analytics dashboard for climate trend visualization and quality assurance.

## 🚀 Key Features

* **Interactive Map Selection**: Select any global location using an integrated Folium map to fetch precise coordinates.
* **Automated .WTH Generation**: Converts API data into the specialized DSSAT file format with accurate headers and `YYYYDDD` date formatting.
* **Geopy Integration**: Automatically performs reverse geocoding to retrieve city, state, and country names for professional file headers.
* **Dynamic Analytics Dashboard**:
* **Temperature Analysis**: Monthly boxplots and daily line charts for  and .
* **Rainfall Analysis**: Bar charts for daily and monthly precipitation totals.
* **Outlier Detection**: Visual identification of temperature anomalies using the Interquartile Range (IQR) method.
* **Smart Scaling**: Graphs automatically scale their Y-axis based on data peaks plus 10 units for optimal visibility.


* **Quality Assurance**: Automated checks for missing values (-99.0) and logical consistency (e.g., ).

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/weather_dashboard.git
cd weather_dashboard

```


2. **Create a virtual environment (Recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies**:
```bash
pip install streamlit pandas numpy matplotlib seaborn requests folium streamlit-folium plotly geopy

```



## 📖 Usage

Run the application locally using the following command:

```bash
streamlit run weather_dashboard.py

```

1. **Generate**: Click on the map to select a location, choose your start and end dates, and click **Generate Weather Data**.
2. **Download**: Once generated, click the **Download .WTH File** button to save the DSSAT-compliant file.
3. **Analyze**: Use the dashboard filters to explore climate patterns, view summary statistics, and check for outliers.
4. **Upload**: Have an existing file? Use the **Upload** tab to visualize any standard DSSAT `.WTH` file.

## 📦 Deployment

The best way to share this app is through **Streamlit Community Cloud**:

1. Push your code to a public GitHub repository.
2. Include a `requirements.txt` file in the root directory.
3. Connect your GitHub to [share.streamlit.io](https://share.streamlit.io/) and deploy.

## 📄 Requirements

* Python 3.9 - 3.13
* Internet connection (to access Open-Meteo and Open-Elevation APIs)

---

*Created for agricultural researchers and climate analysts.*
