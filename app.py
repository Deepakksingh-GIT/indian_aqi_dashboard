import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="India AQI Dashboard", layout="wide")

# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("india_city_aqi_2015_2023.csv")
    return df

df = load_data()

st.title("🌍 India City AQI Dashboard (2015–2023)")

# -------------------------
# DATA CLEANING
# -------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🔎 Filter Options")

# City Filter
if "City" in df.columns:
    cities = st.sidebar.multiselect(
        "Select City",
        options=df["City"].unique(),
        default=df["City"].unique()
    )
    df = df[df["City"].isin(cities)]

# Year Filter
if "Year" in df.columns:
    years = st.sidebar.multiselect(
        "Select Year",
        options=sorted(df["Year"].dropna().unique()),
        default=sorted(df["Year"].dropna().unique())
    )
    df = df[df["Year"].isin(years)]

# Select Pollutant (Numeric columns)
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

pollutant = st.sidebar.selectbox(
    "Select Pollutant / AQI Metric",
    numeric_cols
)

# Top N Cities
top_n = st.sidebar.slider("Top N Cities", 1, 20, 5)

# Chart Type
chart_type = st.sidebar.selectbox(
    "Select Chart Type",
    ["Bar", "Line", "Pie", "Histogram", "Box", "Scatter"]
)

# -------------------------
# KPI SECTION
# -------------------------
st.subheader("📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Average AQI", f"{df[pollutant].mean():.2f}")
col2.metric("Maximum AQI", f"{df[pollutant].max():.2f}")
col3.metric("Minimum AQI", f"{df[pollutant].min():.2f}")

# -------------------------
# TOP N CITY ANALYSIS
# -------------------------
st.subheader("🏆 Top Cities Analysis")

if "City" in df.columns:
    top_cities = (
        df.groupby("City")[pollutant]
        .mean()
        .reset_index()
        .sort_values(by=pollutant, ascending=False)
        .head(top_n)
    )
else:
    top_cities = df.head(top_n)

fig = None

try:
    if chart_type == "Bar":
        fig = px.bar(top_cities, x="City", y=pollutant)

    elif chart_type == "Line":
        fig = px.line(top_cities, x="City", y=pollutant)

    elif chart_type == "Pie":
        fig = px.pie(top_cities, names="City", values=pollutant)

    elif chart_type == "Histogram":
        fig = px.histogram(df, x=pollutant)

    elif chart_type == "Box":
        fig = px.box(df, x="City", y=pollutant)

    elif chart_type == "Scatter":
        if "Year" in df.columns:
            fig = px.scatter(df, x="Year", y=pollutant, color="City")
        else:
            fig = px.scatter(df, x="City", y=pollutant)

    if fig:
        st.plotly_chart(fig, use_container_width=True)

except:
    st.error("⚠️ Unable to generate chart. Please change selection.")

# -------------------------
# TREND ANALYSIS
# -------------------------
if "Year" in df.columns:
    st.subheader("📈 Yearly Trend Analysis")

    yearly_trend = (
        df.groupby("Year")[pollutant]
        .mean()
        .reset_index()
    )

    fig_trend = px.line(yearly_trend, x="Year", y=pollutant)
    st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# DATA TABLE
# -------------------------
st.subheader("📄 Filtered Data")
st.dataframe(df)

# -------------------------
# DOWNLOAD BUTTON
# -------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv,
    file_name="filtered_aqi_data.csv",
    mime="text/csv"
)
