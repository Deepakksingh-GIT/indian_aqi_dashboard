import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Advanced India AQI Dashboard", layout="wide")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("india_city_aqi_2015_2023.csv")
    return df

df = load_data()

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

# AQI Classification
def classify_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

if "AQI" in df.columns:
    df["AQI_Category"] = df["AQI"].apply(classify_aqi)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔎 Filters")

if "City" in df.columns:
    selected_cities = st.sidebar.multiselect(
        "Select City",
        df["City"].unique(),
        default=df["City"].unique()
    )
    df = df[df["City"].isin(selected_cities)]

if "Year" in df.columns:
    selected_years = st.sidebar.multiselect(
        "Select Year",
        sorted(df["Year"].dropna().unique()),
        default=sorted(df["Year"].dropna().unique())
    )
    df = df[df["Year"].isin(selected_years)]

top_n = st.sidebar.slider("Top N Cities", 1, 20, 5)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🗺 Map",
    "🔮 Prediction"
])

# ===================================================
# TAB 1 - OVERVIEW
# ===================================================
with tab1:

    st.title("🌍 India AQI Advanced Dashboard")

    st.subheader("📌 Dynamic KPI Selector")

    kpi_option = st.selectbox(
        "Select KPI",
        ["Average AQI", "Maximum AQI", "Worst City", "YoY Growth %"]
    )

    if kpi_option == "Average AQI":
        st.metric("Average AQI", f"{df['AQI'].mean():.2f}")

    elif kpi_option == "Maximum AQI":
        st.metric("Maximum AQI", f"{df['AQI'].max():.2f}")

    elif kpi_option == "Worst City":
        worst_city = df.groupby("City")["AQI"].mean().idxmax()
        st.metric("Most Polluted City", worst_city)

    elif kpi_option == "YoY Growth %":
        yearly = df.groupby("Year")["AQI"].mean().reset_index()
        if len(yearly) >= 2:
            growth = ((yearly.iloc[-1]["AQI"] - yearly.iloc[-2]["AQI"])
                      / yearly.iloc[-2]["AQI"]) * 100
            st.metric("YoY Growth %", f"{growth:.2f}%")
        else:
            st.warning("Not enough data for YoY calculation")

    # ------------------------------
    # Top N Cities Bar Chart
    # ------------------------------
    st.subheader("🏆 Top Polluted Cities")

    top_cities = (
        df.groupby("City")["AQI"]
        .mean()
        .reset_index()
        .sort_values(by="AQI", ascending=False)
        .head(top_n)
    )

    fig_bar = px.bar(
        top_cities,
        x="City",
        y="AQI",
        color="AQI",
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------
    # AQI Category Distribution
    # ------------------------------
    st.subheader("🌡 AQI Category Distribution")

    category_counts = (
        df["AQI_Category"]
        .value_counts()
        .reset_index()
    )

    category_counts.columns = ["AQI_Category", "Count"]

    fig_pie = px.pie(
        category_counts,
        names="AQI_Category",
        values="Count",
        color="AQI_Category",
        color_discrete_map={
            "Good": "green",
            "Satisfactory": "lightgreen",
            "Moderate": "yellow",
            "Poor": "orange",
            "Very Poor": "red",
            "Severe": "darkred"
        }
    )

    st.plotly_chart(fig_pie, use_container_width=True)

# ===================================================
# TAB 2 - TRENDS
# ===================================================
with tab2:

    st.subheader("📈 Yearly AQI Trend")

    yearly_trend = (
        df.groupby("Year")["AQI"]
        .mean()
        .reset_index()
    )

    fig_trend = px.line(yearly_trend, x="Year", y="AQI")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ------------------------------
    # Month Heatmap
    # ------------------------------
    st.subheader("📅 Monthly AQI Heatmap")

    if "Month" in df.columns:
        heatmap_data = (
            df.groupby(["Month", "City"])["AQI"]
            .mean()
            .reset_index()
        )

        fig_heat = px.density_heatmap(
            heatmap_data,
            x="Month",
            y="City",
            z="AQI",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(fig_heat, use_container_width=True)

# ===================================================
# TAB 3 - MAP
# ===================================================
with tab3:

    st.subheader("🗺 City AQI Map")

    if {"Latitude", "Longitude"}.issubset(df.columns):

        city_avg = (
            df.groupby(["City", "Latitude", "Longitude"])["AQI"]
            .mean()
            .reset_index()
        )

        fig_map = px.scatter_mapbox(
            city_avg,
            lat="Latitude",
            lon="Longitude",
            size="AQI",
            color="AQI",
            color_continuous_scale="Reds",
            zoom=4,
            height=500
        )

        fig_map.update_layout(mapbox_style="open-street-map")

        st.plotly_chart(fig_map, use_container_width=True)

    else:
        st.warning("Latitude & Longitude columns not found in dataset.")

# ===================================================
# TAB 4 - PREDICTION
# ===================================================
with tab4:

    st.subheader("🔮 AQI Forecast Using Linear Regression")

    yearly = df.groupby("Year")["AQI"].mean().reset_index()

    if len(yearly) >= 2:

        X = yearly[["Year"]]
        y = yearly["AQI"]

        model = LinearRegression()
        model.fit(X, y)

        next_year = np.array([[yearly["Year"].max() + 1]])
        prediction = model.predict(next_year)

        st.metric(
            f"Predicted AQI for {int(next_year[0][0])}",
            f"{prediction[0]:.2f}"
        )

    else:
        st.warning("Not enough data for prediction.")

# ---------------------------------------------------
# DATA TABLE
# ---------------------------------------------------
st.subheader("📄 Filtered Data")
st.dataframe(df)

# ---------------------------------------------------
# DOWNLOAD BUTTON
# ---------------------------------------------------
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Filtered Data",
    csv,
    "filtered_aqi_data.csv",
    "text/csv"
)
