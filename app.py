import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Ultimate India AQI Dashboard", layout="wide")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("india_city_aqi_2015_2023.csv")

df = load_data()

# --------------------------------------------------
# DATA PREPROCESSING
# --------------------------------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

def classify_aqi(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

if "AQI" in df.columns:
    df["AQI_Category"] = df["AQI"].apply(classify_aqi)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

if "City" in df.columns:
    cities = st.sidebar.multiselect(
        "Select City",
        df["City"].unique(),
        default=df["City"].unique()
    )
    df = df[df["City"].isin(cities)]

if "Year" in df.columns:
    years = st.sidebar.multiselect(
        "Select Year",
        sorted(df["Year"].dropna().unique()),
        default=sorted(df["Year"].dropna().unique())
    )
    df = df[df["Year"].isin(years)]

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

pollutant = st.sidebar.selectbox("Select Pollutant", numeric_cols)

top_n = st.sidebar.slider("Top N", 1, 20, 5)

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 KPI Dashboard",
    "📈 Visual Analytics",
    "🗺 Map & Comparison",
    "🔮 Prediction & Insights"
])

# ==================================================
# TAB 1 - KPI DASHBOARD
# ==================================================
with tab1:

    st.title("🌍 India AQI KPI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average AQI", f"{df[pollutant].mean():.2f}")
    col2.metric("Maximum AQI", f"{df[pollutant].max():.2f}")
    col3.metric("Minimum AQI", f"{df[pollutant].min():.2f}")
    col4.metric("Std Deviation", f"{df[pollutant].std():.2f}")

    worst_city = df.groupby("City")[pollutant].mean().idxmax()
    best_city = df.groupby("City")[pollutant].mean().idxmin()

    st.write(f"🔴 Most Polluted City: **{worst_city}**")
    st.write(f"🟢 Cleanest City: **{best_city}**")

    # Top N Chart
    top_data = (
        df.groupby("City")[pollutant]
        .mean()
        .reset_index()
        .sort_values(by=pollutant, ascending=False)
        .head(top_n)
    )

    fig_bar = px.bar(top_data, x="City", y=pollutant, color=pollutant)
    st.plotly_chart(fig_bar, use_container_width=True)

# ==================================================
# TAB 2 - 15 CHART OPTIONS
# ==================================================
with tab2:

    st.subheader("📊 Advanced Chart Studio (15 Options)")

    chart_type = st.selectbox("Select Chart Type", [
        "Bar",
        "Line",
        "Area",
        "Pie",
        "Histogram",
        "Box",
        "Violin",
        "Scatter",
        "Density Heatmap",
        "Sunburst",
        "Treemap",
        "Strip",
        "ECDF",
        "Funnel",
        "3D Scatter"
    ])

    x_axis = st.selectbox("Select X Axis", df.columns)
    y_axis = st.selectbox("Select Y Axis", numeric_cols)

    try:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == "Area":
            fig = px.area(df, x=x_axis, y=y_axis)
        elif chart_type == "Pie":
            fig = px.pie(df, names=x_axis, values=y_axis)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis)
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis)
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_axis, y=y_axis)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == "Density Heatmap":
            fig = px.density_heatmap(df, x=x_axis, y=y_axis)
        elif chart_type == "Sunburst":
            fig = px.sunburst(df, path=[x_axis], values=y_axis)
        elif chart_type == "Treemap":
            fig = px.treemap(df, path=[x_axis], values=y_axis)
        elif chart_type == "Strip":
            fig = px.strip(df, x=x_axis, y=y_axis)
        elif chart_type == "ECDF":
            fig = px.ecdf(df, x=y_axis)
        elif chart_type == "Funnel":
            fig = px.funnel(df, x=y_axis, y=x_axis)
        elif chart_type == "3D Scatter":
            if len(numeric_cols) >= 3:
                z_axis = st.selectbox("Select Z Axis", numeric_cols)
                fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis)
            else:
                st.warning("Need at least 3 numeric columns for 3D")

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Chart could not be generated. Change axes.")

# ==================================================
# TAB 3 - MAP & COMPARISON
# ==================================================
with tab3:

    st.subheader("City Comparison")

    selected_compare = st.multiselect(
        "Select Cities to Compare",
        df["City"].unique()
    )

    if selected_compare:
        compare_df = df[df["City"].isin(selected_compare)]
        fig_compare = px.line(compare_df, x="Year", y=pollutant, color="City")
        st.plotly_chart(fig_compare, use_container_width=True)

    if {"Latitude", "Longitude"}.issubset(df.columns):

        city_avg = (
            df.groupby(["City", "Latitude", "Longitude"])[pollutant]
            .mean()
            .reset_index()
        )

        fig_map = px.scatter_mapbox(
            city_avg,
            lat="Latitude",
            lon="Longitude",
            size=pollutant,
            color=pollutant,
            zoom=4,
            height=500
        )

        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)

# ==================================================
# TAB 4 - PREDICTION & INSIGHTS
# ==================================================
with tab4:

    st.subheader("🔮 AQI Prediction (Linear Regression)")

    yearly = df.groupby("Year")[pollutant].mean().reset_index()

    if len(yearly) >= 2:
        X = yearly[["Year"]]
        y = yearly[pollutant]

        model = LinearRegression()
        model.fit(X, y)

        next_year = np.array([[yearly["Year"].max() + 1]])
        prediction = model.predict(next_year)

        st.metric(
            f"Predicted AQI for {int(next_year[0][0])}",
            f"{prediction[0]:.2f}"
        )

    st.subheader("🧠 Automated Insights")

    avg_aqi = df[pollutant].mean()
    worst_city = df.groupby("City")[pollutant].mean().idxmax()
    best_city = df.groupby("City")[pollutant].mean().idxmin()

    st.write(f"""
    • Average AQI is **{avg_aqi:.2f}**
    • Worst affected city is **{worst_city}**
    • Cleanest city is **{best_city}**
    • Pollution shows seasonal spike in winter months
    • Urban metropolitan regions consistently show higher AQI
    """)

# --------------------------------------------------
# DATA TABLE & DOWNLOAD
# --------------------------------------------------
st.subheader("📄 Filtered Data")
st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Data", csv, "filtered_data.csv", "text/csv")
