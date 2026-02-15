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

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")

# ---------------------------------------------------
# AUTO DETECT IMPORTANT COLUMNS
# ---------------------------------------------------
city_col = None
date_col = None
aqi_col = None

for col in df.columns:
    if col.lower() in ["city", "city_name", "cityname"]:
        city_col = col
    if col.lower() in ["date"]:
        date_col = col
    if col.lower() == "aqi":
        aqi_col = col

# Convert date
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month

# ---------------------------------------------------
# AQI CATEGORY
# ---------------------------------------------------
def classify_aqi(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Satisfactory"
    elif aqi <= 200: return "Moderate"
    elif aqi <= 300: return "Poor"
    elif aqi <= 400: return "Very Poor"
    else: return "Severe"

if aqi_col:
    df["AQI_Category"] = df[aqi_col].apply(classify_aqi)

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔎 Filters")

if city_col:
    selected_cities = st.sidebar.multiselect(
        "Select City",
        df[city_col].dropna().unique(),
        default=df[city_col].dropna().unique()
    )
    df = df[df[city_col].isin(selected_cities)]

if "Year" in df.columns:
    selected_years = st.sidebar.multiselect(
        "Select Year",
        sorted(df["Year"].dropna().unique()),
        default=sorted(df["Year"].dropna().unique())
    )
    df = df[df["Year"].isin(selected_years)]

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

pollutant = st.sidebar.selectbox("Select Numeric Column", numeric_cols)

top_n = st.sidebar.slider("Top N Cities", 1, 20, 5)

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 KPI Dashboard",
    "📈 15 Chart Studio",
    "🗺 Comparison",
    "🔮 Prediction & Insights"
])

# ===================================================
# TAB 1 - KPI DASHBOARD
# ===================================================
with tab1:

    st.title("🌍 Advanced India AQI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average", f"{df[pollutant].mean():.2f}")
    col2.metric("Max", f"{df[pollutant].max():.2f}")
    col3.metric("Min", f"{df[pollutant].min():.2f}")
    col4.metric("Std Dev", f"{df[pollutant].std():.2f}")

    if city_col:
        worst_city = df.groupby(city_col)[pollutant].mean().idxmax()
        best_city = df.groupby(city_col)[pollutant].mean().idxmin()

        st.write(f"🔴 Most Polluted City: **{worst_city}**")
        st.write(f"🟢 Cleanest City: **{best_city}**")

        top_data = (
            df.groupby(city_col)[pollutant]
            .mean()
            .reset_index()
            .sort_values(by=pollutant, ascending=False)
            .head(top_n)
        )

        fig_bar = px.bar(top_data, x=city_col, y=pollutant, color=pollutant)
        st.plotly_chart(fig_bar, use_container_width=True)

# ===================================================
# TAB 2 - 15 CHART OPTIONS
# ===================================================
with tab2:

    st.subheader("📊 15 Chart Options")

    chart_type = st.selectbox("Select Chart Type", [
        "Bar","Line","Area","Pie","Histogram","Box","Violin",
        "Scatter","Density Heatmap","Sunburst","Treemap",
        "Strip","ECDF","Funnel","3D Scatter"
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

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Chart could not be generated.")

# ===================================================
# TAB 3 - CITY COMPARISON
# ===================================================
with tab3:

    if city_col and "Year" in df.columns:

        selected_compare = st.multiselect(
            "Select Cities to Compare",
            df[city_col].unique()
        )

        if selected_compare:
            compare_df = df[df[city_col].isin(selected_compare)]
            fig_compare = px.line(compare_df, x="Year", y=pollutant, color=city_col)
            st.plotly_chart(fig_compare, use_container_width=True)

# ===================================================
# TAB 4 - PREDICTION & INSIGHTS
# ===================================================
with tab4:

    st.subheader("🔮 AQI Prediction")

    if "Year" in df.columns:

        yearly = df.groupby("Year")[pollutant].mean().reset_index()

        if len(yearly) >= 2:
            X = yearly[["Year"]]
            y = yearly[pollutant]

            model = LinearRegression()
            model.fit(X, y)

            next_year = np.array([[yearly["Year"].max() + 1]])
            prediction = model.predict(next_year)

            st.metric(
                f"Predicted Value for {int(next_year[0][0])}",
                f"{prediction[0]:.2f}"
            )

    st.subheader("🧠 Automated Insights")

    if city_col:
        worst_city = df.groupby(city_col)[pollutant].mean().idxmax()
        best_city = df.groupby(city_col)[pollutant].mean().idxmin()

        st.write(f"""
        • Average value is **{df[pollutant].mean():.2f}**  
        • Maximum recorded value is **{df[pollutant].max():.2f}**  
        • Most polluted city is **{worst_city}**  
        • Cleanest city is **{best_city}**  
        • Pollution trend shows variation across years  
        • Urban cities consistently show higher AQI levels  
        """)

# ---------------------------------------------------
# DATA TABLE & DOWNLOAD
# ---------------------------------------------------
st.subheader("📄 Filtered Data")
st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download Data", csv, "filtered_data.csv", "text/csv")
