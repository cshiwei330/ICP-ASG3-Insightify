# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import plotly.express as px
import pydeck as pdk
import geopandas as gpd
import shapely.geometry as shp

# Import Snowflake modules
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Window
from snowflake.snowpark.functions import col
from datetime import datetime

# Get account credentials from a json file
with open("account.json") as f:
    data = json.load(f)
    username = data["username"]
    password = data["password"]
    account = data["account"]

# Specify connection parameters
connection_parameters = {
    "account": account,
    "user": username,
    "password": password,
}

# Create Snowpark session
session = Session.builder.configs(connection_parameters).create()

# Define the app title and favicon
st.set_page_config(page_title='ICP ASG 3', page_icon="streamlit\favicon.ico")

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['SW', 'Ernest', 'Gwyneth', 'GF', 'KK'])

with tab1:
    st.title('Predicting Future Sales')
    
    # User Input 1: Select Customer Cluster
    # Display the dropdown box
    selected_cluster = st.selectbox(
        "Select Customer Cluster:",
        ("1 month", "2 months", "3 months")
    )
    
    # User Input 1: Select Timeframe
    # Display the dropdown box
    selected_months = st.selectbox(
        "Select the range of months for prediction:",
        ("1 month", "2 months", "3 months")
    )

    # User Input 2:
    # Display checkboxes for key metrics
    st.write("Select the metrics to view:")
    show_total_revenue = st.checkbox("Total Predicted Revenue")
    show_avg_spending = st.checkbox("Average Predicted Spending per Customer")

    # User Input 3:
    # Display the "Predict" button
    if st.button("Predict"):
        # Create the dummy data
        predicted_sales_data = pd.DataFrame({
            "city": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
            "total_revenue": [100000, 80000, 60000, 40000, 30000],
            "avg_spending": [500, 400, 300, 200, 100],
            "latitude": [40.7127, 34.0522, 41.8781, 29.7604, 33.4484],
            "longitude": [-74.0059, -118.2437, -87.6298, -95.3693, -112.0740]
        })

        # Create a GeoDataFrame
        predicted_sales_geo_df = gpd.GeoDataFrame(
            predicted_sales_data,
            geometry=[shp.Point(lon, lat) for lon, lat in zip(predicted_sales_data["longitude"], predicted_sales_data["latitude"])]
        )

        # Create a map of USA
        cities = predicted_sales_geo_df["city"].unique()
        locations = predicted_sales_geo_df[["longitude", "latitude"]].values

        # Create a layer for the map
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=predicted_sales_geo_df,
            get_position="geometry.coordinates",
            get_fill_color="total_revenue",
            get_radius="avg_spending",
        )

        # Render the map
        view_state = pdk.ViewState(latitude=37, longitude=-95, zoom=5)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

        # Add a hover handler to the map
        def hover(info):
            city = info.pick["city"]
            if show_total_revenue:
                st.text(f"Total Predicted Revenue for {city}: {info.pick['total_revenue']}")
            if show_avg_spending:
                st.text(f"Average Predicted Spending per Customer for {city}: {info.pick['avg_spending']}")

        layer.hover_callback = hover
    
with tab2:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab3:
    st.title('Predicting Customer Churn')

with tab4:
    st.title('Uplift Revenue of Churn/Non-Churn Customers')
    st.subheader('Sub Title')
    
with tab5:
    st.title('Inventory Management')
    st.subheader('Truck')