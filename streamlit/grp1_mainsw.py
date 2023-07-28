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
import plotly.graph_objects as go

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
    
    ## Define function to load the customer's cluster results
    def load_cust_cluster():
        data_cust_cluster = pd.read_csv("./sw_datasets/cluster_results.csv") 
        data_cust_cluster = pd.DataFrame(data_cust_cluster)
        return data_cust_cluster
    
    ## Define function to load the uplift prediction model
    def load_uplift_1M():
        data_1m = pd.read_csv("./sw_datasets/UpliftPrediction[1M].csv") 
        data_1m = pd.DataFrame(data_1m)
        # Load customer cluster data
        data_cust_cluster = load_cust_cluster()
        data_1m = pd.merge(data_1m, data_cust_cluster, on='CUSTOMER_ID')
        # Return merged data
        return data_1m
    
    def load_uplift_3M():
        data_3m = pd.read_csv("./sw_datasets/UpliftPrediction[3M].csv") 
        data_3m = pd.DataFrame(data_3m)
        # Load customer cluster data
        data_cust_cluster = load_cust_cluster()
        data_3m = pd.merge(data_3m, data_cust_cluster, on='CUSTOMER_ID')
        return data_3m
    
    ## Create a DataFrame with city, longitude, and latitude information
    city_coordinates = pd.DataFrame({
        'CITY_FREQUENCY': [10613, 10016, 9261, 9122, 7288],
        'CITY': ['San Mateo', 'New York City', 'Boston', 'Denver', 'Seattle'],
        'LATITUDE': [37.5625, 40.7128, 42.3601, 39.7392, 47.6062],
        'LONGITUDE': [-122.3229, -74.0060, -71.0589, -104.9903, -122.3321]
    })

    ## Define user input functions
    # User Input 1: Select Customer Cluster
    def get_cust_cluster():
        # Display the dropdown box
        cluster_selection = ['1 - Low Value (Customers who buy less frequently and generate lower sales)', 
                             '2 - Middle Value (Customers who make average purchases)', 
                             '3 - High Value (Customers who make frequent purchases and generate higher sales.)']
        selected_cluster = st.selectbox(
            "Select Customer Cluster:", cluster_selection)
        if selected_cluster == '1 - Low Value':
            return 1
        elif selected_cluster == '2 - Middle Value':
            return 0
        else:
            return 2
    
    # User Input 2: Select Timeframe
    def get_timeframe():
        # Display the dropdown box
        timeframe_selection = ['1 month', '2 months', '3 months']
        selected_months = st.selectbox(
            "Select the range of months for prediction:", timeframe_selection)
        return selected_months
    
    # User Input 3: Select Metric
    def get_selected_metrics():
        # Display checkboxes for key metrics
        st.write("Select the metrics to view:")
        show_total_revenue = st.checkbox("Total Predicted Revenue")
        show_avg_spending = st.checkbox("Average Predicted Spending per Customer")
        selected_metrics = []
        if show_total_revenue:
            selected_metrics.append("0")
        if show_avg_spending:
            selected_metrics.append("1")
        return selected_metrics
    
    ## Define function to get results to display
    def process_data(data, cluster_input):
    # Filter the data based on cluster_input
        filtered_data_cluster = data[data['CLUSTER'] == cluster_input]
        filtered_data_city = filtered_data_cluster.groupby('CITY_FREQUENCY').aggregate(sum)

        # Keep only columns that are useful
        filtered_data_city = filtered_data_city[['PREDICT_SALES', 'CLUSTER', 'MONETARY_M3_HO']]

        # Merge city_coordinates with filtered_data_city
        final_df = pd.merge(filtered_data_city, city_coordinates, on='CITY_FREQUENCY')

        # Round the values to two decimal places
        final_df['PREDICT_SALES_Rounded'] = final_df['PREDICT_SALES'].round(2)
        final_df['AVG_SPENDING_Rounded'] = (final_df['PREDICT_SALES'] / final_df['CLUSTER']).round(2)
        final_df['UPLIFT_PERCENTAGE'] = ((final_df['PREDICT_SALES'] - final_df['MONETARY_M3_HO']) / final_df['MONETARY_M3_HO']) * 100

        return final_df
    
    ## Define function to display results in a map
    def display_map(final_df):
        # Create the map figure
        fig = go.Figure()

        # Add the map trace
        fig.add_trace(
            go.Scattergeo(
                locationmode='USA-states',
                lon=final_df['LONGITUDE'],
                lat=final_df['LATITUDE'],
                text=final_df['CITY'],
                customdata=final_df[['PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']],
                hovertemplate='<b>%{text}</b><br><br>' +
                            'Total Predicted Revenue: $%{customdata[0]:.2f}<br>' +
                            'Average Predicted Spending per Customer: $%{customdata[1]:.2f}<br>' +
                            'Percentage Uplift: %{customdata[2]:.2f}%',
                mode='markers',
                marker=dict(
                    size=10,
                    color='blue',
                    opacity=0.8
                )
            )
        )

        # Customize the layout
        fig.update_layout(
            title='Predicted Sales by City',
            geo=dict(
                scope='usa',
                showland=True,
                landcolor='rgb(217, 217, 217)',
                subunitcolor='rgb(255, 255, 255)',
                countrycolor='rgb(255, 255, 255)',
                showlakes=True,
                lakecolor='rgb(255, 255, 255)',
                showsubunits=True,
                showcountries=True,
                resolution=110,
                projection=dict(type='albers usa'),
                lonaxis=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    range=[-125.0, -66.0],
                    dtick=5
                ),
                lataxis=dict(
                    showgrid=True,
                    gridwidth=0.5,
                    range=[25.0, 50.0],
                    dtick=5
                )
            )
        )

        # Display the map in the Streamlit tab
        st.plotly_chart(fig)
        
    ## Get user inputs
    cluster_input = get_cust_cluster()
    timeframe_input = get_timeframe()
    metrics_input = get_selected_metrics()

    # Display the "Predict" button
    if st.button("Predict"): 
        if (timeframe_input == '1 month'):
            ## Load data based on selected timeframe
            data_1M = load_uplift_1M()
            
            # Filter and process the data based on user input
            final_df = process_data(data_1M, cluster_input)
            
            # Display the map in the Streamlit tab
            display_map(final_df)
            
        else:
            ## Load data based on selected timeframe
            data_3M = load_uplift_3M()
            
            # Filter and process the data based on user input
            final_df = process_data(data_3M, cluster_input)
            
            # Display the map in the Streamlit tab
            display_map(final_df)
            

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