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
    description = """
    Welcome to the 'Predict Future Sales' tab! 
    This dashboard is designed to help Tasty Byte's team analyze and predict future sales, aligning with the company's ambitious goal of achieving 25% YoY growth over the next 5 years.

    With this interactive tool, you can explore valuable insights that will contribute to your strategic decision-making process. 
    Gain a deeper understanding of sales trends, identify growth opportunities, and make data-driven decisions to propel Tasty Byte towards its long-term vision."""
    st.write(description)
    
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
    
    def load_uplift_2M():
        data_2m = pd.read_csv("./sw_datasets/UpliftPrediction[2M].csv") 
        data_2m = pd.DataFrame(data_2m)
        # Load customer cluster data
        data_cust_cluster = load_cust_cluster()
        data_2m = pd.merge(data_2m, data_cust_cluster, on='CUSTOMER_ID')
        # Return merged data
        return data_2m
    
    def load_uplift_3M():
        data_3m = pd.read_csv("./sw_datasets/UpliftPrediction[3M].csv") 
        data_3m = pd.DataFrame(data_3m)
        # Load customer cluster data
        data_cust_cluster = load_cust_cluster()
        data_3m = pd.merge(data_3m, data_cust_cluster, on='CUSTOMER_ID')
        return data_3m
    
    def get_predicted_sales(data):
        predicted_sales = data['PREDICT_SALES'].sum().round(2)
        return predicted_sales
    
    ## Create a DataFrame with city, longitude, and latitude information
    city_coordinates = pd.DataFrame({
        'CITY_FREQUENCY': [10613, 10016, 9261, 9122, 7288],
        'CITY': ['San Mateo', 'New York City', 'Boston', 'Denver', 'Seattle'],
        'LATITUDE': [37.5625, 40.7128, 42.3601, 39.7392, 47.6062],
        'LONGITUDE': [-122.3229, -74.0060, -71.0589, -104.9903, -122.3321]
    })
    
    ## Visualisation 1: Display bar chart 
    # Create dataframe that stores sales values
    nov = get_predicted_sales(load_uplift_1M())
    dec = get_predicted_sales(load_uplift_2M())
    jan = get_predicted_sales(load_uplift_3M())
    data = {
    'Month': ['October 2022', 'November 2022', 'December 2022', 'January 2023'],
    'sales': [4837269.25, nov, dec, jan]}
    sales_df = pd.DataFrame(data)
    
    # Create bar chart
    st.subheader('Sales Trend and Prediction')
    fig_1 = go.Figure()

    fig_1.add_trace(
        go.Bar(
            x=sales_df['Month'],
            y=sales_df['sales'],
            marker_color='blue',
            text=sales_df['sales'],
            textposition='auto',
        )
    )

    # Customize the layout
    fig_1.update_layout(
        xaxis_title='Month',
        yaxis_title='Sales Amount ($)',
        showlegend=False
    )
    
    # Display the bar chart in the Streamlit tab
    st.plotly_chart(fig_1)
    
    ## Present insights based on the bar chart
    st.subheader('Insights:')
    st.write("Based on the sales trend and predictions, here are some key observations:")

    # Create a list of insights
    insights = [
        "There is a positive growth trend in sales over the three upcoming months, which aligns with Tasty Byte's growth goal.",
        "The predicted sales for January 2023 shows significant growth and is a positive indicator of progress towards the 25% YoY growth goal.",
    ]

    # Display the insights in a bullet point format
    st.write("✓ " + insights[0])
    st.write("✓ " + insights[1])
    
    ## Visualisation 2: Display pie chart 
    # Create DataFrame
    cluster_results_df = pd.DataFrame(load_cust_cluster())
    
    # Map numeric cluster values to labels
    cluster_results_df['CLUSTER'] = cluster_results_df['CLUSTER'].replace({0: 'Middle Value', 
                                                                           1: 'Low Value', 
                                                                           2: 'High Value'})

    # Count the number of customers in each cluster
    cluster_counts = cluster_results_df['CLUSTER'].value_counts()
    
    # Create a pie chart
    st.subheader('Customer Cluster Distribution')
    fig_2 = go.Figure()

    fig_2.add_trace(
        go.Pie(
            labels=cluster_counts.index,
            values=cluster_counts.values,
            textinfo='percent+label',
            marker=dict(colors=['gold', 'mediumturquoise', 'darkorange']),
        )
    )

    # Customize the layout
    fig_2.update_layout(
        showlegend=False,
    )

    # Display the pie chart in the Streamlit tab
    st.plotly_chart(fig_2)
    
    ## Predicted Future Sales Based On Customer Cluster
    st.subheader('Predict Future Sales Based On Customer Cluster')

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
        show_total_sales = st.checkbox("Total Predicted sales")
        show_avg_spending = st.checkbox("Average Predicted Spending per Customer")
        selected_metrics = []
        if show_total_sales:
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

        # Add the map trace using scattermapbox
        fig.add_trace(
            go.Scattermapbox(
                lon=final_df['LONGITUDE'],
                lat=final_df['LATITUDE'],
                text=final_df['CITY'],
                customdata=final_df[['PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']],
                hoverinfo='text',  # Display the text when hovering over the markers
                mode='markers',
                marker=dict(
                    size=final_df['PREDICT_SALES_Rounded'] / 1000000,  # Adjust marker size based on sales (scaled down for aesthetics)
                    sizemode='diameter',
                    sizeref=0.03,  # Adjust the scaling factor for marker size
                    color=final_df['UPLIFT_PERCENTAGE'],  # Use percentage uplift as the marker color
                    colorscale='Viridis',  # Choose a suitable colorscale
                    colorbar=dict(
                        title='Percentage Uplift',
                        thickness=15,
                        len=0.5,
                    ),
                    opacity=0.8
                ),
                showlegend=False  # Set showlegend to False to hide the legend
            )
        )

        # Add custom labels to the map
        labels = []
        for index, row in final_df.iterrows():
            label = f"Total Predicted Sales: ${row['PREDICT_SALES_Rounded']:.2f}<br>" + \
                    f"Average Predicted Spending per Customer: ${row['AVG_SPENDING_Rounded']:.2f}<br>" + \
                    f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            labels.append(label)

        fig.add_trace(
            go.Scattermapbox(
                lon=final_df['LONGITUDE'],
                lat=final_df['LATITUDE'],
                mode='text',
                text=labels,
                textfont=dict(size=10, color='white'),
                showlegend=False  # Set showlegend to False to hide the legend
            )
        )

        # Customize the layout
        fig.update_layout(
            title='Predicted Sales by City',
            mapbox=dict(
                style='carto-positron',  
                zoom=2.5,  # Set the initial zoom level
                center=dict(lat=44, lon=-95.7129),  # Set the initial center of the map
            )
        )

        # Display the map in the Streamlit tab
        st.plotly_chart(fig)

    ## Define Function to display results in table
    def display_table(data):
        final_df = data[['CITY', 'PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']]
        final_df.rename(columns={'CITY':'City',
                                'PREDICT_SALES_Rounded': 'Predicted Sales ($)',
                                'AVG_SPENDING_Rounded': 'Average Sales Per Customer ($)',
                                'UPLIFT_PERCENTAGE': 'Uplife Percentage (%)'}, inplace=True)
        st.write(final_df)
        return

     
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
            
            # Display table result
            display_table(final_df)
            
        elif (timeframe_input == '2 months'):
            ## Load data based on selected timeframe
            data_2M = load_uplift_2M()
            
            # Filter and process the data based on user input
            final_df = process_data(data_2M, cluster_input)
            
            # Display the map in the Streamlit tab
            display_map(final_df)
            
            # Display table result
            display_table(final_df)
            
        else:
            ## Load data based on selected timeframe
            data_3M = load_uplift_3M()
            
            # Filter and process the data based on user input
            final_df = process_data(data_3M, cluster_input)
            
            # Display the map in the Streamlit tab
            display_map(final_df)
            
            # Display table result
            display_table(final_df)
            
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