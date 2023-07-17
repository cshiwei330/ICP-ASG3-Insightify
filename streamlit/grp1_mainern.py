# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import json
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
    #"role": "ACCOUNTADMIN",
    #"warehouse": "tasty_ds_wh",
    #"database": "NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE",
    #"schema": "analytics",
}

# Create Snowpark session
session = Session.builder.configs(connection_parameters).create()

# Define the app title and favicon
st.set_page_config(page_title='ICP ASG 3', page_icon="favicon.ico")

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['SW', 'Ernest', 'Gwyneth', 'GF', 'KK'])

with tab1:
    st.title('Overall')
    st.subheader('Sub Title')
    
with tab2:
    st.title('Predicting Customer Spending')
    st.subheader('Sub Title')

    # Define function to load the uplift prediction model
    def load_Uplift_Churn_1M():
        data = pd.read_csv("./uplift/UpliftPrediction[1M].csv") 
        return data
    
    # Define function to load the cluster sales
    def load_cluster_sales_1M():
        data = pd.read_csv("./uplift/clusterSales[1M].csv") 
        return data
    def load_city_enc():
        data = pd.read_csv("./uplift/city_enc.csv") 
        return data

    # User inpput
    def get_city():
        city_selection = st.multiselect("Select City:", options = load_city_enc())
        city_list = list(city_selection)
        return city_list
    def get_city_int(city_input):
        # Load city mapping 
        city_dict = load_city_enc()
        city_int = [] # Store selected city frequency 
        for each in city_input:
            city_int.append(city_dict[each]['CITY_FREQUENCY'])
        return city_int
    
    # Load the 
    load_Uplift_Churn_1M = load_Uplift_Churn_1M()
    load_cluster_sales_1M = load_cluster_sales_1M()
    city_input = get_city() # Select City
    city_int = get_city_int(city_input)


    # Function to find uplift
    def find_uplift(pred_df,cluster_sales, uplift_predictions_duration):
        
            pred_df["PREDICT_SALES_ST"] = (pred_df[0] * cluster_sales['Cluster0MeanSales'][0]) + (pred_df[1] * cluster_sales['Cluster1MeanSales'][0])
            
            # Merge pred_df and filteredata
            result_df = uplift_predictions_duration.merge(pred_df, left_index=True, right_index=True, how='right')
            st.write(result_df)
            
            # Total sales
            totalSales = result_df['PREDICT_SALES_ST'].sum()
            # Total uplift
            totalUplift = result_df['PREDICT_SALES_ST'].sum() - result_df['MONETARY_M3_HO'].sum()
            # Calculation for change in revenue
            percentUplift = ((result_df['PREDICT_SALES_ST'].sum()- result_df['MONETARY_M3_HO'].sum())/ result_df['MONETARY_M3_HO'].sum()) * 100
            
            st.write("In the next month, the chosen group of customers will generate $ {: 0,.2f}".format(totalSales))
            st.write("suggesting a $ {: 0,.2f} increase in revenue".format(totalUplift))
            st.write("Which is a {:.2f}% increase".format(percentUplift))
    
    # Define the user input fields
    city_input = get_city() # Select City 

    if st.button('Predict Uplift'):
        # Load the 1M Uplift Model
            uplift_predictions_1M = load_Uplift_Churn_1M()
            
            # Get city int  
            city_int = get_city_int(city_input)
            
            # Filter by user input 
            filtered_data = uplift_predictions_1M[(uplift_predictions_1M['CITY_FREQUENCY'].isin(city_int)) ]

            filtered_data = filtered_data.drop(columns=['CUSTOMER_ID','MONETARY_M3_HO','LTVCluster','PREDICTED_PROBA_0','PREDICTED_PROBA_1','PREDICT_SALES','INDEX'])
            st.write(filtered_data)
            # Make a prediction, write as dataframe
            pred = uplift_2W.predict_proba(filtered_data)
            pred_df = pd.DataFrame(pred, index=filtered_data.index)
            cluster_sales = load_cluster_sales_2W()
            
            find_uplift(pred_df, cluster_sales, uplift_predictions_1M)










    with open('./uplift/Uplift_1M.pkl', 'rb') as file:
        uplift_1M = pickle.load(file)

    

    
with tab3:
    st.title('Predicting Customer Churn')
    st.subheader('Sub Title')
    
with tab4:
    st.title('Uplift Revenue of Churn/Non-Churn Customers')
    st.subheader('Sub Title')

with tab5:
    st.title('Inventory Management')
    st.subheader('Truck')