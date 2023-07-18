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

    # Get customer details
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df = customer_df.filter(F.col("COUNTRY") == "United States")
    us_customer_df = us_customer_df.to_pandas()

    # Define function to load the uplift prediction model
    def load_Uplift_Churn_1M():
        data = pd.read_csv("./uplift/UpliftPrediction[1M].csv") 
        return data
    
    # Load the model
    upliftdata = load_Uplift_Churn_1M()
    
    # Define function to load the cluster sales
    def load_cluster_sales_1M():
        data = pd.read_csv("./uplift/clusterSales[1M].csv") 
        return data
    def load_city_enc():
        data = pd.read_csv("./uplift/city_enc.csv") 
        city_dict = data.set_index('CITY').T.to_dict('dict')
        return city_dict

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

    # Select City
    city_input = get_city() 

    # RFM input
    col1, col2, col3 = st.columns(3)

    # Recency
    col1.subheader("Recency Cluster")
    col1.caption("0 for less recent customers")
    col1.caption("1 for recent customers")
    rCluster =  col1.selectbox(label="Recency", options = upliftdata['CUST_REC_CLUSTER'].unique())

    # Frequency
    col2.subheader("Frequency Cluster")
    col2.caption("0 for infrequent customers")
    col2.caption("1 for moderately frequent customers")
    col2.caption("2 for frequent customers")
    rFrequency =  col2.selectbox(label="Frequency", options = upliftdata['CUST_FREQ_CLUSTER'].unique())

    col3.subheader("Monetary Cluster")
    col3.caption("0 for low spending customers")
    col3.caption("1 for high spending customers")
    rMonetary =  col3.selectbox(label="Monetary", options = upliftdata['CUST_MONETARY_CLUSTER'].unique())

    # Filter csv data
    def filterdata(data):
        # Predicted total sales
        predictedsales = data['PREDICT_SALES'].sum()

        # Actual sales
        actualsales = data['MONETARY_M3_HO'].sum()

        # Uplift
        uplift = predictedsales - actualsales

        # Percent uplift
        percentuplift = ((predictedsales - actualsales) / actualsales) * 100

        st.write("In the next month, the selected group of customer will generate $ {:0,.2f}".format(predictedsales))
        if (predictedsales > actualsales):
            st.write("Which is an increase of $ {:0,.2f}".format(uplift))
            st.write("from $ {:0,.2f}".format(actualsales))
            st.write("This is an increase of {:.2f}% increase".format(percentuplift))
        else:
            st.write("Which is an decrease of $ {:0,.2f}".format(uplift))
            st.write("from $ {:0,.2f}".format(actualsales))
            st.write("This is an decrease of {:.2f}% increase".format(percentuplift))

    if st.button('Predict Uplift'):
        # City input
        city_int = get_city_int(city_input)

        # Filtering data
        filtered_data = upliftdata[(upliftdata['CITY_FREQUENCY'].isin(city_int)) & (upliftdata['CUST_REC_CLUSTER'] == rCluster) & (upliftdata['CUST_FREQ_CLUSTER'] == rFrequency) & (upliftdata['CUST_MONETARY_CLUSTER'] == rMonetary)]

        filterdata(filtered_data)

        lower = filtered_data[filtered_data['MONETARY_M3_HO'] > filtered_data['PREDICT_SALES']]

        lower = lower.merge(us_customer_df, on = "CUSTOMER_ID", how = "left")

        lower = lower[["CUSTOMER_ID", "FIRST_NAME", "LAST_NAME", "BIRTHDAY_DATE", "E_MAIL", "PHONE_NUMBER"]]

        lower = lower.sort_values(by=['CUSTOMER_ID'])

        # Show dataframe
        st.write(lower)

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