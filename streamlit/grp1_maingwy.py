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
    st.title('Title')
    st.subheader('Sub Title')
    
with tab3:
    st.title('Predicting Customer Churn')

    # select churn/not churn
    city_option = st.selectbox(
    'Select to see likely to churn/not churn customers',
    ('Churn', 'Not Churn'))

    # show details of cust likely to churn/not churn
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df = customer_df.filter(F.col("COUNTRY")=="United States")
    us_customer_df = us_customer_df.to_pandas()
    us_customer_df = us_customer_df[["FIRST_NAME", "LAST_NAME", "GENDER", "MARITAL_STATUS", "CHILDREN_COUNT", "BIRTHDAY_DATE", "E_MAIL", "PHONE_NUMBER"]]
    cust_to_show = us_customer_df.head()
    st.dataframe(cust_to_show)

    # select city
    city_option = st.selectbox(
    'Select a city',
    ('San Mateo', 'Denver', 'Seattle', 'New York City', 'Boston'))

    st.write('You selected:', city_option)

    # monetary value slider
    money_option = st.slider(
    'Select the range of monetary value of customer',
    150, 1350)

    st.write('You selected:', money_option)

    # years with us slider
    years_with_us_option = st.slider(
    'Select the range of years of customer with Tasty Bytes',
    1, 4)

    st.write('You selected:', years_with_us_option)

    # select no of orders
    total_orders_option = st.slider(
    'Select the range of orders of customer',
    18, 90)

    st.write('You selected:', total_orders_option)

    # input last purchase date
    date = st.date_input("Enter the customer's last purchase date")

    st.write('You selected:', date)

    # predict churn button
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    st.button('Predict', on_click=click_button)

    if st.session_state.clicked:
        st.write("Customer is likely to churn"); 
        
        




    
with tab4:
    st.title('Uplift Revenue of Churn/Non-Churn Customers')
    st.subheader('Sub Title')
    
with tab5:
    st.title('Inventory Management')
    st.subheader('Truck')
    