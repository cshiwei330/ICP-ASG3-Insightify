# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import json
import math
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
    st.subheader('Details of Churn Customers')

    # loading of dataset 
    def load_next_purchase_cust_seg():
        data = pd.read_csv("./churn/NextPurchaseCustSeg.csv")
        return data
    
    # filter csv based on customer segment chosen 
    def filter_cust_seg(data):
        filtered_cust_seg = next_purchase_cust_seg[next_purchase_cust_seg['CLUSTER']==data]
        return filtered_cust_seg
    
    # filter csv based on whether customer likely to churn or not
    def filter_cust_churn(data):
        filtered_cust_churn = filtered_cust_seg[filtered_cust_seg["CHURN_STATUS"] == data]
        return filtered_cust_churn
    
    # convert dataframe to csv 
    def convert_df(df):
       return df.to_csv(index=False).encode('utf-8')
    
    next_purchase_cust_seg = load_next_purchase_cust_seg()
    next_purchase_cust_seg.rename(columns={'CHURN': 'CHURN_STATUS'}, inplace=True)

    cust_seg_label_mapping = {0: 'Middle Value', 1: 'Low Value', 2:'High Value'}
    next_purchase_cust_seg['CLUSTER'] = next_purchase_cust_seg['CLUSTER'].map(cust_seg_label_mapping)
    # Information about customer segments
    st.write("Low Value: This cluster represents customers who buy less frequently and generate lower sales.")
    st.write("Middle Value: This cluster represents customers who make average purchases.")
    st.write("High Value: This cluster represents customers who make frequent purchases and generate higher sales.")
    # select customer segment
    cust_seg = st.selectbox(
    'Select the information of the customer segment that you would like to view',
    options = np.sort(next_purchase_cust_seg['CLUSTER'].unique()))
    
    # show percentage of churn and not churn of customer segment chosen using bar charts
    filtered_cust_seg = filter_cust_seg(cust_seg)
    churn_label_mapping = {0: 'Churn', 1: 'Not Churn'}
    filtered_cust_seg['CHURN_STATUS'] = filtered_cust_seg['CHURN_STATUS'].map(churn_label_mapping)
    cust_churn_bar = filtered_cust_seg['CHURN_STATUS'].value_counts()
    st.bar_chart(data = cust_churn_bar)

    # show information of churn and not churn customers
    churn, not_churn = st.columns(2)

    # churn
    churn.subheader("Information of Customers Likely to Churn")
    churn_cust = filter_cust_churn("Churn")

    # monetary value
    min_churn_monetary = str(min(round(churn_cust['MONETARY_VALUE'], 2)))
    max_churn_monetary = str(max(round(churn_cust['MONETARY_VALUE'], 2)))
    avg_churn_monetary = str(round(churn_cust['MONETARY_VALUE'].mean(),2))
    # total order
    min_churn_order = str(math.floor(churn_cust['TOTAL_ORDER'].min()))
    max_churn_order = str(math.floor(churn_cust['TOTAL_ORDER'].max()))
    avg_churn_order = str(math.floor(churn_cust['TOTAL_ORDER'].mean()))
    # recency
    min_churn_recency = str(math.floor(churn_cust['RECENCY_DAYS'].min()))
    max_churn_recency = str(math.floor(churn_cust['RECENCY_DAYS'].max()))
    avg_churn_recency = str(math.floor(churn_cust['RECENCY_DAYS'].mean()))

    churn.caption("Range of Customers Likely to Churn Monetary Value: $" + min_churn_monetary + "-"+ max_churn_monetary)
    churn.caption("Range of Customers Likely to Churn Total Orders: " + min_churn_order + "-"+ max_churn_order)
    churn.caption("Range of Customers Likely to Churn Recency Days: " + min_churn_recency + "-"+ max_churn_recency)
    
    churn.caption("Average of Customers Likely to Churn Monetary Value: $" + avg_churn_monetary)
    churn.caption("Average of Customers Likely to Churn Total Orders: " + avg_churn_order)
    churn.caption("Average of Customers Likely to Churn Recency Days: " + avg_churn_recency)

    # not churn
    not_churn.subheader("Information of Customers not Likely to Churn")
    not_churn_cust = filter_cust_churn("Not Churn")

    # monetary value
    min_not_churn_monetary = str(min(round(not_churn_cust['MONETARY_VALUE'], 2)))
    max_not_churn_monetary = str(max(round(not_churn_cust['MONETARY_VALUE'], 2)))
    avg_not_churn_monetary = str(round(not_churn_cust['MONETARY_VALUE'].mean(),2))
    # total order
    min_not_churn_order = str(math.floor(not_churn_cust['TOTAL_ORDER'].min()))
    max_not_churn_order = str(math.floor(not_churn_cust['TOTAL_ORDER'].max()))
    avg_not_churn_order = str(math.floor(not_churn_cust['TOTAL_ORDER'].mean()))
    # recency
    min_not_churn_recency = str(math.floor(not_churn_cust['RECENCY_DAYS'].min()))
    max_not_churn_recency = str(math.floor(not_churn_cust['RECENCY_DAYS'].max()))
    avg_not_churn_recency = str(math.floor(not_churn_cust['RECENCY_DAYS'].mean()))

    not_churn.caption("Range of Customers not Likely to Churn Monetary Value: $" + min_not_churn_monetary + "-"+ max_not_churn_monetary)
    not_churn.caption("Range of Customers not Likely to Churn Total Orders: " + min_not_churn_order + "-"+ max_not_churn_order)
    not_churn.caption("Range of Customers not Likely to Churn Recency Days: " + min_not_churn_recency + "-"+ max_not_churn_recency)
    
    not_churn.caption("Average of Customers not Likely to Churn Monetary Value: $" + avg_not_churn_monetary)
    not_churn.caption("Average of Customers not Likely to Churn Total Orders: " + avg_not_churn_order)
    not_churn.caption("Average of Customers not Likely to Churn Recency Days: " + avg_not_churn_recency)

    # show details of cust likely to churn 
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df = customer_df.filter(F.col("COUNTRY")=="United States")
    us_customer_df = us_customer_df.to_pandas()
    us_customer_df = us_customer_df[us_customer_df['CUSTOMER_ID'].isin(churn_cust['CUSTOMER_ID'])]
    cust_to_show = us_customer_df[["FIRST_NAME", "LAST_NAME", "GENDER", "MARITAL_STATUS", "CHILDREN_COUNT", "BIRTHDAY_DATE", "E_MAIL", "PHONE_NUMBER"]]
    st.dataframe(cust_to_show)

    csv = convert_df(cust_to_show)

    st.download_button(
       "Press to Download Details of Customers Likely to Churn",
       csv,
       "churn_cust_" + str(cust_seg) +".csv",
       "text/csv",
       key='download-csv')

   

    st.subheader('Predicting whether customers churn')

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
    