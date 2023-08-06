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
    st.title('Predicting Customer Spending :moneybag:')

    st.markdown("This tab allows you to make predictions on the customer spending in the next month based on their city, recency cluster(R), frequency cluster(F) and monetary cluster(M).")
    st.markdown("At the bottom, it allow users to mannually input values to get the customer spending in the next month and find out what customer group they belong to.")

    # Get customer details
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df = customer_df.filter(F.col("COUNTRY") == "United States")

    # Get order details
    order_header_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_pos.order_header")
    order_header_df = order_header_df.na.drop(subset="CUSTOMER_ID")
    customer_US = us_customer_df.select(F.col("CUSTOMER_ID"))
    order_header_df = order_header_df.join(customer_US, order_header_df.CUSTOMER_ID == customer_US.CUSTOMER_ID, lsuffix = "", rsuffix = "_US")

    # Convert to pandas
    us_customer_df = us_customer_df.to_pandas()
    order_header_df = order_header_df.to_pandas()

    # Define function to load the uplift prediction model
    def load_Uplift_Churn_1M():
        data = pd.read_csv("./uplift/UpliftPrediction[1M].csv") 
        return data
    
    # Load the model
    upliftdata = load_Uplift_Churn_1M()
    upliftdata_copy = load_Uplift_Churn_1M()
    
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
        city_selection = st.multiselect("Select City*", options = load_city_enc())
        city_list = list(city_selection)
        return city_list
    def get_city_int(city_input):
        # Load city mapping 
        city_dict = load_city_enc()
        city_int = [] # Store selected city frequency 
        for each in city_input:
            city_int.append(city_dict[each]['CITY_FREQUENCY'])
        return city_int
    
    # Bar chart 
    order_year = order_header_df[['ORDER_ID', 'ORDER_TS', 'ORDER_AMOUNT']]

    # Extract the year from the 'ORDER_TS' column
    order_year['year'] = order_year['ORDER_TS'].dt.year

    # Group the data by 'year' and calculate the sum of 'ORDER_AMOUNT'
    sum_by_year = order_year.groupby('year')['ORDER_AMOUNT'].sum()

    # Convert the result into a DataFrame
    bar_df = sum_by_year.reset_index()
    # Set 'year' as the index
    bar_df.set_index('year', inplace=True)

    # Display bar chart
    st.subheader("Bar Chart of Revenue across Year")
    st.bar_chart(bar_df)
    st.markdown("From the bar chart, we are able to see that the yearly revenue for Tasty Bytes has been increasing. However, will it be able to achieve the goal of increasing the revenue by 25% year on year across 5 years?")
    st.markdown("Use the predictor below to find out! :money_mouth_face:")

    # Based on RFM
    st.subheader('Based on RFM')

    # Select City
    city_input = get_city() 

    # RFM input
    col1, col2, col3 = st.columns(3)

    # Setting the input
    cluster_mapping_r = {0: 'Old Customer', 1: 'Recent Customers'}
    upliftdata['CUST_REC_CLUSTER'] = upliftdata['CUST_REC_CLUSTER'].map(cluster_mapping_r)

    cluster_mapping_f = {0: 'Moderately Frequent Customer', 1: 'Infrequent Customers', 2: 'Frequent Customers'}
    upliftdata['CUST_FREQ_CLUSTER'] = upliftdata['CUST_FREQ_CLUSTER'].map(cluster_mapping_f)

    cluster_mapping_m = {0: 'Low Spending Customer', 1: 'High Spending Customers'}
    upliftdata['CUST_MONETARY_CLUSTER'] = upliftdata['CUST_MONETARY_CLUSTER'].map(cluster_mapping_m)

    # Recency
    col1.subheader("Recency Cluster")
    #col1.caption("0 for less recent customers")
    #col1.caption("1 for recent customers")
    rCluster =  col1.selectbox(label="Recency", options = upliftdata['CUST_REC_CLUSTER'].unique())

    # Frequency
    col2.subheader("Frequency Cluster")
    #col2.caption("0 for infrequent customers")
    #col2.caption("1 for moderately frequent customers")
    #col2.caption("2 for frequent customers")
    rFrequency =  col2.selectbox(label="Frequency", options = upliftdata['CUST_FREQ_CLUSTER'].unique())

    col3.subheader("Monetary Cluster")
    #col3.caption("0 for low spending customers")
    #col3.caption("1 for high spending customers")
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

        st.write("In the next month, the selected group of customer will generate ${:0,.2f}.".format(predictedsales))
        if (predictedsales > actualsales):
            st.write("This is an increase by ${:0,.2f}".format(uplift) + " which is an increase of {:.2f}%. :smile:".format(percentuplift))

        else:
            st.write("Which is an decrease of ${:0,.2f}".format(uplift) + " which is a decrease of {:.2f}%. :pensive:".format(percentuplift))

    # convert dataframe to csv 
    def convert_df(df):
       return df.to_csv(index=False).encode('utf-8')

    if st.button('Predict Uplift'):
        # City input
        city_int = get_city_int(city_input)

        # Filtering data
        filtered_data = upliftdata[(upliftdata['CITY_FREQUENCY'].isin(city_int)) & (upliftdata['CUST_REC_CLUSTER'] == rCluster) & (upliftdata['CUST_FREQ_CLUSTER'] == rFrequency) & (upliftdata['CUST_MONETARY_CLUSTER'] == rMonetary)]

        filterdata(filtered_data)

    if st.checkbox('Show customers that did not contribute to the uplift'):
        # City input
        city_int = get_city_int(city_input)

        # Filtering data
        filtered_data = upliftdata[(upliftdata['CITY_FREQUENCY'].isin(city_int)) & (upliftdata['CUST_REC_CLUSTER'] == rCluster) & (upliftdata['CUST_FREQ_CLUSTER'] == rFrequency) & (upliftdata['CUST_MONETARY_CLUSTER'] == rMonetary)]

        lower = filtered_data[filtered_data['MONETARY_M3_HO'] > filtered_data['PREDICT_SALES']]

        lower = lower.merge(us_customer_df, on = "CUSTOMER_ID", how = "left")

        lower = lower[["CUSTOMER_ID", "FIRST_NAME", "LAST_NAME", "E_MAIL", "PHONE_NUMBER"]]

        lower = lower.sort_values(by=['CUSTOMER_ID'])
        st.write(lower) 
        csv = convert_df(lower)

        st.download_button(
        label="Download details of customer that did not contribute to the uplift",
        data=csv,
        file_name='customer_details.csv',
        mime='text/csv',
    )

    # Based on Specific Value
    st.subheader("Based on Specific Value")

    # Load the model
    with open('./uplift/Uplift_1M.pkl', 'rb') as file:
        uplift_1M = pickle.load(file)
    
    # Define function to load the cluster sales
    def load_cluster_sales_1M():
        data = pd.read_csv("./uplift/clusterSales[1M].csv") 
        return data
    
    # User input
    def get_city2():
        city_selection2 = st.selectbox("Select City*", options = load_city_enc())
        city_list2 = [city_selection2]
        return city_list2
    def get_city_int2(city_input2):
        # Load city mapping 
        city_dict = load_city_enc()
        city_int2 = [] # Store selected city frequency 
        for each in city_input2:
            city_int2.append(city_dict[each]['CITY_FREQUENCY'])
        return city_int2

    # Select City
    city_input2 = get_city2() 

    # Years with us
    timeframe_value =  list(range(0, 5))
    years_pickle = st.select_slider("Years as Member",options=timeframe_value)
    st.write('You selected:', years_pickle)

    # Recency
    recency_value = list(range(0, 151))
    recency_pickle = st.select_slider("Days since last purchase", options=recency_value)
    st.write('You selected:', recency_pickle)

    # Frequency
    frequency_value = list(range(5, 51))
    frequency_pickle = st.select_slider("Number of orders yearly", options=frequency_value)
    st.write('You selected:', frequency_pickle)

    # Monetary
    monetary_value = list(range(150, 1501))
    monetary_pickle = st.select_slider("Total amount spent yearly", options=monetary_value)
    st.write('You selected:', monetary_pickle)

    # Total Spent
    totalspent_pickle = monetary_pickle * years_pickle

    # Dataframe
    data = {
        'TOTAL_SPENT': [totalspent_pickle],
        'YEARS_WITH_US': [years_pickle],
        'MONETARY_VALUE': [monetary_pickle],
        'CUSTOMER_FREQUENCY': [frequency_pickle],
        'TOTAL_ORDER': [0],
        'RECENCY_DAYS': [recency_pickle],
        'CITY_FREQUENCY': get_city_int2(city_input2),
        'ORDER_TOTAL_S1': [0],
        'ORDER_TOTAL_S2': [0],
        'ORDER_TOTAL_S3': [0],
        'CHANGE_FROM_S1_TO_S2': [0],
        'CHANGE_FROM_S2_TO_S3': [0]
    }

    final = pd.DataFrame(data)

    if (recency_pickle <21):
        final['CUST_REC_CLUSTER'] = 1
    else:
        final['CUST_REC_CLUSTER'] = 0

    if (frequency_pickle <= 16):
        final['CUST_FREQ_CLUSTER'] = 1
    elif (frequency_pickle > 16 and frequency_pickle <21):
        final['CUST_FREQ_CLUSTER'] = 0
    else:
        final['CUST_FREQ_CLUSTER'] = 2
    
    if (monetary_pickle <= 680):
        final['CUST_MONETARY_CLUSTER'] = 0
    else:
        final['CUST_MONETARY_CLUSTER'] = 1

    final['OVERALL_SCORE'] = final['CUST_REC_CLUSTER'] + final['CUST_FREQ_CLUSTER'] + final['CUST_MONETARY_CLUSTER']

    if st.button('Predict'):
        # Get data based on customer cluster
        filter_value_r = int(final.iloc[0]['CUST_REC_CLUSTER'])
        filter_value_f = int(final.iloc[0]['CUST_FREQ_CLUSTER'])
        filter_value_m = int(final.iloc[0]['CUST_MONETARY_CLUSTER'])

        filtered_data_2 = upliftdata_copy[(upliftdata_copy['CUST_REC_CLUSTER'] == filter_value_r) & (upliftdata_copy['CUST_FREQ_CLUSTER'] == filter_value_f) &
                                          (upliftdata_copy['CUST_MONETARY_CLUSTER'] == filter_value_m) & (upliftdata_copy['CITY_FREQUENCY'].isin(get_city_int2(city_input2)))]
        

        filter_value_order = filtered_data_2["TOTAL_ORDER"].mean()
        filter_value_s1 = filtered_data_2["ORDER_TOTAL_S1"].mean()
        filter_value_s2 = filtered_data_2["ORDER_TOTAL_S2"].mean()
        filter_value_s3 = filtered_data_2["ORDER_TOTAL_S3"].mean()

        final["TOTAL_ORDER"] = filter_value_order
        final["ORDER_TOTAL_S1"] = filter_value_s1
        final["ORDER_TOTAL_S2"] = filter_value_s2
        final["ORDER_TOTAL_S3"] = filter_value_s3
        final["CHANGE_FROM_S1_TO_S2"] = (filter_value_s2 - filter_value_s1)
        final["CHANGE_FROM_S2_TO_S3"] = (filter_value_s3 - filter_value_s2)

        cluster_sales = load_cluster_sales_1M()
        pred = uplift_1M.predict_proba(final)
        pred = pd.DataFrame(pred)
        final["PREDICT_SALES_ST"] = (pred[0] * cluster_sales['Cluster0MeanSales'][0]) + (pred[1] * cluster_sales['Cluster1MeanSales'][0])
        amountSpent = final["PREDICT_SALES_ST"]

        st.write("This specific customer would be spending ${:0,.2f} in the next month".format(amountSpent[0]))
        
        if (final["CUST_REC_CLUSTER"][0] == 1):
            st.write("This customer belongs to the Recent Customer cluster.")
        else:
            st.write("This customer belongs to the Old Customer cluster.")

        if (final["CUST_FREQ_CLUSTER"][0] == 1):
            st.write("This customer belongs to the Moderately Frequent Customer cluster .")
        elif (final["CUST_FREQ_CLUSTER"][0] == 0):
             st.write("This customer belongs to the Infrequent Customer cluster.")
        else:
            st.write("This customer belongs to the Frequent Customer cluster.")

        if (final["CUST_MONETARY_CLUSTER"][0] == 1):
            st.write("This customer belongs to the High Spending Customer cluster.")
        else:
            st.write("This customer belongs to the Low Spending Customer cluster.")
        
    
with tab3:
    st.title('Predicting Customer Churn')
    st.subheader('Sub Title')
    
with tab4:
    st.title('Uplift Revenue of Churn/Non-Churn Customers')
    st.subheader('Sub Title')

with tab5:
    st.title('Inventory Management')
    st.subheader('Truck')