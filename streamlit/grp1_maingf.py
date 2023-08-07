# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import datetime
import pickle
import json
import math
import plotly.express as px
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
    st.subheader('Sub Title')
    
with tab4:
   
    # --- Title & Description for Tab--- 
    st.markdown("###" +' :arrow_up_small: Uplift Analysis for Churn/Non-Churn Customers')
    description = '''
    Using this tab, you can predict the uplift in revenue for both churn and non-churn customers in the United States (US), which plays a crucial role in helping Tasty Byte achieve its goal of attaining 25% year-over-year growth over a period of 5 years. 
    \nThe model employed for these predictions is the AdaBoost Classifier, which was trained on historical data spanning from 1st January, 2019, to 1st November, 2022.

    \nBelow, you will find a histogram  displaying the distribution for "Days to next Purchase." 
    '''
    st.markdown(description)
    
    #-----Functions for loading of files-----#  
    with open('./uplift/Uplift_1M.pkl', 'rb') as file:
        uplift_1M = pickle.load(file)
    with open('./uplift/Uplift_2W.pkl', 'rb') as file:
        uplift_2W = pickle.load(file)
    with open('./uplift/Uplift_3M.pkl', 'rb') as file:
        uplift_3M = pickle.load(file)
    # caching computations that return data
    @st.cache_data
    # Define function to load the uplift prediction model
    def load_Uplift_Churn_2W():
        data = pd.read_csv("./uplift/UpliftPrediction[2W].csv") 
        df = pd.DataFrame(data)
        return df
    @st.cache_data
    def load_Uplift_Churn_1M():
        data = pd.read_csv("./uplift/UpliftPrediction[1M].csv") 
        df = pd.DataFrame(data)
        return df
    @st.cache_data
    def load_Uplift_Churn_3M():
        data = pd.read_csv("./uplift/UpliftPrediction[3M].csv") 
        df = pd.DataFrame(data)
        return df
    
    @st.cache_data
    # Define function to load the cluster sales
    def load_cluster_sales_2W():
        data = pd.read_csv("./uplift/clusterSales[2W].csv")
        return data
    @st.cache_data
    def load_cluster_sales_1M():
        data = pd.read_csv("./uplift/clusterSales[1M].csv") 
        return data
    @st.cache_data
    def load_cluster_sales_3M():
        data = pd.read_csv("./uplift/clusterSales[3M].csv") 
        return data
    @st.cache_data
    def load_next_purchase():
        data = pd.read_csv("./uplift/NextPurchase2.csv")
        return data
    @st.cache_data
    def load_city_enc():
        data = pd.read_csv("./uplift/city_enc.csv") 
        city_dict = data.set_index('CITY').T.to_dict('dict')
        return city_dict
    #=================================================================================================#
    # Define the user input functions 
    def get_churn_days(_cNum):
        data = load_next_purchase() 
        churn_days = _cNum.slider("Determine days for churning", math.floor(data["TARGET"].min()) , 30,  14) # Default value of 14
        return churn_days
    def get_timeframe():
        timeframe_selection = ['2 Weeks', '1 Month', '3 Months']
        timeframe = st.selectbox('Select a timeframe :date:', timeframe_selection)
        return timeframe
    def get_city():
        city_selection = st.multiselect("Select City :cityscape:", options = load_city_enc())
        city_list = list(city_selection)
        return city_list
    def get_city_int(city_input):
        # Load city mapping 
        city_dict = load_city_enc()
        city_int = [] # Store selected city frequency 
        for each in city_input:
            city_int.append(city_dict[each]['CITY_FREQUENCY'])
        return city_int
    def get_years_with_us(cNum):
        # Slider
        timeframe_label =  [1,2,3,4]
        years_with_us_range = cNum.slider("Years as Member",timeframe_label[0],timeframe_label[-1], (timeframe_label[0],timeframe_label[-1]))
        return years_with_us_range
    #=================================================================================================#
    # Functions for analytics
    
    # Function to filter
    def filter_higher(next_purchase_data, uplift_predictions_time, churn_days_higher, years_with_us_range, city_int):
        custGroup = next_purchase_data[ (next_purchase_data["TARGET"] > churn_days_higher)] # Churn days higher 
        filtered_data_higher = uplift_predictions_time[ (uplift_predictions_time['YEARS_WITH_US'] >= years_with_us_range[0]) & (uplift_predictions_time['YEARS_WITH_US'] <= years_with_us_range[1]) 
                                            & (uplift_predictions_time['CITY_FREQUENCY'].isin(city_int)) ]
    
        filtered_data_higher = filtered_data_higher[filtered_data_higher["CUSTOMER_ID"].isin(custGroup["CUSTOMER_ID"])]
        filtered_data_higher = filtered_data_higher.drop(columns=['CUSTOMER_ID','MONETARY_M3_HO','LTVCluster','PREDICTED_PROBA_0','PREDICTED_PROBA_1','PREDICT_SALES','INDEX'])
        #st.write(filtered_data_higher)
        return filtered_data_higher 
    
    def filter_lower(next_purchase_data, uplift_predictions_time, churn_days_lower, years_with_us_range, city_int):
        custGroup = next_purchase_data[ (next_purchase_data["TARGET"] <= churn_days_lower)] # Churn days falling below
        filtered_data_lower = uplift_predictions_time[ (uplift_predictions_time['YEARS_WITH_US'] >= years_with_us_range[0]) & (uplift_predictions_time['YEARS_WITH_US'] <= years_with_us_range[1]) 
                                            & (uplift_predictions_time['CITY_FREQUENCY'].isin(city_int)) ]

        filtered_data_lower = filtered_data_lower[filtered_data_lower["CUSTOMER_ID"].isin(custGroup["CUSTOMER_ID"])]
        filtered_data_lower = filtered_data_lower.drop(columns=['CUSTOMER_ID','MONETARY_M3_HO','LTVCluster','PREDICTED_PROBA_0','PREDICTED_PROBA_1','PREDICT_SALES','INDEX'])
        #st.write(filtered_data_lower)
        return filtered_data_lower  
        
    # Function to find uplift
    def find_uplift(pred_df,cluster_sales, uplift_predictions_duration):
        
        pred_df["PREDICT_SALES_ST"] = (pred_df[0] * cluster_sales['Cluster0MeanSales'][0]) + (pred_df[1] * cluster_sales['Cluster1MeanSales'][0])
        
        # Merge pred_df and filtered data
        result_df = uplift_predictions_duration.merge(pred_df, left_index=True, right_index=True, how='right')
        #st.write(result_df)
        
        # Total sales
        totalSales = result_df['PREDICT_SALES_ST'].sum()
        # Total uplift
        totalUplift = result_df['PREDICT_SALES_ST'].sum() - result_df['MONETARY_M3_HO'].sum()
        # Calculation for change in revenue
        percentUplift = ((result_df['PREDICT_SALES_ST'].sum()- result_df['MONETARY_M3_HO'].sum())/ result_df['MONETARY_M3_HO'].sum()) * 100
        
        return result_df, totalSales, totalUplift, percentUplift
    def recomendation_text(churn_uplift, nonChurn_Uplift):
        st.markdown("#### "+ "Practical Insights :mag:")
        if (churn_uplift > nonChurn_Uplift):
                recommendation = '''Churned customers have yielded a greater uplift when compared to their non-churning counterparts. Churned customers leads to a potential loss in revenue, and the analytics displayed above confirm that they are likely to generate a higher revenue uplift than customers who have not churned. Moreover, retaining existing customers proves to be a more cost-effective strategy than acquiring new ones, as churned customers already possess a certain level of familiarity with Tasty Bytes.
                Churn customers represents potential lost in revenue, and from the analytics shown above, we acknowledged that they are likely to generate a higher uplift in revenue as compared to customers who have not churn.
                
                \nConsequently, re-engaging with them holds the promise of securing purchases and contributing to Tasty Bytes' revenue over the long term, aligning with Tasty Bytes' objective of achieving a 25% year-over-year growth over the next five years.
                \nHence, the marketing team can access the CSV file containing customer details of churned customers. Armed with this information, the team can execute personalized marketing strategies targeting churned customers, cultivating enduring relationships that will help Tasty Bytes in achieving its goal.
                '''
                st.write(recommendation)
        else:
            recommendation = '''Non-churning customers have yielded a greater uplift when compared to customers that have churned. Non-churning customers are engaged and actively contributing to Tasty Bytes's revenue. 
            This group of customers have established their loyalty by contributing consistently to the revenue stream, this allows Tasty Bytes revenue to be more predictable, allowing for better financial planning and forecasting.
            
            \nAs a strong base of non churning customers provides stable growth towards the business therefore, the marketing team can access the CSV file containing customer details of non-churning customers. Armed with this information, the team can execute personalized marketing strategies targeting churned customers, cultivating enduring relationships that will help Tasty Bytes in achieving its goal.
            '''
            st.write(recommendation)
        
    # Function to display model predictions in visulisations, seperated by tabs 
    def display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df):
        
        # Grouping results into different aspects
        totalSales = [nonChurnTotalSales, churnTotalSales]
        totalUplift = [nonChurn_totalUplift, churn_totalUplift]
        percentUplift = [nonChurn_percentUplift, churn_percentUplift]
        
        # Display group of customers
        index = ['Non-Churn','Churn']
        
        # Storing results as dataframes
        df_totalSales = pd.DataFrame({'Total Sales': totalSales}, index=index)
        df_totalUplift = pd.DataFrame({'Total Uplift': totalUplift}, index=index)
        df_percentUplift = pd.DataFrame({'Percentage Uplift': percentUplift}, index=index) 
        
        # Sub tab for user to navigate 
        tab1, tab2, tab3 = st.tabs(["Total Sales", "Total Uplift", "Percentage Uplift"])

        # Total Sales Tab
        with tab1:
            st.write("#### "+ "Analysis of Total Sales for Churn & Non-Churn Customers :heavy_dollar_sign:")
            # Bar chart for total sales analysis 
            totalSalesGraph = go.Figure()

            totalSalesGraph.add_trace(
                go.Bar(
                    x = index,
                    y = df_totalSales['Total Sales'],
                    marker_color='#003f5c',
                    text = df_totalSales['Total Sales'],
                    textposition='inside',
                    texttemplate='$%{y:,.0f}'
                )
            )

            # Customize the layout
            totalSalesGraph.update_layout(
                xaxis_title='Type of Customer',
                yaxis_title='Amount ($)',
                font=dict(
                size=12,
                color='#000000') ,
                showlegend=False
            )
            # Plotting
            st.plotly_chart(totalSalesGraph)
            # Display insights 
            st.write("Total Sales Generated for Non-Churn Customers: ${:,.2f}".format(nonChurnTotalSales))
            st.write("Total Sales Generated for Churn Customers: ${:,.2f}".format(churnTotalSales))

        # Total Uplift Tab
        with tab2:
            st.write("#### "+ "Analysis of Total Uplift for Churn & Non-Churn Customers :chart_with_upwards_trend: ")
            totalUpliftGraph = go.Figure()

            totalUpliftGraph.add_trace(
                go.Bar(
                    x = index,
                    y = df_totalUplift['Total Uplift'],
                    marker_color='#003f5c',
                    text = df_totalUplift['Total Uplift'],
                    textposition='inside',
                    texttemplate='$%{y:,.0f}'
                )
            )

            # Customize the layout
            totalUpliftGraph.update_layout(
                xaxis_title='Type of Customer',
                yaxis_title='Amount ($)',
                font=dict(
                size=12,
                color='#000000') ,
                showlegend=False
            )
            # Plotting
            st.plotly_chart(totalUpliftGraph)
            # Display insights
            
            # Negative uplift for non churn
            if (nonChurn_totalUplift >= 0):
                st.write("Total Uplift for Non-Churn Customers: ${:,.2f} :large_green_circle: ".format(nonChurn_totalUplift))
            else:
                st.write("Total Uplift for Non-Churn Customers: ${:,.2f} :small_red_triangle_down:".format(nonChurn_totalUplift))  
            # Negative uplift for churn
            if (churn_totalUplift >= 0):
                st.write("Total Uplift for Churn Customers: ${:,.2f} :large_green_circle: ".format(churn_totalUplift))
            else:
                st.write("Total Uplift for Churn Customers: ${:,.2f} :small_red_triangle_down:".format(churn_totalUplift))
            
            st.write("Displayed above, the uplift generated by non-churn customers amounts to {:,.2f} stemming from the participation of {} US customers. While the uplift attributed to churn customers stands at {:,.2f} with the involvement of {} US customers.".format(nonChurn_totalUplift, len(nonChurn_df.index), churn_totalUplift, len(churn_df.index)))
            
            # Recomendation and Call-To-Action
            recomendation_text(churn_totalUplift,nonChurn_totalUplift)
            downloadCustomerDetails(nonChurn_df, churn_df,key1="tabTotal1", key2="tabTotal2")
            
            

        # Percentage (%) Tab 
        with tab3:
            st.write("#### "+ "Analysis of Percentage Uplift (%) for Churn & Non-Churn Customers")
            percentUpliftGraph = go.Figure()

            percentUpliftGraph.add_trace(
                go.Bar(
                    x = index,
                    y = df_percentUplift['Percentage Uplift'],
                    marker_color='#003f5c',
                    text = df_percentUplift['Percentage Uplift'],
                    textposition='inside',
                    texttemplate='%{y:,.2f} %'
                )
            )

            # Customize the layout
            percentUpliftGraph.update_layout(
                xaxis_title='Type of Customer',
                yaxis_title='Percentage (%)',
                font=dict(
                size=12,
                color='#000000') ,
                showlegend=False
            )
            # Plotting
            st.plotly_chart(percentUpliftGraph)
            # Display insights
            # Negative uplift for non churn
            if (nonChurn_percentUplift >= 0):
                st.write("Total Percentage Uplift for Non-Churn Customers: {:,.2f} % :large_green_circle: ".format(nonChurn_percentUplift))
            else:
                st.write("Total Percentage Uplift for Non-Churn Customers: {:,.2f} % :small_red_triangle_down:".format(nonChurn_percentUplift))  
            # Negative uplift for churn
            if (churn_percentUplift >= 0):
                st.write("Total Percentage Uplift for Churn Customers: {:,.2f} % :large_green_circle: ".format(churn_percentUplift))
            else:
                st.write("Total Percentage Uplift for Churn Customers: {:,.2f} % :small_red_triangle_down:".format(churn_percentUplift))
            
            # Recomendation and Call-To-Action
            recomendation_text(churn_percentUplift,nonChurn_percentUplift)
            downloadCustomerDetails(nonChurn_df, churn_df,key1="tabPercent1", key2="tabPercent2")
            
            
    # Download filtered customer details     
    def downloadCustomerDetails(nonChurn_df, churn_df, key1, key2):   
        st.markdown("""---""")
        # Description of Callable Action
        st.markdown("###" +' :clipboard: Download Customer Details')
        callToActionDescription = '''
        Analysis on the predictions for churn and non-churn customer groups are displayed above, now you can download customer details that belongs to the respectively group by simply clicking a button :three_button_mouse:. 
        '''
        st.markdown(callToActionDescription)
        
        customerDetails_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.RAW_CUSTOMER.CUSTOMER_LOYALTY").to_pandas()
        
        download_nonChurn , download_churn = st.columns(2) # Position Left Right
        download_nonChurn.download_button(
            label="Non-Churn Customer Details as CSV",
            data= customerDetails_df[customerDetails_df["CUSTOMER_ID"].isin(nonChurn_df["CUSTOMER_ID"])].to_csv().encode('utf-8'),
            file_name='nonChurn_customer_details.csv',
            mime='text/csv',
            key=key1
        )
        download_churn.download_button(
            label="Churn Customers Details as CSV",
            data = customerDetails_df[customerDetails_df["CUSTOMER_ID"].isin(churn_df["CUSTOMER_ID"])].to_csv().encode('utf-8'),
            file_name='churn_customer_details.csv',
            mime='text/csv',
            key=key2
        )
    
    #=================================================================================================#
    # Tab Content
    
    # Load dataset
    days_to_next_purchase = load_next_purchase()

    # Draw the histogram
    histogram_days_to_next_purchase = go.Figure()

    histogram_days_to_next_purchase.add_trace(
        go.Histogram(
            name='Histogram of Days to next Purchase',
            x= days_to_next_purchase['TARGET'], marker_color='#003f5c', histnorm='density'
        )
    )
    # Customize histogram
    histogram_days_to_next_purchase.update_layout(
        title={
            'text' : 'Distribution of Days to Next Purchase',
            'x':0.5,
            'xanchor': 'center',
            "font" :dict(size=20)
        },
        xaxis_title='Days',
        yaxis_title='No. of Customers',
        showlegend=False
    )
    # Display the bar chart in the Streamlit tab
    st.plotly_chart(histogram_days_to_next_purchase)
    
    histogramAnalysis = '''
    Based on the histogram provided, it can be deduced that the distribution for days until the next purchase is skewed to the right. 
                The majority of US customers are anticipated to make a visit to Tasty Bytes within the upcoming 35 days.
    \nBy analyzing this plot, you can determine the number of days it takes for customers to churn, how many years they have been members, and then choose a specific timeframe for making predictions.'''
                
    st.markdown(histogramAnalysis)
    
    # Streamlit Tab Design
    # Define the user input fields
    st.write("###"+ " Predict Uplift")
    user_input_position_1 , user_input_position_2 = st.columns(2) # Position Left Right
    churn_days = get_churn_days(user_input_position_1) # Select customer base
    years_with_us_range = get_years_with_us(user_input_position_2) # Select duration of membership 
    timeframe_input = get_timeframe() # Select Timeframe Model 
    city_input = get_city() # Select City 
    
    
    if st.button('Predict Uplift'):
        # 2 Week Model timeframe 
        if (timeframe_input == '2 Weeks'):
            # Load the 2W Uplift Model
            uplift_predictions_2W = load_Uplift_Churn_2W()
            
            # Load the next purchase date 
            next_purchase_data = load_next_purchase()
            # Get city int  
            city_int = get_city_int(city_input)
            if (len(city_int) == 0):
                st.write("To proceed, please choose at least one city.")
            else:
                cluster_sales = load_cluster_sales_2W()
                # --- Uplift Analysis for Non-Churn Customers --- 
                # Filter for customer that did not churn (lower than churn days)
                filtered_lower = filter_lower(next_purchase_data, uplift_predictions_2W, churn_days, years_with_us_range, city_int)
                pred = uplift_2W.predict_proba(filtered_lower)
                pred_df_lower = pd.DataFrame(pred, index=filtered_lower.index)
                
                nonChurn_df, nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift = find_uplift(pred_df_lower, cluster_sales, uplift_predictions_2W)

                # --- Uplift Analysis for Churn Customers ---
                # Filter for customer that churn (higher than churn days)
                filtered_higher = filter_higher(next_purchase_data, uplift_predictions_2W, churn_days, years_with_us_range, city_int)
                
                # Make a prediction, write as dataframe
                pred = uplift_2W.predict_proba(filtered_higher)
                pred_df_higher = pd.DataFrame(pred, index=filtered_higher.index)

                churn_df, churnTotalSales, churn_totalUplift, churn_percentUplift = find_uplift(pred_df_higher, cluster_sales, uplift_predictions_2W)
                
                # Compare Churn and Non Churn 
                display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df) 
                
        # 1 Month Model timeframe    
        elif (timeframe_input == '1 Month'):
            # Load the 1M Uplift Model
            uplift_predictions_1M = load_Uplift_Churn_1M()
            
            # Load the next purchase date 
            next_purchase_data = load_next_purchase()
            # Get city int  
            city_int = get_city_int(city_input)
            if (len(city_int) == 0):
                st.write("To proceed, please choose at least one city.")
            else:
                cluster_sales = load_cluster_sales_1M()
                # --- Uplift Analysis for Non-Churn Customers --- 
                # Filter for customer that did not churn (lower than churn days)
                filtered_lower = filter_lower(next_purchase_data, uplift_predictions_1M, churn_days, years_with_us_range, city_int)
                pred = uplift_1M.predict_proba(filtered_lower)
                pred_df_lower = pd.DataFrame(pred, index=filtered_lower.index)
    
                nonChurn_df, nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift = find_uplift(pred_df_lower, cluster_sales, uplift_predictions_1M)
                
                # --- Uplift Analysis for Churn Customers ---
                # Filter for customer that churn (higher than churn days)
                filtered_higher = filter_higher(next_purchase_data, uplift_predictions_1M, churn_days, years_with_us_range, city_int)
                
                # Make a prediction, write as dataframe
                pred = uplift_1M.predict_proba(filtered_higher)
                pred_df_higher = pd.DataFrame(pred, index=filtered_higher.index)
    
                churn_df, churnTotalSales, churn_totalUplift, churn_percentUplift = find_uplift(pred_df_higher, cluster_sales, uplift_predictions_1M)
                
                # Compare Churn and Non Churn 
                display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df)
                
        # 3 Month Model timeframe 
        elif (timeframe_input == '3 Months'):
            # Load the 3M Uplift Model
            uplift_predictions_3M = load_Uplift_Churn_3M()
            
            # Load the next purchase date 
            next_purchase_data = load_next_purchase()
            # Get city int  
            city_int = get_city_int(city_input)
            if (len(city_int) == 0):
                st.write("To proceed, please choose at least one city.")
            else:
                cluster_sales = load_cluster_sales_3M()
                # --- Uplift Analysis for Non-Churn Customers --- 
                # Filter for customer that did not churn (lower than churn days)
                filtered_lower = filter_lower(next_purchase_data, uplift_predictions_3M, churn_days, years_with_us_range, city_int)
                pred = uplift_3M.predict_proba(filtered_lower)
                pred_df_lower = pd.DataFrame(pred, index=filtered_lower.index)                
    
                nonChurn_df, nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift = find_uplift(pred_df_lower, cluster_sales, uplift_predictions_3M)
                
                # --- Uplift Analysis for Churn Customers ---
                # Filter for customer that churn (higher than churn days)
                filtered_higher = filter_higher(next_purchase_data, uplift_predictions_3M, churn_days, years_with_us_range, city_int)
                
                # Make a prediction, write as dataframe
                pred = uplift_3M.predict_proba(filtered_higher)
                pred_df_higher = pd.DataFrame(pred, index=filtered_higher.index)
    
                churn_df, churnTotalSales, churn_totalUplift, churn_percentUplift = find_uplift(pred_df_higher, cluster_sales, uplift_predictions_3M)
                
                 # Compare Churn and Non Churn 
                display_prediction_analysis(nonChurnTotalSales, nonChurn_totalUplift, nonChurn_percentUplift, nonChurn_df, churnTotalSales, churn_totalUplift, churn_percentUplift, churn_df)
                 
with tab5:
    menu_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_pos.menu")
    truck_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_pos.truck")
    truck_df = truck_df.with_column('LAST_DATE', F.iff(F.col("TRUCK_ID") == F.col('TRUCK_ID'), "2022-10-18", '0'))
    truck_df = truck_df.withColumn("DAYS_OPENED", F.datediff("day", F.col("TRUCK_OPENING_DATE"), F.col('LAST_DATE')))
    menu_df = menu_df.to_pandas()
    truck_df = truck_df.to_pandas()
    