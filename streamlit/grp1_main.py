# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from datetime import date
import pickle
import json
import math
import plotly.graph_objects as go
# Import Snowflake modules
from snowflake.snowpark import Session
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
from snowflake.snowpark import Window
from snowflake.snowpark.functions import col, date_add, to_date, desc, row_number
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
st.set_page_config(page_title='ICP ASG 3', page_icon="favicon.ico")

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['Predicting Future Sales [Shi Wei]', 'Ernest', 'Gwyneth', 'GF', 'KK'])

with tab1:
    st.title('Predicting Future Sales')
    description = """
    Welcome to the 'Predict Future Sales' tab! 
    This dashboard is designed to help Tasty Byte's team analyze and predict future sales, aligning with the company's ambitious goal of achieving 25% YoY growth over the next 5 years.

    With this interactive tool, you can explore valuable insights that will contribute to your strategic decision-making process. 
    Gain a deeper understanding of sales trends, identify growth opportunities, and make data-driven decisions to propel Tasty Byte towards its long-term vision."""
    st.markdown(description)
    
    ## Define function to load the customer's cluster results
    def SW_load_cust_cluster():
        data_cust_cluster = pd.read_csv("./sw_datasets/cluster_results.csv") 
        data_cust_cluster = pd.DataFrame(data_cust_cluster)
        return data_cust_cluster
    
    ## Define function to load the uplift prediction model
    def SW_load_uplift_1M():
        data_1m = pd.read_csv("./sw_datasets/UpliftPrediction[1M].csv") 
        data_1m = pd.DataFrame(data_1m)
        # Load customer cluster data
        data_cust_cluster = SW_load_cust_cluster()
        data_1m = pd.merge(data_1m, data_cust_cluster, on='CUSTOMER_ID')
        # Return merged data
        return data_1m
    
    def SW_load_uplift_2M():
        data_2m = pd.read_csv("./sw_datasets/UpliftPrediction[2M].csv") 
        data_2m = pd.DataFrame(data_2m)
        # Load customer cluster data
        data_cust_cluster = SW_load_cust_cluster()
        data_2m = pd.merge(data_2m, data_cust_cluster, on='CUSTOMER_ID')
        # Return merged data
        return data_2m
    
    def SW_load_uplift_3M():
        data_3m = pd.read_csv("./sw_datasets/UpliftPrediction[3M].csv") 
        data_3m = pd.DataFrame(data_3m)
        # Load customer cluster data
        data_cust_cluster = SW_load_cust_cluster()
        data_3m = pd.merge(data_3m, data_cust_cluster, on='CUSTOMER_ID')
        return data_3m
    
    def SW_get_predicted_sales(data):
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
    nov = SW_get_predicted_sales(SW_load_uplift_1M())
    dec = SW_get_predicted_sales(SW_load_uplift_2M())
    jan = SW_get_predicted_sales(SW_load_uplift_3M())
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
    st.subheader('Insights')
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
    cluster_results_df = pd.DataFrame(SW_load_cust_cluster())
    
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
    
    ## Predict Future Sales Based On Customer Cluster
    st.subheader('Predict Future Sales Based On Customer Cluster')

    ## Define user input functions
    # User Input 1: Select Customer Cluster
    def SW_get_cust_cluster():
        # Display the dropdown box
        cluster_selection = ['1 - Low Value (Customers who buy less frequently and generate lower sales)', 
                             '2 - Middle Value (Customers who make average purchases)', 
                             '3 - High Value (Customers who make frequent purchases and generate higher sales.)']
        selected_cluster = st.selectbox(
            "Select Customer Cluster:", cluster_selection)
        if selected_cluster == '1 - Low Value (Customers who buy less frequently and generate lower sales)':
            return 1
        elif selected_cluster == '2 - Middle Value (Customers who make average purchases)':
            return 0
        else:
            return 2
    
    # User Input 2: Select Timeframe
    def SW_get_timeframe():
        # Display the dropdown box
        timeframe_selection = ['1 month', '2 months', '3 months']
        selected_months = st.selectbox(
            "Select the range of months for prediction:", timeframe_selection)
        return selected_months
    
    # User Input 3: Select Metric
    def SW_get_selected_metrics():
        # Display checkboxes for key metrics
        st.write("Select the metrics to view:")
        show_total_sales = st.checkbox("Total Predicted sales", value=True)
        show_avg_spending = st.checkbox("Average Predicted Spending per Customer", value=True)
        selected_metrics = []
        if show_total_sales:
            selected_metrics.append("0")
        if show_avg_spending:
            selected_metrics.append("1")
        return selected_metrics
    
    ## Define function to get results to display
    def SW_process_data(data, cluster_input):
    # Filter the data based on cluster_input
        filtered_data_cluster = data[data['CLUSTER'] == cluster_input]
        # Replace all values in 'CLUSTER' so we can find the number of customers
        filtered_data_cluster['CLUSTER'] = 1
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
    

    ## Define function to display results in a map based on selected metric
    def SW_display_map(final_df, metrics_input):
        # Create the map figure
        fig = go.Figure()

        # Adjust bubble size based on sales (scaled down for aesthetics)
        fig.add_trace(
            go.Scattermapbox(
                lon=final_df['LONGITUDE'],
                lat=final_df['LATITUDE'],
                text=final_df['CITY'],
                customdata=final_df[['PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']],
                hoverinfo='text',  # Display the text when hovering over the markers
                mode='markers',
                marker=dict(
                    size=final_df['PREDICT_SALES_Rounded'] / 500000,  # Adjust marker size based on sales (scaled down for aesthetics)
                    sizemode='area',  # Use area to scale the marker size
                    sizeref=0.003,  # Adjust the scaling factor for marker size
                    sizemin=5,  # Set the minimum size for the markers
                    color=final_df['UPLIFT_PERCENTAGE'],  # Use Uplift Percentage as the marker color
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
            if '0' in metrics_input and '1' in metrics_input:
                label = f"Total Predicted Sales: ${row['PREDICT_SALES_Rounded']:.2f}<br>" + \
                        f"Average Predicted Spending per Customer: ${row['AVG_SPENDING_Rounded']:.2f}<br>" + \
                        f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            elif '0' in metrics_input:
                label = f"Total Predicted Sales: ${row['PREDICT_SALES_Rounded']:.2f}<br>" + \
                        f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            elif '1' in metrics_input:
                label = f"Average Predicted Spending per Customer: ${row['AVG_SPENDING_Rounded']:.2f}<br>" + \
                        f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
            else:
                label = f"Percentage Uplift: {row['UPLIFT_PERCENTAGE']:.2f}%"
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
            title={
                'text': 'Predicted Sales by City<br><span style="font-size:10px; font-weight:normal">*The size of each bubble represents the predicted sales amount.</span>',
                'y': 0.95,  # Adjust the vertical position of the title
                'x': 0.5,  # Center the title horizontally
                'xanchor': 'center',
                'yanchor': 'top',
            },
            mapbox=dict(
                style='carto-positron',
                zoom=2.5,  # Set the initial zoom level
                center=dict(lat=44, lon=-95.7129),  # Set the initial center of the map
            ),
            showlegend=True,  # Show the legend
            legend=dict(
                title='Predicted Sales ($)',  # Legend title
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
            )
        )

        # Display the map in the Streamlit tab
        st.plotly_chart(fig)

    ## Define Function to display results in table
    def SW_display_table(data, metrics_input):
        if metrics_input == ['0']:
            final_df = data[['CITY', 'PREDICT_SALES_Rounded', 'UPLIFT_PERCENTAGE']]
            final_df.rename(columns={'CITY':'City',
                                'PREDICT_SALES_Rounded': 'Predicted Sales ($)',
                                'UPLIFT_PERCENTAGE': 'Uplift Percentage (%)'}, inplace=True)
            
        elif metrics_input == ['1']:
            final_df = data[['CITY', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']]
            final_df.rename(columns={'CITY':'City',
                                'AVG_SPENDING_Rounded': 'Average Sales Per Customer ($)',
                                'UPLIFT_PERCENTAGE': 'Uplift Percentage (%)'}, inplace=True)
            
        else:
            final_df = data[['CITY', 'PREDICT_SALES_Rounded', 'AVG_SPENDING_Rounded', 'UPLIFT_PERCENTAGE']]
            final_df.rename(columns={'CITY':'City',
                                'PREDICT_SALES_Rounded': 'Predicted Sales ($)',
                                'AVG_SPENDING_Rounded': 'Average Sales Per Customer ($)',
                                'UPLIFT_PERCENTAGE': 'Uplift Percentage (%)'}, inplace=True)
            
        st.subheader("Predicted Sales and Average Sales Per Customer with Uplift Percentage for Each City")
        st.write(final_df)
        return
    
    ## Get user inputs
    cluster_input = SW_get_cust_cluster()
    timeframe_input = SW_get_timeframe()
    metrics_input = SW_get_selected_metrics()

    # Display the "Predict" button
    if st.button("Predict"): 
        if (timeframe_input == '1 month'):
            ## Load data based on selected timeframe
            data_1M = SW_load_uplift_1M()
            
            # Filter and process the data based on user input
            final_df = SW_process_data(data_1M, cluster_input)
            
            # Display the map in the Streamlit tab
            SW_display_map(final_df, metrics_input)
            
            # Display table result
            SW_display_table(final_df, metrics_input)
            
            # Display actionable insights (These are customized based on user's input)
            st.subheader("Actionable Insights Based on Selection")
            if cluster_input == 1: #Low Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (low value) over a one-month timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   San Mateo has the highest predicted sales amount of $1,234,487.37 and a relatively high uplift percentage of 62.86%. Consider running targeted marketing campaigns in this city to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Sales Team Focus:")
                st.write("   Denver has a relatively lower uplift percentage of 38.84% compared to other cities. Consider directing the sales team's efforts towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   Boston exhibits a significant uplift percentage of 111.19%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
                
            elif cluster_input == 2: #High Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (high value) over a one-month timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Consider running targeted marketing campaigns in San Mateo, which shows a significantly higher uplift percentage of 48.64%. Offer personalized promotions or incentives to attract more customers and increase sales in this area.")

                st.write("2. Customer Retention Efforts:")
                st.write("   Focus on customer retention efforts in Seattle and New York City, where negative uplift percentages (-2.30% and -2.08%) indicate a slight decline in predicted sales. Identify and address reasons for the decline to retain existing customers and prevent further loss in sales.")

                st.write("3. Pricing Strategy:")
                st.write("   Capitalize on the positive uplift percentage (15.44%) observed in Boston by implementing dynamic pricing strategies or limited-time offers. This can help further boost sales in the city.")

            else: #Middle Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (middle value) over a one-month timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Consider running targeted marketing campaigns in Boston, which shows a significantly higher uplift percentage of 48.87%. The company can offer personalized promotions or incentives to attract more customers and increase sales in this area.")
                
                st.write("2. Sales Team Focus")
                st.write("   Denver has a relatively lower uplift percentage of 10.02% compared to other cities. Consider directing the sales team's efforts towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   Boston exhibits a significant uplift percentage of 48.87%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")

        elif (timeframe_input == '2 months'):
            ## Load data based on selected timeframe
            data_2M = SW_load_uplift_2M()
            
            # Filter and process the data based on user input
            final_df = SW_process_data(data_2M, cluster_input)
            
            # Display the map in the Streamlit tab
            SW_display_map(final_df, metrics_input)
            
            # Display table result
            SW_display_table(final_df, metrics_input)
            
            # Display actionable insights (These are customized based on user's input)
            st.subheader("Actionable Insights Based on Selection")
            if cluster_input == 1: #Low Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (low value) over a two-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("  San Mateo has the highest predicted sales amount of $1,236,987.90 and a relatively high uplift percentage of 63.19%. Consider running targeted marketing campaigns in this city to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Sales Team Focus:")
                st.write("   Denver has a relatively lower uplift percentage of 39.43% compared to other cities. Consider directing the sales team's efforts towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   Boston exhibits a significant uplift percentage of 111.69%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
                
            elif cluster_input == 2: #High Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (high value) over a two-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   San Mateo stands out with the predicted sales amount of $7,526.23 and the only positive uplift percentage of 2.00%. Consider running targeted marketing campaigns in San Mateo to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Customer Retention Efforts:")
                st.write("   Focus on customer retention efforts in Denver, where negative uplift percentage (-18.09%) indicate a slight decline in predicted sales. Identify and address reasons for the decline to retain existing customers and prevent further loss in sales.")

                st.write("3. Pricing Strategy:")
                st.write("   Since San Mateo exhibits a positive uplift percentage of 2.00%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")

            else: #Middle Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (middle value) over a two-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Consider running targeted marketing campaigns in Boston, which shows a significantly higher uplift percentage of 49.61%. The company can offer personalized promotions or incentives to attract more customers and increase sales in this area.")
                
                st.write("2. Sales Team Focus")
                st.write("   Denver has a relatively lower uplift percentage of 10.56% compared to other cities. Consider directing the sales team's efforts towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   Boston exhibits a significant uplift percentage of 49.61%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
            
        else:
            ## Load data based on selected timeframe
            data_3M = SW_load_uplift_3M()
            
            # Filter and process the data based on user input
            final_df = SW_process_data(data_3M, cluster_input)
            
            # Display the map in the Streamlit tab
            SW_display_map(final_df, metrics_input)
            
            # Display table result
            SW_display_table(final_df, metrics_input)
            
            # Display actionable insights (These are customized based on user's input)
            st.subheader("Actionable Insights Based on Selection")
            if cluster_input == 1: #Low Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (low value) over a three-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("  San Mateo stands out with the highest predicted sales amount of $2,984,755.02 and a relatively high uplift percentage of 37.65%. The company can run targeted marketing campaigns in this city to capitalize on the high predicted sales and further increase customer engagement. Offering personalized promotions or incentives can attract more customers and drive sales even higher in San Mateo.")

                st.write("2. Sales Team Focus:")
                st.write("   Denver has a relatively lower uplift percentage of 25.58% compared to other cities. The company should consider directing the sales team's efforts towards Denver to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   Boston exhibits a significant uplift percentage of 63.94%. The company should consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
                
            elif cluster_input == 2: #High Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (high value) over a three-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   San Mateo stands out with the predicted sales amount of $3,116.67 and an impressive uplift percentage of 49.12%. Consider running targeted marketing campaigns in San Mateo to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")

                st.write("2. Customer Retention Efforts:")
                st.write("   Focus on customer retention efforts in Denver, where negative uplift percentage (-10.18%) indicate a slight decline in predicted sales. Identify and address reasons for the decline to retain existing customers and prevent further loss in sales.")

                st.write("3. Pricing Strategy:")
                st.write("   Boston exhibits a significant uplift percentage of 16.14%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")

            else: #Middle Value
                st.write("Below are some actionable insights based on the predicted sales and uplift percentage for each city within the customer cluster (middle value) over a three-months timeframe:")
                st.write("1. Targeted Marketing Campaigns:")
                st.write("   Boston and New York City have relatively high predicted sales of $1,526,225.99 and $1,202,009.54, respectively. Consider running targeted marketing campaigns in these cities to capitalize on the high predicted sales and further increase customer engagement. Offer personalized promotions or incentives to attract more customers and drive sales even higher.")
                
                st.write("2. Sales Team Focus")
                st.write("   Denver has a negative uplift percentage of 0.45%. Consider directing the sales team's efforts towards this city to explore opportunities for growth and expansion. The sales team can focus on building relationships with potential customers, understanding their needs, and offering tailored solutions to drive sales in Denver.")

                st.write("3. Pricing Strategy:")
                st.write("   New York City exhibits a significant uplift percentage of 20.89%. Consider implementing dynamic pricing strategies or limited-time offers in this city to take advantage of the positive sales trend. By adjusting prices based on demand and customer behavior, the company can potentially further boost sales and revenue in Boston.")
        
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

    if st.button('Predict customer spending'):
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
    st.write("In this webtab, Tasty Byte management will have the ability to obtain details about churn customers across various customer segments. They can leverage the insights and advice provided to take proactive measures aimed at retaining these customers in order for Tasty Bytes to reach their goal of increasing NPS from 3 to 40 by year end 2023.")
    st.write("Additionally, the management will also be able to experiment with customer's details to predict whether they will be likely to churn or not.")
    st.write("Customers are likely to churn when their predicted days to next purchase is more than 14 days.")
    st.header('Details of Churn Customers')

    # loading of dataset 
    def load_next_purchase_cust_seg():
        data = pd.read_csv("./churn/NextPurchaseCustSeg2.csv")
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
    # select customer segment
    cust_seg = st.selectbox(
    'Select the information of the customer segment that you would like to view',
    options = ['Low Value (Customers who buy less frequently and generate lower sales)', 
                             'Middle Value (Customers who make average purchases)', 
                             'High Value (Customers who make frequent purchases and generate higher sales)'])
    
    # show percentage of churn and not churn of customer segment chosen using bar charts
    cust_seg_option = cust_seg.split('(')[0].strip()
    filtered_cust_seg = filter_cust_seg(cust_seg_option)
    churn_label_mapping = {0: 'Not Churn', 1: 'Churn'}
    filtered_cust_seg['CHURN_STATUS'] = filtered_cust_seg['CHURN_STATUS'].map(churn_label_mapping)
    cust_churn_bar = filtered_cust_seg['CHURN_STATUS'].value_counts()
    st.bar_chart(data = cust_churn_bar)

    # show details of cust likely to churn 
    st.write("Details of customers likely to churn")
    churn_cust = filter_cust_churn("Churn")
    not_churn_cust = filter_cust_churn("Not Churn")
    customer_df = session.table("NGEE_ANN_POLYTECHNIC_FROSTBYTE_DATA_SHARE.raw_customer.customer_loyalty")
    us_customer_df_sf = customer_df.filter(F.col("COUNTRY")=="United States")
    us_customer_df = us_customer_df_sf.to_pandas()
    us_customer_df = us_customer_df[us_customer_df['CUSTOMER_ID'].isin(churn_cust['CUSTOMER_ID'])]
    us_customer_df = pd.merge(us_customer_df, churn_cust, on='CUSTOMER_ID', how='inner')
    us_customer_df = us_customer_df.sort_values(by='PREDICTED')
    us_customer_df = us_customer_df.reset_index(drop=True)
    cust_to_show = us_customer_df[["FIRST_NAME", "LAST_NAME", "GENDER", "MARITAL_STATUS", "CHILDREN_COUNT", "BIRTHDAY_DATE", "E_MAIL", "PHONE_NUMBER", "TOTAL_SPENT", "TOTAL_ORDER", "YEARS_WITH_US", "PREDICTED"]]
    cust_to_show.rename(columns={'PREDICTED': 'PREDICTED_DAYS_TO_NEXT_PURCHASE'}, inplace=True)
    cust_to_show['YEARS_WITH_US'] = cust_to_show['YEARS_WITH_US'].round().astype(int)
    cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'] = cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].astype(int)
    st.dataframe(cust_to_show)

    # give insights about the churn customers
    avg_churn_recency = str(math.floor(churn_cust['RECENCY_DAYS'].mean()))
    avg_not_churn_recency = str(math.floor(not_churn_cust['RECENCY_DAYS'].mean())) 
    min_predicted = str(cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].min())
    max_predicted = str(cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].max())
    avg_predicted = str(math.floor(cust_to_show['PREDICTED_DAYS_TO_NEXT_PURCHASE'].mean()))
    st.subheader("Insights")
    st.write("Out of all the " + str(cust_seg_option) + " customers, "+ str(len(cust_to_show)) + " of them are likely to churn as the average time since their last order is approximately " + str(avg_churn_recency) + " days, compared to unlikely to churn customers of " + str(avg_not_churn_recency) + " days.")
    st.write("These customers have a predicted "+ min_predicted + "-"+ max_predicted + " days to next purchase range, with the average customers having a predicted " + avg_predicted + " days to next purchase.")
    csv = convert_df(cust_to_show)

    # give advice on how to retain the churn customers
    st.subheader("Advice to retain the churn customers")
    if cust_seg_option == "High Value":
        st.write("Since these customers are of high value, it will be crucial to implement targeted retention strategies to address their potential churn as they contribute to a significant portion of Tasty Bytes revenue.")
        st.write("The reasons behind why these customers are showing signs of potential churn despite contributing so much to Tasty Bytes’ revenue and making frequent orders should be investigated.")
        st.write("To retain these customers, exclusive menu items can be offered to these customers to provide them with a unique and premium experience, creating a sense of loyalty and making them less likely to switch to competitors.")
        st.write("Another suggestion is to focus on the high value customers that are more likely to purchase in the next "+ min_predicted + "-" + avg_predicted + " days, rather than customers predicted to purchase in eg. " + max_predicted + " days. This can impact overall retention rates more effectively and generate quicker positive results compared to those with longer predicted purchase timelines.")  
    elif cust_seg_option == "Middle Value":
        st.write("Even though these customers are of middle value, they still play a significant role in the overall business. It is still essential to address their potential churn to maintain a healthy customer base.")
        st.write("Feedback can be gathered from these customers through surveys or feedback forms to help identify areas for improvement and tailor services to better meet their needs. Responding to their concerns and suggestions can demonstrate that their opinions are valued, fostering a positive customer experience.")
        st.write("To retain these customers, implementing personalised special offers and discounts based on their preferences and order history can be a strategic approach. This will encourage repeat business and foster a sense of appreciation amongst these customers.")
        st.write("Another suggestion is to focus on the middle value customers that are more likely to purchase in the next "+ min_predicted + "-" + avg_predicted + " days, rather than customers predicted to purchase in eg. " + max_predicted + " days. This can impact overall retention rates more effectively and generate quicker positive results compared to those with longer predicted purchase timelines.") 
    else:
        st.write("While low value customers may not contribute as much revenue as high or middle value customers, it is still important to address their potential churn and explore ways to retain them as they still represent a portion of Tasty Bytes’ customer base.")
        st.write("Analysing these customer’s order history and  feedback through surveys or feedback forms can help to identify customer’s preferences, buying behaviour and pain points to be addressed to improve the overall customer experience.")
        st.write("To retain these customers, attractive discounts or promotions such as cost-effective deals can be offered to incentivize repeat purchases and even an increase in order frequency. This may also potentially convert some of them into higher value customers in the long run, contributing positively to the overall business growth.")
        st.write("Another suggestion is to focus on the low value customers that are more likely to purchase in the next "+ min_predicted + "-" + avg_predicted + " days, rather than customers predicted to purchase in eg. " + max_predicted + " days. This can impact overall retention rates more effectively and generate quicker positive results compared to those with longer predicted purchase timelines.") 
    st.write("With customer details, targeted marketing strategies such as email marketing can be implemented to deliver personalised messages, promotions and offers that resonate with each customer. This makes the emails become more engaging and relevant, fostering a sense of value and loyalty amongst customers.")

    st.download_button(
       "Press to Download Details of " + cust_seg_option + " Customers Likely to Churn",
       csv,
       "churn_cust_" + str(cust_seg_option) +".csv",
       "text/csv",
       key='download-csv')



    st.header('Predicting whether customers churn')

    # loading model
    with open('./churn/NextPurchase2.pkl', 'rb') as file:
        npm = pickle.load(file)

    # total spending input
    spending_option = st.number_input("Input Total Spending of Customer", min_value=1)

    st.write('You selected:', spending_option)

    # years with us input
    max_year = datetime.today().year - 2019
    years_list = [str(year) for year in range(1, max_year + 1)]
    years_with_us_option = st.selectbox(
    'Select the Number of Years the Customer has been with Tasty Bytes',
    years_list
)

    st.write('You selected:', years_with_us_option)

    # select no of orders
    total_orders_option = st.number_input("Input Number of Orders", min_value=1)

    st.write('You selected:', total_orders_option)

    # input last purchase date
    first_date = datetime(2019, 1, 1)
    today_date = datetime(2022, 11, 1)
    date = st.date_input("Enter the customer's last purchase date", first_date, first_date,today_date,today_date)

    st.write('You selected:', date)

    # predict churn button
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    st.button('Predict Whether Customer Will Churn', on_click=click_button)

    if st.session_state.clicked:
        #calculate average of trans_datediff1
        trans_datediff1 = next_purchase_cust_seg['TRANS_DATEDIFF1'].mean()

        #calculate average of trans_datedif2
        trans_datediff2 = next_purchase_cust_seg['TRANS_DATEDIFF2'].mean()

        #calculate average of avg(days_between)
        avg_days_between = next_purchase_cust_seg['AVG(DAYS_BETWEEN)'].mean()
   
        #calculate average of min(days_between)
        min_days_between = next_purchase_cust_seg['MIN(DAYS_BETWEEN)'].mean()

        #calculate average of max(days_between)
        max_days_between= next_purchase_cust_seg['MAX(DAYS_BETWEEN)'].mean()

        #calculate monetary value 
        monetary = spending_option / int(years_with_us_option)

        #calculate frequency
        frequency = int(total_orders_option) / int(years_with_us_option)

        #calculate recency
        recency = (today_date.date() - date).days

        #calculate monetary cluster 
        if monetary <= 566:
            monetary_cluster = 0
        elif monetary <= 795:
            monetary_cluster = 1
        else:
            monetary_cluster = 2
        
        #calculate frequency cluster 
        if frequency <= 14:
            frequency_cluster = 0
        elif frequency <= 20:
            frequency_cluster = 1
        else:
            frequency_cluster = 2

        #calculate recency cluster 
        if recency <= 12:
            recency_cluster = 2
        elif recency <= 33:
            recency_cluster = 1
        else:
            recency_cluster = 0

        #calculate overall score 
        if recency <= 12 and frequency >= 30 and monetary >= 1259:
            overall_score = 6
        elif recency <= 33 and frequency >= 28 and monetary >= 1247:
            overall_score = 5
        elif recency <= 86 and frequency >= 24 and monetary >= 1006:
            overall_score = 4
        elif recency <= 96 and frequency >= 14 and frequency <20 and monetary >= 566 and monetary <794:
            overall_score = 1
        elif recency <= 98 and frequency >= 20 and monetary >= 794:
            overall_score = 3
        elif recency <= 115 and frequency >= 19 and monetary >= 776:
            overall_score = 2
        else:
            overall_score = 0 

        #making of dataframe to input to model 
        data = [[spending_option, years_with_us_option, monetary, frequency, total_orders_option, recency, max_days_between, min_days_between, avg_days_between, trans_datediff1, trans_datediff2, recency_cluster, frequency_cluster, monetary_cluster, overall_score]]
        final = pd.DataFrame(data, columns = ['TOTAL_SPENT','YEARS_WITH_US','MONETARY_VALUE','CUSTOMER_FREQUENCY','TOTAL_ORDER','RECENCY_DAYS','MAX(DAYS_BETWEEN)','MIN(DAYS_BETWEEN)','AVG(DAYS_BETWEEN)','TRANS_DATEDIFF1','TRANS_DATEDIFF2','CUST_REC_CLUSTER','CUST_FREQ_CLUSTER','CUST_MONETARY_CLUSTER','OVERALL_SCORE'])

        pred = npm.predict(final)
        pred = pred.round().astype(int)
        if pred[-1] <= 14:
            st.write("Customer is not likely to churn.")
        else:
            st.write("Customer is likely to churn. It is predicted that they are likely to make a purchase in the next " + str(pred[-1]) + " days, exceeding the 14-day benchmark for potential churn by " + str(pred[-1] - 14) + " days.") 
    
with tab4:
    st.title('Uplift Revenue of Churn/Non-Churn Customers')
    st.subheader('Sub Title')
    
with tab5:
    st.title('Title')
    st.subheader('Sub Title')