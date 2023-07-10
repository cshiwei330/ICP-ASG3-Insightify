# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle

# Define the app title and favicon
st.set_page_config(page_title='ICP ASG 3', page_icon="favicon.ico")

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['SW', 'Ernest', 'Gwyneth', 'GF', 'KK'])

with tab1:
    st.title('Overall')
    st.subheader('Sub Title')
    
    # Function 1: Show predicted revenue 
    # Function 1.1: Allow user to select a timeframe
    # Define the range of months
    start_month = datetime.date(2023, 1, 1)
    end_month = datetime.date(2023, 12, 1)

    # Display the slider
    selected_months = st.slider(
        "Select a range of months",
        min_value=start_month,
        max_value=end_month,
        value=(start_month, end_month),
        format="MMM YYYY",
        step=datetime.timedelta(days=31)  # Adjust the step size as needed
    )

    # Extract the selected range
    start_selected_month, end_selected_month = selected_months

    # Display the selected range
    st.write("Start month:", start_selected_month.strftime("%b %Y"))
    st.write("End month:", end_selected_month.strftime("%b %Y"))
    
    #Function 1.2: Allow user to select what metrics they want us to predict
    # Display checkboxes for key metrics
    show_total_revenue = st.checkbox("Total Predicted Revenue")
    show_avg_spending = st.checkbox("Average Predicted Spending per Customer")

    # Display the "Predict" button
    if st.button("Predict"):
        # Check the status of checkboxes and display selected metrics
        if show_total_revenue:
            # Code to display total predicted revenue
            st.write("Total Predicted Revenue: $1,000,000")

        if show_avg_spending:
            # Code to display average predicted spending per customer
            st.write("Average Predicted Spending per Customer: $100")

    
with tab2:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab3:
    st.title('Predicting Customer Churn')
    st.subheader('Sub Title')
    
with tab4:
    st.title('Uplift Revenue of Churn/Non-Churn Customers')
    st.subheader('Sub Title')
    
    with open('Uplift_1M.pkl', 'rb') as file:
        uplift_1M = pickle.load(file)
    
    def load_Uplift_Churn_1W():
    # Load the uplift prediction model
        data = pd.read_csv("UpliftPrediction[1W].csv") 
        return data
    
    # Load the 1W Uplift Model
    uplift_predictions = load_Uplift_Churn_1W()
    uplift_predictions = pd.DataFrame(uplift_predictions)
    
    # Slicer
    years_with_us_range = st.slider("Years Range", uplift_predictions['YEARS_WITH_US'].min(), 
                                    uplift_predictions['YEARS_WITH_US'].max(), (uplift_predictions['YEARS_WITH_US'].min(), uplift_predictions['YEARS_WITH_US'].max()))

    st.subheader(years_with_us_range)
    filtered_data = uplift_predictions[(uplift_predictions['YEARS_WITH_US'] >= years_with_us_range[0]) & (uplift_predictions['YEARS_WITH_US'] < years_with_us_range[1])]
    filtered_data = filtered_data.drop(columns=['CUSTOMER_ID','MONETARY_M3_HO','PREDICTED_PROBA_0','PREDICTED_PROBA_1'])
    
    if st.button("Predict", key='4'):
        pred = uplift_1M.predict_proba(filtered_data)
        st.dataframe(pred)
    
with tab5:
    st.title('Inventory Management')
    st.subheader('Sub Title')
    
    
    def load_Uplift_Churn():
    # First load the original airbnb listtings dataset
        data = pd.read_csv("UpliftPrediction[1W].csv") #use this for the original dataset, before transformations and cleaning
        return data
    uplift_predictions = load_Uplift_Churn()
    print(uplift_predictions)
    uplift_predictions = pd.DataFrame(uplift_predictions)
    
    years_with_us_range = st.slider("Years Range", uplift_predictions['YEARS_WITH_US'].min(), 
                                    uplift_predictions['YEARS_WITH_US'].max(), (uplift_predictions['YEARS_WITH_US'].min(), uplift_predictions['YEARS_WITH_US'].max()), key='5')

    def show_widgets(uplift_predictions):
        if st.button('Show Q2 Data'):
            st.table(uplift_predictions)
        else:
            st.table(uplift_predictions)
        if st.checkbox('Select years with us'):
            st.line_chart(uplift_predictions)
        else:
            st.line_chart(uplift_predictions)
        quarter = st.radio('Which quarter?', ('Q1', 'Q2'))
        if quarter == 'Q1':
            st.line_chart(uplift_predictions)
        elif quarter == 'Q2':
            st.line_chart(uplift_predictions)
        selected_quarter = st.selectbox('Which quarter?', ('Q1', 'Q2'))
        if selected_quarter == 'Q1':
            st.area_chart(uplift_predictions)
        elif selected_quarter == 'Q2':
            st.area_chart(uplift_predictions)
                      
    st.subheader(years_with_us_range)
    filtered_data = uplift_predictions[(uplift_predictions['YEARS_WITH_US'] >= years_with_us_range[0]) & (uplift_predictions['YEARS_WITH_US'] < years_with_us_range[1])]
    st.dataframe(uplift_predictions)
    
    st.subheader('Result')
    st.dataframe(filtered_data)