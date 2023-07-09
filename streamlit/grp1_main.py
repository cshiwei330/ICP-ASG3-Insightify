# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np
import datetime

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
    st.title('Title')
    st.subheader('Sub Title')
    
with tab4:
    st.title('Title')
    st.subheader('Sub Title')
    
with tab5:
    st.title('Inventory Management')
    st.subheader('Sub Title')