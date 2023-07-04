# Import neccessary packages
import streamlit as st
import pandas as pd
import numpy as np

# Define the app title and favicon
st.set_page_config(page_title='ICP ASG 3', page_icon="favicon.ico")

# Tabs set-up
tab1, tab2, tab3, tab4, tab5 = st.tabs(['tab1', 'tab2', 'tab3', 'tab4', 'tab5'])

with tab1:
    st.title('Title')
    st.subheader('Sub Title')
    
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
    st.title('Title')
    st.subheader('Sub Title')