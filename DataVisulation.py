import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import json
import requests
from streamlit_lottie import st_lottie
from lotti import lottie_report
def report():
    col1,col2 = st.columns([3,3])
    with col1:
        st.subheader("Exploring Spam Dataset with Pandas Profiling 📝")
        st.write("""
        Welcome to the "Data Visualization 📝" app! This interactive tool is designed to help you gain valuable insights into a dataset related to spam messages. By leveraging the power of Pandas Profiling, the app generates a detailed report that unveils the dataset's characteristics, patterns, and statistical summaries.""")
    with col2:
            lottie_report
            report = st_lottie(lottie_report,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)
    df = pd.read_csv("spam.csv")
    
    profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
    st_profile_report(profile)

