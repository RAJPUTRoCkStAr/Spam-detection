import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import io
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import json
import requests
from streamlit_lottie import st_lottie
from lotti import lottie_clouds
def dataover():
    # Load dataset
    df = pd.read_csv('spam.csv')
    st.title("Spam Detection and Model Comparison App")
    st.subheader("Explore and analyze spam detection data.")
    col1, col2 = st.columns([3, 3])
    with col1:
        st.subheader("Data Exploration:")
        st.write("""
        Data exploration is essential in the analysis process, unraveling a dataset's patterns, relationships, and insights. Tasks include inspecting data types, summarizing statistics, and visualizing distributions. It guides feature engineering and informs analytical model development. In this Streamlit app, users dynamically explore aspects of the spam detection dataset.""")
    with col2:
            lottie_clouds
            clouds = st_lottie(lottie_clouds,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)

        # Options for data exploration
    options = st.multiselect(
             'Select Options',
             ['Head', 'Info', 'Describe', 'Tail', 'Columns', "Unique & Duplicated"]
        )

        # Data exploration based on user selection
    if "Head" in options:
            st.write(df.head())
    if "Info" in options:
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.write("DataFrame Information:")
            st.text(info_str)
    if "Describe" in options:
        st.write(df.describe())
    if "Tail" in options:
        st.write(df.tail())
    if "Columns" in options:
        st.write(df.columns)
    if "Unique & Duplicated" in options:
            unique_rows_count = len(df.drop_duplicates())
            st.write(f"Number of unique rows: {unique_rows_count}")
            duplicated_rows = df.duplicated().sum()
            st.write("Number of duplicated rows:", duplicated_rows, " in our dataset")
            agree = st.checkbox('Compare according to graphs 📈')
            if agree:
                plt.figure(figsize=(20, 4))
                plt.subplot(1, 2, 2)
                sns.barplot(x=['Duplicated', 'Unique'], y=[duplicated_rows, df.shape[0]], palette="viridis", width=0.5)
                plt.title('Duplicated Rows')
                plt.tight_layout()
                st.pyplot(plt)
    if not options:
        st.write("You have not selected anything to show")
