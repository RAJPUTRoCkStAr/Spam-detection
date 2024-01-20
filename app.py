import streamlit as st
from streamlit_option_menu import option_menu
import Home,Dataoverview,DataVisulation,ModelEvaluation,WordClouds,Predict,Chabo

st.set_page_config(
                page_title="Spam Detection App",
                page_icon="✉️",
                layout="wide")
class MultiApp:
    def __init__(self):
        self.apps = []
    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })
    def run():
        with st.sidebar:      
                app = option_menu(
                        menu_title="Main Menu",
                        options=[
                                'Home',
                                'Spam Exploration',
                                'Word Cloud Analysis',
                                'Model Insights',
                                'Spam Prediction',
                                'Data Visualization',
                                'JARVIS'
                                ],
                                icons=['house-fill', 'journal','rocket-takeoff','amd','activity','bar-chart-steps','chat-square-dots'],
                                menu_icon="cast"
                        ) 
        if app == "Home":
            Home.home()
        if app == "Spam Exploration":
          Dataoverview.dataover()
        if app == "Word Cloud Analysis":
          WordClouds.word()  
        if app == "Model Insights":
          ModelEvaluation.model()
        if app == "Spam Prediction":
          Predict.pred()
        if app == "Data Visualization":
          DataVisulation.report()   
        if app == "JARVIS":
          Chabo.jar()
    run()   
