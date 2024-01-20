import streamlit as st
import json
import requests
from streamlit_lottie import st_lottie
def load_lottiefile(filepath:str):
    with open (filepath, 'r') as f:
        return json.load(f)
def load_lottieurl(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_dover = load_lottiefile("lottiefiles/dataoverview.json")
lottie_model = load_lottiefile("lottiefiles/modeleva.json")
lottie_clouds = load_lottiefile("lottiefiles/wordcouds.json")
lottie_hello = load_lottiefile("lottiefiles/hello.json")
lottie_report = load_lottiefile("lottiefiles/report.json")
lottie_predict = load_lottiefile("lottiefiles/predict.json")
lottie_about = load_lottiefile("lottiefiles/About.json")
lottie_future = load_lottiefile("lottiefiles/future.json")
lottie_jarvis = load_lottiefile("lottiefiles/Jarvis.json")
lottie_robot = load_lottiefile("lottiefiles/robot.json")


