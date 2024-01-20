import streamlit as st
from streamlit_lottie import st_lottie
from lotti import lottie_jarvis, lottie_robot

st.markdown(f"# Jarvis",st_lottie(lottie_jarvis, speed=1, reverse=True, loop=True, quality='medium', height=80, width=80, key=None))
