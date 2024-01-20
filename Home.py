import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie
from lotti import lottie_dover,lottie_about,lottie_future,lottie_clouds
def home():
        col1, col2 = st.columns([3, 3])
        with col1:
                st.subheader("Welcome to the Spam Detection App👋")
                st.title("Spam Detection & Model Evaluation")
                st.write("This app allows you to explore and analyze spam detection data.")
                st.write("Spam detection is the process of identifying and filtering out unwanted, unsolicited, and often malicious messages, known as spam, from legitimate and relevant messages. These messages can be found in various forms, including emails, text messages, and comments.")
        with col2:
            lottie_dover
            dover = st_lottie(lottie_dover,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)
        tab = option_menu(None, ["Overview 🗒️", "About Me 👨‍💻", "Future Enhancements 📈"], orientation="horizontal")
        if tab == "Overview 🗒️":
                col3,col4 = st.columns([3,3])
                with col3:
                        st.subheader("Overview: 🗒️")
                        st.markdown("Explore the features of this app and learn how to use it for spam detection.")
                        st.subheader("Key Features:")
                        st.markdown("- **Data Exploration:** Analyze the dataset by exploring its head, info, describe, and tail.")
                        st.markdown("- **Word Clouds:** Visualize word clouds for ham and spam messages.")
                        st.markdown("- **Model Evaluation:** Evaluate the performance of different machine learning models.")
                        st.markdown("- **Predict:** Algorithms like decision trees, random forests, support vector machines, and neural networks are used to predict whether a message is spam based on learned features.")
                        st.markdown("- **Data Visualization:** Explore the data and report of our csv file using ydata-profiling.")
                        st.subheader("How to Use:")
                        st.markdown("1. Navigate to the 'Home' section to explore about the page.")
                        st.markdown("1. Go to 'Main' section to explore the dataset summary.")
                        st.markdown("2. Visit 'Word Clouds' to visualize word clouds for ham and spam messages.")
                        st.markdown("3. Go to 'Model Evaluation' to assess the performance of machine learning models.")
                        st.markdown("4. In the 'Predict' section, enter a message and select a model to predict if it's spam or ham.")
                        st.markdown("4. In the 'Data Visualization' section, explore the report using ydata-profiling.")
                with col4:
                        lottie_clouds
                        clouds = st_lottie(lottie_clouds,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)
                
        elif tab == "About Me 👨‍💻":
                col5,col6 = st.columns([3,3])
                with col5:
                        st.subheader("About Me: 👨‍💻")
                        st.write("Hello, I'm Sumit Kumar Singh, the creator of Spam Detection App.")
                        st.write("I have a passion for data science and machine learning, and I built this app to showcase the power of these technologies in spam detection.")
                with col6:
                        lottie_about
                        about = st_lottie(lottie_about,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)
        elif tab == "Future Enhancements 📈":  
                col7,col8 = st.columns([3,3])
                with col7:
                        st.subheader("Future Enhancements: 📈")
                        st.markdown("As we evolve, exciting enhancements are in the works to make the Spam Detection App even more powerful:")
                        st.markdown("✅  ==> **Real-time Prediction:** Get instant predictions for messages as they arrive, ensuring timely insights.")
                        st.markdown("    ==> **User Authentication:** Personalize your experience by creating an account, unlocking exclusive features, and tailoring the app to your needs.")
                        st.markdown("    ==> **Enhanced Visualizations:** Dive deeper into data exploration with advanced visualization tools for a richer analytical experience.")
                        st.markdown("✅ ==> **Chatbot Integration:** Develop a chatbot to assist users in understanding different models and their definitions.")
                        st.markdown("✅ ==> **Integration with Gmail API:** Currently, we're hard at work integrating with the Gmail API for seamless spam and ham filtering directly from your Gmail account.")
                        st.markdown("✅ ==> **Effortless Sync:** Keep your model updated with the latest email patterns, ensuring accurate predictions.")
                        st.markdown("   ==> **Explore Other APIs:** Look out for integrations with additional APIs, expanding our approach to comprehensive spam detection.")
                        st.write("Stay tuned for these transformative updates!")
                with col8:
                        lottie_future
                        future = st_lottie(lottie_future,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)
        st.write("⭐Feel free to explore the app and stay tuned for future updates!")
