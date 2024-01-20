import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from wordcloud import WordCloud
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
def word():
        df = pd.read_csv('spam.csv')
        st.title("Word Clouds")
        tab1 = option_menu(None, ["Ham", "Spam", "Graph 📈"], orientation="horizontal")
        df['Category'] = df['Category'].replace({'ham': 1, 'spam': 0})
        if "Ham" in tab1:
                ham_messages = df[df['Category'] == 1]['Message']
                ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(ham_messages))
                st.title("Analysis of 'Ham' Messages")
                st.write("""
                The word cloud below visualizes the most frequently occurring words in messages categorized as "ham" (non-spam). Each word's size in the cloud is proportional to its frequency in the dataset. This visualization offers a quick and intuitive overview of common themes, communication patterns, and frequently used words within the non-spam messages.

                ### Key Insights:
                - **Frequent Terms:** Larger and bolder words represent those that appear most often in the "ham" messages.
                - **Content Overview:** The word cloud provides a snapshot of the common topics and subjects found in non-spam communication.
                - **Communication Patterns:** Explore the language and phrases commonly used in regular messages.

                This visualization serves as an exploratory tool for understanding the textual content of "ham" messages at a glance.
                """)
                st.image(ham_wordcloud.to_image())
        if "Spam" in tab1:
                spam_messages = df[df['Category'] == 0]['Message']
                spam_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(spam_messages))
                st.title("Analysis of 'Spam' Messages")
                st.write("""
                The word cloud below visualizes the most frequently occurring words in messages categorized as "spam". Each word's size in the cloud is proportional to its frequency in the dataset. This visualization offers a quick and intuitive overview of common themes, communication patterns, and frequently used words within the spam messages.

                ### Key Insights:
                - **Frequent Terms:** Larger and bolder words represent those that appear most often in the spam messages.
                - **Content Overview:** The word cloud provides a snapshot of the common topics and subjects found in spam communication.
                - **Communication Patterns:** Explore the language and phrases commonly used in spam messages.

                This visualization serves as an exploratory tool for understanding the textual content of spam messages at a glance.
                """)
                st.image(spam_wordcloud.to_image())
        if "Graph 📈" in tab1:
                encoder = LabelEncoder()
                df['Category'] = encoder.fit_transform(df['Category'])
                groups = df.groupby(by="Category").count().Message
                Count_HAM = groups[0]
                Count_SPAM = groups[1]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=["Ham", "Spam"],
                    y=[Count_HAM, Count_SPAM],
                    marker_color="indianred",
                    width=[0.4, 0.4],
                    text=[f"Count_HAM: {Count_HAM}", f"CNT_SPAM: {Count_SPAM}"]
                ))
                fig.update_layout(title="Comparing with Graph")

                st.plotly_chart(fig)
                st.title("Distribution of Classes in the Dataset")
                st.write(f"""
                The bar chart below illustrates the distribution of messages in the dataset across different classes, specifically "Ham" (non-spam) and "Spam". Each bar represents the count of messages in the respective category.

                ### Key Insights:
                - **Ham Messages:** The number of messages categorized as "Ham" is represented by the first bar.
                - **Spam Messages:** The second bar represents the count of messages categorized as "Spam".

                This visualization provides a quick overview of the class distribution, allowing us to understand the balance or imbalance between non-spam and spam messages in the dataset.

                ### Class Counts:
                - **Ham Messages Count:** {Count_HAM}
                - **Spam Messages Count:** {Count_SPAM}

                Explore the chart to gain insights into the relative abundance of each class in the dataset.
                """)
