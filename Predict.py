import imaplib
import time
import email
from email.header import decode_header
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from streamlit_lottie import st_lottie
from lotti import lottie_predict
import Robot
class MultiApp:
    def __init__(self):
        self.apps = []
    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })
def pred():
    df = pd.read_csv('spam.csv')
    tab9, tab10 = st.columns([3, 3])
    with tab9:
        st.title("🛡️ Spam Prediction")
        st.write("""
            Welcome to the Spam Prediction app! This tool utilizes machine learning models to detect whether a given message is spam or not. Simply enter a message, choose a classification model, and let the app predict its status.\n
            Example Output:\n
            🎉 Not Spam: If the result is 'ham,' the message is identified as not spam.\n
            🚨 Spam Detected: If the result is 'spam,' the message is identified as spam.
            """)
    with tab10:
        lottie_predict
        predict = st_lottie(lottie_predict,speed=1,reverse=True,loop=True,quality='medium',height=None,width=None,key=None)
    st.markdown("### Detect spam messages with machine learning!")
    def fetch_latest_emails(email_user, email_pass, num_emails=5, num_lines=5):
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_user, email_pass)
        mail.select("inbox")
        status, messages = mail.search(None, "ALL")
        messages = messages[0].split()
        latest_emails = []
        for mail_id in messages[-num_emails:]:
            _, msg_data = mail.fetch(mail_id, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True)
            else:
                body = msg.get_payload(decode=True)
            body_text = body.decode("utf-8") if body else None
            if body_text:
                body_lines = body_text.splitlines()[:num_lines]
                truncated_body = "\n".join(body_lines)
            else:
                truncated_body = "No body content available."
            latest_emails.append({"ID": mail_id, "Subject": subject, "Body": truncated_body})
        mail.logout()
        return latest_emails
    latest_emails = None
    email_user = st.text_input("Enter your email address") 
    email_pass = st.text_input("Enter your email api",placeholder="asithnkeyiopghue") 
    st.info(" 🚀 Chat with our helpful bot JARVIS for quick assistance")
    numemail = st.slider("Select number of email you want to display",max_value=20)
    submission = st.button("Submit")
    if submission :
        if email_user and email_pass:
            latest_emails = fetch_latest_emails(email_user, email_pass, num_emails=numemail, num_lines=5)

        if latest_emails is not None:
            st.header("Select an email to predict:")
            selected_email = st.selectbox("Select an email:", [f"{email['Subject']} ({email['ID'].decode('utf-8')})" for email in latest_emails])
            selected_email_id = selected_email.split('(')[1].split(')')[0]

            selected_model = st.selectbox("Select a model:", [
                "Decision Tree",
                "Logistic Regression",
                "XGBClassifier",
                "Support Vector Machine",
            ])
            model = None
        if st.button("Predict"):
                X_full = df["Message"]
                y_full = df["Category"]
                if selected_model == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif selected_model == "Logistic Regression":
                    model = LogisticRegression()
                elif selected_model == "XGBClassifier":
                    model = XGBClassifier()
                elif selected_model == "Support Vector Machine":
                    model = svm.SVC(kernel="linear")
                if not model:
                    st.warning("Please select a model")
                encoder = LabelEncoder()
                df["Category"] = encoder.fit_transform(df["Category"])
                x = df["Message"]
                y = df["Category"]
                cv = CountVectorizer(decode_error="ignore")
                df["Message"].fillna("", inplace=True)
                X = cv.fit_transform(df["Message"])
                xtrain, xtest, ytrain, ytest = train_test_split(X, y)
                model.fit(xtrain, ytrain)
                for email_data in latest_emails:
                    if email_data["ID"].decode() == selected_email_id:
                        with st.status("lets predict your email message", expanded=True) as status:
                            st.info("Checking your email...")
                            st.text(f"Selected Email: {email_data['Subject']}")
                            st.text(email_data["Body"])
                            time.sleep(4)
                            st.info("Implementing machine learning algorithm")
                            email_body_vectorized = cv.transform([email_data["Body"]])
                            prediction = model.predict(email_body_vectorized)[0]
                            result = encoder.inverse_transform([prediction])[0]
                            st.write('Machine Learned from your email Messages ')
                            time.sleep(2)
                            st.write("Predicting your email message ....")
                            time.sleep(2)
                            if result == 0:
                                st.success('🎉 Info! This message is identified as Ham.')
                            elif result == 1:
                                st.error('🚨 Alert! This message is identified as Spam.') 
                        st.session_state.predicted_result = result
        if st.button("Save Prediction"):
            if 'predicted_result' in st.session_state:
                        new_data = pd.DataFrame({"Message": [selected_email], "Category": [st.session_state.predicted_result]})
                        df = pd.concat([df, new_data], ignore_index=True)
                        df.to_csv('spam.csv', index=False)
                        st.success("Prediction saved successfully!")
            else:
                        st.warning("No prediction to save. Make a prediction first.")
