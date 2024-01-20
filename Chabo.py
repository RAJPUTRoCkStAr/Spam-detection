import streamlit as st
import random
from streamlit_lottie import st_lottie
from lotti import lottie_jarvis,lottie_robot
def jar():
    training_data = [
        ("hello", "greeting"),
        ("hi", "greeting"),
        ("hey", "greeting"),
        ("how are you", "care"),
        ("what is Spam detection", "aim"),
        ("what is gamil api", "gmail"),
        ("Usage spam detection", "use"),
        ("what is api", "api"),
        ("what is api password","apipass"),
        ("how to get api password","getapi"),
        ("what is logistic regression", "lr"),
        ("what is decision tree","dt"),
        ("what is XGBClassifier","xgb"),
        ("what is Support Vector Machine","svm"), 
        ("code of logistic regression", "lrcode"),
        ("code of lr", "lrcode"),
        ("code of Lr", "lrcode"),
        ("code of decision Tree", "dtcode"),
        ("code of Decision Tree", "dtcode"),
        ("code of dt", "dtcode"),
        ("code of Dt", "dtcode"),
        ("code of XGBClassifier", "xgbcode"),
        ("code of XGB", "xgbcode"),
        ("code of xgb", "xgbcode"),
        ("code of Support Vector Machine", "svmcode"), 
        ("code of SVM", "svmcode"), 
        ("code of svm", "svmcode"), 
        ("tell me a joke", "humor"),
        ("who won the world series", "sports"),
        ("how s the weather today", "weather"),
        ("recommend a book", "recommendation"),
    ]

    intent_dict = dict(training_data)

    responses_dict = {
        "greeting": ["Hi there! How can I help you?", "Hello!", "Hey!"],
        "care": ["I m just a program, but Im doing well. Thanks for asking!"],
        "aim":["Spam detection is identifying and filtering out unwanted or irrelevant messages using techniques like machine learning, content analysis, and blacklists."],
        "gmail":["Gmail API is Google's tool for developers to programmatically interact with Gmail, enabling tasks like email management and automation."],
        "apipass":["An API password typically refers to a secure key or token generated for authentication when accessing an API (Application Programming Interface)."],
        "getapi":["To generate an App Password for Gmail:\n Enable two-step verification in Google Account Security.\n Create an App Password under 'https://myaccount.google.com/apppasswords'"],
        "api":["An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate and interact."],
        "lr":["Logistic Regression is a binary classification method using a logit function, predicting probabilities and separating classes with a boundary."],
        "lrcode":["""
                    # Logistic Regression Model Training Code
                    from sklearn.linear_model import LogisticRegression
                    lr_model = LogisticRegression().fit(xtrain, ytrain)
                    lr_pred = lr_model.predict(xtest)
                """],
        "dt":["A Decision Tree is a tree-like model that makes decisions based on features, recursively splitting data to classify or regress."],
        "dtcode":["""
                    # Decision Tree Model Training Code
                    from sklearn.tree import DecisionTreeClassifier
                    dt_model = DecisionTreeClassifier().fit(xtrain, ytrain)
                    dt_pred= dt_model.predict(xtest)
                """],
        "xgb":["XGBoost (Extreme Gradient Boosting) Classifier is a machine learning algorithm that utilizes gradient boosting to enhance decision tree models for classification tasks."],
        "xgbcode":["""
                    # XGBClassifier Model Training Code
                    from xgboost import XGBClassifier 
                    xgb_model = XGBClassifier().fit(xtrain, ytrain)
                    xgb_pred = xgb_model.predict(xtest)
                """],
        "svm":["Support Vector Machine (SVM) is a machine learning algorithm for classification and regression tasks. It finds a hyperplane in a high-dimensional space to separate data points into different classes."],
        "svmcode":["""
                    # Support Vector Machine Model Training Code
                    from sklearn import svm
                    svm_model = svm.SVC(kernel="linear").fit(xtrain, ytrain)
                    svm_pred = svm_model.predict(xtest)
                """],
        "use":["Discover spam data insights, visualize word clouds, evaluate ML models, predict messages spam status, and chat with JARVIS for assistance"],
        "humor": ["Why don't scientists trust atoms? Because they make up everything!", "Here's a joke: ..."],
        "sports": ["I'm not sure, I don't follow sports closely."],
        "unknown": ["I didn't understand that. Can you ask me something else?"],
        "weather": ["I don't have real-time weather information. You might want to check a weather website."],
        "recommendation": ["Sure, I recommend 'The Hitchhiker's Guide to the Galaxy' by Douglas Adams."]
    }

    def classify_intent(user_input):
        return intent_dict.get(user_input.lower(), "unknown")

    def chatbot_response(user_input):
        intent = classify_intent(user_input)
        return responses_dict.get(intent, responses_dict["unknown"])

    st.header(f"Hello i m jarvis!",lottie_jarvis)
    jarvis = st_lottie(lottie_jarvis,speed=1,reverse=True,loop=True,quality='medium',height=80,width=80,key=None)
    centered_text = "<h1 style='text-align: center;'>Centered Content</h1>"
    st.markdown(centered_text, unsafe_allow_html=True)
    lottie_robot
    robot = st_lottie(lottie_robot,speed=1,reverse=True,loop=True,quality='medium',height=480,width=480,key=None)
    conversation_history = st.session_state.get("conversation_history", [])
    user_input = st.chat_input("You:")
    
    if user_input:
        bot_responses = chatbot_response(user_input)
        bot_response = random.choice(bot_responses)    
        conversation_history.append({"user": user_input, "bot": bot_response})
        for entry in conversation_history:
            st.text(f"You: {entry['user']}")
            st.text(f"Jarvis: {entry['bot']}")
        # st.image("https://i.imgur.com/XOcXTIw.png", width=100)
        # st.markdown("### JARVIS:")
        # for entry in conversation_history:
        #     st.text(f"**You:** {entry['user']}")
        #     st.text(f"**JARVIS:** {entry['bot']}")
        #     st.markdown("---")
        st.session_state.conversation_history = conversation_history