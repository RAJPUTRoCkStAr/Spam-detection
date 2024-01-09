import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import time
from streamlit_lottie import st_lottie
from lotti import lottie_model
def model():
        col1,col2 = st.columns([3,3])
        with col1:
                st.title("Model Evaluation")
                st.write("""
                Model evaluation assesses the performance of a machine learning model. Common metrics include accuracy, precision, recall, and F1 score. It helps gauge how well the model generalizes to new data. A comprehensive evaluation involves examining a confusion matrix, ROC curves, and precision-recall curves. Understanding these metrics is crucial for optimizing models and making informed decisions about their deployment.""")
        with col2:
                lottie_model
                model = st_lottie(lottie_model,speed=1,reverse=True,loop=True,quality='medium',key=None)
        df = pd.read_csv('spam.csv')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        encoder = LabelEncoder()
        df['Category'] = encoder.fit_transform(df['Category'])
        x = df['Message']
        y = df['Category']
        cv = CountVectorizer(decode_error='ignore')
        X = cv.fit_transform(df['Message'])
        st.subheader('lets train the model with test_size and random_state')
        test_size = st.slider('Select Test Size', min_value=0.0, max_value=1.0, step=0.10,value=0.2)
        st.info(f"test_size for training and testing is ➡️{test_size*100}%")
        random_state = st.slider('Select random_state Size',1,100)
        st.info(f"Random State for training and testing is ➡️{random_state}")
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=random_state)
        on = st.checkbox('Show Data Shapes')
        if on:
            st.subheader('Shape of the training and testing data:')
            st.write(f'Training data shape: {xtrain.shape}')
            st.write(f'Testing data shape: {xtest.shape}')
            st.write(f'Training data shape: {ytrain.shape}')
            st.write(f'Testing data shape: {ytest.shape}')
        st.subheader('lets train the various model')
        model = option_menu(None,["Decision Tree Classifier", "Logistic Regression",'XGBClassifier','Support Vector Machine'],styles={"nav-link": {"font-size": "13px"}},orientation="horizontal")
        if model == "Random Forest Classifier":
            n_estimators = st.slider('Select random N_estimator',1,100)
            lrandom_state = st.slider('Select random forest_state Size',1,100)
        st.write('You selected:', model)
        mbutton = st.button('Train Model')
        with st.status("Downloading data...", expanded=False) as status:
            st.write("Preparing data...")
            time.sleep(2)
            st.write("Trainig Data")
            time.sleep(1)
            st.write("Model Training")
            time.sleep(2)
            status.update(label=f"Model Trained ➡️ {model}!✨", state="complete", expanded=False)
        if mbutton:
            if model == 'logistic_regression':
                lr_model = LogisticRegression().fit(xtrain,ytrain)
                lr_pred = lr_model.predict(xtest)
                lr_score = accuracy_score(ytest , lr_pred)*100
                st.info(f'Logistic Regression accuracy score: {lr_score:.2f}%')
                cmlr = confusion_matrix(ytest, lr_pred)
                cm_lr = pd.DataFrame(cmlr, index=lr_model.classes_, columns=lr_model.classes_)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_lr, annot=True, cmap='Blues', fmt='g')
                plt.title('Confusion Matrix - LogisticRegression')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot()
                classification_lr = classification_report(ytest, lr_pred)
                st.text(f"Classification Report for Logistic Regression :\n{classification_lr}")
            if model == 'Decision Tree Classifier':
                dt_model=DecisionTreeClassifier().fit(xtrain,ytrain)
                dt_pred= dt_model.predict(xtest)
                dt_score = accuracy_score(ytest,dt_pred)*100
                st.info(f'Decision Tree Classifier accuracy score: {dt_score:.2f}%')
                cmdt = confusion_matrix(ytest, dt_pred)
                cm_dt = pd.DataFrame(cmdt, index=dt_model.classes_, columns=dt_model.classes_)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_dt, annot=True, cmap='Blues', fmt='g')
                plt.title('Confusion Matrix - Decision Tree')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot()
                classification_dt = classification_report(ytest, dt_pred)
                st.text(f"Classification Report for Decision Tree:\n{classification_dt}")
            if  model == 'Support Vector Machine':
                svm_model = svm.SVC(kernel='linear').fit(xtrain, ytrain)
                svm_pred = svm_model.predict(xtest)
                svm_score = accuracy_score(ytest, svm_pred)*100
                st.info(f'Support Vector Machine accuracy score: {svm_score:.2f}%')
                cmsvm = confusion_matrix(ytest, svm_pred)
                cm_svm = pd.DataFrame(cmsvm, index=svm_model.classes_, columns=svm_model.classes_)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_svm, annot=True, cmap='Blues', fmt='g')
                plt.title('Confusion Matrix - Support Vector Machine')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot()
                classification_svm = classification_report(ytest, svm_pred)
                st.text(f"Classification Report for Support Vector Machine:\n{classification_svm}")
            if model == 'XGBClassifier':
                xgb_model = XGBClassifier().fit(xtrain,ytrain)
                xgb_pred = xgb_model.predict(xtest)
                xgb_score = accuracy_score(ytest , xgb_pred)*100
                st.info(f'XGBClassifier accuracy score: {xgb_score:.2f}%')
                cmxgb = confusion_matrix(ytest, xgb_pred)
                cm_xgb = pd.DataFrame(cmxgb, index=xgb_model.classes_, columns=xgb_model.classes_)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_xgb, annot=True, cmap='Blues', fmt='g')
                plt.title('Confusion Matrix - XGBClassifier')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot()
                classification_xgb = classification_report(ytest, xgb_pred)
                st.text(f"Classification Report for XGBClassifier:\n{classification_xgb}")
        scode = option_menu(None,["Compare","Show Code"],styles={"nav-link": {"font-size": "13px"}},orientation="horizontal")
        if scode == "Compare":
            tab2= st.selectbox('Choose How you wanna comapre it',("Accuracy Score 📈","Confusion Matrix 📈"),index=None,placeholder="Choose a option")
            if "Accuracy Score 📈" in tab2:
                dt_model = DecisionTreeClassifier().fit(xtrain, ytrain)
                lr_model = LogisticRegression().fit(xtrain, ytrain)
                xgb_model = XGBClassifier().fit(xtrain, ytrain)
                svm_model = svm.SVC(kernel='linear').fit(xtrain, ytrain)
                dt_pred = dt_model.predict(xtest)
                lr_pred = lr_model.predict(xtest)
                xgb_pred = xgb_model.predict(xtest)
                svm_pred = svm_model.predict(xtest)
                dt_score = accuracy_score(ytest, dt_pred) * 100
                lr_score = accuracy_score(ytest, lr_pred) * 100
                xgb_score = accuracy_score(ytest, xgb_pred) * 100
                svm_score = accuracy_score(ytest, svm_pred) * 100
                models = pd.DataFrame({
                    'Model': ['Logistic Regression', 'Decision Tree', 'XGBoost Classifier', 'Support Vector Machine'],
                    'Score': [lr_score, dt_score, xgb_score, svm_score]
                }).round(2)
                models = models.sort_values(by='Score', ascending=False)
                models['Text'] = [f'{model}: {score:.2f}' for model, score in zip(models['Model'], models['Score'])]
                fig = px.bar(models, y="Model", x="Score", color="Model", text='Text', orientation='h', title="Comparing ML Algorithms")
                fig.update_layout(xaxis_title="Accuracy", xaxis_title_font_size=13)
                fig.update_layout(yaxis_title="Model", yaxis_title_font_size=14)
                st.plotly_chart(fig)
            if "Confusion Matrix 📈" in tab2:
                dt_model = DecisionTreeClassifier().fit(xtrain, ytrain)
                lr_model = LogisticRegression().fit(xtrain, ytrain)
                xgb_model = XGBClassifier().fit(xtrain, ytrain)
                svm_model = svm.SVC(kernel='linear').fit(xtrain, ytrain)
                dt_pred = dt_model.predict(xtest)
                lr_pred = lr_model.predict(xtest)
                xgb_pred = xgb_model.predict(xtest)
                svm_pred = svm_model.predict(xtest)
                models = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'XGBoost', 'Support Vector Machine']
                predictions = [dt_pred, lr_pred, xgb_pred, svm_pred]
                true_positives = []
                false_positives = []
                true_negatives = []
                false_negatives = []
                for model, prediction in zip(models, predictions):
                    true_labels = ytest
                    cm = confusion_matrix(true_labels, prediction)
                    tp, fp, fn, tn = cm.ravel()
                    true_positives.append(tp)
                    false_positives.append(fp)
                    true_negatives.append(tn)
                    false_negatives.append(fn)
                st.bar_chart({
                    'True Positives': true_positives,
                    'False Positives': false_positives,
                    'True Negatives': true_negatives,
                    'False Negatives': false_negatives
                }, use_container_width=True)
                st.text('Models: ' + ', '.join(models))
                pds = pd.DataFrame({"Model":['Decision Tree','Logistic Regression', 'XGBoost', 'Support Vector Machine']})
                st.dataframe(pds)
                st.title('The best Model is Support Vector Machine is the best performer and impress us as compare to other algorithms')
        elif scode=="Show Code":
            st.subheader("Model Training Code")
            if model == 'Logistic Regression':
                st.code("""
                    # Logistic Regression Model Training Code
                    from sklearn.linear_model import LogisticRegression
                    lr_model = LogisticRegression().fit(xtrain, ytrain)
                    lr_pred = lr_model.predict(xtest)
                """)
            elif model == 'Decision Tree Classifier':
                st.code("""
                    # Decision Tree Model Training Code
                    from sklearn.tree import DecisionTreeClassifier
                    dt_model = DecisionTreeClassifier().fit(xtrain, ytrain)
                    dt_pred= dt_model.predict(xtest)
                """)
            elif model == 'Support Vector Machine':
                st.code("""
                    # Support Vector Machine Model Training Code
                    from sklearn import svm
                    svm_model = svm.SVC(kernel='linear').fit(xtrain, ytrain)
                    svm_pred = svm_model.predict(xtest)
                """)
            elif model == 'XGBClassifier':
                st.code("""
                    # XGBClassifier Model Training Code
                    from xgboost import XGBClassifier 
                    xgb_model = XGBClassifier().fit(xtrain, ytrain)
                    xgb_pred = xgb_model.predict(xtest)
                """)
