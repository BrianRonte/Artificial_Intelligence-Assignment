import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 
from xgboost.sklearn import XGBClassifier
import pickle
import streamlit as st


@st.cache(persist=True)
def load_data():
    data = pd.read_csv('Telco-Customer-Churn.csv')
    return data

@st.cache(persist=True)
def preprocess(df):
    # feature selection
    data = df[['gender','SeniorCitizen','InternetService','PaymentMethod','tenure','MonthlyCharges','Churn']]
    label_encoder = LabelEncoder()
    for column in data.select_dtypes(include='object').columns:
        values = np.array(data[column])
        data[column] = label_encoder.fit_transform(values)

    X = data.drop(['Churn'], axis=1)
    y = data['Churn']
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2, random_state=0)
    return Xtrain,Xtest,ytrain,ytest,label_encoder


@st.cache(suppress_st_warning=True)
def xgbclassifier(Xtrain,Xtest,ytrain,ytest):
    clf = XGBClassifier()
    clf.fit(Xtrain,ytrain)
    y_pred = clf.predict(Xtest)
    acc_score = metrics.accuracy_score(ytest,y_pred) * 100
    auc_score = metrics.roc_auc_score(ytest, y_pred)
    # saving the model
    pickle_in = open("xgb_model.pkl", "wb")
    xgb_model = pickle.dump(clf, pickle_in)
    pickle_in.close()
    return acc_score,auc_score


def accept_user_data():
    gender = st.selectbox('Gender',('Male','Female'))
    seniorcitizen = st.selectbox('SeniorCitizen',('Yes','No'))
    internetService = st.selectbox('Internet Service',('Fiber Optic','DSL','No Internet Service'))
    paymentMethod = st.selectbox('Payment Method',('Electronic Cheque','Mailed Cheque','Bank Transfer (Automatic)','Credit Card (Automatic)'))
    tenure = st.number_input('Tenure')
    monthlyCharge = st.number_input('Monthly Charge')
    # making changes to match label encoded data in the dataframe
    if(gender=='Male'):
        gender = 1
    elif(gender=='Female'):
        gender=0
    if(seniorcitizen=='Yes'):
        seniorcitizen = 1
    elif(seniorcitizen=='No'):
        seniorcitizen=0
    if(internetService=='Fiber Optic'):
        internetService = 1
    elif(internetService=='DSL'):
        internetService = 2
    else:
        internetService = 0
    if(paymentMethod=='Electronic Cheque'):
        paymentMethod = 0
    elif(paymentMethod=='Mailed Cheque'):
        paymentMethod = 2
    elif(paymentMethod=='Bank Transfer (Automatic)'):
        paymentMethod = 1
    else:
        paymentMethod = 3
    # store all the variables in a numpy array
    user_data = np.array([gender,seniorcitizen,internetService,paymentMethod,tenure,monthlyCharge]).reshape(1,-1)
    return user_data

def main():
    # loading the data
    data = load_data()
    Xtrain,Xtest,ytrain,ytest,label_encoder = preprocess(data)
    accuracy, auc_score = xgbclassifier(Xtrain,Xtest,ytrain,ytest)
    # loading the presaved model
    pickle_out = open('xgb_model.pkl', "rb")
    model = pickle.load(pickle_out)
    pickle_out.close()

    st.title("Introduction to building in Streamlit")

    st.sidebar.image('laura.jpg')
    st.sidebar.title("This is the sidebar")
    st.sidebar.markdown("Let's start with an XGBoost")

    if(st.checkbox("Display data", False)):
        st.subheader("Showing data now...")
        st.write(f'The dataset has a shape of {data.shape}')
        st.write(data.head())

    if(st.checkbox("Display metrics summary")):
        st.subheader("Display XGB Classifier metrics...")
        st.write("Model Accuracy : ", accuracy.round(2))
        st.write("Auc Score : ", auc_score)

    if(st.checkbox("Want to input your own data? Tick if yes")):
        user_data = accept_user_data()
        if st.button("Classify"):
            prediction = model.predict(user_data)
            pred = label_encoder.inverse_transform(prediction)
            success_string = f"Classification result : {pred}"
            st.write("Is the customer likely to churn?")
            st.success(success_string)

if __name__ == "__main__":
    main()