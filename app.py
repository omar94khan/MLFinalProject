import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title('Fraud Detector')

    file = st.file_uploader("Choose a file")

    if file is not None:
        classifier = pickle.load(open('RandomForrestClassifier_df4x.pkl', 'rb'))
        
        df = pd.read_csv(file)
        st.write("The dataset you uploaded is:")
        st.write(df)

        for col in ['Class']:
            df = df.loc[:,df.columns != col]
        df = pd.DataFrame(MinMaxScaler().fit(df).transform(df), columns=df.columns)


        result_df = pd.DataFrame(classifier.predict_proba(df)[1])
            
        st.write("Output DataFrame depicting probability of the transaction being fraudulant.")
        st.write(result_df)
    
main()