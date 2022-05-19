   
import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title('Fraud Detector')

    file = st.file_uploader("Choose a file")

    if file is not None:
        classifier = pickle.load(open('RandomForrestClassifier_df4x.pkl', 'rb'))
        
        df = pd.read_csv(file)
        for col in ['Class']:
            df = df.loc[:,df.columns != col]
        df = pd.DataFrame(MinMaxScaler().fit(df).transform(df), columns=df.columns)
        st.write("The dataset you uploaded is:")
        st.write(df)

        result_df = pd.DataFrame(classifier.predict_proba(df))
            
        st.write("Output DataFrame depicting probability of the transaction being fraudulant.")
        st.write(result_df)
    
main()