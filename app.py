import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

def main():
    st.title('Fraud Detector')

    file = st.file_uploader("Choose a file. Please ensure the file only has 30 columns including Time, V1-V28, and Amount; all amounts unscaled.")

    if file is not None:
        classifier = pickle.load(open('RandomForrestClassifier_df4x.pkl', 'rb'))
        
        df = pd.read_csv(file)
        st.write("The dataset you uploaded is:")
        st.write(df)

        df2 = pd.DataFrame()
        
        columns = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12',
                    'V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
                    'V24','V25','V26','V27','V28','Amount']

        for col in columns:
            df2[col] = df[col]
        df2 = pd.DataFrame(MinMaxScaler().fit(df2).transform(df2), columns=df2.columns)
#        for col in ['Class']:
#            df = df.loc[:,df.columns != col]
#        df = pd.DataFrame(MinMaxScaler().fit(df).transform(df), columns=df.columns)


        result_df = pd.DataFrame(classifier.predict_proba(df2))[1]
            
        st.write("Output DataFrame depicting probability of the transaction being fraudulant.")
        st.write(result_df)
    
main()