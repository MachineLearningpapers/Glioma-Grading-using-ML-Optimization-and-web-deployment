import numpy as np
import streamlit as st
import pickle
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame

from google.oauth2 import service_account
#from gsheetsdb import connect
from gspread_pandas import Spread,Client



#scope = ["https://www.googleapis.com/auth/spreadsheets"]

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=scope,
)
#conn = connect(credentials=credentials)
client = Client(scope = scope, creds = credentials)
spreadsheet_name = 'Glioma_Database'
spread = Spread(spreadsheet_name, client = client)

sh = client.open(spreadsheet_name)
worksheet_list = sh.worksheets()

#st.write(spread.url)
#@st.cache_data(ttl=600)

def load_the_spreadsheet(spreadsheet_name):
    worksheet = sh.worksheet(spreadsheet_name)
    df = DataFrame(worksheet.get_all_records())
    return df

def update_the_spreadsheet(spreadsheet_name, dataframe):
    col = ['Age_at_diagnosis','Gender',	'Race',	'IDH1',	'TP53',	'ATRX',	'PTEN',	'EGFR',	'CIC', 'MUC16',	'PIK3CA', 'NF1', 'PIK3R1', 'FUBP1',	'RB1',
           'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
    #spread.df_to_sheet(dataframe[col], sheet = spreadsheet_name, index = False)
    spread.df_to_sheet(dataframe, sheet = spreadsheet_name, index = False)
    st.sidebar.info('Updated to Googlesheet')


# Load our mnodel
model_load = pickle.load(open('my_model_glioma_xgb.sav', 'rb'))
X_train = pd.read_csv('X_train_.csv')
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# Identify categorical columns
categorical_cols = X_train.select_dtypes('object').columns.tolist()
encoder.fit(X_train[categorical_cols])
encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
X_train[encoded_cols] = encoder.fit_transform(X_train[categorical_cols])

# Define a prediction function

def prediction_model(input_data):

    prediction = model_load.predict(input_data)

    if prediction[0] == 0:
        return "GBM", "The patient has a High Grade Glioma(Glioblastoma)"
    elif prediction[0] == 1:
        return "LGG", 'The patient has a Low Grade Glioma(LGG)'
    



def main():

        # App title
    #st.set_page_config(page_title="Glioma prediction App")

    # Replicate Credentials
    with st.sidebar:
        st.title('Glioma prediction App')
        st.markdown('This Web App has been designed to give users the test our Glioma prediction model and provide any feedback they may have. You can fill the information about the patient and Click on the Glioma Test result button to get the result')
        st.image('glioma_picture.jpeg')
        
        
    

    # Give a title to the App
    st.title('Glioma prediction App')

    col1, col2 = st.columns(2)
    # Getting the Input from the user
    
    Age = col1.text_input('Age of the patient')
    Gender = col1.selectbox('Gender of the patient(Male/Female)', ['Male', 'Female'], key='Gender')
    Race = col1.selectbox('Select the race of the patient',['white', 'asian', 'black or african american','american indian or alaska native'], key='Race' )
    IDH1 = col1.selectbox('Select the IDH1(isocitrate dehydrogenase) value', ['NOT_MUTATED', 'MUTATED'], key='IDH1')
    TP53 = col1.selectbox('Select the TP53(tumor protein p53) value', ['NOT_MUTATED', 'MUTATED'], key='TP53')
    ATRX = col1.selectbox('Select the ATRX(ATRX chromatin remodeler) value', ['NOT_MUTATED', 'MUTATED'], key='ATRX')
    PTEN = col1.selectbox('Select the PTEN(phosphatase and tensin homolog) value', ['NOT_MUTATED', 'MUTATED'], key='PTEN')
    EGFR = col1.selectbox('Select the EGFR(epidermal growth factor receptor) value', ['NOT_MUTATED', 'MUTATED'], key='EGFR')
    CIC = col1.selectbox('Select the CIC(capicua transcriptional repressor) value', ['NOT_MUTATED', 'MUTATED'], key='CIC')
    MUC16 = col1.selectbox('Select the MUC16(mucin 16, cell surface associated) value', ['NOT_MUTATED', 'MUTATED'], key='MUC16')
    PIK3CA = col1.selectbox('Select the PIK3CA(phosphatidylinositol-4,5-bisphosphate 3-kinase catalytic subunit alpha) value', ['NOT_MUTATED', 'MUTATED'], key='PIK3CA')
    NF1 = col2.selectbox('Select the NF1(neurofibromin 1) value', ['NOT_MUTATED', 'MUTATED'], key='NF1')
    PIK3R1 = col2.selectbox('Select the PIK3R1(phosphoinositide-3-kinase regulatory)', ['NOT_MUTATED', 'MUTATED'], key='PIK3R1')#subunit 1 value
    FUBP1 = col2.selectbox('Select the FUBP1( value', ['NOT_MUTATED', 'MUTATED'], key='FUBP1') #far upstream element binding protein 1)
    RB1 = col2.selectbox('Select the RB1(RB transcriptional corepressor 1) value', ['NOT_MUTATED', 'MUTATED'], key='RB1')
    NOTCH1 = col2.selectbox('Select the NOTCH1(notch receptor 1) value', ['NOT_MUTATED', 'MUTATED'], key='NOTCH1')
    BCOR = col2.selectbox('Select the BCOR(BCL6 corepressor) value', ['NOT_MUTATED', 'MUTATED'], key='BCOR')
    CSMD3 = col2.selectbox('Select the CSMD3(CUB and Sushi multiple domains 3)', ['NOT_MUTATED', 'MUTATED'], key='CSMD3') #value
    SMARCA4 = col2.selectbox('Select the SMARCA4 value', ['NOT_MUTATED', 'MUTATED'], key='SMARCA4') #(SWI/SNF related, matrix associated, actin dependent regulator of chromatin, subfamily a, member 4)
    GRIN2A = col2.selectbox('Select the RIN2A value', ['NOT_MUTATED', 'MUTATED'], key='RIN2A') #(glutamate ionotropic receptor NMDA type subunit 2A)
    IDH2 = col2.selectbox('Select the IDH2  value', ['NOT_MUTATED', 'MUTATED'], key='IDH2') #(isocitrate dehydrogenase (NADP(+)) 2)
    FAT4 = col2.selectbox('Select the FAT4(FAT atypical cadherin 4) value', ['NOT_MUTATED', 'MUTATED'], key='FAT4')
    PDGFRA = st.selectbox('Select the PDGFRA(platelet-derived growth factor receptor alpha) value', ['NOT_MUTATED', 'MUTATED'], key='PDGFRA')
    
    data = {
    'Age_at_diagnosis': [Age],
    'Gender': [Gender],
    'Race': [Race],
    'IDH1': [IDH1],
    'TP53': [TP53],
    'ATRX': [ATRX],
    'PTEN': [PTEN],
    'EGFR': [EGFR],
    'CIC': [CIC],
    'MUC16': [MUC16],
    'PIK3CA': [PIK3CA],
    'NF1': [NF1],
    'PIK3R1': [PIK3R1],
    'FUBP1': [FUBP1],
    'RB1': [RB1],
    'NOTCH1': [NOTCH1],
    'BCOR': [BCOR],
    'CSMD3': [CSMD3],
    'SMARCA4': [SMARCA4],
    'GRIN2A': [GRIN2A],
    'IDH2': [IDH2],
    'FAT4': [FAT4],
    'PDGFRA': [PDGFRA],
    
     }

    df_ = pd.DataFrame(data)
    df_['Age_at_diagnosis'] = df_['Age_at_diagnosis'].astype(float)
    #df_['Age_at_diagnosis'] = float(df_['Age_at_diagnosis'].iloc[0])#.apply(pd.to_numeric, errors='coerce')
    numeric_cols = df_.select_dtypes(include=np.number).columns.tolist()
    df_[encoded_cols] = encoder.transform(df_[categorical_cols])
    X = df_[numeric_cols + encoded_cols]
    

    opt_df = DataFrame(data)
    opt_df['Grade'] = " "
    df = load_the_spreadsheet('Sheet1')
    #new_df = pd.concat([df, opt_df], ignore_index=True)
    #new_df = df.append(opt_df, ignore_index =  True)



    # variable of prediction
    result = ''

    if st.button('Glioma test result'):
        label, result = prediction_model(X)
        opt_df.iloc[:,-1] = label
        new_df = pd.concat([df, opt_df], ignore_index=True)
        update_the_spreadsheet('Sheet1', new_df)
        st.write(new_df)

    # Display the result
    st.success(result)

if __name__ == '__main__':
    main()
