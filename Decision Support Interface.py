import streamlit as st
import pickle
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
raw_data=pd.read_csv(r'C:/Users/Shijin Chen/Desktop/UR Summer/Python/heloc_dataset_v1.csv')
raw_data.drop_duplicates()

X=raw_data.iloc[:,1:]
Y=(raw_data.iloc[:,0]=='Bad').astype(int)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
X_train,X_validation,Y_train,Y_validation=train_test_split(X_train,Y_train,test_size=0.25,random_state=1)
from sklearn.impute import SimpleImputer
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import MinMaxScaler
from  sklearn.pipeline import Pipeline

imputer=SimpleImputer(strategy='median')
scaler=MinMaxScaler()
data_pipeline=Pipeline([('Replace -7',SimpleImputer(missing_values=-7,strategy='median')),
                        ('Replac -8',SimpleImputer(missing_values=-8,strategy='median')),
                        ('Replace -9',SimpleImputer(missing_values=-9,strategy='median')),
                        ('Sacler',MinMaxScaler()),
                        ])
X_train_scale=pd.DataFrame(data_pipeline.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
X_validation_scale=pd.DataFrame(data_pipeline.transform(X_validation),columns=X_test.columns,index=X_validation.index)
X_test_scale=pd.DataFrame(data_pipeline.transform(X_test),columns=X_test.columns,index=X_test.index)
best_svc = SVC(C=12, gamma='auto', probability=True).fit(X_train_scale,Y_train)

st.title('Risk Prediction')
st.write('Please input your data below:')
x1=st.slider('select the value of ExternalRiskEstimate',min_value=30,max_value=100)
x2=st.slider('select the value of MSinceOldestTradeOpen',min_value=0,max_value=800)
x3=st.slider('select the value of MSinceMostRecentTradeOpen',min_value=0,max_value=400)
x4=st.slider('select the value of AverageMInFile',min_value=0,max_value=400)
x5=st.slider('select the value of NumSatisfactoryTrades',min_value=0,max_value=100)
x6=st.slider('select the value of NumTrades60Ever2DerogPubRec',min_value=0,max_value=20)
x7=st.slider('select the value of NumTrades90Ever2DerogPubRec',min_value=0,max_value=20)
x8=st.slider('select the value of PercentTradesNeverDelq',min_value=0,max_value=100)
x9=st.slider('select the value of MSinceMostRecentDelq',min_value=0,max_value=100)
x10=st.slider('select the value of MaxDelq2PublicRecLast12M',min_value=0,max_value=9)
x11=st.slider('select the value of MaxDelqEver',min_value=1,max_value=9)
x12=st.slider('select the value of NumTotalTrades',min_value=0,max_value=100)
x13=st.slider('select the value of NumTradesOpeninLast12M',min_value=0,max_value=12)
x14=st.slider('select the value of PercentInstallTrades',min_value=0,max_value=100)
x15=st.slider('select the value of MSinceMostRecentInqexcl7days',min_value=0,max_value=24)
x16=st.slider('select the value of NumInqLast6M',min_value=0,max_value=70)
x17=st.slider('select the value of NumInqLast6Mexcl7days',min_value=0,max_value=70)
x18=st.slider('select the value of NetFractionRevolvingBurden',min_value=0,max_value=230)
x19=st.slider('select the value of NetFractionInstallBurden',min_value=0,max_value=470)
x20=st.slider('select the value of NumRevolvingTradesWBalance',min_value=0,max_value=30)
x21=st.slider('select the value of NumInstallTradesWBalance',min_value=1,max_value=23)
x22=st.slider('select the value of NumBank2NatlTradesWHighUtilization',min_value=0,max_value=18)
x23=st.slider('select the value of PercentTradesWBalance',min_value=0,max_value=100)
x=pd.DataFrame(np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23]).reshape([1,-1]),columns=X.columns)
with open('predict_model','wb')as f:
    pickle.dump(best_svc,f)
with open('predict_model','rb')as f:
    loaded_model=pickle.load(f)
yp=loaded_model.predict(x)
if yp==1:
    st.write('The prediction result is Bad')
else:
    st.write('The prediction result is Good')

