import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("heart_disease_uci.csv")
print(df.describe())
print(df.isnull().sum())
print(df.info())

df["restecg"]=df["restecg"].fillna(df["restecg"].mode()[0])
df["chol"]=df["chol"].fillna(df["chol"].median())
df["trestbps"]=df["trestbps"].fillna(df["trestbps"].median())
df["thalch"]=df["thalch"].fillna(df["thalch"].median())
df["exang"]=df["exang"].fillna(df["exang"].mode()[0])
df["oldpeak"]=df["oldpeak"].fillna(df["oldpeak"].median())
df["slope"]=df["slope"].fillna(df["slope"].mode()[0])
df["fbs"]=df["fbs"].fillna(df["fbs"].mode()[0])
df=df.drop("thal",axis=1)
df=df.drop("ca",axis=1)

print(df.isnull().sum())

print(df["sex"].unique())
print(df["dataset"].unique())
print(df["slope"].unique())
print(df["restecg"].unique())
print(df["fbs"].unique())
print(df["exang"].unique())
print(df["cp"].unique())

df=pd.get_dummies(df,columns=["dataset","slope","restecg","cp"],drop_first=True)
print(df.columns)

le=LabelEncoder()
df["sex"]=le.fit_transform(df["sex"])
df["fbs"]=le.fit_transform(df["fbs"])
df["exang"]=le.fit_transform(df["exang"])





# x=df[['dataset_Hungary', 'dataset_Switzerland',
#        'dataset_VA Long Beach', 'slope_flat', 'slope_upsloping',
#        'restecg_normal', 'restecg_st-t abnormality', 'cp_atypical angina',
#        'cp_non-anginal', 'cp_typical angina','chol','trestbps','thalch','oldpeak','sex','fbs','exang']]

df["num"]=df["num"].apply(lambda x:0 if x==0 else 1)
x=df.drop("num",axis=1)
y=df['num']

x_train,x_test,y_train,y_test=train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)


scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)




model=LogisticRegression(max_iter=1000,class_weight="balanced")
model.fit(x_train,y_train)
prediction=model.predict(x_test)

print("Accuracy Score: ",accuracy_score(y_test,prediction))
print("confusion Matrix: ",confusion_matrix(y_test,prediction))
print("classification report: ",classification_report(y_test,prediction))