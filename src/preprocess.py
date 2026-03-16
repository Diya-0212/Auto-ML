import pandas as pd
import numpy as np
def preprocess_data(df,Target):

    #drop obvious non useful columns if they exist
    drop_cols=["PassengerId","Name","Ticket","Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns],errors="ignore")

    #Separate features and target
    x=df.drop(columns=[Target])
    y=df[Target]

    #identify column types
    numeric_cols=x.select_dtypes(include=[np.number]).columns
    categorical_cols=x.select_dtypes(include=["object"]).columns
    
    #handles missing values
    x[numeric_cols]=x[numeric_cols].fillna(x[numeric_cols].mean())
    x[categorical_cols]=x[categorical_cols].fillna(x[categorical_cols].mode().iloc[0])

    #encode categorical variables
    x=pd.get_dummies(x,columns=categorical_cols)

    return x,y