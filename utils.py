import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_heart_data():

    df = pd.read_csv("archive/heart_disease_uci.csv")

    df = df.dropna()  # Removes any rows with missing values

    # Convert binary categorical variables
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["fbs"] = df["fbs"].map({"TRUE": 1, "FALSE": 0})
    df["exang"] = df["exang"].map({"TRUE": 1, "FALSE": 0})

    # One-hot encode multi-class categorical variables to make sense of them
    df = pd.get_dummies(df, columns=["cp", "restecg", "slope", "thal"])



    X = df.drop(columns=["id", "dataset", "num"]).values
    # we had to do this because the answer was 0 to 4 but we are building a binary classification model
    # we are reshaping because we need it to be in format (xxx, 1) so that we can perform matrix operartions
    y = df["num"].apply(lambda x: 1 if x > 0 else 0).values.reshape(-1, 1)

    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = np.nan_to_num(X) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size  = 0.2, random_state = 42)

    
    return X_train, X_test, y_train, y_test