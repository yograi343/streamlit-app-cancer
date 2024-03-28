import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle
import os


def get_clean_data():
    data = pd.read_csv("..\\Data\\data.csv")

    data = data.drop(["Unnamed: 32", "id"], axis=1)

    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    return data


def create_model(data):
    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"].copy()

    # split the data into test, train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=42
    )

    # scale up the data sets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # create a model instance and train the model
    lg_reg = LogisticRegression()
    lg_reg.fit(X_train_scaled, y_train)

    # test a model
    y_pred = lg_reg.predict(X_test_scaled)
    print("Accuracy of our Model:", accuracy_score(y_pred, y_test))
    print("Classification Report: ", classification_report(y_pred, y_test))

    return lg_reg, scaler


def main():
    # get the cleaned data
    data = get_clean_data()

    # create a model
    model, scaler = create_model(data)

    # saving the model
    # if not os.path.exists("model"):
    #     os.makedirs("model")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
