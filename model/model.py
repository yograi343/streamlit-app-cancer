import pandas as pd


def main():
    data = pd.read_csv("D:\\1 Learning\\cancer\\streamlit-app-cancer\\Data\\data.csv")
    data = data.drop(["Unnamed: 32", "id"], axis=1)


if __name__ == "__main__":
    main()
