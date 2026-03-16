import pandas as pd
from src.preprocess import preprocess_data
from src.train_models import train_and_evaluate

def load_data(path):
    df=pd.read_csv(path)
    return df

def analyze_data(df):
    print("\nDataset shape:", df.shape)
    print("\nColumns:",df.columns)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)

def choose_target(df):
    print("\nAvailable columns:")
    print(list(df.columns))
    target=input("\nEnter the target column: ")
    return target


if __name__=="__main__":
    df=load_data("data/Titanic-Dataset.csv")

    analyze_data(df)
    target=choose_target(df)
    x,y=preprocess_data(df,target)
    print("\nAfter preprocessing:")
    print("\nFeature shape:", x.shape)
    print("\nTarget shape:", y.shape)

    results=train_and_evaluate(x,y)
    print("\nModel Performance:")

    for model,score in results.items():
        print(f"{model}: {score:.3f}")

    best_model=max(results,key=results.get)
    print("\nBest Model:",best_model)