import pandas as pd

def load_dataset(filename="../data/train.csv"):
    df = pd.read_csv(filename)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    return df
