import pandas as pd

def load_dataset(filename="../data/train.csv"):
    return pd.read_csv(filename)
