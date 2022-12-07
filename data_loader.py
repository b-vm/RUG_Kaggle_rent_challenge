import pandas as pd

def load_dataset(filename="../data/train.csv"):
    # df = pd.read_csv(filename)
    df = pd.read_csv(filename, index_col='id')

    return df
