import pandas as pd

def load_dataset(filename="../data/train.csv"):
    df = pd.read_csv(filename, index_col='id')
    # if 'id' in df.columns: #Why this?
        # df = df.drop('id', axis=1)
    return df
