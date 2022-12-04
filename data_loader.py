import pandas as pd

def load_dataset(filename="../data/train.csv"):
    # df = pd.read_csv(filename)
    df = pd.read_csv(filename, index_col='id')

    # I'm using this instead of the given IDs, such that the ids go from 1->~28000
    # If you don't the ids are in a random order, making it harder to add a different dataframe with different ids
    # if 'id' in df.columns:
    #     df = df.drop('id', axis=1)
    return df
