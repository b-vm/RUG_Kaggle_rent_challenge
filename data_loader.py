import pandas as pd

<<<<<<< HEAD
def load_dataset(filename="../data/train.csv"):
    df = pd.read_csv(filename, index_col='id')
    # if 'id' in df.columns: #Why this?
        # df = df.drop('id', axis=1)
=======
def load_dataset(filename="./data/train.csv"):
    df = pd.read_csv(filename)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
>>>>>>> 06b19ab04c5473affcd4fea06685d7a025212a11
    return df
