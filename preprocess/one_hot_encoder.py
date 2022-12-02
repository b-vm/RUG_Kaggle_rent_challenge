import pandas as pd

from logger import log

def to_one_hot_encoding(df, columns):
    for column in columns:
        log.info(f"One hot encoding column '{column}'...")
        # get the dummies and store it in a variable
        dummies = pd.get_dummies(df[column])
        dummies = dummies.rename(columns={x: column+"_"+x for x in dummies.columns})

        # Concatenate the dummies to original dataframe
        df = pd.concat([df, dummies], axis='columns')

        # drop the values
        df = df.drop(column, axis='columns')
        log.info(f"Finished one hot encoding column '{column}'")
    return df
