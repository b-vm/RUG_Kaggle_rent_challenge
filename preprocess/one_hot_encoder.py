import pandas as pd

def to_one_hot_encoding(df, columns):
    for column in columns:
        # get the dummies and store it in a variable
        dummies = pd.get_dummies(df[column])

        # Concatenate the dummies to original dataframe
        df = pd.concat([df, dummies], axis='columns')

        # drop the values
        df = df.drop(column, axis='columns')
    return df
