import pandas as pd

def discretize_categories(df, categorical_to_code_columns):
    df[categorical_to_code_columns] = df[categorical_to_code_columns].astype('category')
    df[categorical_to_code_columns] = df[categorical_to_code_columns].apply(lambda x: x.cat.codes)
    return df
