import pandas as pd

from logger import log

def discretize_categories(df, categorical_to_code_columns):
    log.info(f"Discretizing categories to code for columns: '{categorical_to_code_columns}...'")
    df[categorical_to_code_columns] = df[categorical_to_code_columns].astype('category')
    df[categorical_to_code_columns] = df[categorical_to_code_columns].apply(lambda x: x.cat.codes)
    log.info(f"Finished discretizing categories to code for columns: '{categorical_to_code_columns}'")
    return df
