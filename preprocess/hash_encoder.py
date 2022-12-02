import category_encoders as ce
import pandas as pd

from logger import log

def find_large_enough_power_of_2(x):
    powers = []
    i = 1
    while i <= x:
        if i & x:
            powers.append(i)
        i <<= 1
    return len(powers) + 1

def hash_encode_columns(df, columns):
    for column in columns:
        log.info(f"Hash encoding column '{column}'...")
        n_components = find_large_enough_power_of_2(len(df[column].unique()))

        encoder=ce.HashingEncoder(cols=column,n_components=n_components)
        new_cols = encoder.fit_transform(df[column])
        new_cols = new_cols.rename(columns={x: column+"_"+x for x in new_cols.columns})
        df = pd.concat([df, new_cols], axis =1)
        log.info(f"Finished hash encoding column '{column}'")
        df = df.drop(column, axis="columns")

    return df

if __name__=="__main__":
    n = 63
    print(find_large_enough_power_of_2(n))
