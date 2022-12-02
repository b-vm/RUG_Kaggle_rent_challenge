import pandas as pd
from data_loader import load_dataset

def to_one_hot_encoding(df, columns):
    for column in columns:
        # get the dummies and store it in a variable
        dummies = pd.get_dummies(df[column])

        # Concatenate the dummies to original dataframe
        df = pd.concat([df, dummies], axis='columns')

        # drop the values
        df = df.drop(column, axis='columns')
    return df

def discretize_categories(df, categorical_to_code_columns):
    df[categorical_to_code_columns] = df[categorical_to_code_columns].astype('category')
    df[categorical_to_code_columns] = df[categorical_to_code_columns].apply(lambda x: x.cat.codes)
    return df

def main():
    df = load_dataset()
    print(df.columns)

    categorical_to_code_columns = ['city', 'isRoomActive']
    df = discretize_categories(df, categorical_to_code_columns)

    one_hot_encoded_columns = ['matchLanguages']
    df = to_one_hot_encoding(df, one_hot_encoded_columns)

    # print the dataframe
    print(df.head())
    df.to_csv("./output.csv")

if __name__ == '__main__':
    main()
