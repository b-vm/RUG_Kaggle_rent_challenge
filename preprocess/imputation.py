import pandas as pd

from logger import log


def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename)


def impute_most_common(df, column_name, to_replace_values=[]):
    # Get most common value
    most_common_value = get_most_common_value(df, column_name, to_replace_values)
    # replace NA values
    df[column_name] = df[column_name].fillna(most_common_value)
    # Replace other values
    for value in to_replace_values:
        df[column_name] = df[column_name].replace(value, most_common_value)
    return df


# def impute_na_with_most_common(df, column_name, blacklist_values=[]):
#     most_common_value = get_most_common_value(df, column_name, blacklist_values)
#     df[column_name] = df[column_name].fillna(most_common_value)
#     return df


def get_most_common_value(df, column_name, blacklist_values=[]):
    value_counts = df[column_name].value_counts()
    for value in value_counts.index.tolist():
        if value in blacklist_values:
            continue
        else:
            return value


# def count_missing(df, column_name):
#     missing_mask = df.isna().sum()
#     # missing_count = missing_mask.count()
#     print(missing_mask)


# def get_missing_mask(df, column_name):
#     return df[column_name].isna()


def print_helpers(df):
    print(df.head())
    print(df.shape)
    print(df.columns)


def impute_data(df, columns):
    # ugly master function
    # impute_most_common takes a to_replace_values, which if left empty only na values are replaced. Any values placed in the list are also replaced
    # So don't make a loop, because you want to define custom values to replace
    for column in columns:
        df = impute_most_common(df, column)
    # df = impute_most_common(df, "gender", ["Unknown"])
    # df = impute_most_common(df, "internet")
    # df = impute_most_common(df, "roommates")
    # df = impute_most_common(df, "shower")
    # df = impute_most_common(df, "toilet")
    # df = impute_most_common(df, "kitchen")
    # df = impute_most_common(df, "living")
    # df = impute_most_common(df, "matchCapacity")
    # df = impute_most_common(df, "isRoomActive")
    # print(df)
    return df


def main():
    # column_name = "gender"
    # df = impute_most_common(df, column_name, ["Unknown"])
    # print(df[column_name].value_counts())

    # count_missing(df, column_name)
    df = load_dataset()
    contains_nan_values = df.columns[df.isna().any()].tolist()
    print(f"Updating for the following columns: {contains_nan_values}")
    df = impute_data(df, contains_nan_values)
    df.to_csv("./data/imputed_data.csv")


if __name__ == "__main__":
    main()
