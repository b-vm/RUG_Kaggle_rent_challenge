from util import is_nan

from logger import log

def preprocess_match_columns(df, columns):
    for column in columns:
        log.info(f"Running preprocessing on match column '{column}'...")
        unique_option_pairs = df[column].unique()

        split_options = []

        for option_pair in unique_option_pairs:
            if option_pair == "Not important":
                continue
            if is_nan(option_pair):
                continue
            if column == "matchStatus":
                split_options.extend(option_pair.split(', '))
            else:
                split_options.extend(option_pair.split(' '))


        split_options = [x for x in split_options if (not x.isdigit() and not x == '+' and not x =="more")]

        split_options = set(split_options)

        for option in split_options:
            df[column+"_"+option] = 0

        for idx, option_pair in enumerate(df[column]):
            if is_nan(option_pair):
                continue
            for option in split_options:
                if option_pair == "Not important":
                    df.iloc[idx, df.columns.get_loc(column+"_"+option)] = 1
                    continue
                if option in option_pair:
                    df.iloc[idx, df.columns.get_loc(column+"_"+option)] = 1
                    continue
        df = df.drop(column, axis="columns")
        log.info(f"Finished preprocessing on match column '{column}'")

    return df

def preprocess_match_age(df):
    column = 'matchAge'
    log.info(f"Running preprocessing on match column '{column}'...")
    unique_option_pairs = df[column].unique()
    print(unique_option_pairs)

    split_options = []

    for option_pair in unique_option_pairs:
        if option_pair == "Not important":
            continue
        if is_nan(option_pair):
            continue
        split_options.extend(option_pair.split(' - '))

    for option in ["minMatchAge", "maxMatchAge"]:
        df[option] = 0

    for idx, option_pair in enumerate(df[column]):
        if is_nan(option_pair):
            continue

        if option_pair == "Not important":
            df.iloc[idx, df.columns.get_loc("minMatchAge")] = 1
            df.iloc[idx, df.columns.get_loc("maxMatchAge")] = 1

        option_pair.split(' - ')
        for option in ["minMatchAge", "maxMatchAge"]:
            df.iloc[idx, df.columns.get_loc(option)] = 1

    for idx, option_pair in enumerate(df[column]):
        if is_nan(option_pair):
            continue
        for option in split_options:
            if option_pair == "Not imp":
                df.iloc[idx, df.columns.get_loc(option)] = 1
                continue
            if option in option_pair:
                df.iloc[idx, df.columns.get_loc(option)] = 1
                continue
    df = df.drop(column, axis="columns")
    log.info(f"Finished preprocessing on match column '{column}'")

    return df

