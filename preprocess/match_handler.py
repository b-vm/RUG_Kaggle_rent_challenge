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

    df["minMatchAge"] = 0
    df["maxMatchAge"] = 100

    for idx, option_pair in enumerate(df[column]):
        if is_nan(option_pair):
            continue

        option_split = option_pair.split(' - ')
        if option_split[0] == "Not important" and option_split[1] == "Not important":
            continue
        if option_split[1] != "Not important":
            df.iloc[idx, df.columns.get_loc("maxMatchAge")] = int(option_split[1][:-6])
        if option_split[0] != "Not important":
            df.iloc[idx, df.columns.get_loc("minMatchAge")] = int(option_split[0][:-6])

    df = df.drop(column, axis="columns")
    log.info(f"Finished preprocessing on match column '{column}'")

    return df
