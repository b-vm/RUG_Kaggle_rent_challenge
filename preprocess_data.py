#     import sys
#     sys.path.append('..')

from logger import log

import pandas as pd

from data_loader import load_dataset
from preprocess.imputation import impute_data, impute_with_set_value
from preprocess.location_mapping import (
    calculate_city_centers,
    get_distance_to_city_center,
)
from preprocess.one_hot_encoder import to_one_hot_encoding
from preprocess.discretization_encoding import discretize_categories
from preprocess.match_handler import preprocess_match_columns, preprocess_match_age, preprocess_match_capacity
from preprocess.roommates import preprocess_roommates
from preprocess.text_analysis import merge_df_from_file
from preprocess.language import preprocess_language

from preprocess.hash_encoder import hash_encode_columns
from preprocess.normalize import normalize_dataframe
from preprocess.time_handler import (
    columns_to_timestamp,
    preprocess_posted_ago,
    preprocess_raw_availability,
)
from preprocess.text_analysis import (
    predict_price_with_nlp_test,
    predict_price_with_nlp_train,
    preprocess_nlp_stuff,
)


def preprocess_data(df: pd.DataFrame, is_test_set: bool = False):

    # print(len(df["title"].unique()))
    # exit()

    df = preprocess_nlp_stuff(df, is_test_set, 1)

    log.info(f"Applying preprocessing on the following columns:")
    orig_df = df
    if "Unnamed: 0" in df.columns:
        orig_df = df.drop("Unnamed: 0", axis=1)
    if 'rent' in df:
        df = orig_df.drop("rent", axis=1)
    for column in df.columns:
        print(f" -- {column}")
    original_column_length = len(df.columns)

    ## Change missing data
    # Impute missing data
    # TODO: Some data is missing on purpose, check which columns should not be auto-imputed
    # contains_nan_values = df.columns[df.isna().any()].tolist()
    to_impute = [
        'isRoomActive', 'postedAgo', 'furnish', 'internet', 'pets',
        'smokingInside', 'matchCapacity'
    ]

    log.info(f"Imputing missing data")
    df = impute_data(df, to_impute)

    df = impute_with_set_value(df, "shower", "Unknown")
    df = impute_with_set_value(df, "kitchen", "Unknown")
    df = impute_with_set_value(df, "living", "Unknown")
    df = impute_with_set_value(df, "toilet", "Unknown")
    df = impute_with_set_value(df, "matchAge", "Not important - Not important")
    df = impute_with_set_value(df, "matchGender", "Unknown")
    df = impute_with_set_value(df, "matchStatus", "Not important")
    df = impute_with_set_value(df, "matchLanguages", "Not important")

    # Add some extra data
    log.info(f"Adding postalCodeInt to dataframe")
    df["postalCodeInt"] = df["postalCode"].apply(lambda x: int(x[:-2]))
    # Distance to city centers
    log.info(f"Adding city centers")
    city_centers = calculate_city_centers(df)
    df = get_distance_to_city_center(df, city_centers)

    # Add Image based rent
    log.info("Adding image based rent estimation...")
    imageBasedDf = None
    if is_test_set:
        imageBasedDf = pd.read_csv("./test_imageBasedRent.csv")
    else:
        imageBasedDf = pd.read_csv("./imageBasedRent.csv")
    imageBasedDf = imageBasedDf.fillna(
        imageBasedDf.loc[:, "imageBasedRent"].mean(), axis=1)
    df["imageBasedRent"] = imageBasedDf["imageBasedRent"].astype("float32")
    log.info("Finished adding image based rent estimation")

    # print(df.columns)

    # Find out the best way to impute for missing values in the nlp-based rent estimation

    ## Transform Data
    # Simple ordinal discretization
    # categorical_to_code_columns = [
    #     "isRoomActive",
    #     "energyLabel",
    #     "internet",
    #     "shower",
    #     "toilet",
    #     "kitchen",
    #     "living",
    #     "pets",
    #     "smokingInside",
    #     # "rentDetail",
    # ]
    # df = discretize_categories(df, categorical_to_code_columns)
    # One hot encoding
    # one_hot_encoded_columns = ["propertyType", "furnish", "gender"]
    # df = to_one_hot_encoding(df, one_hot_encoded_columns)

    # Hash encoding
    # df = hash_encode_columns(df, ["city", "matchCapacity", "roommates", "postalCode"])

    ## Some special cases
    # match columns
    df = preprocess_language(df)
    # df = preprocess_match_columns(df, ["matchLanguages", "matchStatus", "matchGender"])
    df = preprocess_match_columns(df, ["matchStatus", "matchGender"])
    df = preprocess_match_age(df)
    df = preprocess_match_capacity(df)

    # time related columns
    df = preprocess_posted_ago(df)
    df = columns_to_timestamp(df, ["firstSeenAt", "lastSeenAt"])
    df = preprocess_raw_availability(df)

    df = preprocess_roommates(df)

    if 'rent' in orig_df:
        df["rent"] = orig_df["rent"]

    # drop spurious rows
    if 'rent' in df:
        df = df[df.rent != 1]
    df = df.drop(columns=[
        "coverImageUrl", "rentInText", "matchLanguages",
        "descriptionTranslated", "descriptionNonTranslated"
    ],
                 errors='ignore')

    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # print the dataframe
    log.info(
        f"Changed from {original_column_length} number of columns to {len(df.columns)} number of columns"
    )

    print(max(df.index))

    return df


def preprocess_and_save_data(df, is_test_set):
    df = preprocess_data(df, is_test_set)

    output_filename = "./data/preprocessed_data.csv" if not is_test_set else "./data/preprocessed_data_test.csv"
    log.info(f"Saving new dataset to {output_filename}")
    df.to_csv(output_filename, index_label="id")


if __name__ == "__main__":
    using_test_set = True
    data_filename = "./data/train.csv" if not using_test_set else "./data/test.csv"
    loaded_df = load_dataset(filename=data_filename)
    preprocess_and_save_data(loaded_df, is_test_set=using_test_set)
