#     import sys
#     sys.path.append('..')

from logger import log

from data_loader import load_dataset
from preprocess.imputation import impute_data
from preprocess.location_mapping import calculate_city_centers, get_distance_to_city_center
from preprocess.one_hot_encoder import to_one_hot_encoding
from preprocess.discretization_encoding import discretize_categories
from preprocess.match_handler import preprocess_match_columns, preprocess_match_age

from preprocess.hash_encoder import hash_encode_columns
from preprocess.normalize import normalize_dataframe
from preprocess.time_handler import columns_to_timestamp, preprocess_posted_ago, preprocess_raw_availability

def preprocess_data(df):
    log.info(f"Applying preprocessing on the following columns:")
    orig_df = df
    if 'Unnamed: 0' in df.columns:
        orig_df = df.drop('Unnamed: 0', axis=1)
    df = orig_df.drop('rent', axis=1)
    for column in df.columns:
        print(f" -- {column}")
    original_column_length = len(df.columns)

    ## Change missing data
    # Impute missing data
    contains_nan_values = df.columns[df.isna().any()].tolist()
    log.info(f"Imputing missing data for the following columns: {contains_nan_values}")
    df = impute_data(df, contains_nan_values)
    # Add some extra data
    log.info(f"Adding postalCodeInt to dataframe")
    df['postalCodeInt'] = df['postalCode'].apply(lambda x: int(x[:-2]))
    # Distance to city centers
    log.info(f"Adding city centers")
    city_centers = calculate_city_centers(df)
    df = get_distance_to_city_center(df, city_centers)


    ## Transform Data
    # Simple ordinal discretization
    categorical_to_code_columns = ['isRoomActive', 'energyLabel', "internet", "shower", "toilet", "kitchen", "living", "pets", "smokingInside", "rentDetail"]
    df = discretize_categories(df, categorical_to_code_columns)
    # One hot encoding
    one_hot_encoded_columns = ['propertyType', 'furnish', 'gender']
    df = to_one_hot_encoding(df, one_hot_encoded_columns)
    # Hash encoding
    df = hash_encode_columns(df, ["city", "matchCapacity", "roommates", "postalCode"])

    ## Some special cases
    # match columns
    df = preprocess_match_columns(df, ["matchLanguages", "matchStatus", "matchGender"])
    df = preprocess_match_age(df)
    # time related columns
    df = preprocess_posted_ago(df)
    df = columns_to_timestamp(df, ["firstSeenAt", "lastSeenAt"])
    df = preprocess_raw_availability(df)

    # print the dataframe
    log.info(f"Changed from {original_column_length} number of columns to {len(df.columns)} number of columns")

    df = normalize_dataframe(df)

    orig_df[df.columns] = df

    output_filename = "./data/preprocessed_data.csv"
    log.info(f"Saving new dataset to {output_filename}")
    df.to_csv(output_filename)

if __name__ == '__main__':
    loaded_df = load_dataset(filename="./data/train.csv")
    preprocess_data(loaded_df)
