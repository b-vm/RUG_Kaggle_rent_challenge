if __name__ == "__main__":
    import sys
    sys.path.append('..')

from logger import log


from data_loader import load_dataset
from one_hot_encoder import to_one_hot_encoding
from discretization_encoding import discretize_categories
from match_handler import preprocess_match_columns, preprocess_match_age

from hash_encoder import hash_encode_columns
from time_handler import columns_to_timestamp, preprocess_posted_ago, preprocess_raw_availability

def preprocess(df):
    log.info(f"Applying preprocessing on the following columns:")
    for column in df.columns[:-1]:
        print(f" -- {column}")
    original_column_length = len(df.columns)

    # Add some extra data
    df['postalCodeInt'] = df['postalCode'].apply(lambda x: int(x[:-2]))


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
    # print(df.head())

    output_filename = "./enhanced_data.csv"
    log.info(f"Saving new dataset to {output_filename}")
    df.to_csv(output_filename)

if __name__ == '__main__':
    loaded_df = load_dataset()
    preprocess(loaded_df)
