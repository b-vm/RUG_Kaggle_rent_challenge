if __name__ == "__main__":
    import sys
    sys.path.append('..')

from data_loader import load_dataset
from one_hot_encoder import to_one_hot_encoding
from discretization_encoding import discretize_categories
from match_handler import preprocess_match_columns, preprocess_match_age

from category_encoders import BinaryEncoder
from util import cartesian_coord

def preprocess(df):
    print(df.columns)

    # print(len(df.index))
    # print(len([x for x in df.duplicated(keep=False) if x]))

    # categorical_to_code_columns = ['isRoomActive']
    # df = discretize_categories(df, categorical_to_code_columns)

    # one_hot_encoded_columns = ['matchLanguages']
    # df = to_one_hot_encoding(df, one_hot_encoded_columns)

    # enc = BinaryEncoder(cols=["matchLanguages"]).fit(df)
    # df = enc.transform(df)

    # print(df['matchLanguages'])
    df = preprocess_match_columns(df, ["matchLanguages", "matchStatus", "matchGender"])
    # df = preprocess_match_columns(df, ["matchStatus"])
    # preprocess_match_age(df)

    # print the dataframe
    # print(df.head())
    # print(df['isRoomActive'].unique())
    # print(df.columns)
    df.to_csv("./output.csv")

if __name__ == '__main__':
    loaded_df = load_dataset()
    preprocess(loaded_df)
