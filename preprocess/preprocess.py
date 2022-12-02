from data_loader import load_dataset
from one_hot_encoder import to_one_hot_encoding
from discretization_encoding import discretize_categories

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
