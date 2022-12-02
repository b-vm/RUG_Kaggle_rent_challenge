from sklearn.decomposition import PCA
import pandas as pd

from logger import log

def load_dataset(filename="./completed_data.csv"):
    return pd.read_csv(filename)


def get_pca(df):
    pca = PCA(.95)
    pca.fit_transform(df, y="rent")
    return pca


def get_feature_rank_pca(df, feature):
    pass


def get_feature_impact_score_pca(df, feature):
    pass


def apply_pca(df):
    df = df.select_dtypes(exclude=['object'])
    pca = get_pca(df)

    transformed_df = pca.transform(df)
    return transformed_df


if __name__ == "__main__":
    df = load_dataset(filename="./data/preprocessed_data.csv")
    print(apply_pca(df))

