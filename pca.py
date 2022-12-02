from sklearn.decomposition import PCA
import pandas as pd


def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename)


def get_pca(df):
    pca = PCA()
    pca.fit(df, y="rent")
    return pca


def get_feature_rank_pca(df, feature):
    pass


def get_feature_impact_score_pca(df, feature):
    pass


def main():
    df = load_dataset()
    pca = get_pca(df)


if __name__ == "__main__":
    main()
