from sklearn.decomposition import PCA
import pandas as pd

from logger import log

def load_dataset(filename="./completed_data.csv"):
    return pd.read_csv(filename)

def get_pca(df):
    pca = PCA(.95)
    pca.fit_transform(df, y="rent")
    return pca

def apply_pca(df):
    print("running this?")
    df = df.select_dtypes(exclude=['object'])
    pca = get_pca(df)
    print(f"Variance explained by pca: {pca.explained_variance_ratio_}")
    print(f"Number of principal components: pca.n_components_")
    principal_df = pd.DataFrame(data = pca
             , columns = ['principal component ' + x for x in range(pca.n_components_) ])
    print(principal_df)

    transformed_df = pca.transform(df)
    return transformed_df

if __name__ == "__main__":
    df = load_dataset(filename="./data/preprocessed_data.csv")
    apply_pca(df)

