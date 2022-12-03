from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from data_loader import load_dataset

from logger import log

def get_pca(df):
    pca = PCA(.95)
    # Separating out the features
    x = df.drop('rent', axis=1).values# Separating out the target
    y = df.loc[:,['rent']].values
    pca = pca.fit(x)
    components = pca.transform(x)
    return pca, components, x, y

def apply_pca(df):
    log.info("Running PCA")
    df = df.select_dtypes(exclude=['object'])
    pca, principal_components, x, y = get_pca(df)
    log.info(f"Variance explained by pca: {sum(pca.explained_variance_ratio_)}")
    log.info(f"Number of principal components: {pca.n_components_}")
    principal_df = pd.DataFrame(data=[[np.corrcoef(df[c],principal_components[:,n])[1,0]
                                       for n in range(pca.n_components_)] for c in df],
                                columns=['PC ' + str(x) for x in range(pca.n_components_)],
                index = df.columns)
    # print(principal_df)

    principal_df = pd.DataFrame(data=principal_components,
                                columns=['PC ' + str(x) for x in range(pca.n_components_)])

    # log.info(principal_df.sort_values(['PC ' + str(x) for x in range(pca.n_components_)], axis=0))

    final_df = pd.concat([principal_df, df[['rent']]], axis = 1)
    return final_df

if __name__ == "__main__":
    df = load_dataset(filename="./data/preprocessed_data.csv")
    final_df = apply_pca(df)
    log.info(final_df)

