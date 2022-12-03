from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from data_loader import load_dataset
from matplotlib import pyplot as plt

from logger import log

def get_pca(df):
    pca = PCA(.95)
    # pca = PCA(.95)
    # Separating out the features
    x = df.drop('rent', axis=1).values# Separating out the target
    y = df.loc[:,['rent']].values
    pca = pca.fit(x)
    components = pca.transform(x)
    return pca, components, x, y

def plot_pca(finalDf):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    ax.scatter(finalDf.loc[:, 'PC 0']
            , finalDf.loc[:, 'PC 1']
            , finalDf.loc[:, 'PC 2']
            , c = 'r'
            , s = 50)
    ax.legend(['rent'])
    ax.grid()
    plt.show()

def apply_pca(df):
    log.info("Running PCA")
    df = df.select_dtypes(exclude=['object'])
    pca, principal_components, x, y = get_pca(df)
    log.info(f"Variance explained by pca: {sum(pca.explained_variance_ratio_)}")
    log.info(f"Number of principal components: {pca.n_components_}")
    # principal_df = pd.DataFrame(data=[[np.corrcoef(df[c],principal_components[:,n])[1,0]
    #                                    for n in range(pca.n_components_)] for c in df],
    #                             columns=['PC ' + str(x) for x in range(pca.n_components_)],
    #             index = df.columns)
    # print(principal_df)

    principal_df = pd.DataFrame(data=principal_components,
                                columns=['PC ' + str(x) for x in range(pca.n_components_)])
    # plot_pca(principal_df)

    # log.info(principal_df.sort_values(['PC ' + str(x) for x in range(pca.n_components_)], axis=0))

    final_df = pd.concat([principal_df, df[['rent']]], axis = 1)
    return final_df

if __name__ == "__main__":
    df = load_dataset(filename="./data/preprocessed_data.csv")
    final_df = apply_pca(df)
    log.info(final_df)

