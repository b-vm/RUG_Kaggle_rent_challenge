import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_dataframe(df):
    scale = StandardScaler()

    scaled_df = df.select_dtypes(exclude=['object', 'category'
                                          ])  #.drop(columns="rentFromNLP")
    scaled_df = scaled_df.drop(['imageBasedRent', 'rent', 'rentFromNLP'],
                               axis=1,
                               errors='ignore')
    scaled_df_cols = scaled_df.columns
    scaled_df = pd.DataFrame(scale.fit_transform(scaled_df.values),
                             columns=scaled_df.columns,
                             index=scaled_df.index)
    df[scaled_df.columns] = scaled_df

    return df, scale, scaled_df_cols
