import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


if __name__ == "__main__":
    train_data = pd.read_csv(os.path.join("data", "train.csv"))
    print("train data", train_data.shape)

    for col in train_data:
        if train_data[col].dtype == 'object':
            train_data[col] = train_data[col].astype('category')

    bst = XGBRegressor(tree_method='hist', objective='reg:squarederror', enable_categorical=True)

    scores = []
    kf = KFold(n_splits=5)
    for train_split_idxs, val_split_idxs in kf.split(train_data):
        train_split = train_data.iloc[train_split_idxs]
        val_split = train_data.iloc[val_split_idxs]

        bst.fit(train_split.loc[:, train_split.columns != 'rent'], train_split['rent'])
        preds = bst.predict(val_split.loc[:, val_split.columns != 'rent'])

        score = (val_split['rent'] - preds).abs().mean()
        scores.append(score)
        print(f"split {len(scores)} mean absolute difference: €{score}")

    print(f"total mean absolute difference: €{np.array(scores).mean()}")