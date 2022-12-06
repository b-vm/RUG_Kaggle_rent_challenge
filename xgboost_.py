import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from preprocess.pca import apply_pca
import xgboost as xgb
from xgboost import XGBRegressor


def train_kfold(model, train_data):
    start_time = time.monotonic()
    scores = []
    kf = KFold(n_splits=5)
    for train_split_idxs, val_split_idxs in kf.split(train_data):
        train_split = train_data.iloc[train_split_idxs]
        val_split = train_data.iloc[val_split_idxs]

        model.fit(train_split.loc[:, train_split.columns != 'rent'], train_split['rent'])
        preds = model.predict(val_split.loc[:, val_split.columns != 'rent'])

        score = (val_split['rent'] - preds).abs().mean()
        scores.append(score)
        print(f"split {len(scores)} mean absolute difference: €{score}")

    print(f"total mean absolute difference: €{np.array(scores).mean():0.2f}  in {time.monotonic()-start_time:0.3f} seconds")


def train_all_data(model, train_data):
    model.fit(train_data.loc[:, train_data.columns != 'rent'], train_data['rent'])
    return model

def get_importances(model, train_data):
    model = train_all_data(model, train_data)
    importances = pd.DataFrame({
        "feature": train_data.loc[:, train_data.columns != 'rent'].columns,
        "importance": model.feature_importances_
    })

    sorted = importances.sort_values('importance')
    sorted.to_csv(os.path.join("out", "xgboost_importances.csv"))
    print(sorted)


if __name__ == "__main__":
    # train_data = pd.read_csv(os.path.join("data", "train.csv"))
    # train_data = pd.read_csv(os.path.join("data", "output.csv"), index_col='id')
    train_data = pd.read_csv(os.path.join("data", "preprocessed_data.csv"), index_col='id')

    print("train data shape", train_data.shape)

    # transformed_train_data = pd.DataFrame(apply_pca(train_data))
    # transformed_train_data['rent'] = train_data['rent']
    # train_data = transformed_train_data
    # print(train_data.head())
    # exit()


    # for col in train_data:
    #     if train_data[col].dtype == 'object':
    #         train_data[col] = train_data[col].astype('category')


    # param_grid = {
    #     "n_estimators": [100],
    #     "max_depth": [16],
    #     "learning_rate": [0.05],
    #     "gamma": [15, 20],
    #     "min_child_weight": [5],
    #     "colsample_bytree": [0.5, 0.6]
    # }

    best_params = {
        "n_estimators": [100], #300
        "max_depth": [18],
        "learning_rate": [0.05],
        "subsample": [1],
        "gamma": [20],
        "min_child_weight": [5],
        "colsample_bytree": [0.7]
    }
    param_grid = best_params


    # model = XGBRegressor(tree_method='gpu_hist', objective='reg:squarederror', enable_categorical=True)
    model = XGBRegressor(tree_method='hist', objective='reg:squarederror', enable_categorical=True)

    gs = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', verbose=3)
    gs.fit(train_data.loc[:, train_data.columns != 'rent'], train_data['rent'])

    sweep_params = {param:[setting[param] for setting in gs.cv_results_['params']] for param in gs.cv_results_['params'][0]}

    results = pd.DataFrame({
        **sweep_params,
        "mean_test_score": gs.cv_results_['mean_test_score'],
        "rank_test_score": gs.cv_results_['rank_test_score']
    })
    print(results.sort_values('rank_test_score'))
    results.to_csv(os.path.join("out", "parameter_sweep.csv"))


    # best_params = {key:best_params[key][0] for key in best_params}
    # model = XGBRegressor(tree_method='hist', objective='reg:squarederror', enable_categorical=True, **best_params)
    # get_importances(model, train_data)
    # model.save(os.path.join("out", "xgboost_model"))

# to beat, 87.51
