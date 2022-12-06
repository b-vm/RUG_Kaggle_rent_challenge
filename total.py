from data_loader import load_dataset
from preprocess_data import preprocess_data
from preprocess.normalize import normalize_dataframe

import os
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor


def main():
    using_test_set = False
    data_filename = "./data/preprocessed_data.csv" if not using_test_set else "./data/preprocessed_data_test.csv"
    df = load_dataset(filename=data_filename)

    # df = preprocess_data(df)

    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    df, scaler = normalize_dataframe(df)

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

    model = XGBRegressor(tree_method='hist', objective='reg:squarederror', enable_categorical=True)

    gs = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', verbose=3)
    gs.fit(df.loc[:, df.columns != 'rent'], df['rent'])

    sweep_params = {param:[setting[param] for setting in gs.cv_results_['params']] for param in gs.cv_results_['params'][0]}

    print(gs.cv_results_['mean_test_score'])
    print(scaler.inverse_transform(gs.cv_results_['mean_test_score']))
    exit()

    results = pd.DataFrame({
        **sweep_params,
        "mean_test_score": gs.cv_results_['mean_test_score'],
        "rank_test_score": gs.cv_results_['rank_test_score']
    })
    print(results.sort_values('rank_test_score'))
    results.to_csv(os.path.join("out", "parameter_sweep.csv"))


# to beat, 87.51 (global 84)
if __name__ == '__main__':
    main()
