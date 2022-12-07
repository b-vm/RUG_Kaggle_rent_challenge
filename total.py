from data_loader import load_dataset
from preprocess.text_analysis import filter_predictions
from preprocess_data import preprocess_data
from preprocess.normalize import normalize_dataframe

import numpy as np

import os
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
import xgboost as xgb
from xgboost import XGBRegressor


def invTransform(scaler, data, colName, colNames):
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))),
                         columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values


def main():
    using_test_set = True
    data_filename = "./data/preprocessed_data.csv"
    df = load_dataset(filename=data_filename)

    for col in df:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        if df[col].dtype == 'bool':
            df[col] = df[col].astype('float64')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('float64')
    # df = df.select_dtypes(exclude=['object'])  #.drop(columns="rentFromNLP")

    test_filename = "./data/preprocessed_data_test.csv"
    test_df = load_dataset(filename=test_filename)

    for col in test_df:
        if test_df[col].dtype == 'object':
            test_df[col] = test_df[col].astype('category')
        if test_df[col].dtype == 'bool':
            test_df[col] = test_df[col].astype('float64')
        if test_df[col].dtype == 'int64':
            test_df[col] = test_df[col].astype('float64')
    # test_df = test_df.select_dtypes(exclude=['object'])  #.drop(columns="rentFromNLP")

    test_df, _, _ = normalize_dataframe(test_df)

    df, _, _ = normalize_dataframe(df)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=1001)
    orig_test = test_df
    test_df = test_df.sample(frac=1, random_state=1001)

    if 'Unnamed: 0' in test_df:
        test_df = test_df.drop('Unnamed: 0', axis = 1)

    test_df = test_df[df.loc[:, df.columns != 'rent'].columns]


    best_params = {
        "n_estimators": [300],  #300
        "max_depth": [18],
        "learning_rate": [0.05],
        "subsample": [1],
        "gamma": [20],
        "min_child_weight": [5],
        "colsample_bytree": [0.7],
    }
    param_grid = best_params

    model = XGBRegressor(tree_method='hist',
                         objective='reg:squarederror',
                         enable_categorical=True)

    gs = GridSearchCV(model,
                      param_grid,
                      cv=5,
                      n_jobs=-1,
                      scoring='neg_mean_absolute_error',
                      verbose=3,
                      refit=True)
    gs.fit(df.loc[:, df.columns != 'rent'], df['rent'])
    # model.fit(df.loc[:, df.columns != 'rent'], df['rent'])

    gs.predict(df.loc[:, df.columns != 'rent'])
    sweep_params = {
        param: [setting[param] for setting in gs.cv_results_['params']]
        for param in gs.cv_results_['params'][0]
    }

    # gs_results = [gs.cv_results_[f'split{x}_test_score'][0] for x in range(5)]
    # results = pd.DataFrame(np.zeros((len(gs_results), len(scaled_df_cols))))
    # results.iloc[:, -1] = gs_results
    # print(results)
    # results = pd.DataFrame(scaler.inverse_transform(results))
    # print(results)
    # print(results.iloc[:, -1].mean())

    # print(gs.cv_results_['mean_test_score'])
    # print(scaler.inverse_transform(gs.cv_results_['mean_test_score']))
    # exit()

    results = pd.DataFrame({
        **sweep_params, "mean_test_score":
        gs.cv_results_['mean_test_score'],
        "rank_test_score":
        gs.cv_results_['rank_test_score']
    })
    print(results.sort_values('rank_test_score'))
    results.to_csv(os.path.join("out", "parameter_sweep.csv"))

    if not using_test_set:
        return
    print("Testing...")


    print("Generating submission...")
    predictions = gs.predict(test_df)
    # test_df['rent'] = predictions
    pred_df = pd.DataFrame(predictions, columns=["rent"], index=test_df.index)

    pred_df = pred_df.reindex(orig_test.index)

    # nlp_df = load_dataset(filename="./data/train_with_nlp_prediction.csv")


    # TODO: Overlay the NLP dataset
    # better_nlp_df = filter_predictions(nlp_df)
    # index_list = better_nlp_df.index.tolist()
    # print(nlp_df['rentFromNLP'].unique())
    # exit()

    # print(max(better_nlp_df.index))
    # print(max(pred_df.index))
    # print(type(index_list))
    # for idx in index_list:
    #     try:
    #         pred_df.iloc[idx, pred_df.columns.get_loc('rent')] = 0
    #         print(better_nlp_df.iloc[idx, better_nlp_df.columns.get_loc('rentFromNLP')])
    #     except:
    #         print("Skipping one")
    #         continue
    pred_df.to_csv("./new_submission.csv")


# to beat, 87.51 (global 84)
# Next to beat, 85.97
if __name__ == '__main__':
    main()
