if __name__=="__main__":
    import sys
    sys.path.append('..')
    

import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re
from preprocess.imputation import impute_with_set_value, impute_average_value
from tqdm import tqdm
import numpy as np


def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename, index_col="id")


def save_dataset(df, filename="./data/train_location_mapped.csv"):
    df.to_csv(filename)


def encode_text(model, tokenizer, texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, return_tensors="pt")  # here
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)  # here
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


def get_encoding_tokenizer_and_model():
    print("Loading Model and Tokenizer")

    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model


def encode_descriptions(df):
    tokenizer, model = get_encoding_tokenizer_and_model()
    print("Encoding Descriptions")
    df["descriptionVector"] = df.apply(
        lambda row: encode_text(model, tokenizer, df.descriptionNonTranslated), axis=1
    )
    return df


def get_rent_from_text(model, text):
    prompt = {
        "question": "What is the price of the rent?",
        "context": str(text),
    }
    result = model(prompt)
    return result["answer"]


def check_if_rent_is_in_text(df):
    df["rentInText"] = df.apply(
        lambda row: val_in_text(row.rent, row.descriptionNonTranslated), axis=1
    )
    print(df["rentInText"].value_counts())
    return df


def get_rent_with_nlp(df, model):
    df["rentFromNLP"] = df.progress_apply(
        lambda row: strip_rent_text(
            get_rent_from_text(model, row.descriptionNonTranslated)
        ),
        axis=1,
    )
    # print(df["rentFromNLP"].value_counts())
    return df


def is_str_in_text(df, value):
    df["rentInText"] = df.apply(
        lambda row: val_in_text(value, row.descriptionNonTranslated), axis=1
    )
    print(df["rentInText"].value_counts())
    return df


def val_in_text(val, text):
    if str(val) in str(text):
        return True
    return False


def strip_rent_text(rent_text):
    result = re.search(r"([0-9]+),[0-9]+", rent_text)
    if result == None:
        result = re.search(r"EUR ([0-9]+)", rent_text)
    if result == None:
        result = re.search(r"euro ([0-9]+)", rent_text)
    if result == None:
        result = re.search(r"([0-9]+) euro", rent_text)
    if result == None:
        result = re.search(r"([0-9]+) euro", rent_text)
    if result == None:
        result = re.search(r"€([0-9]+)", rent_text)
    if result == None:
        result = re.search(r"€ ([0-9]+)", rent_text)
    if result == None:
        result = re.search(r"([0-9]+),-", rent_text)

    # print(result.group(1))
    if result == None:
        return ""
    return result.group(1)


def check_accuracy(df):
    df["predictionCorrect"] = df.apply(
        lambda row: str(row.rentFromNLP) == str(row.rent), axis=1
    )
    print("Correct predictions:")
    print(df["predictionCorrect"].value_counts())
    return df


def get_nlp_model():
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return nlp


# def check_accuracy(df):
def merge_df_from_file(df, filename):
    df_2 = load_dataset(filename)
    df = pd.merge(df, df_2, how="outer", on=list(df.columns))
    return df


def merge_df(df1, df2):
    print(df1.columns)
    df = pd.merge(df1, df2, how="outer", on=list(df1.columns))
    return df


def predict_price_with_nlp_train(df):
    tqdm.pandas()
    df = check_if_rent_is_in_text(df)
    model = get_nlp_model()
    df = get_rent_with_nlp(df.loc[df["rentInText"] == True], model)
    # df = check_accuracy(df)
    return df


def predict_price_with_nlp_test(df):
    tqdm.pandas()
    model = get_nlp_model()
    df = get_rent_with_nlp(df, model)
    return df


def predict_price_with_nlp_test_as_file(from_filepath, to_filepath):
    tqdm.pandas()
    df = load_dataset(from_filepath)
    model = get_nlp_model()
    df = get_rent_with_nlp(df, model)
    save_dataset(df, to_filepath)


def predict_price_with_nlp_train_as_file(from_filepath, to_filepath):
    tqdm.pandas()
    df = load_dataset(from_filepath)
    df = check_if_rent_is_in_text(df)
    model = get_nlp_model()
    df = get_rent_with_nlp(df.loc[df["rentInText"] == True], model)
    # df = check_accuracy(df)
    save_dataset(df, to_filepath)


def mae_on_prediction(df):
    df["predictionError"] = df.apply(
        lambda row: abs(row.rent - row.rentFromNLP) if row.rentFromNLP > 0 else 0.0,
        axis=1,
    )
    print("Mean prediction error:")
    print(df["predictionError"].max())
    print(df["predictionError"].mean())
    return df


def filter_predictions(df):
    df.loc[df["rentFromNLP"] > 6000, "rentFromNLP"] = 0.0
    df.loc[df["rentFromNLP"] <= 10, "rentFromNLP"] = 0.0
    return df


def floats_to_ints(df):

    return df


def preprocess_nlp_stuff(df, is_test_set: bool = False, nlp_impute_method: int = 0):
    # Add the nlp-based rent estimation
    nlp_predict_df = (
        load_dataset("./test_with_nlp_prediction.csv")
        if is_test_set
        else load_dataset("./train_with_nlp_prediction.csv")
    )

    # df = (
    #     merge_df_from_file(df, "./test_with_nlp_prediction.csv")
    #     if is_test_set
    #     else merge_df_from_file(df, "./train_with_nlp_prediction.csv")
    # )
    nlp_predict_df = filter_predictions(nlp_predict_df)
    nlp_predict_df.dropna()
    nlp_predict_df["rentFromNLP"] = nlp_predict_df["rentFromNLP"].astype("Int64")

    df = merge_df(df, nlp_predict_df)

    # method 0 - dont do anything
    if nlp_impute_method == 1:
        # method 1 - set average rent
        average_rent_in_train_set = 670
        df = impute_with_set_value(df, "rentFromNLP", average_rent_in_train_set)
    # elif nlp_impute_method == 2:
    #     # method 2 - set average of predicted, probably better in case of class imbalance
    #     df = impute_average_value(df, "rentFromNLP")

    return df


def main():
    print("Loading Dataset")
    df = load_dataset("../data/train.csv")
    df = encode_descriptions(df.head(10))
    print(df[["descriptionNonTranslated", "descriptionVector"]])

    exit()

    df.rent = df.rent.astype(float)
    df = filter_predictions(df)
    # df["rentFromNLP"] = df["rentFromNLP"].astype("Int64")
    df["rentFromNLP"].fillna(0.0)
    print(df)
    mae_on_prediction(df)
    df = check_accuracy(df)
    print(df["rentFromNLP"].value_counts())
    print(df["rentInText"].value_counts())

    # train = load_dataset()
    # df = merge_df_from_file(train, "./train_with_nlp_prediction.csv")
    # print(df.head())
    # exit()

    # predict_price_with_nlp_train_as_file(
    #     "./data/train.csv", "./data/train_with_nlp_prediction.csv"
    # )
    # predict_price_with_nlp_test_as_file(
    #     "./data/test.csv", "./data/test_with_nlp_prediction.csv"
    # )

    # df = load_dataset()
    # df = df.head(10000)
    # # df = is_str_in_text(df, "euro")
    # df = check_if_rent_is_in_text(df)
    # # nlp_model()
    # model = get_nlp_model()
    # df = get_rent_with_nlp(df.loc[df["rentInText"] == True], model)
    # df = check_accuracy(df)
    # print(df[["rent", "rentFromNLP", "predictionCorrect"]])


# def nlp_model():
#     model_name = "deepset/roberta-base-squad2"

#     # a) Get predictions
#     nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
#     QA_input = {
#         "question": "What is the rent?",
#         "context": "Spacious fully furnished room available for a female tenant for at least one year– registering is possible – This room is available from 01-04-2020 onwards – The apartment is 110 square meters at the Gouden Leeuw in Amsterdam – Shoppingcenter Ganzenpoort is only minutes away, train, metro, bus and the Hoge school Diemen are in this neighbourhood- Within 10 minutes from the city center of Amsterdam. You are sharing the apartment with three other young ladies. Nicely furnished apartment in a quiet environment. De price is 495 euro all inn, except extra heating costs, internet, and taxes. The room can be locked and has central heating. Taxes are shared with the other tenants. The deposit is two months rent. The room is for one person only.",
#     }
#     res = nlp(QA_input)
#     print(res)
#     # b) Load model & tokenizer
#     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    main()
