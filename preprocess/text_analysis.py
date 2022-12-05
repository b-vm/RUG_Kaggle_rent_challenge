import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re
from tqdm import tqdm


def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename, index_col="id")


def save_dataset(df, filename="./data/train_location_mapped.csv"):
    df.to_csv(filename)


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


def get_mlp_model():
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return nlp


# def check_accuracy(df):
def merge_df_from_file(df, filename):
    df_2 = load_dataset(filename)
    df = pd.merge(df, df_2, how="outer", on=list(df.columns))
    return df


def predict_price_with_nlp_train(df):
    tqdm.pandas()
    df = check_if_rent_is_in_text(df)
    model = get_mlp_model()
    df = get_rent_with_nlp(df.loc[df["rentInText"] == True], model)
    # df = check_accuracy(df)
    return df


def predict_price_with_nlp_test(df):
    tqdm.pandas()
    model = get_mlp_model()
    df = get_rent_with_nlp(df, model)
    return df


def predict_price_with_nlp_test_as_file(from_filepath, to_filepath):
    tqdm.pandas()
    df = load_dataset(from_filepath)
    model = get_mlp_model()
    df = get_rent_with_nlp(df, model)
    save_dataset(df, to_filepath)


def predict_price_with_nlp_train_as_file(from_filepath, to_filepath):
    tqdm.pandas()
    df = load_dataset(from_filepath)
    df = check_if_rent_is_in_text(df)
    model = get_mlp_model()
    df = get_rent_with_nlp(df.loc[df["rentInText"] == True], model)
    # df = check_accuracy(df)
    save_dataset(df, to_filepath)


def main():

    train = load_dataset()
    df = merge_df_from_file(train, "./train_with_nlp_prediction.csv")
    print(df.head())
    exit()

    predict_price_with_nlp_train_as_file(
        "./data/train.csv", "./data/train_with_nlp_prediction.csv"
    )
    predict_price_with_nlp_test_as_file(
        "./data/test.csv", "./data/test_with_nlp_prediction.csv"
    )

    # df = load_dataset()
    # df = df.head(10000)
    # # df = is_str_in_text(df, "euro")
    # df = check_if_rent_is_in_text(df)
    # # nlp_model()
    # model = get_mlp_model()
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
