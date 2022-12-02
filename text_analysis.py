import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import re


def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename)


def get_rent_from_text(model, text):
    prompt = {
        "question": "What is the rent?",
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
    df["rentFromNLP"] = df.apply(
        lambda row: strip_rent_text(
            get_rent_from_text(model, row.descriptionNonTranslated)
        ),
        axis=1,
    )
    print(df["rentFromNLP"].value_counts())
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
    result = re.sub(r"[^0-9]", "", rent_text)
    return result


def check_accuracy(df):
    df["predictionCorrect"] = df.apply(
        lambda row: str(row.rentFromNLP) == str(row.rent), axis=1
    )
    print(df["predictionCorrect"].value_counts())
    return df


def get_mlp_model():
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return nlp


# def check_accuracy(df):


def main():
    df = load_dataset()
    df = df.head(1000)
    # df = is_str_in_text(df, "euro")
    df = check_if_rent_is_in_text(df)
    # nlp_model()
    model = get_mlp_model()
    df = get_rent_with_nlp(df.loc[df["rentInText"] == True], model)
    df = check_accuracy(df)
    print(df[["rent", "rentFromNLP", "predictionCorrect"]])


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
