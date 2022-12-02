import pandas as pd
import numpy as np
from math import sqrt


def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename)


def calculate_city_centers(df):
    cities = df["city"].unique()
    city_centers = {}
    for city in cities:
        latitude = df.loc[df["city"] == city]["latitude"].mean()
        longitude = df.loc[df["city"] == city]["longitude"].mean()
        city_centers[city] = (latitude, longitude)
    return city_centers


def get_distance_to_city_center(df, city_centers):
    df["distToCity"] = df.apply(
        lambda row: calculate_distance(
            (row.latitude, row.longitude), city_centers[row.city]
        ),
        axis=1,
    )
    return df


def calculate_distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    # return np.linalg.norm(a - b)


def main():
    df = load_dataset()
    city_centers = calculate_city_centers(df)
    df = get_distance_to_city_center(df, city_centers)
    print(df.head)


if __name__ == "__main__":
    main()
