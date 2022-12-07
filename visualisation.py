import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load a simple dataset as a pandas DataFrame
def load_dataset(filename="./data/train.csv"):
    return pd.read_csv(filename, index_col="id")


def main():
    example()
    exit()
    df = load_dataset()

    alt.Chart(df).mark_point().encode(
        x="area_sqm",
        y="rent",
        color="city",
    ).interactive()

    # plt.show()


def example():
    x = np.arange(100)
    source = pd.DataFrame({"x": x, "f(x)": np.sin(x / 5)})

    chart = alt.Chart(source).mark_line().encode(x="x", y="f(x)")
    chart.display()


if __name__ == "__main__":
    main()
