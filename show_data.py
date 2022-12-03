
from data_loader import load_dataset
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="ticks", color_codes=True)

def show_data(df):
    g = sns.pairplot(df)
    plt.show()

if __name__ == '__main__':
    df = load_dataset("./data/preprocessed_data.csv")
    show_data(df)
