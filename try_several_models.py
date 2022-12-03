from data_loader import load_dataset
from pca import apply_pca

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from logger import log

def train_and_run_model(train_data, train_label, test_data, test_label, model):
    log.info(f"Running {model.__class__.__name__}")

    log.info("fitting model...")
    model.fit(train_data, train_label)

    log.info("testing model...")
    y_pred = model.predict(test_data)
    average_error = mean_absolute_error(y_pred, test_label)

    acc = model.score(test_data, test_label)

    print(f"{model.__class__.__name__} accuracy: {acc}")
    print(f"Mean Absolute Error: {average_error}")

def apply_models(df):
    models = [LogisticRegression(solver = 'lbfgs'), KNeighborsRegressor(), MLPRegressor()]
    train_data, test_data, train_label, test_label = train_test_split( df.drop('rent', axis=1), df['rent'], test_size=0.2, random_state=42)

    for model in reversed(models):
        train_and_run_model(train_data, train_label, test_data, test_label, model)

if __name__ == '__main__':
    df = load_dataset("./data/preprocessed_data.csv")
    # df = apply_pca(df)
    df = df.select_dtypes(exclude=['object'])
    apply_models(df)
