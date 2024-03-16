import logging
import os
import pickle
import argparse
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class My_Classifier_Model:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None

    def train(self, dataset_filename):
        df = pd.read_csv(dataset_filename)
        df = df.drop(["Cabin", "Name"], axis=1)
        df["Total expenses"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"]
        df["Total expenses"] = df["Total expenses"] + df["VRDeck"]
        df = df.drop(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], axis=1)

        x = df["Total expenses"].mean()
        mask1 = df["Total expenses"] >= x
        df.loc[mask1, "Total expenses"] = 1
        mask2 = df["Total expenses"] != 1
        df.loc[mask2, "Total expenses"] = 0
        df = df.dropna()
        df = df.reset_index()
        df = df.drop("index", axis=1)

        df["Transported"] = df["Transported"].astype("int")
        df["CryoSleep"] = df["CryoSleep"].astype("int")
        df["VIP"] = df["VIP"].astype("int")
        df["Total expenses"] = df["Total expenses"].astype("int")
        df["HomePlanet"].value_counts()

        HomePlanet_is_Earth = df["HomePlanet"] == "Earth"
        HomePlanet_is_Europa = df["HomePlanet"] == "Europa"
        HomePlanet_is_Mars = df["HomePlanet"] == "Mars"
        df.loc[HomePlanet_is_Earth, "HomePlanet"] = 0
        df.loc[HomePlanet_is_Europa, "HomePlanet"] = 1
        df.loc[HomePlanet_is_Mars, "HomePlanet"] = 2

        df["Destination"].value_counts()
        Destination_is_TRAPPIST = df["Destination"] == "TRAPPIST-1e"
        Destination_is_Cancri = df["Destination"] == "55 Cancri e"
        Destination_is_PSO = df["Destination"] == "PSO J318.5-22"
        df.loc[Destination_is_TRAPPIST, "Destination"] = 0
        df.loc[Destination_is_Cancri, "Destination"] = 1
        df.loc[Destination_is_PSO, "Destination"] = 2
        df["HomePlanet"] = df["HomePlanet"].astype("int")
        df["Destination"] = df["Destination"].astype("int")
        df["Age"] = df["Age"].astype("int")

        y = df["Transported"]
        x = df.drop(["Transported", "PassengerId"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        self.model = LogisticRegression(random_state=0, max_iter=1000)
        self.model.fit(X_train, y_train)

        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        model_filepath = os.path.join(model_dir, 'model.pkl')
        with open(model_filepath, 'wb') as f:
            pickle.dump(self.model, f)

        self.logger.info('Model training completed.')
        self.logger.info(f'Training accuracy: {self.model.score(X_train, y_train)}')
        self.logger.info(f'Test accuracy: {self.model.score(X_test, y_test)}')

    def predict(self, dataset_filename, model_name):
        if model_name == 'model':
            model_filename = "model/model.pkl"
        elif model_name == 'tuned_model':
            model_filename = "model/tuned_model.pkl"
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        df = pd.read_csv(dataset_filename)
        test_ids = df["PassengerId"]

        df = df.drop(["Cabin", "Name", "PassengerId"], axis=1) if model_name == 'model' else df.drop(["Cabin", "Name"], axis=1)
        df["Total expenses"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"]
        df["Total expenses"] = df["Total expenses"] + df["VRDeck"]
        df = df.drop(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], axis=1)
        df["HomePlanet"] = df["HomePlanet"].map({"Earth": 0, "Europa": 1, "Mars": 2})
        df["Destination"] = df["Destination"].map({"TRAPPIST-1e": 0, "55 Cancri e": 1, "PSO J318.5-22": 2})

        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        submission_pred = model.predict(df)

        dfinal = pd.DataFrame({"PassengerId": test_ids.values, "Transported": submission_pred})
        dfinal["Transported"] = dfinal["Transported"].astype("bool")

        results_dir = './data'
        os.makedirs(results_dir, exist_ok=True)
        results_filepath = os.path.join(results_dir, 'results.csv')
        dfinal.to_csv(results_filepath, index=False)

        self.logger.info(f'Прогноз сохранен в {results_filepath}')


def tune_hyperparameters(dataset_filename):
    df = pd.read_csv(dataset_filename)

    # Подготовка данных для обучения
    df = df.drop(["Cabin", "Name", "PassengerId"], axis=1)
    df["Total expenses"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"]
    df["Total expenses"] = df["Total expenses"] + df["VRDeck"]
    df = df.drop(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], axis=1)
    df["HomePlanet"] = df["HomePlanet"].map({"Earth": 0, "Europa": 1, "Mars": 2})
    df["Destination"] = df["Destination"].map({"TRAPPIST-1e": 0, "55 Cancri e": 1, "PSO J318.5-22": 2})

    imputer = SimpleImputer(strategy='mean')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    X = df.drop(["Transported"], axis=1)
    y = df["Transported"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "C": trial.suggest_loguniform("C", 0.001, 1000),
            "max_iter": trial.suggest_int("max_iter", 100, 1000),
        }
        model = LogisticRegression(random_state=0, **params)

        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, pred)

        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    best_params = study.best_trial.params

    model = LogisticRegression(random_state=0, **best_params)
    model.fit(X_train, y_train)

    model_dir = './model'
    os.makedirs(model_dir, exist_ok=True)
    model_filepath = os.path.join(model_dir, 'tuned_model.pkl')
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Best hyperparameters found: {best_params}")
    print(f"Model saved to {model_filepath}")


def main():
    parser = argparse.ArgumentParser(description='Model training and prediction')
    parser.add_argument('mode', choices=['train', 'predict','tune'], help='Mode: train, tune or predict')
    parser.add_argument('--dataset', required=True, help='Path to dataset file')
    parser.add_argument('--model',choices=['tuned_model','model'], required=True, help='Name of the model')
    args = parser.parse_args()

    model = My_Classifier_Model()

    if args.mode == 'train':
        model.train(args.dataset)
    elif args.mode == 'predict':
        model.predict(args.dataset, args.model)
    elif args.mode == 'tune':
        tune_hyperparameters(args.dataset)


if __name__ == '__main__':
    main()
