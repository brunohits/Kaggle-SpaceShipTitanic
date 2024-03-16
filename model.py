import logging
import os
import pickle
import argparse
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logging.basicConfig(filename='./data/log_file.log', level=logging.INFO)
class My_Classifier_Model:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None

    def train(self, dataset_filename):
        df = pd.read_csv(dataset_filename)
        df.HomePlanet.value_counts(dropna=False)
        df.CryoSleep.value_counts(dropna=False)
        df.VIP.value_counts(dropna=False)
        passenger_attributes = df.drop(columns=["Transported"])  # X
        passenger_labels = df["Transported"]  # y
        passenger_attributes.drop(columns=["Name", "Cabin", "PassengerId"], inplace=True)
        passenger_attributes = pd.get_dummies(passenger_attributes, drop_first=True)
        passenger_attributes = passenger_attributes.dropna()
        passenger_labels = passenger_labels[passenger_attributes.index]

        # Initialize Logistic Regression models
        model = LogisticRegression()

        # Train the models
        model.fit(passenger_attributes, passenger_labels)

        # Evaluate model performance
        score = model.score(passenger_attributes, passenger_labels)  # score on test set
        # Print scores
        print("Score :", score)

        # Save the trained models
        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        model_filepath = os.path.join(model_dir, 'model.pkl')
        with open(model_filepath, 'wb') as f:
            pickle.dump(model, f)

        self.logger.info('Model training completed.')

    def predict(self, dataset_filename, model_name):
        self.logger.info('Prediction started.')
        if model_name == 'model':
            model_filepath = "model/model.pkl"
        elif model_name == 'tuned_model':
            model_filepath = "model/tuned_model.pkl"

        with open(model_filepath, 'rb') as f:
            model = pickle.load(f)

        df_test = pd.read_csv(dataset_filename)
        df_test = df_test.set_index("PassengerId")
        df_test.drop(columns=["Name", "Cabin"], inplace=True)
        df_test = pd.get_dummies(df_test, drop_first=True)
        df_test = df_test.fillna(0)

        # Load train data to get true labels
        df_train = pd.read_csv("data/train.csv")
        df_train = df_train.set_index("PassengerId")
        true_labels = df_train["Transported"]

        predictions = model.predict(df_test)
        # accuracy = accuracy_score(true_labels, predictions)
        # print("Accuracy:", accuracy)  Uncomment to get accurancy

        df_test["Transported"] = predictions
        submission = df_test[["Transported"]]

        # Save submission.csv in the data directory
        results_dir = './data'
        os.makedirs(results_dir, exist_ok=True)
        result_filepath = os.path.join(results_dir, 'result.csv')
        submission.to_csv(result_filepath)

        self.logger.info(f'Prediction saved in ./data/result.csv')

    def tune_hyperparameters(self, dataset_filename):
        self.logger.info('Hyperparameter tuning started.')
        df = pd.read_csv(dataset_filename)
        df.HomePlanet.value_counts(dropna=False)
        df.CryoSleep.value_counts(dropna=False)
        df.VIP.value_counts(dropna=False)
        passenger_attributes = df.drop(columns=["Transported"])  # X
        passenger_labels = df["Transported"]  # y
        passenger_attributes.drop(columns=["Name", "Cabin", "PassengerId"], inplace=True)
        passenger_attributes = pd.get_dummies(passenger_attributes, drop_first=True)
        passenger_attributes = passenger_attributes.dropna()
        passenger_labels = passenger_labels[passenger_attributes.index]

        def objective(trial):
            # Define space of hyperparameters to search
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.5),
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 100)
            }

            # Initialize CatBoostClassifier model
            model = CatBoostClassifier(**params)

            # Split dataset
            X_train, X_valid, y_train, y_valid = train_test_split(passenger_attributes, passenger_labels, test_size=0.2)

            # Train the model
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)

            # Evaluate model
            score = model.score(X_valid, y_valid)

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=8)

        best_params = study.best_params

        # Initialize and train the model with the best hyperparameters
        tuned_model = CatBoostClassifier(**best_params)
        tuned_model.fit(passenger_attributes, passenger_labels, verbose=False)

        # Save the trained model
        model_dir = './model'
        os.makedirs(model_dir, exist_ok=True)
        model_filepath = os.path.join(model_dir, 'tuned_model.pkl')
        with open(model_filepath, 'wb') as f:
            pickle.dump(tuned_model, f)

        logging.info('Hyperparameter tuning completed.')


def main():
    parser = argparse.ArgumentParser(description='Model training and prediction')
    parser.add_argument('mode', choices=['train', 'predict', 'tune'], help='Mode: train, tune or predict')
    parser.add_argument('--dataset', required=True, help='Path to dataset file')
    parser.add_argument('--model', choices=['tuned_model', 'model'], required=False, help='Name of the model')
    args = parser.parse_args()

    model = My_Classifier_Model()

    if args.mode == 'train':
        model.train(args.dataset)
    elif args.mode == 'predict':
        model.predict(args.dataset, args.model)
    elif args.mode == 'tune':
        model.tune_hyperparameters(args.dataset)


if __name__ == '__main__':
    main()
