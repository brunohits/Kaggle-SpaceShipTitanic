import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2, random_state=42)
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations",100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate",0.001,1.0),
        "depth": trial.suggest_int("depth",1,10)
    }
    model = CatBoostClassifier(**params)

    model.fit(X_train,y_train,verbose=False)

    pred = model.predict(X_val)

    accuracy = accuracy_score(y_val,pred)

    return accuracy


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=100,verbose=False)

best_params = study.best_trial.params

model = CatBoostClassifier(
    iterations = best_params["iterations"],
    learning_rate=best_params["learning_rate"],
    depth = best_params["depth"]
)