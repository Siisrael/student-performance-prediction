from sklearn.ensemble import GradientBoostingRegressor
import joblib



def train_model(X_train, y_train, params = None):

    if params is None:

        params= {
        "learning_rate": 0.1,
        "max_depth" : 3,
        "min_samples_split" : 30,
        "n_estimators" : 200
        }


    print("Entrenando modelo")

    model = GradientBoostingRegressor(**params)

    model.fit(X_train,y_train)


    print("OK: Modelo entrenado")


    return model

