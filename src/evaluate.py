import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from pathlib import Path


def evaluate_model(model, X_test, y_test):



    y_pred = model.predict(X_test)

    metrics = {
        "R2" : r2_score(y_test, y_pred),
        "MAE" : mean_absolute_error(y_test,y_pred),
        "RMSE" : np.sqrt(mean_squared_error(y_test, y_pred))
    }

    print("----Metricas en test set del modelo--------")
    print(f"R2 score : {metrics["R2"]:.3f}")
    print(f"MAE : {metrics["MAE"]:.3f}")
    print(f"RMSE : {metrics["RMSE"]:.3f}")


    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Métricas guardadas en {'metrics.json'}")

    return metrics