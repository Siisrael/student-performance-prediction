from src.data import cargar_data, validation,data_cleaning
from src.preprocessing import split_data,preprocess_features
from src.train import train_model
from src.evaluate import evaluate_model


from pathlib import Path
import joblib


processed_dir = Path('datos/processed')
RAW_DATA_PATH = Path("datos/StudentPerformanceFactors.csv")
CLEAN_DATA_PATH = Path("datos/interim/clean_data.csv")


def main():

    TARGET = "Exam_Score"
    
    COLUMNAS_PARA_LOG = ["Hours_Studied", "Tutoring_Sessions", "Physical_Activity"]

    df = cargar_data(RAW_DATA_PATH)
    
    validation(RAW_DATA_PATH)


    ## Limpio el dataset basandome en el notebook y guardo el nuevo dataset como clean_data.csv

    df_clean = data_cleaning(df)

    validation(CLEAN_DATA_PATH)
  
    X_train, X_test, y_train, y_test = split_data(df_clean, TARGET, test_size=0.2, random_state=42, stratify=True)


    X_train_processed, X_test_processed, preprocessor = preprocess_features(X_train,X_test,COLUMNAS_PARA_LOG,numerical_strategy='standard', categorical_strategy='onehot')

    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_train_processed, processed_dir / 'X_train.pkl')
    joblib.dump(X_test_processed, processed_dir / 'X_test.pkl')
    joblib.dump(y_train, processed_dir / 'y_train.pkl')
    joblib.dump(y_test, processed_dir / 'y_test.pkl')
    joblib.dump(preprocessor, processed_dir / 'preprocessor.pkl')


    print(f"OK: Datos procesados guardados")


    model = train_model(X_train_processed, y_train)

    joblib.dump(model, 'models/GB_model.pkl')

    print("OK: Modelo guardado")

    metrics = evaluate_model(
        model=model,  
        X_test=X_test_processed,
        y_test=y_test,
    )

    print(f"\nMétricas finales: {metrics}")

if __name__ == "__main__":
    main()