
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
import joblib
from pathlib import Path



def split_data(df, target_col, test_size=0.2, random_state=42, stratify=True):
    """
    Divide los datos en conjuntos de entrenamiento y prueba
    
    Args:
        df: DataFrame con los datos limpios
        target_col: Nombre de la columna objetivo
        test_size: Proporción del conjunto de prueba
        random_state: Semilla para reproducibilidad
        stratify: Si True, hace split estratificado (útil para clasificación)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Determinar si hacer split estratificado
    stratify_param = y if stratify and _is_classification(y) else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"Train set: {X_train.shape[0]} muestras")
    print(f"Test set: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test


def identify_column_types(df):
    """
    Identifica automáticamente qué columnas son numéricas y categóricas
    
    Args:
        df: DataFrame
    
    Returns:
        dict con 'numerical' y 'categorical' columns
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


def create_preprocessor(X_train, cols_to_log, numerical_strategy='standard', categorical_strategy='onehot'):
    """
    Crea un pipeline de preprocesamiento con ColumnTransformer
    """
    
  
    column_types = identify_column_types(X_train) 
    all_numeric = column_types['numerical']
    categorical_cols = column_types['categorical']
    
    ### Separo las columnas numericas en columnas a las q les debo aplicarles la tranformacion log con las q no
    numeric_normal_cols = [c for c in all_numeric if c not in cols_to_log]
    
    transformers = []

    
    if cols_to_log:
        log_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('power', PowerTransformer(method='yeo-johnson', standardize=True))
        ])
        transformers.append(('log', log_transformer, cols_to_log))
        

    if numeric_normal_cols and numerical_strategy != 'none':
        steps = [('imputer', SimpleImputer(strategy='median'))]
        
        if numerical_strategy == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif numerical_strategy == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
            
        num_pipeline = Pipeline(steps)
        transformers.append(('num', num_pipeline, numeric_normal_cols))
    
   
    if categorical_cols and categorical_strategy != 'none':
        steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
        
        if categorical_strategy == 'onehot':
            steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        elif categorical_strategy == 'label':
            
            steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            
        cat_pipeline = Pipeline(steps)
        transformers.append(('cat', cat_pipeline, categorical_cols))

    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
        verbose_feature_names_out=False 
    ).set_output(transform="pandas") 
    
    return preprocessor


def preprocess_features(X_train, X_test,COLUMNAS_PARA_LOG, numerical_strategy='standard', 
                       categorical_strategy='onehot'):
    """
    Aplica preprocesamiento a los datos de train y test
    IMPORTANTE: Fit solo en train, transform en ambos
    
    Args:
        X_train: DataFrame de entrenamiento
        X_test: DataFrame de prueba
        numerical_strategy: Estrategia para columnas numéricas
        categorical_strategy: Estrategia para columnas categóricas
    
    Returns:
        X_train_processed, X_test_processed, preprocessor
    """
    # Crea preprocessor
    preprocessor = create_preprocessor(X_train,COLUMNAS_PARA_LOG, numerical_strategy, categorical_strategy)
    
    # FIT solo con datos de entrenamiento
    print("Ajustando preprocessor con datos de entrenamiento...")
    preprocessor.fit(X_train)
    
    # TRANSFORM en ambos conjuntos
    print("Transformando datos de entrenamiento...")
    X_train_processed = preprocessor.transform(X_train)
    
    print("Transformando datos de prueba...")
    X_test_processed = preprocessor.transform(X_test)
    
    # Convertir a DataFrame
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    
    return X_train_processed, X_test_processed, preprocessor


def encode_target(y_train, y_test, task_type='classification'):
    """
    Codifica la variable objetivo si es necesario (clasificación)
    
    Args:
        y_train: Serie/array con target de entrenamiento
        y_test: Serie/array con target de prueba
        task_type: 'classification' o 'regression'
    
    Returns:
        y_train_encoded, y_test_encoded, encoder (o None si es regresión)
    """
    if task_type == 'regression':
        return y_train, y_test, None
    
    # Para clasificación con labels categóricos
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        
        print(f"Target codificado: {dict(enumerate(encoder.classes_))}")
        return y_train_encoded, y_test_encoded, encoder
    
    return y_train, y_test, None



def _is_classification(y):
    """Determina si es un problema de clasificación"""
    return y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() < 20