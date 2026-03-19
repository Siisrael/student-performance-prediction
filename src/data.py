import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def cargar_data(path):

    df = pd.read_csv(path)

    print("OK: Dataset cargado.")
    return df


def validation(path):

    df = pd.read_csv(path)
    

    schema = pa.DataFrameSchema({
    "Hours_Studied": Column(int, unique=False, nullable=True),
    "Attendance": Column(int, Check.in_range(0, 100)),  # porcetaje de asistencia
    "Sleep_Hours": Column(int, Check.in_range(0, 24)),  
    "Previous_Scores" : Column(int, Check.in_range(0,100)),
    "Tutoring_Sessions" : Column(int, unique=False, nullable= True),
    "Physical_Activity" : Column(int, unique= False, nullable=True),
    "Exam_Score" : Column(int, Check.in_range(0,101),nullable=True),
    "Parental_Involvement": Column(str, Check.isin(['Low', 'Medium', 'High']),coerce=True),
    "Access_to_Resources": Column(str, Check.isin(['Low', 'Medium', 'High']),coerce=True),
    "Motivation_Level": Column(str, Check.isin(['Low' ,'Medium', 'High']),coerce=True),
    "Family_Income": Column(str, Check.isin(['Low' ,'Medium' ,'High']),coerce=True),
    "Teacher_Quality" : Column(str, Check.isin(['Low', 'Medium' ,'High']),coerce=True,nullable=True),
    "Extracurricular_Activities" : Column(str, Check.isin(['No' ,'Yes']),coerce=True),
    "Internet_Access" : Column(str, Check.isin(['No', 'Yes']),coerce=True),
    "School_Type" : Column(str, Check.isin(['Public' ,'Private']),coerce=True),
    "Peer_Influence" : Column(str, Check.isin(['Positive', 'Negative', 'Neutral']),coerce=True),
    "Learning_Disabilities" : Column(str, Check.isin(['No' ,'Yes']),coerce=True),
    "Parental_Education_Level" : Column(str, Check.isin(['High School' ,'College', 'Postgraduate']),coerce=True,nullable=True),
    "Distance_from_Home" : Column(str, Check.isin(['Near', 'Moderate' ,'Far']),coerce=True,nullable=True),
    "Gender" : Column(str, Check.isin(['Male' ,'Female']),coerce=True),
    
    })
    report = {
        'is_valid': True,
        'issues': [],
        'missing_summary': {},
        'duplicates': 0
    }
    strict_mode=False
    try:
        print("Validando datos...")
        schema.validate(df, lazy=True)


    except pa.errors.SchemaErrors as e:
        report['is_valid'] = False
        report['issues'].append("Errores de schema encontrados")
        
        print(f"\n⚠️ Errores de validación encontrados:")
        print(e.failure_cases)
        
        if strict_mode:
            raise e  
    
    
    # NaN
    missing_count = df.isna().sum()
    missing_with_values = missing_count[missing_count > 0]
    report['missing_summary'] = missing_with_values.to_dict()
    
    if len(missing_with_values) > 0:
        print(f"\n📊 NaN encontrados:")
        for col, count in missing_with_values.items():
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.1f}%)")
    else:
        print("\n✓ Sin NaN")
    
    # Duplicados
    duplicates = df.duplicated().sum()
    report['duplicates'] = duplicates
    
    if duplicates > 0:
        print(f"\n {duplicates} filas duplicadas encontradas")
        report['issues'].append(f"{duplicates} duplicados")
        
    else:
        print("✓ Sin duplicados")
    
    # Resumen
    print(f"\n{'='*60}")
    if report['is_valid']:
        print("✅ VALIDACIÓN EXITOSA")
    else:
        print("❌ VALIDACIÓN FALLÓ")
        print(f"Problemas encontrados: {len(report['issues'])}")
        for issue in report['issues']:
            print(f"  - {issue}")
    
    print(f"{'='*60}\n")

    print("OK: Datos validados.")
    
    return report

    


def data_cleaning(data):


    print("Iniciando limpieza de datos....")

    data_clean = data.copy()

    for col in data.columns:

        missing_pct = (data_clean[col].isna().sum() / len(data)) * 100

        if missing_pct > 70:
            print("EL porcenaje de NaN es mayor al 70%, procedo a dropearlos")

            data_clean = data_clean[col].dropna(columns=[col])
            

        if missing_pct > 5:

            data_clean[f"{col}_was_missing"] = data_clean[col].isna().astype(int)

            print("Remplazo por categoria is missing")

        if data_clean[col].dtype in ['int64', 'float64']:
                # Numérica: mediana por la presencia de outliers

            data_clean[col].fillna(data_clean[col].median(), inplace=True)
            print("Remplazo por mediana")

        else:
        # Categórica: moda o "Unknown"
            if data_clean[col].mode().empty:
                data_clean[col].fillna('Unknown')
            else:
                data_clean[col].fillna(data_clean[col].mode()[0], inplace=True)

    data_clean.to_csv(BASE_DIR / "datos" / "interim" / "clean_data.csv", index=False)

    print("OK: Data cleaning terminado")
    
    return data_clean