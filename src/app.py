from flask import Flask,request,jsonify
import joblib
import pandas as pd

## Uso esto para simular una prediccion a traves de una API

app = Flask(__name__)

VARIABLES = ['Hours_Studied', 'Attendance', 'Parental_Involvement',
       'Access_to_Resources', 'Extracurricular_Activities', 'Sleep_Hours',
       'Previous_Scores', 'Motivation_Level', 'Internet_Access',
       'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality', 'School_Type',
       'Peer_Influence', 'Physical_Activity', 'Learning_Disabilities',
       'Parental_Education_Level', 'Distance_from_Home', 'Gender']


print("Cargando modelo...")
modelo = joblib.load("../models/GB_model.pkl")
preprocessor = joblib.load("../datos/processed/preprocessor.pkl")
print("¡Modelo listo!")


@app.route("/predecir", methods = ["POST"])
def predecir():
    
    datos_json = request.get_json()

    df = pd.DataFrame([datos_json]).reindex(columns=VARIABLES)

    df_processed = preprocessor.transform(df)

    prediccion = modelo.predict(df_processed)

    return jsonify({'prediccion': str(prediccion[0])})
    


print("Iniciando servidor...")
app.run(debug=True)
