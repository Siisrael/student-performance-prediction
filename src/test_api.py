import requests

url = 'http://127.0.0.1:5000/predecir'


payload = {
    'Hours_Studied': 25,
    'Attendance': 90,
    'Parental_Involvement': 'High',     
    'Access_to_Resources': 'High',      
    'Extracurricular_Activities': 'Yes', 
    'Sleep_Hours': 8,
    'Previous_Scores': 85,
    'Motivation_Level': 'High',         
    'Internet_Access': 'Yes',
    'Tutoring_Sessions': 2,
    'Family_Income': 'Middle',          
    'Teacher_Quality': 'High',          
    'School_Type': 'Public',            
    'Peer_Influence': 'Positive',       
    'Physical_Activity': 4,
    'Learning_Disabilities': 'No',
    'Parental_Education_Level': 'High School', 
    'Distance_from_Home': 'Near',       
    'Gender': 'Male'                    
}

print(f"Enviando datos a {url}...")
print(f"Datos: {payload}")

try:
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        json_respuesta = response.json()
        print("\n✅ ¡ÉXITO!")
        print(f"La predicción es: {json_respuesta['prediccion']}")
    else:
        print("\n❌ HUBO UN PROBLEMA")
        print(f"Código de error: {response.status_code}")
        print(f"Mensaje del servidor: {response.text}")

except Exception as e:
    print(f"\n❌ Error de conexión: {e}")
    print("¿Está corriendo app.py en la otra terminal?")