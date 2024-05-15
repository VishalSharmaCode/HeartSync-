import uvicorn
from fastapi import FastAPI
from Heart import Heart
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("vajra.pkl","rb")
rf = pickle.load(pickle_in)

@app.get('/')
def index():
    return{'massage': 'Hello, Vishal'}

@app.post('/predict')
def predict_heart(data:Heart):
    data = data.dict()
    print(data)
    print("hello")
    
    
    male: data['male']
    age: data['age']
    diabetes: data['diabetes']
    totChol: data['totChol']
    sysBP: data['sysBP']
    diaBP: data['diaBP']
    BMI: data['BMI']
    heartRate: data['heartRate']
    glucose: data['glucose']
    prediction = rf.predict([[male, age, diabetes, totChol,sysBP, diaBP, BMI, heartRate, glucose]])
    if(prediction[0]>0.5):
        prediction = 'Chance of heart diseas'
    else:
        prediction = 'No Heart disease'
    return{
        'prediction': prediction
        }
if __name__ =='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)