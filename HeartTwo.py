from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json



app = FastAPI()
class model_input(BaseModel):
    male: int
    age: int
    currentSmoker: int
    BPMeds:int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: int
    glucose: int
    
    
heart_APIModel = pickle.load(open('heart_model_one.sav','rb'))


@app.post('/heart_prede') 
def heart_prede(input_parameters :model_input):
    input_data = input_parameters.json()
    input_dicrionary = json.loads(input_data)
    
    gend = input_dicrionary['male']
    age_of = input_dicrionary['age']
    somker = input_dicrionary['currentSmoker']
    bpeds = input_dicrionary['BPMeds']
    pstorke = input_dicrionary['prevalentStroke']
    phyp = input_dicrionary['prevalentHyp']
    dia = input_dicrionary['diabetes']
    chol = input_dicrionary['totChol']
    sBP = input_dicrionary['sysBP']
    dBP = input_dicrionary['diaBP']
    bmi = input_dicrionary['BMI']
    BPM = input_dicrionary['heartRate']
    glu = input_dicrionary['glucose']
    
    
    input_list =[gend, age_of, somker,bpeds,pstorke,phyp,dia, chol, sBP, dBP, bmi, BPM, glu]
    
    prediction = heart_APIModel.predict([input_list])
    
    if prediction[0] ==0:
        return 'No disease'
    else:
        return 'Disease'
