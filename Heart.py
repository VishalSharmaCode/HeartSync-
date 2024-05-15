from pydantic import BaseModel

class Heart(BaseModel):
    male: int
    age: float
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float