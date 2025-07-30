import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

model = joblib.load("covid_diag.pkl")  # If it's in the same folder as main.py

class inp_data(BaseModel):
    Age: int
    Gender: int
    Fever: int
    Cough: int
    Fatigue: int  
    Breathlessness: int
    Comorbidity: int 
    Stage: int   
    Type: int
    Tumor_Size: float
app = FastAPI()
@app.get("/")
def root_msg():
    return {"Message": "Welcome to the Covid Diagnosis API"}
@app.post("/predict")
def predict(Data: inp_data):
    #inp=pd.DataFrame([data.dict()])
    inp=np.array([[Data.Age, Data.Gender, Data.Fever, Data.Cough, Data.Fatigue, Data.Breathlessness, Data.Comorbidity, Data.Stage, Data.Type, Data.Tumor_Size]])
    prd=model.predict(inp)[0]
    return {"Prediction": prd}
    
    
