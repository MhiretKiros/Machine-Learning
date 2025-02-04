from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS Middleware
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model, scaler, and encoders
rf_model = joblib.load('travel_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data model
class TravelData(BaseModel):
    Age: int
    Gender: str
    Budget: str
    Preferred_Climate: str
    Interest: str
    Travel_Duration: str
    Accommodation_Preference: str
    Transportation_Mode: str
    Activity_Level: str
    Food_Preference: str
    Travel_Type: str

# Prediction function
def predict_destination(data: dict):
    for col, le in label_encoders.items():
        if col in data:
            if data[col] in le.classes_:
                data[col] = le.transform([data[col]])[0]
            else:
                raise HTTPException(status_code=400, detail=f"Invalid value '{data[col]}' for feature '{col}'. Expected one of: {list(le.classes_)}")

    new_df = pd.DataFrame([data])
    new_df_scaled = scaler.transform(new_df)
    pred_encoded = rf_model.predict(new_df_scaled)[0]
    pred_destination = label_encoders['Destination'].inverse_transform([pred_encoded])[0]
    return pred_destination

# API endpoint
@app.post("/predict")
def predict(travel_data: TravelData):
    try:
        data_dict = travel_data.dict()
        prediction = predict_destination(data_dict)
        return {"predicted_destination": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run: uvicorn travel_api:app --reload
