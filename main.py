from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

# Load scaler dan model
scaler = joblib.load("scaler.pkl")
model = joblib.load("xgb_best_model.pkl")

feature_names = scaler.feature_names_in_
region_columns = [feat for feat in feature_names if feat.startswith("region_")]

# Mapping
type_mapping = {"conventional": 0, "organic": 1}

season_mapping = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}


# Model input
class InputData(BaseModel):
    total_volume: float = Field(..., example=1000.0)
    f4046: float = Field(..., alias="_4046", example=300.0)
    f4225: float = Field(..., alias="_4225", example=400.0)
    f4770: float = Field(..., alias="_4770", example=300.0)
    total_bags: float = Field(..., example=50.0)
    small_bags: float = Field(..., example=30.0)
    large_bags: float = Field(..., example=15.0)
    xlarge_bags: float = Field(..., example=5.0)
    type: str = Field(..., example="organic")
    year: int = Field(..., example=2025)
    month: int = Field(..., example=4)
    season: str = Field(..., example="spring")
    region: str = Field(..., example="Boston")

    class Config:
        allow_population_by_field_name = True  # Supaya FastAPI bisa nerima field alias


app = FastAPI()


@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert string jadi angka
        type_val = type_mapping.get(data.type.lower())
        season_val = season_mapping.get(data.season.lower())

        if type_val is None:
            raise HTTPException(status_code=400, detail="Invalid type input.")
        if season_val is None:
            raise HTTPException(status_code=400, detail="Invalid season input.")

        # One-hot encoding untuk region
        region_vector = [0] * len(region_columns)
        region_input = f"region_{data.region.replace(' ', '').replace('-', '').lower()}"

        found = False
        for idx, region_col in enumerate(region_columns):
            region_clean = region_col.replace(" ", "").replace("-", "").lower()
            if region_input == region_clean:
                region_vector[idx] = 1
                found = True
                break

        if not found:
            raise HTTPException(
                status_code=400, detail=f"Region '{data.region}' not found."
            )

        input_features = [
            data.total_volume,
            data.f4046,
            data.f4225,
            data.f4770,
            data.total_bags,
            data.small_bags,
            data.large_bags,
            data.xlarge_bags,
            type_val,
            data.year,
            data.month,
            season_val,
        ] + region_vector

        # Check jumlah fitur
        if len(input_features) != len(feature_names):
            raise HTTPException(
                status_code=500,
                detail=f"Input feature mismatch: expected {len(feature_names)}, got {len(input_features)}.",
            )

        # Scaling + predict
        scaled_input = scaler.transform([input_features])
        prediction = model.predict(scaled_input)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
