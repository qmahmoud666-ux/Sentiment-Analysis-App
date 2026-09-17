from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(title="Sentiment Analysis API")

# تحميل النماذج والملفات
nlp_model = joblib.load('nlp_sentiment_model.pkl')
feature_model = joblib.load('feature_sentiment_model.pkl')
scaler = joblib.load('scaler.pkl')
num_cols = joblib.load('num_columns_list.pkl')

label_map = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}

# هيكل البيانات المستقبلة للنصوص
class TextRequest(BaseModel):
    text: str

# هيكل البيانات المستقبلة للـ Features
class FeaturesRequest(BaseModel):
    features: List[float]

@app.get("/get-columns")
def get_columns():
    return {"num_cols": num_cols}

@app.post("/predict-text")
def predict_text(data: TextRequest):
    pred = nlp_model.predict([data.text])[0]
    probs = nlp_model.predict_proba([data.text])[0].tolist()
    classes = [label_map.get(c, str(c)) for c in nlp_model.classes_]
    
    return {
        "prediction": label_map.get(pred, str(pred)),
        "raw_pred": int(pred) if isinstance(pred, (int, np.integer)) else str(pred),
        "confidence": dict(zip(classes, probs))
    }

@app.post("/predict-features")
def predict_features(data: FeaturesRequest):
    scaled_feats = scaler.transform([data.features])
    pred = feature_model.predict(scaled_feats)[0]
    probs = feature_model.predict_proba(scaled_feats)[0].tolist()
    classes = [label_map.get(c, str(c)) for c in feature_model.classes_]
    
    return {
        "prediction": label_map.get(pred, str(pred)),
        "raw_pred": int(pred) if isinstance(pred, (int, np.integer)) else str(pred),
        "confidence": dict(zip(classes, probs))
    }