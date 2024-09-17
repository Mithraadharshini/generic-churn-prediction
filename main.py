from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split

from .model import train_model, evaluate_model
from .data_utils import load_and_preprocess_data

app = FastAPI()

class DataInput(BaseModel):
    data: dict
    dataset: str

# Global model and preprocessor storage
models = {}
preprocessors = {}

@app.on_event("startup")
async def startup_event():
    # Update this list with paths to your datasets
    datasets = [
        'data/Bank_churn.csv',
        'data/BankChurners.csv',
        'data/churn-bigml-80.csv',
        'data/orange_telecom.csv'
    ]
    
    global models
    global preprocessors

    models = {}
    preprocessors = {}

    for dataset in datasets:
        X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(dataset)
        model, _, _ = train_model(X_train, y_train)
        models[dataset] = model
        preprocessors[dataset] = preprocessor
        evaluate_model(model, X_test, y_test)

@app.post("/predict/")
def predict(data: DataInput):
    try:
        df = pd.DataFrame([data.data])
        preprocessor = preprocessors.get(data.dataset)
        if preprocessor is None:
            raise HTTPException(status_code=404, detail="Preprocessor not found for dataset")
        
        X = preprocessor.transform(df)
        model = models.get(data.dataset)
        if model is None:
            raise HTTPException(status_code=404, detail="Model not found for dataset")
        
        prediction = model.predict(X)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/global/")
def explain_global_view(dataset: str):
    try:
        model = models.get(dataset)
        preprocessor = preprocessors.get(dataset)
        if model is None or preprocessor is None:
            raise HTTPException(status_code=404, detail="Dataset or model not found")
        
        # Example: Explain for a sample
        df = pd.read_csv(dataset)
        X, _, _ = load_and_preprocess_data(dataset)
        explanation = explain_global(model, X)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/local/")
def explain_local_view(dataset: str, instance_index: int):
    try:
        model = models.get(dataset)
        preprocessor = preprocessors.get(dataset)
        if model is None or preprocessor is None:
            raise HTTPException(status_code=404, detail="Dataset or model not found")
        
        df = pd.read_csv(dataset)
        X, _, _ = load_and_preprocess_data(dataset)
        instance = X[instance_index]
        explanation = explain_local(model, X, instance)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
