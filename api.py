from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
import uuid
import numpy as np
from models.feedforward import FeedforwardNeuralNetwork
from models.convolutional import ConvolutionalNeuralNetwork
from models.recurrent import RecurrentNeuralNetwork
from utils.visualization import visualize_network, plot_training_loss
import os

app = FastAPI()

API_KEY_NAME = "api_key"
API_KEY = os.getenv("API_KEY", "your-secure-api-key")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Depends(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Could not validate credentials"
        )

ModelType = Literal["Feedforward", "Convolutional", "Recurrent"]

class CreateModelRequest(BaseModel):
    model_type: ModelType = Field(...)
    layers: Optional[List[int]] = None  # For Feedforward
    input_size: Optional[int] = None    # For Recurrent
    output_size: Optional[int] = None   # For Recurrent

    class Config:
        protected_namespaces = ()

class TrainModelRequest(BaseModel):
    model_id: str = Field(...)
    inputs: List[List[float]]
    targets: List[List[float]]
    learning_rate: float = 0.1
    epochs: int = 100

    class Config:
        protected_namespaces = ()

class ModelResponse(BaseModel):
    model_id: str = Field(...)
    model_type: ModelType

    class Config:
        protected_namespaces = ()

class TrainingResponse(BaseModel):
    loss_history: List[float]

    class Config:
        protected_namespaces = ()

class SaveLoadModelRequest(BaseModel):
    model_id: str = Field(...)
    file_path: str

    class Config:
        protected_namespaces = ()

# In-memory database
models: Dict[str, Any] = {}

@app.post("/models/", response_model=ModelResponse, dependencies=[Depends(get_api_key)])
def create_model(request: CreateModelRequest):
    model_id = str(uuid.uuid4())
    if request.model_type == "Feedforward":
        model = FeedforwardNeuralNetwork(request.layers)
    elif request.model_type == "Convolutional":
        model = ConvolutionalNeuralNetwork()
    elif request.model_type == "Recurrent":
        if not request.input_size or not request.output_size:
            raise HTTPException(status_code=400, detail="Input and output sizes are required for recurrent models")
        model = RecurrentNeuralNetwork(request.input_size, request.output_size)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    models[model_id] = model
    return ModelResponse(model_id=model_id, model_type=request.model_type)

@app.post("/models/train/", response_model=TrainingResponse, dependencies=[Depends(get_api_key)])
def train_model(request: TrainModelRequest):
    model = models.get(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    inputs = np.array(request.inputs)
    targets = np.array(request.targets)

    loss_history = model.train(inputs, targets, request.learning_rate, request.epochs)
    
    return TrainingResponse(loss_history=loss_history)

@app.get("/models/", response_model=List[ModelResponse], dependencies=[Depends(get_api_key)])
def list_models():
    return [ModelResponse(model_id=model_id, model_type=model.__class__.__name__) for model_id, model in models.items()]

@app.delete("/models/{model_id}", dependencies=[Depends(get_api_key)])
def delete_model(model_id: str):
    if model_id in models:
        del models[model_id]
        return {"detail": "Model deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Model not found")

@app.post("/models/save/", dependencies=[Depends(get_api_key)])
def save_model(request: SaveLoadModelRequest):
    model = models.get(request.model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model.save_model(request.file_path)
    return {"detail": "Model saved successfully"}

@app.post("/models/load/", dependencies=[Depends(get_api_key)])
def load_model(request: SaveLoadModelRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine the model type from the file path or metadata
    # Assuming the model type is part of the file name for simplicity
    model_type = None
    if "feedforward" in request.file_path:
        model_type = "Feedforward"
        model = FeedforwardNeuralNetwork([1, 1, 1])  # Placeholder sizes
    elif "convolutional" in request.file_path:
        model_type = "Convolutional"
        model = ConvolutionalNeuralNetwork()
    elif "recurrent" in request.file_path:
        model_type = "Recurrent"
        model = RecurrentNeuralNetwork(1, 1)  # Placeholder sizes
    
    if not model_type:
        raise HTTPException(status_code=400, detail="Invalid model type in file path")
    
    model.load_model(request.file_path)
    model_id = str(uuid.uuid4())
    models[model_id] = model
    return ModelResponse(model_id=model_id, model_type=model_type)

@app.get("/models/visualize/{model_id}", dependencies=[Depends(get_api_key)])
def visualize_model(model_id: str):
    model = models.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    visualize_network(model)
    return {"detail": "Model visualization created"}

@app.get("/models/plot_loss/{model_id}", dependencies=[Depends(get_api_key)])
def plot_loss(model_id: str):
    model = models.get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if hasattr(model, 'loss_history') and model.loss_history:
        plot_training_loss(model.loss_history)
        return {"detail": "Training loss plot created"}
    else:
        raise HTTPException(status_code=400, detail="No training loss history available")
