from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  # Import BaseModel from pydantic
import os
import torch
import gpt2_qa_predict
import t5_qa_predict
os.environ['USE_XFORMERS'] = '0'
# Global model and tokenizer storage
models = {}
tokenizers = {}
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins; adjust for more restrictive policies if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    model_name: str
    question: str

class ModelRequest(BaseModel):
    model_name: str

@app.post("/predict/")
async def predict(data: InputData):
    model_name = data.model_name
    question = data.question
    print(model_name)
    # Check if the model is loaded
    if model_name == 'nagthgr8/gpt2-qa':
       return {"predicted_answer": gpt2_qa_predict.generate_answer(question)}
    else:
       return {"predicted_answer": t5_qa_predict.predict_answer(question)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
