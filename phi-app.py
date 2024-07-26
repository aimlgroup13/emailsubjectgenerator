from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM  # Adjust import as needed
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins; adjust for more restrictive policies if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer
model_name = "nagthgr8/subject-phi3"  # Change to your model path
model = AutoModelForCausalLM.from_pretrained(model_name)  # Use generic model class
tokenizer = AutoTokenizer.from_pretrained(model_name)

class InputData(BaseModel):
    email_body: str

# Serve the 'index.html' file when accessing the root URL
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict/")
async def predict(data: InputData):
    input_text = data.email_body  # Extract email body from request data
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"subject_line": generated_text}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
