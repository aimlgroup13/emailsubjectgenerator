from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  # Import BaseModel from pydantic
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
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
# Configuration parameters
model_dir = "nagthgr8/subject-mistrel"
tokenizer_file = f"{model_dir}/tokenizer.model"
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# Load the model and tokenizer
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_dir,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
model = model.to("cuda")

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
    # Format the input for the model
    formatted_input = f"User: {data.email_body}\nAssistant:"
    inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True).to("cuda")  # Move inputs to GPU

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False  # Ensure cache usage is enabled
        )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_subject": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
