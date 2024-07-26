from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load model and tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "/home/ramch/AI-AUTOMATED-QA/saved-gpt2model"

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load the model
model = GPT2LMHeadModel.from_pretrained(model_path)

# Move the model to GPU if available
if torch.cuda.is_available():
    model.to('cuda')

@app.get("/", response_class=HTMLResponse)
def get_form():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Prediction</title>
    </head>
    <body>
        <h1>Submit Text for Prediction</h1>
        <form action="/predict" method="post">
            <textarea name="input_text" rows="4" cols="50" placeholder="Enter your text here..."></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(input_text: str = Form(...)):
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    outputs = model.generate(inputs['input_ids'], max_length=512)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    subject_marker = '@subject'
    start_index = prediction.find(subject_marker)
    subject = ""
    if start_index != -1:
        # Extract the text after '@subject'
        start_index += len(subject_marker)
        subject = prediction[start_index:].strip()
        subject = subject.split('\n', 1)[0].strip()
    else:
        subject = "Failed to predict"
    print(subject)
    return {"prediction": subject}
