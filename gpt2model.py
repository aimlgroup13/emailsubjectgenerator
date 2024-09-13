from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load model and tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_path = "/home/ramch/AI-AUTOMATED-QA/saved-gpt2model"
#model_path = "/home/ramch/AI-AUTOMATED-QA/model-gpt2-july14"

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

#@app.post("/predict")
def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    if torch.cuda.is_available():
        inputs = inputs.to('cuda')
    outputs = model.generate(
       inputs['input_ids'], 
       attention_mask=inputs['attention_mask'],  # Include attention mask
       max_length=512, 
       pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
       no_repeat_ngram_size=3,  # To prevent repetition
       early_stopping=True  # To prevent excessive generation
    )
    subject_line = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    #print("Generated Subject:", subject_line)
    return  subject_line

email_text = """
Michelle Here are my very minor comments.
However we still need to wait on any additions, based on meeting with SME's today.
One concern is the firing of the learner who performs  bad in the final two scenarios.
Do we face any copyright issues using the CNN type themes?
In addition, I think we need to stay clear of anything that remotely seems like California or anything tha>#(i.e.So-cal Waha) In addition, comments on regulatory issues may be a problem (i.e.California Legislature).#Sheri  When you read all the scripts together and due to the similar mechanics being taught it appears ver>#Thus I do believe we need to maybe use a "Dateline" type theme for one, and a "60 Minute" type theme for a>#In the last two scenarios can we include a promotion out of the associate program for the stellar performe>#Cheers Kirk
"""

result = predict(email_text)
print(result)
