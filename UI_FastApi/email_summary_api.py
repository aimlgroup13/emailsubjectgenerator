from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SubjectRequest(BaseModel):
    email_body: str
    model: str

@app.post("/generate_subject/")
def generate_subject(request: SubjectRequest):
    model_name = request.model.lower()
    if model_name == "bart":
        from bart_model import summarize_text
        subject = summarize_text(request.email_body)
    elif model_name == "mistral":
        subject = f"from mistral"
    elif model_name == "llama":
        subject = f"from llama"
    else:
        subject = f"from {model_name}"
    
    return {"subject": subject}