from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel  # Import BaseModel from pydantic
import os
import torch
import bart_pred2
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
# Configuration parameters
#model_dir = "sbtraining2020/esubjectgen_llama31_clean"
#tokenizer_file = f"{model_dir}/tokenizer.model"
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# Load the model and tokenizer
from unsloth import FastLanguageModel
#model, tokenizer = FastLanguageModel.from_pretrained(
#    model_name = model_dir,
#    max_seq_length = max_seq_length,
#    dtype = dtype,
#    load_in_4bit = load_in_4bit,
#)
#model = model.to("cuda").eval()
#tokenizer_padding_side = "left"
#FastLanguageModel.for_inference(model) # Enable native 2x faster inference

class InputData(BaseModel):
    model_name: str
    email_body: str

class ModelRequest(BaseModel):
    model_name: str

@app.post("/load_model")
async def load_model(request: ModelRequest):
    model_name = request.model_name
    model_dir = model_name
    tokenizer_file = f"{model_dir}/tokenizer.model"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        model = model.to("cuda").eval()
        tokenizer_padding_side = "left"
        FastLanguageModel.for_inference(model)

        # Store model and tokenizer in global variables
        models[model_name] = model
        tokenizers[model_name] = tokenizer

        return {"message": f"Model '{model_name}' loaded successfully!"}
    except Exception as e:
        raise e

# Serve the 'index.html' file when accessing the root URL
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Serve static files from the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

#@app.post("/predict/")
#async def predict(data: InputData):
#    # Format the input for the model
#    formatted_input = f"User: {data.email_body}\nAssistant:"
#    print("Getting inputs thru tokenizer...")
#    inputs = tokenizer(formatted_input, return_tensors="pt", truncation=True).to("cuda")  # Move inputs to GPU
#    print("Generating the outputs...")
#    # Generate output
#    with torch.no_grad():
#        outputs = model.generate(
#            input_ids=inputs["input_ids"],
#            attention_mask=inputs["attention_mask"],
#            max_new_tokens=64,
#            num_return_sequences=1,
#            eos_token_id=tokenizer.eos_token_id,
#            pad_token_id=tokenizer.pad_token_id,
#            use_cache=True  # Ensure cache usage is enabled
#        )
#    print("decoding outputs..")
#    # Decode the generated output
#    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#    return {"generated_subject": generated_text}

email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

#inputs = tokenizer(
#[
#	email_prompt.format(
#           "Please help summarize the provided email body and generate email subject", # instruction
#           "Phillip,   Could you please do me a favor?I would like  to read your current title policy to see what it says about easements.You  should have received a copy during your closing",
#           "", # output - leave this blank for generation!
#        )
#], return_tensors = "pt").to("cuda")
#outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
#pred = tokenizer.batch_decode(outputs)
#print(pred)

#@app.post("/predict/")
#async def predict(data: InputData):
#    print("Getting inputs through tokenizer...")
#    inputs = tokenizer(
#	[
#          email_prompt.format(
#             "Please help summarize the provided email body and generate email subject",
#             data.email_body,
#             ""
#          )
#	], return_tensors="pt").to("cuda")
#    print("Inputs:", inputs)
#    print("Generating the outputs...")
#    try:
#        outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
#        print("Decoding outputs...")
#        decoded_output = tokenizer.batch_decode(outputs)
#        return {"predicted_subject": decoded_output[0]}
#    except Exception as e:
#        print(f"An error occurred during generation: {e}")
#        raise e
@app.post("/predict/")
async def predict(data: InputData):
    model_name = data.model_name
    email_body = data.email_body
    print(model_name)
    # Check if the model is loaded
    if model_name != 'sbtraining2020/email_bart_1' and model_name not in models:
        # Load model if not already loaded
        model_load_response = await load_model(ModelRequest(model_name=model_name))
        if 'message' not in model_load_response:
            raise HTTPException(status_code=500, detail="Model loading failed")

    if model_name != 'sbtraining2020/email_bart_1':
        model = models[model_name]
        tokenizer = tokenizers[model_name]
        print("Getting inputs through tokenizer...")
        inputs = tokenizer(
          [
            email_prompt.format(
            "Please help summarize the provided email body and generate email subject",
            email_body,
            ""
            )
          ], return_tensors="pt").to("cuda")

        print("Inputs:", inputs)
        print("Generating the outputs...")
        try:
            outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
            print("Decoding outputs...")
            decoded_output = tokenizer.batch_decode(outputs)
            return {"predicted_subject": decoded_output[0]}
        except Exception as e:
            print(f"An error occurred during generation: {e}")
        raise e
    else:
        return {"predicted_subject": bart_pred2.predict_bart(email_body)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
