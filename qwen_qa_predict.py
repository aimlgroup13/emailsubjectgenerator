import json
from transformers import pipeline
from evaluate import load

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "sbtraining2020/qwen_qa1", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    model = model.to("cuda").eval()
    tokenizer_padding_side = "left"
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
with open('/home/ramch/AI-AUTOMATED-QA/testdataset.json', 'r') as f:
    testdataset = json.load(f)

# Define email prompt and EOS token
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def predict_answer(question, max_length=256):
    # Tokenize the question
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            "Please help summarize the provided email body and generate email subject", # instruction
            question, "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    # Decode the predictions
    decoded_output = tokenizer.batch_decode(outputs)
    return decoded_output

#inputs = tokenizer(
#[
#    alpaca_prompt.format(
#        "Please help summarize the provided email body and generate email subject", # instruction
#        "Are there any variants other than back propagation, that can replace Gradient Descent for neural networks?", # input
#        "", # output - leave this blank for generation!
#    )
#], return_tensors = "pt").to("cuda")

#outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
#pred = tokenizer.batch_decode(outputs)
#print(pred)
