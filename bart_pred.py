import json
from transformers import BartForConditionalGeneration, BartTokenizer
model_name = "nagthgr8/subject-bart"

# Load the model
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load the tokenizer
tokenizer = BartTokenizer.from_pretrained(model_name)
model = model.to("cuda").eval()
email_prompt = """Below is an instruction that describes a task, paired with an input that provides further>

### Instruction:
{}

### Input:
{}

### Response:
{}"""
def predict_bart(input_data):
    print(input_data)
    inputs = tokenizer(
    [
    	email_prompt.format(
          "Please help summarize the provided email body and generate email subject", # instruction
          input_data,
          "",
    	)
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    pred = tokenizer.batch_decode(outputs)
    print(pred)
    return pred
