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
inputs = tokenizer(
[
    email_prompt.format(
        "Please help summarize the provided email body and generate email subject", # instruction
        "Phillip,   Could you please do me a favor?I would like  to read your current title policy to see what it says about easements.You  should have received a copy during your closing I don't know how many  pages it will be but let me know how you want to handle getting a copy  made.I'll be happy to make the copy, or whatever makes it easy for  you.Thanks,",
        "",
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
pred = tokenizer.batch_decode(outputs)
print(pred)
