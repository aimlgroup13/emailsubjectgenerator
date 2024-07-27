import json
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
# Load the ROUGE metric
rouge = load("rouge")

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "nagthgr8/subject-mistrel", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        
    )
    model = model.to("cuda").eval()
    tokenizer_padding_side = "left"
    # email_prompt = Copied from above
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
with open('/home/ramch/AI-AUTOMATED-QA/testdataset.json', 'r') as f:
    testdataset = json.load(f)

# Define email prompt and EOS token
email_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

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
        "Phillip,   Could you please do me a favor?I would like  to read your current title policy to see what it says about easements.You  should have received a copy during your closing.I don't know how many  pages it will be but let me know how you want to handle getting a copy  made.I'll be happy to make the copy, or whatever makes it easy for  you.Thanks,", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
pred = tokenizer.batch_decode(outputs)
print(pred)
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def format_input(instruction, input_text, response_text=""):
    return email_prompt.format(instruction, input_text, response_text) + EOS_TOKEN

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

all_scores = []
print("Calculating the scores..")
# Use index-based access for Dataset objects
for i in range(len(testdataset[:100])):
    example = testdataset[i]  # Access element like a list
    instruction = example['instruction']
    input_text = example['input']
    references = example['output']  # List of references
    formatted_input = format_input(instruction, input_text)
    # Tokenize input text and generate prediction
    inputs = tokenizer(
    [
        email_prompt.format(
           "Please help summarize the provided email body and generate email subject", # instruction
           input_text,
           "", # output - leave this blank for generation!
        )
    ], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache = True)
    prediction = tokenizer.batch_decode(outputs)
    print("Input Text:")
    print(input_text)
    print("\nGenerated Output Tokens:")
    print(outputs[0])  # This prints the raw token IDs, which might not be very readable

    print("\nDecoded Prediction:")
    print(prediction)
    # Calculate ROUGE scores against each reference and store the best
    best_scores = {'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0}, 
                   'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0}, 
                   'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}}
    print(references)
    for ref in references:
        print("ref", ref)
        scores = scorer.score(prediction[0], ref)
        print(scores)
        for metric, score in scores.items():
            if score.fmeasure > best_scores[metric]['fmeasure']:
                best_scores[metric] = {'fmeasure': score.fmeasure, 
                                       'precision': score.precision, 
                                       'recall': score.recall}
    print(best_scores)
    all_scores.append(best_scores)

# Aggregate scores (e.g., calculate average)
avg_scores = {
    'rouge1': {'fmeasure': 0, 'precision': 0, 'recall': 0},
    'rouge2': {'fmeasure': 0, 'precision': 0, 'recall': 0},
    'rougeL': {'fmeasure': 0, 'precision': 0, 'recall': 0}
}
print("Calculating avg scores..")
for scores in all_scores:
    for metric in avg_scores:
        for key in ['fmeasure', 'precision', 'recall']:
            if key in scores[metric]:
                avg_scores[metric][key] += scores[metric][key]

for metric in avg_scores:
    for key in ['fmeasure', 'precision', 'recall']:
        avg_scores[metric][key] /= len(all_scores)

# Print the average ROUGE scores
print(avg_scores)

with open('mistral_avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
