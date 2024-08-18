import json
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import pipeline
from evaluate import load
from rouge_score import rouge_scorer
# Load the ROUGE metric
rouge = load("rouge")

model_name = "nagthgr8/subject-bart"

# Load the model
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load the tokenizer
tokenizer = BartTokenizer.from_pretrained(model_name)
model = model.to("cuda").eval()
with open('/home/ramch/AI-AUTOMATED-QA/testdataset.json', 'r') as f:
    testdataset = json.load(f)

# Define email prompt and EOS token
email_prompt = """Below is an instruction that describes a task, paired with an input that provides further>

### Instruction:
{}

### Input:
{}

### Response:
{}"""

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

with open('bart_avg_scores.json', 'w') as f:
    json.dump(avg_scores, f, indent=4)
